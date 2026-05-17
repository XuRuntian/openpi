# robodriver_robot_boostert1_aio_ros2/node.py

import threading
from typing import Dict, Any, Optional

import cv2
import numpy as np
import logging_mp
import rclpy

from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from std_msgs.msg import Header, Float32, Float32MultiArray
from sensor_msgs.msg import Image, JointState, CompressedImage
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    qos_profile_sensor_data,
)

logger = logging_mp.get_logger(__name__)

CONNECT_TIMEOUT_FRAME = 10

QOS_BEST_EFFORT = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# 由生成脚本根据 JSON 自动生成
NODE_CONFIG = {

    "follower_joint_topics": {
        "follower_joint_states": {
            "topic": "/joint_states",
            "msg": "JointState"
        },
        "follower_dexhand": {
            "topic": "/dexhand_state",
            "msg": "JointState"
        },
    },

    "camera_topics": {
        "image_top": {
            "topic": "/head_camera/image_raw",
            "msg": "Image"
        }
    }
}


class BoosterT1AioRos2Node(Node):
    """
    ROS2 → 本地缓存
    - leader: Float32MultiArray / Float32 -> np.ndarray
    - follower: JointState -> dict(name->position)
    - camera: Image/CompressedImage -> RGB np.ndarray
    """

    def __init__(
        self,
        leader_joint_topics: Optional[Dict[str, Dict[str, str]]] = None,
        follower_joint_topics: Optional[Dict[str, Dict[str, str]]] = None,
        camera_topics: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        super().__init__("boostert1_aio_ros2_direct")

        self.leader_joint_cfgs = leader_joint_topics or NODE_CONFIG["leader_joint_topics"]
        self.follower_joint_cfgs = follower_joint_topics or NODE_CONFIG["follower_joint_topics"]
        self.camera_cfgs = camera_topics or NODE_CONFIG["camera_topics"]

        if not self.leader_joint_cfgs:
            raise RuntimeError("leader_joint_topics is empty")
        if not self.follower_joint_cfgs:
            raise RuntimeError("follower_joint_topics is empty")

        # 缓存
        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}

        self.recv_follower: Dict[str, Any] = {}
        self.recv_follower_status: Dict[str, int] = {}

        self.recv_leader: Dict[str, Any] = {}
        self.recv_leader_status: Dict[str, int] = {}

        self.lock = threading.Lock()
        self.running = False

        # ---------- follower subscriptions ----------
        for comp_name, cfg in self.follower_joint_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "JointState")

            if msg_name == "JointState":
                msg_cls = JointState
                callback = lambda msg, cname=comp_name: self._joint_callback_follower(cname, msg)
            elif msg_name == "Pose":
                msg_cls = Pose
                callback = lambda msg, cname=comp_name: self._pose_callback_follower(cname, msg)
            elif msg_name == "Odometry":
                msg_cls = Odometry
                callback = lambda msg, cname=comp_name: self._odom_callback_follower(cname, msg)
            elif msg_name == "Float32MultiArray":
                msg_cls = Float32MultiArray
                expect_len = int(cfg.get("len", 0))
                callback = lambda msg, cname=comp_name, el=expect_len: self._f32marray_callback_follower(cname, msg, el)
            elif msg_name == "Float32":
                msg_cls = Float32
                callback = lambda msg, cname=comp_name: self._f32_callback_follower(cname, msg)
            else:
                raise RuntimeError(f"Unsupported follower msg type: {msg_name}")

            # follower 强烈建议 BEST_EFFORT，避免 QoS incompatible
            self.create_subscription(msg_cls, topic, callback, QOS_BEST_EFFORT)
            logger.info(f"[Direct] Follower subscriber '{comp_name}' at {topic} ({msg_name})")

        # ---------- leader subscriptions ----------
        for comp_name, cfg in self.leader_joint_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "JointState")

            if msg_name == "JointState":
                msg_cls = JointState
                callback = lambda msg, cname=comp_name: self._joint_callback_leader(cname, msg)
            elif msg_name == "Pose":
                msg_cls = Pose
                callback = lambda msg, cname=comp_name: self._pose_callback_leader(cname, msg)
            elif msg_name == "Odometry":
                msg_cls = Odometry
                callback = lambda msg, cname=comp_name: self._odom_callback_leader(cname, msg)
            elif msg_name == "Float32MultiArray":
                msg_cls = Float32MultiArray
                expect_len = int(cfg.get("len", 0))
                callback = lambda msg, cname=comp_name, el=expect_len: self._f32marray_callback_leader(cname, msg, el)
            elif msg_name == "Float32":
                msg_cls = Float32
                callback = lambda msg, cname=comp_name: self._f32_callback_leader(cname, msg)
            else:
                raise RuntimeError(f"Unsupported leader msg type: {msg_name}")

            self.create_subscription(msg_cls, topic, callback, QOS_BEST_EFFORT)
            logger.info(f"[Direct] Leader subscriber '{comp_name}' at {topic} ({msg_name})")

        # （保留你原来的发布器：如果你确实要发 /joint_states）
        self.pub_action_joint_states = self.create_publisher(JointState, "/joint_states", 10)

        # ---------- camera subscriptions ----------
        self.camera_subs = []
        for cam_name, cfg in self.camera_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "Image")

            if msg_name == "Image":
                sub = self.create_subscription(
                    Image,
                    topic,
                    lambda msg, cname=cam_name: self._image_callback(cname, msg),
                    qos_profile_sensor_data,
                )
            elif msg_name == "CompressedImage":
                sub = self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda msg, cname=cam_name: self._compressed_image_callback(cname, msg),
                    qos_profile_sensor_data,
                )
            else:
                raise RuntimeError(f"Unsupported camera msg type: {msg_name}")

            self.camera_subs.append(sub)
            logger.info(f"[Direct] Camera '{cam_name}' subscribed: {topic} ({msg_name})")

        # executor for spin thread
        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None

        logger.info("[Direct] READY (ROS2 subscriptions created).")

    # ======================
    # callbacks: camera
    # ======================

    def _image_callback(self, cam_name: str, msg: Image):
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            h, w = msg.height, msg.width
            encoding = (msg.encoding or "").lower()

            frame = None
            if encoding == "bgr8":
                frame = data.reshape((h, w, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif encoding == "rgb8":
                frame = data.reshape((h, w, 3))
            else:
                # 不认识就直接忽略，避免异常刷屏
                return

            with self.lock:
                self.recv_images[cam_name] = frame
                self.recv_images_status[cam_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Image callback error ({cam_name}): {e}")

    def _compressed_image_callback(self, cam_name: str, msg: CompressedImage):
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                logger.error(f"CompressedImage decode failed: {cam_name}, format={msg.format}")
                return
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            with self.lock:
                self.recv_images[cam_name] = frame
                self.recv_images_status[cam_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"CompressedImage callback error ({cam_name}): {e}")

    # ======================
    # callbacks: JointState
    # ======================

    def _joint_callback_follower(self, comp_name: str, msg: JointState):
        try:
            with self.lock:
                self.recv_follower[comp_name] = {
                    name: pos for name, pos in zip(msg.name, msg.position)
                }
                self.recv_follower_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Joint callback error (follower:{comp_name}): {e}")

    def _joint_callback_leader(self, comp_name: str, msg: JointState):
        try:
            with self.lock:
                self.recv_leader[comp_name] = {
                    name: pos for name, pos in zip(msg.name, msg.position)
                }
                self.recv_leader_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Joint callback error (leader:{comp_name}): {e}")

    # ======================
    # callbacks: Float32 / Float32MultiArray
    # ======================

    def _f32marray_callback_follower(self, comp_name: str, msg: Float32MultiArray, expect_len: int = 0):
        try:
            data = list(msg.data) if msg.data is not None else []
            if expect_len > 0 and len(data) != expect_len:
                logger.warning(f"[follower:{comp_name}] Float32MultiArray len mismatch: got {len(data)} expect {expect_len}")
            with self.lock:
                self.recv_follower[comp_name] = np.array(data, dtype=float)
                self.recv_follower_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Float32MultiArray callback error (follower:{comp_name}): {e}")

    def _f32marray_callback_leader(self, comp_name: str, msg: Float32MultiArray, expect_len: int = 0):
        try:
            data = list(msg.data) if msg.data is not None else []
            if expect_len > 0 and len(data) != expect_len:
                logger.warning(f"[leader:{comp_name}] Float32MultiArray len mismatch: got {len(data)} expect {expect_len}")
            with self.lock:
                self.recv_leader[comp_name] = np.array(data, dtype=float)
                self.recv_leader_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Float32MultiArray callback error (leader:{comp_name}): {e}")

    def _f32_callback_follower(self, comp_name: str, msg: Float32):
        try:
            with self.lock:
                # 统一成 vector，避免 len(float) 崩
                self.recv_follower[comp_name] = np.array([float(msg.data)], dtype=float)
                self.recv_follower_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Float32 callback error (follower:{comp_name}): {e}")

    def _f32_callback_leader(self, comp_name: str, msg: Float32):
        try:
            with self.lock:
                # 统一成 vector，避免 len(float) 崩
                self.recv_leader[comp_name] = np.array([float(msg.data)], dtype=float)
                self.recv_leader_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Float32 callback error (leader:{comp_name}): {e}")

    # ======================
    # callbacks: Pose / Odom (保留)
    # ======================

    def _pose_callback_follower(self, comp_name: str, msg: Pose):
        try:
            vec = np.array(
                [msg.position.x, msg.position.y, msg.position.z,
                 msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                dtype=float,
            )
            with self.lock:
                self.recv_follower[comp_name] = vec
                self.recv_follower_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Pose callback error (follower:{comp_name}): {e}")

    def _pose_callback_leader(self, comp_name: str, msg: Pose):
        try:
            vec = np.array(
                [msg.position.x, msg.position.y, msg.position.z,
                 msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                dtype=float,
            )
            with self.lock:
                self.recv_leader[comp_name] = vec
                self.recv_leader_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Pose callback error (leader:{comp_name}): {e}")

    def _odom_callback_follower(self, comp_name: str, msg: Odometry):
        try:
            vec = np.array(
                [
                    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
                    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                    msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z,
                ],
                dtype=float,
            )
            with self.lock:
                self.recv_follower[comp_name] = vec
                self.recv_follower_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Odometry callback error (follower:{comp_name}): {e}")

    def _odom_callback_leader(self, comp_name: str, msg: Odometry):
        try:
            vec = np.array(
                [
                    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
                    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                    msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z,
                ],
                dtype=float,
            )
            with self.lock:
                self.recv_leader[comp_name] = vec
                self.recv_leader_status[comp_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Odometry callback error (leader:{comp_name}): {e}")

    # ======================
    # publisher helper
    # ======================

    def ros2_send(self, action: Dict[str, Any]):
        """
        action: {joint_name: value, ...}
        这里仍然发布到 /joint_states（保留你的原逻辑）
        """
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(action.keys())
        msg.position = [float(v) for v in action.values()]
        msg.velocity = []
        msg.effort = []
        self.pub_action_joint_states.publish(msg)

    # ======================
    # spin control
    # ======================

    def start(self):
        if self.running:
            return
        self.running = True

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)

        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        logger.info("[ROS2] Node started (executor spin thread running)")

    def _spin_loop(self):
        try:
            while self.running and rclpy.ok():
                # 小超时，避免卡死，便于 stop
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as e:
            logger.error(f"[ROS2] Spin error: {e}")

    def stop(self):
        if not self.running:
            return
        self.running = False

        # 不要 rclpy.shutdown()（避免整个进程后续无法创建 future）
        try:
            if self._executor is not None:
                try:
                    self._executor.remove_node(self)
                except Exception:
                    pass
                try:
                    self._executor.shutdown()
                except Exception:
                    pass
        finally:
            self._executor = None

        try:
            self.destroy_node()
        except Exception:
            pass

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None

        logger.info("[ROS2] Node stopped.")
