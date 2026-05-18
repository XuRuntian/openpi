import time
import logging_mp
import numpy as np
import rclpy

from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config import BoosterT1AioRos2RobotConfig
from .node import BoosterT1AioRos2Node


logger = logging_mp.get_logger(__name__)


class BoosterT1AioRos2Robot(Robot):
    config_class = BoosterT1AioRos2RobotConfig
    name = "boostert1_aio_ros2"

    def __init__(self, config: BoosterT1AioRos2RobotConfig):

        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones


        self.leader_motors = config.leader_motors
        self.follower_motors = config.follower_motors["follower_joint_states"]
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras = ["image_pika_pose"]

        # self.status = GalaxeaLiteEEposeROS2RobotStatus()

        self.robot_ros2_node = BoosterT1AioRos2Node()
        # self.robot_ros2_node.start()

        self.connected = False
        self.logs = {}

    # ========= features =========

    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {
            f"follower_{motor}.pos": float
            for motor in self.follower_motors
        }
    

    @property
    def _leader_motors_ft(self) -> dict[str, type]:
        return {
            f"leader_{joint}.pos": float
            for _comp_name, joints in self.leader_motors.items()
            for joint in joints.keys()
        }



    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (
                self.config.cameras[cam].height,
                self.config.cameras[cam].width,
                3,
            )
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        return {**self._follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return self._leader_motors_ft

    @property
    def is_connected(self) -> bool:
        return self.connected

    # ========= connect / disconnect =========

    def connect(self):
        timeout = 5
        start_time = time.perf_counter()

        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 约定：node 里有 recv_images / recv_follower / recv_leader
        conditions = [
            # 摄像头图像
            (
                lambda: all(
                    name in self.robot_ros2_node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras
                ),
                lambda: [
                    name
                    for name in self.cameras
                    if name not in self.robot_ros2_node.recv_images
                    and name not in self.connect_excluded_cameras
                ],
                "等待摄像头图像超时",
            ),
            (
                lambda: all(
                    (comp_name in self.robot_ros2_node.recv_leader) and
                    (
                        # vec: np.ndarray / list / tuple
                        (hasattr(self.robot_ros2_node.recv_leader[comp_name], "__len__") and
                        len(self.robot_ros2_node.recv_leader[comp_name]) >= len(joints))
                    )
                    for comp_name, joints in self.leader_motors.items()
                ),
                lambda: sorted([
                    # 缺组件 or 长度不够都算 missing（展开到 joint 名方便你看）
                    joint_name
                    for comp_name, joints in self.leader_motors.items()
                    for joint_name in joints.keys()
                    if (
                        (comp_name not in self.robot_ros2_node.recv_leader) or
                        (not hasattr(self.robot_ros2_node.recv_leader[comp_name], "__len__")) or
                        (len(self.robot_ros2_node.recv_leader[comp_name]) < len(joints))
                    )
                ]),
                "等待主臂数据超时",
            ),



            # 从臂
            (
                lambda: all(
                    any(name in key for _follower_name, follower in self.robot_ros2_node.recv_follower.items() for key in follower)
                    for name in self.follower_motors
                ),
                lambda: [
                    name
                    for name in self.follower_motors
                    if not any(name in key for _follower_name, follower in self.robot_ros2_node.recv_follower.items() for key in follower)
                ],
                "等待从臂数据超时",
            ),
        ]

        completed = [False] * len(conditions)

        while True:
            for i, (cond, _get_missing, _msg) in enumerate(conditions):
                if not completed[i] and cond():
                    completed[i] = True

            if all(completed):
                break

            if time.perf_counter() - start_time > timeout:
                failed_messages = []
                for i, (cond, get_missing, base_msg) in enumerate(conditions):
                    if completed[i]:
                        continue

                    missing = get_missing()
                    if cond() or not missing:
                        completed[i] = True
                        continue

                    if i == 0:
                        received = [
                            name
                            for name in self.cameras
                            if name not in missing
                        ]
                    elif i == 1:
                        received = [
                            name
                            for name in self.leader_motors
                            if name not in missing
                        ]
                    else:
                        received = [
                            name
                            for name in self.follower_motors
                            if name not in missing
                        ]

                    msg = (
                        f"{base_msg}: 未收到 [{', '.join(missing)}]; "
                        f"已收到 [{', '.join(received)}]"
                    )
                    failed_messages.append(msg)

                if not failed_messages:
                    break

                raise TimeoutError(
                    f"连接超时，未满足的条件: {'; '.join(failed_messages)}"
                )


            time.sleep(0.01)

        # 成功日志
        success_messages = []

        if conditions[0][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in self.robot_ros2_node.recv_images
                and name not in self.connect_excluded_cameras
            ]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        if conditions[1][0]():
            leader_received = []
            for comp_name, joints in self.leader_motors.items():
                if comp_name not in self.robot_ros2_node.recv_leader:
                    continue
                vec = self.robot_ros2_node.recv_leader[comp_name]
                if hasattr(vec, "__len__") and len(vec) >= len(joints):
                    leader_received.append(comp_name)

            success_messages.append(f"主臂数据: {', '.join(leader_received)}")

        if conditions[2][0]():
            follower_received = [
                name
                for name in self.follower_motors
                if any(name in key for _follower_name, follower in self.robot_ros2_node.recv_follower.items() for key in follower)
            ]
            success_messages.append(f"从臂数据: {', '.join(follower_received)}")

        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)

        self.connected = True
    
    def get_node(self):
        return self.robot_ros2_node

    def disconnect(self):
        if not self.connected:
            raise DeviceNotConnectedError()
        self.connected = False

    def __del__(self):
        if getattr(self, "connected", False):
            self.disconnect()

    # ========= calibrate / configure =========

    def calibrate(self):
        pass

    def configure(self):
        pass

    @property
    def is_calibrated(self):
        return True

    # ========= obs / action =========

    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict: dict[str, Any] = {}

        # for name in self.follower_motors:
        #     for _follower_name, follower in self.robot_ros2_node.recv_follower.items():
        #         for key, value in follower.items():
        #             if name == key:
        #                 obs_dict[f"follower_{name}.pos"] = float(value)

        # ---- 逐组件展开，然后逐 joint 填入 ----
        joint_map_all = {}
        for _comp_name, joint_map in self.robot_ros2_node.recv_follower.items():
            if isinstance(joint_map, dict):
                joint_map_all.update(joint_map)

        missing = []
        for joint_name in self.follower_motors.keys():   # AAHead_yaw, Head_pitch, ...
            val = None

            # 1) 先精确匹配
            if joint_name in joint_map_all:
                val = joint_map_all[joint_name]
            else:
                # 2) 再做一次“子串匹配”（有些 joint_states 可能带前后缀）
                for k, v in joint_map_all.items():
                    if joint_name in k:
                        val = v
                        break

            if val is None:
                missing.append(joint_name)
                obs_dict[f"follower_{joint_name}.pos"] = float("nan")  
            else:
                if joint_name in ["left_thumb", "right_thumb"]:
                    val = (float(val) - 200.0) / 600.0
                    val = max(0.0, min(1.0, val))
                obs_dict[f"follower_{joint_name}.pos"] = float(val)

        if missing:
            logger.warning(f"Missing follower joints in /joint_states: {missing[:8]}"
                        + (f" (+{len(missing)-8} more)" if len(missing) > 8 else ""))

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")

        # ---- 摄像头图像保持不变 ----
        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            for name, val in self.robot_ros2_node.recv_images.items():
                if cam_key == name or cam_key in name:
                    obs_dict[cam_key] = val
                    break
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return obs_dict
    
    def get_action(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        act_dict: dict[str, Any] = {}

        # for name in self.leader_motors:
        #     for _leader_name, leader in self.robot_ros2_node.recv_leader.items():
        #         for key, value in leader.items():
        #             if name == key:
        #                 act_dict[f"leader_{name}.pos"] = float(value)

        # ---- 逐组件展开，然后逐 joint 填入 ----
        for comp_name, joints in self.leader_motors.items():

            # node 中按组件名存放 leader 数据，可能是 vector，也可能是 JointState 展开的 dict
            data = self.robot_ros2_node.recv_leader.get(comp_name)
            if data is None:
                continue

            joint_names = list(joints.keys())

            if isinstance(data, dict):
                for joint in joint_names:
                    val = None
                    if joint in data:
                        val = data[joint]
                    else:
                        for k, v in data.items():
                            if joint in k:
                                val = v
                                break
                    if val is not None:
                        act_dict[f"leader_{joint}.pos"] = float(val)
                continue

            for idx, joint in enumerate(joint_names):
                if idx >= len(data):
                    break
                act_dict[f"leader_{joint}.pos"] = float(data[idx])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f} ms")

        return act_dict

    # ========= send_action =========

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """接收上层环境传来的干净 action 字典，通过中层代理话题发送给机器人"""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )

        # 1. 动态初始化中层控制话题的发布器（若未初始化）
        if not hasattr(self.robot_ros2_node, "pub_q14"):
            from std_msgs.msg import Float32, Float64MultiArray
            
            self.robot_ros2_node.pub_q14 = self.robot_ros2_node.create_publisher(
                Float64MultiArray, "/ik/dual_arm_q14", 10
            )
            self.robot_ros2_node.pub_left_gripper = self.robot_ros2_node.create_publisher(
                Float32, "/vr/left_gripper", 10
            )
            self.robot_ros2_node.pub_right_gripper = self.robot_ros2_node.create_publisher(
                Float32, "/vr/right_gripper", 10
            )
            
            # 定义严格对应遥操作驱动的 14 轴顺序
            self._q14_names = [
                "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
                "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
                "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
                "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll"
            ]

        from std_msgs.msg import Float32, Float64MultiArray

        # 2. 组装并发送 14 轴双臂姿态（修复键名不匹配问题）
        q14_values = []
        for name in self._q14_names:
            # 先尝试原始键名，再尝试LeRobot标准的follower_*.pos格式
            val = action.get(name, 0.0)
            if val == 0.0:
                val = action.get(f"follower_{name}.pos", 0.0)
            q14_values.append(float(val))
        
        # 调试打印：确认发送的关节值正确
        print(f"📤 发送关节指令: {[f'{v:.4f}' for v in q14_values[:3]]}...")

        msg_q14 = Float64MultiArray()
        msg_q14.data = q14_values
        self.robot_ros2_node.pub_q14.publish(msg_q14)

        # 3. 处理左夹爪发送 (将 0.0~1.0 映射到 0~1000)
        left_gripper_val = 0.0
        if "left_gripper" in action:
            left_gripper_val = action["left_gripper"]
        elif "follower_left_gripper.pos" in action:
            left_gripper_val = action["follower_left_gripper.pos"]
        
        msg_lg = Float32()
        target_pos = max(0.0, min(1000.0, left_gripper_val * 1000.0))
        msg_lg.data = float(target_pos)
        self.robot_ros2_node.pub_left_gripper.publish(msg_lg)

        # 4. 处理右夹爪发送 (将 0.0~1.0 映射到 0~1000)
        right_gripper_val = 0.0
        if "right_gripper" in action:
            right_gripper_val = action["right_gripper"]
        elif "follower_right_gripper.pos" in action:
            right_gripper_val = action["follower_right_gripper.pos"]
        
        msg_rg = Float32()
        target_pos = max(0.0, min(1000.0, right_gripper_val * 1000.0))
        msg_rg.data = float(target_pos)
        self.robot_ros2_node.pub_right_gripper.publish(msg_rg)

        # 保持原接口返回值不变，避免LeRobot内部记录Log时中断
        return {f"{arm_motor}.pos": val for arm_motor, val in action.items()}