from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("boostert1_aio_ros2")
@dataclass
class BoosterT1AioRos2RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    # 只放 VR 主手：这些会进入 action
    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
        }
    )

    # 机器人本体状态：进入 observation.state
    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "follower_joint_states": {
                "Left_Shoulder_Pitch": Motor(1, "robot_motor", norm_mode_body),
                "Left_Shoulder_Roll": Motor(2, "robot_motor", norm_mode_body),
                "Left_Elbow_Pitch": Motor(3, "robot_motor", norm_mode_body),
                "Left_Elbow_Yaw": Motor(4, "robot_motor", norm_mode_body),
                "Left_Wrist_Pitch": Motor(5, "robot_motor", norm_mode_body),
                "Left_Wrist_Yaw": Motor(6, "robot_motor", norm_mode_body),
                "Left_Hand_Roll": Motor(7, "robot_motor", norm_mode_body),
                "left_thumb": Motor(8, "robot_motor", norm_mode_body),

                "Right_Shoulder_Pitch": Motor(9, "robot_motor", norm_mode_body),
                "Right_Shoulder_Roll": Motor(10, "robot_motor", norm_mode_body),
                "Right_Elbow_Pitch": Motor(11, "robot_motor", norm_mode_body),
                "Right_Elbow_Yaw": Motor(12, "robot_motor", norm_mode_body),
                "Right_Wrist_Pitch": Motor(13, "robot_motor", norm_mode_body),
                "Right_Wrist_Yaw": Motor(14, "robot_motor", norm_mode_body),
                "Right_Hand_Roll": Motor(15, "robot_motor", norm_mode_body),
                "right_thumb": Motor(16, "robot_motor", norm_mode_body),
            },
        }
    )


    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(default_factory=lambda: {})