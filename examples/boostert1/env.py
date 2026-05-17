from typing import Dict, List, Optional
import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

# ==============================================================================
# 严格的关节控制序列（必须与你训练模型或使用的数据集特征顺序完全一致）
# 依据你的 config.py，左右臂各包含 7 个核心关节。
# ==============================================================================
LEFT_ARM_JOINTS = [
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Wrist_Pitch",
    "Left_Wrist_Yaw",
    "Left_Hand_Roll"
]

RIGHT_ARM_JOINTS = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Right_Wrist_Pitch",
    "Right_Wrist_Yaw",
    "Right_Hand_Roll"
]

class Boostert1RealEnvironment(_environment.Environment):
    """基于 LeRobot 规范与 ROS2 异步节点构建的机器人推理环境适配器 (Follower 直接控制端)"""

    def __init__(
        self,
        robot,  # 接收从 main.py 传入的 BoosterT1AioRos2Robot 实例
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        self._robot = robot
        self._render_height = render_height
        self._render_width = render_width
        
        self.left_joints = LEFT_ARM_JOINTS
        self.right_joints = RIGHT_ARM_JOINTS
        
        # 动作空间维度：左臂(7) + 左夹爪(1) + 右臂(7) + 右夹爪(1) = 16 维
        # 💡 注意：如果你的推理模型使用的是标准的 Aloha 14维 (6+1+6+1) 策略，请务必根据模型的实际维度裁剪此处的数组结构。
        self.joint_dim = len(self.left_joints) + 1 + len(self.right_joints) + 1

    @override
    def reset(self) -> None:
        """每个 Episode 开始前的初始化逻辑"""
        # 如果你的底盘或机械臂有安全的初始待机姿态，可以在这里通过调用 robot 实例使其复位
        pass

    @override
    def is_episode_complete(self) -> bool:
        # 在线实时推理时，通常由外部或人工中断，此处默认返回 False
        return False

    @override
    def get_observation(self) -> dict:
        """
        从底层的 robot.py 中读取最新缓存的数据，并转换为 VLA 模型标准期望的 Numpy 格式
        """
        # 1. 调用你封装好的方法获取底层状态字典
        raw_obs = self._robot.get_observation()

        # 2. 图像处理管道 (对齐模型输入尺寸，执行 HWC -> CHW 转换)
        images = {}
        if "image_top" in raw_obs and raw_obs["image_top"] is not None:
            # 缩放并自动进行 Padding 处理
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(raw_obs["image_top"], self._render_height, self._render_width)
            )
            # 调整维度顺序以符合大模型对 Tensor 的输入期望
            images["cam_high"] = einops.rearrange(img, "h w c -> c h w")
        else:
            raise RuntimeError("错误: 未能在状态字典中获取到 'image_top' 顶置相机的图像。")

        # 3. 组装 1D 状态向量 (State Vector)
        state = np.zeros(self.joint_dim, dtype=np.float32)
        idx = 0
        
        # 填充左臂 7 轴
        for joint in self.left_joints:
            state[idx] = raw_obs.get(f"follower_{joint}.pos", 0.0)
            idx += 1
            
        # 填充左夹爪反馈状态（暂时用灵巧手的大拇指位置替代反馈，后续可在 robot.py 中细化映射）
        state[idx] = raw_obs.get("follower_left_thumb.pos", 0.0)
        idx += 1

        # 填充右臂 7 轴
        for joint in self.right_joints:
            state[idx] = raw_obs.get(f"follower_{joint}.pos", 0.0)
            idx += 1
            
        # 填充右夹爪反馈状态
        state[idx] = raw_obs.get("follower_right_thumb.pos", 0.0)
        idx += 1

        return {
            "state": state,
            "images": images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        """
        接收大模型推理输出的 1D Action 数组，将其转化为底层能够识别的 Follower 关节名字并下发
        """
        model_action = action["actions"]  # 获取大模型输出的 1D 动作向量 (np.ndarray)
        
        action_dict = {}
        idx = 0
        
        # 1. 拆解左臂 7 轴目标姿态
        for joint in self.left_joints:
            action_dict[f"follower_{joint}.pos"] = float(model_action[idx])
            idx += 1
            
        # 2. 拆解左夹爪目标信号 (通常为 0.0~1.0 归一化值)
        action_dict["follower_left_gripper.pos"] = float(model_action[idx])
        idx += 1
        
        # 3. 拆解右臂 7 轴目标姿态
        for joint in self.right_joints:
            action_dict[f"follower_{joint}.pos"] = float(model_action[idx])
            idx += 1
            
        # 4. 拆解右夹爪目标信号
        action_dict["follower_right_gripper.pos"] = float(model_action[idx])
        idx += 1

        # 将翻译好的 Follower 目标姿态字典通过机器人实例下发
        self._robot.send_action(action_dict)