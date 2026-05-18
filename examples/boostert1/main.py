import dataclasses
import logging
import rclpy # 新增

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

# 导入你自己的 robot 和 config
from examples.boostert1 import env as _env
from examples.boostert1.config import BoosterT1AioRos2RobotConfig
from examples.boostert1.robot import BoosterT1AioRos2Robot

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    action_horizon: int = 25
    num_episodes: int = 1
    max_episode_steps: int = 1000
    prompt: str = "do something"

def main(args: Args) -> None:
    # 1. 初始化 ROS2
    rclpy.init()

    try:
        # 2. 实例化并连接你的真实机器人
        config = BoosterT1AioRos2RobotConfig()
        robot = BoosterT1AioRos2Robot(config)
        
        # 启动 ROS2 节点线程
        robot.robot_ros2_node.start()
        logging.info("Connecting to robot...")
        robot.connect()
        logging.info("Robot connected successfully!")

        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

        # 3. 将 robot 实例传入你的 Env
        runtime = _runtime.Runtime(
            environment=_env.Boostert1RealEnvironment(
                robot=robot, # 传入机器人实例
                render_height=224, 
                render_width=224,
                prompt=args.prompt
            ),
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[],
            max_hz=50,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

        runtime.run()
        
    finally:
        # 4. 优雅退出
        logging.info("Shutting down robot and ROS2...")
        if 'robot' in locals():
            robot.robot_ros2_node.stop()
            robot.disconnect()
        rclpy.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)