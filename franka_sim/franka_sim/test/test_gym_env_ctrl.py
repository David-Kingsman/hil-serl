# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np

import imageio
from franka_sim import envs
import gymnasium as gym

# import joystick wrapper
from franka_env.envs.wrappers import JoystickIntervention, KbdIntervention
from franka_env.spacemouse.spacemouse_expert import ControllerType

from franka_sim.utils.viewer_utils import DualMujocoViewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="xbox", help="Controller type. xbox|ps5")

    args = parser.parse_args()
    controller_type = ControllerType[args.controller.upper()]

# env = envs.PandaPickCubeGymEnv(render_mode="human", image_obs=True)
env = gym.make("PandaPickCubeVision-v0", render_mode="human", image_obs=True)

env = KbdIntervention(env)

env.reset()
m = env.unwrapped.model
d = env.unwrapped.data

# Create the dual viewer
dual_viewer = DualMujocoViewer(env.unwrapped.model, env.unwrapped.data)

frames1 = []
frames2 = []


# 初始化视频写入器
front_video_writer = imageio.get_writer(
    "/home/agilex/chenjin/hil-serl-sim/video/franka_lift_cube_render_front.mp4", fps=200
)
wrist_video_writer = imageio.get_writer(
    "/home/agilex/chenjin/hil-serl-sim/video/franka_lift_cube_render_wrist.mp4", fps=200
)

with dual_viewer as viewer:
    for i in range(100000):
        obs, rew, done, truncated, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        images = obs["images"]

        # 获取当前帧
        front_frame = np.array(images["front"])
        wrist_frame = np.array(images["wrist"])

        # 将当前帧写入对应的视频文件
        front_video_writer.append_data(front_frame)
        wrist_video_writer.append_data(wrist_frame)

        viewer.sync()

# 关闭视频写入器
front_video_writer.close()
wrist_video_writer.close()