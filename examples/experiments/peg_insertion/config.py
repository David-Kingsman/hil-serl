import os
import jax
import jax.numpy as jnp
import numpy as np

from serl_robot_infra.ur_env.envs.ur30_env import DefaultEnvConfig  
from serl_robot_infra.ur_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
) 
from serl_robot_infra.ur_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper 
from serl_launcher.wrappers.chunking import ChunkingWrapper 
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.peg_insertion.wrapper import PEGEnv


class EnvConfig(DefaultEnvConfig):
    # robot config
    ROBOT_IP: str = "172.22.22.2" 
    CONTROLLER_HZ: int = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])
    
    # task related pose definitions
    TARGET_POSE = np.array([0.5, 0.0, 0.3, np.pi, 0, 0])  # peg insertion target pose, need to be changed
    GRASP_POSE = np.array([0.4, 0.0, 0.25, np.pi, 0, 0])  # peg grasp pose, need to be changed
    RESET_POSE = np.array([0.5, 0.0, 0.35, np.pi, 0, 0])  # reset pose, need to be changed
    
    # joint reset configuration
    RESET_Q = np.array([[-1.771, -1.943, 2.005, -3.244, -1.597, -1.5412]])
    RANDOM_RESET = False
    RANDOM_POSITION_RANGE = (0.05, 0.05, 0.05)
    RANDOM_ROT_RANGE = (0.03,)
    
    # workspace limits
    ABS_POSE_LIMIT_HIGH = np.array([0.1, 0.68, 0.75, 0.15, 0.2, 0.1])
    ABS_POSE_LIMIT_LOW = np.array([-0.1, 0.4, 0.55, -0.15, -0.2, -0.1])
    ABS_POSE_RANGE_LIMITS = np.array([0.4, 0.8])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)
    
    # UR impedance control parameters (hardcoded in ur30_controller.py, here not needed)
    # UR impedance control through the following parameters:
    # - kp, kd: set in ur30_controller.py
    # - ERROR_DELTA: error limit
    # - FORCEMODE_DAMPING: force mode damping

    
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_2": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_state"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = PEGEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
            camera_mode="rgb",
            max_episode_length=100,
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                # obs['state'][0, 2] is z coordinate (tcp_pose[2])
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 2] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env