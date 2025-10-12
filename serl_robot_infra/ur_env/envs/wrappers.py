import gymnasium as gym
import numpy as np
from agentlace import action
import time
from typing import Tuple, List
from gymnasium.spaces import Box

from ur_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from ur_env.spacemouse.fake_spacemouse import FakeSpaceMouseExpert
from scipy.spatial.transform import Rotation as R

from ur_env.utils.rotations import quat_2_euler, quat_2_mrp, omega_to_mrp_dot

# 添加sigmoid函数
sigmoid = lambda x: 1 / (1 + np.exp(-x))

ROT90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
ROT_GENERAL = np.array([np.eye(3), ROT90, ROT90 @ ROT90, ROT90.transpose()])


class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, gripper_action_span=3, device_number: int = 0):
        super().__init__(env)
        self.gripper_enabled = True

        try:
            self.expert = SpaceMouseExpert(device_number=device_number)
        except Exception as e:
            self.expert = FakeSpaceMouseExpert()
            print(f"openend fake SpacemouseExpert since: {e}")

        self.last_intervene = 0
        self.left = np.array([False] * gripper_action_span, dtype=np.bool_)
        self.right = self.left.copy()

        self.invert_axes = [-1, -1, 1, -1, -1, 1]
        self.deadspace = 0.15

    def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a = self.get_deadspace_action()

        if np.linalg.norm(
                expert_a) > 0.001 or self.left.any() or self.right.any():  # also read buttons with no movement
            self.last_intervene = time.time()

        if self.gripper_enabled:
            gripper_action = np.zeros((1,)) + int(self.left.any()) - int(self.right.any())
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if time.time() - self.last_intervene < 0.5:
            # expert_a = self.adapt_spacemouse_output(expert_a)
            return expert_a, True

        return action, False

    def get_deadspace_action(self) -> np.ndarray:
        expert_a, buttons = self.expert.get_action()

        positive = np.clip((expert_a - self.deadspace) / (1. - self.deadspace), a_min=0.0, a_max=1.0)
        negative = np.clip((expert_a + self.deadspace) / (1. - self.deadspace), a_min=-1.0, a_max=0.0)
        expert_a = positive + negative

        self.left, self.right = np.roll(self.left, -1), np.roll(self.right, -1)  # shift them one to the left
        self.left[-1], self.right[-1] = tuple(buttons)

        return np.array(expert_a, dtype=np.float32)

    def adapt_spacemouse_output(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - expert_a: spacemouse raw output
        Output:
        - expert_a: spacemouse output adapted to force space (action)
        """

        # position = super().get_wrapper_attr("curr_pos")  # get position from ur_env
        position = self.unwrapped.curr_pos
        z_angle = np.arctan2(position[1], position[0])  # get first joint angle

        z_rot = R.from_rotvec(np.array([0, 0, z_angle]))
        action[:6] *= self.invert_axes  # if some want to be inverted
        action[:3] = z_rot.apply(action[:3])  # z rotation invariant translation

        # TODO add tcp orientation to the equation (extract z rotation from tcp pose)
        action[3:6] = z_rot.apply(action[3:6])  # z rotation invariant rotation

        return action

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left.any()
        info["right"] = self.right.any()
        return obs, rew, done, truncated, info


class DualSpaceMouseIntervention(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.expert_left = SpacemouseIntervention(env, device_number=1)
        self.expert_right = SpacemouseIntervention(env, device_number=4)

    def step(self, action):
        action_left = action[:7]
        action_right = action[7:]

        new_action_left, replaced_left = self.expert_left.action(action_left)
        new_action_right, replaced_right = self.expert_right.action(action_right)
        new_action = np.concatenate((new_action_left, new_action_right), axis=0)

        obs, rew, done, truncated, info = self.env.step(new_action)

        if replaced_left or replaced_right:
            info["intervene_action"] = new_action

        info["left"] = self.expert_left.left.any() or self.expert_right.left.any()  # Whether the left button is pressed.
        info["right"] = self.expert_left.right.any() or self.expert_right.right.any()  # Whether the right button is pressed.

        return obs, rew, done, truncated, info


class Quat2EulerWrapper(gym.ObservationWrapper):  # not used anymore (stay away from euler angles!)
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_euler(tcp_pose[3:]))
        )
        return observation


class ToMrpWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to mrp angles
    """

    def __init__(self, env: gym.Env, transform_obs=True):
        super().__init__(env)
        self.transform_obs = transform_obs
        # from xyz + quat to xyz + mrp
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to mrp
        tcp_pose = observation["state"]["tcp_pose"]
        tcp_pose_mrp = np.concatenate((tcp_pose[:3], quat_2_mrp(tcp_pose[3:])))
        observation["state"]["tcp_pose"] = tcp_pose_mrp

        if self.transform_obs:
            # Map angular velocity (in reset/body frame) to MRP rate using current MRP
            sigma = tcp_pose_mrp[3:6]
            omega = observation["state"]["tcp_vel"][3:6]
            observation["state"]["tcp_vel"][3:6] = omega_to_mrp_dot(sigma, omega)
            # If EMA velocity exists, convert its angular part as well
            if "ema_tcp_vel" in observation["state"]:
                omega_ema = observation["state"]["ema_tcp_vel"][3:6]
                observation["state"]["ema_tcp_vel"][3:6] = omega_to_mrp_dot(sigma, omega_ema)
        return observation


def rotate_state(state: np.ndarray, num_rot: int):
    assert len(state.shape) == 1 and state.shape[0] % 3 == 0
    state = state.reshape((-1, 3)).transpose()
    rotated = np.dot(ROT_GENERAL[num_rot % 4], state).transpose()
    return rotated.reshape((-1))


class ObservationRotationWrapper(gym.Wrapper):
    """
    Convert every observation into the first quadrant of the Relative Frame
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        print("Observation Rotation Wrapper enabled!")
        self.num_rot_quadrant = -1

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = self.rotate_observation(obs, random=True)  # rotate initial state random
        return obs, info

    def step(self, action: np.ndarray):
        action = self.rotate_action(action=action)
        obs, reward, done, truncated, info = self.env.step(action)
        # print("\nquadrant: ", self.num_rot_quadrant)
        rotated_obs = self.rotate_observation(obs)
        return rotated_obs, reward, done, truncated, info

    def rotate_observation(self, observation, random=False):
        if not random:
            x, y = (observation["state"]["tcp_pose"][:2])
            self.num_rot_quadrant = int(x < 0.) * 2 + int(x * y < 0.)  # save quadrant info
        else:
            self.num_rot_quadrant = int(time.time_ns()) % 4  # do not mess with seeded np.random

        for state in observation["state"].keys():
            if state == "gripper_state":
                continue
            elif state == "action":
                observation["state"][state][:6] = rotate_state(observation["state"][state][:6], self.num_rot_quadrant)
            else:
                observation["state"][state][:] = rotate_state(observation["state"][state],
                                                              self.num_rot_quadrant)  # rotate

        if "images" in observation:
            for image_keys in observation["images"].keys():
                observation["images"][image_keys][:] = np.rot90(
                    observation["images"][image_keys],
                    axes=(0, 1),
                    k=self.num_rot_quadrant
                )
        return observation

    def rotate_action(self, action):
        rotated_action = action.copy()
        rotated_action[:6] = rotate_state(action[:6], 4 - self.num_rot_quadrant)  # rotate
        return rotated_action



class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """
    def __init__(self, env: gym.Env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info


class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received))  # 环境结束或所有奖励满足
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # 从 xyz + quat 到 xyz + 旋转矩阵前两列
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # 从 xyz + quat 到 xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """
    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"]["gripper_state"][0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"]["gripper_state"][0]
        return observation, reward, terminated, truncated, info


class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0  # assume gripper starts opened
        self.last_gripper_pos_right = 0  # assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left == 0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left == 1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right == 0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right == 1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info
