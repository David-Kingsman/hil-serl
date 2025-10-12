import copy
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
from pynput import keyboard

from serl_robot_infra.ur_env.envs.ur30_env import UR30Env

def euler_2_quat(euler):
    """make euler to quat"""
    return R.from_euler("xyz", euler).as_quat()

class PEGEnv(UR30Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False

        def on_press(key):
            if str(key) == "Key.f1":
                self.should_regrasp = True

        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """move to target position - UR controller automatically handles interpolation"""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        # UR controller directly sets target position, no need for manual interpolation
        self.set_target_pose(goal)
        time.sleep(timeout)

    def go_to_reset(self, joint_reset=False):
        """
        Move to the rest position - adapt to UR control方式
        """
        # UR uses joint space reset, no need for complex Cartesian path planning
        if joint_reset:
            print("JOINT RESET")
            # use UR's joint reset
            reset_Q = self.config.RESET_Q[0].copy()  # get reset joint angles
            self.set_reset_angles(reset_Q)
        else:
            # use UR's Cartesian reset
            reset_pose = self.config.RESET_POSE.copy()
            if self.randomreset:  # randomize reset position
                reset_pose[:2] += np.random.uniform(
                    -self.config.RANDOM_POSITION_RANGE[0], 
                    self.config.RANDOM_POSITION_RANGE[0], 
                    (2,)
                )
                euler_random = self.config.RESET_POSE[3:].copy()
                euler_random[-1] += np.random.uniform(
                    -self.config.RANDOM_ROT_RANGE[0], 
                    self.config.RANDOM_ROT_RANGE[0]
                )
                reset_pose[3:] = euler_2_quat(euler_random)
            
            # directly set target position, UR controller handles motion planning
            self.set_target_pose(reset_pose)
        
        time.sleep(0.5)


    def regrasp(self):
        """regrasp peg - adapt to UR control"""
        # 抬升到安全高度
        reset_pose = self.config.RESET_POSE.copy()
        reset_pose[2] += 0.04  # lift 4cm
        self.set_target_pose(reset_pose)
        time.sleep(1.0)

        # user interaction: release gripper
        input("Press enter to release gripper...")
        self.set_gripper_pos(np.array(1.0))  # open gripper
        
        # user interaction: re-place peg
        input("Place peg in holder and press enter to grasp...")
        
        # move to grasp position above
        top_pose = self.config.GRASP_POSE.copy()
        top_pose[2] += 0.05  # above grasp position 5cm
        top_pose[0] += np.random.uniform(-0.005, 0.005)  # add small random offset
        self.set_target_pose(top_pose)
        time.sleep(1.0)

        # descend to grasp position
        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.05
        self.set_target_pose(grasp_pose)
        time.sleep(0.5)

        # close gripper
        self.set_gripper_pos(np.array(-1.0))  # close gripper
        self.last_gripper_act = time.time()
        time.sleep(2)

        # lift
        self.set_target_pose(top_pose)
        time.sleep(0.5)

        # move to reset position
        self.set_target_pose(self.config.RESET_POSE)
        time.sleep(1.0)


    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        # check if need to regrasp
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False

        # execute reset process
        self._recover()
        self.go_to_reset(joint_reset=False)
        self._recover()
        self.curr_path_length = 0

        # get initial observation
        self.update_currpos()  # UR30Env's method name
        obs = self._get_obs(np.zeros(self.action_space.shape))  # UR30Env's _get_obs needs action parameter
        self.terminate = False
        return obs, {}