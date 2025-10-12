"""Gym Interface for UR30"""
import time
import threading
import copy
import numpy as np
import gymnasium as gym
import cv2
import queue
import warnings
from typing import Dict, Tuple
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
from ur_env.camera.video_capture import VideoCapture
from ur_env.camera.rs_capture import RSCapture

from robot_controllers.ur30_controller import UrImpedanceController
from robot_controllers.controller_client import ControllerClientWithGripper


class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [v for k, v in img_array.items() if "full" not in k], axis=0
            )
            cv2.namedWindow("RealSense Cameras", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RealSense Cameras", 300, 700)
            cv2.imshow("RealSense Cameras", frame)
            cv2.waitKey(1)



##############################################################################


class DefaultEnvConfig:
    """Default configuration for UR30Env. Fill in the values below."""

    RESET_Q = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_POSITION_RANGE = (0.0,)
    RANDOM_Z_RANGE = (0.0)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ABS_POSE_RANGE_LIMITS = np.zeros((2,))
    ACTION_SCALE = np.zeros((3,), dtype=np.float32)

    ROBOT_IP: str = "localhost"
    CONTROLLER_HZ: int = 0
    GRIPPER_TIMEOUT: int = 0  # in milliseconds
    ERROR_DELTA: float = 0.
    FORCEMODE_DAMPING: float = 0.
    FORCEMODE_TASK_FRAME = np.zeros(6, )
    FORCEMODE_SELECTION_VECTOR = np.ones(6, )
    FORCEMODE_LIMITS = np.zeros(6, )

    REALSENSE_CAMERAS: Dict = {
        "shoulder": "",
        "wrist": "",
    }
    CALIBRATION_PATH: str = ""


##############################################################################


class UR30Env(gym.Env):
    def __init__(
            self,
            hz: int = 10,
            fake_env=False,
            config=DefaultEnvConfig,
            max_episode_length: int = 100,
            save_video: bool = False,
            camera_mode: str = "rgb",  # one of (rgb, grey, depth, both(rgb depth), none)
            visualize_camera_mode: bool = True,
            ema_alpha: float = 0.2,
    ):
        self.max_episode_length = max_episode_length
        self.curr_path_length = 0
        self.action_scale = config.ACTION_SCALE

        self.config = config
        self.ema_alpha = float(ema_alpha)

        self.resetQ = config.RESET_Q
        self.curr_reset_pose = np.zeros((7,), dtype=np.float32)

        self.curr_pos = np.zeros((7,), dtype=np.float32)
        self.curr_vel = np.zeros((6,), dtype=np.float32)
        self.curr_Q = np.zeros((6,), dtype=np.float32)
        self.curr_Qd = np.zeros((6,), dtype=np.float32)
        self.curr_force = np.zeros((3,), dtype=np.float32)
        self.curr_torque = np.zeros((3,), dtype=np.float32)
        self.curr_timestamp_diff = np.zeros((1,), dtype=np.float32)
        self.ema_force = np.zeros((6,), dtype=np.float32)
        self.ema_tcp_vel = np.zeros((6,), dtype=np.float32)

        self.last_state_timestamp = None
        self.curr_timestamp = None
        self.neutral_gripper_command = False

        self.gripper_state = np.zeros((2,), dtype=np.float32)
        self.random_reset = config.RANDOM_RESET
        self.random_position_range = config.RANDOM_POSITION_RANGE
        self.random_z_range = config.RANDOM_Z_RANGE
        self.random_rot_range = config.RANDOM_ROT_RANGE
        self.hz = hz
        np.random.seed(0)        # fix seed for fixed (random) initial rotations

        camera_mode = None if camera_mode.lower() == "none" else camera_mode
        if camera_mode is not None and save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []
        self.camera_mode = camera_mode
        self.visualize_camera_mode = visualize_camera_mode

        self.cost_infos = {}

        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.xy_range = gym.spaces.Box(
            config.ABS_POSE_RANGE_LIMITS[0],
            config.ABS_POSE_RANGE_LIMITS[1],
            dtype=np.float64,
        )
        self.mrp_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        self.last_action = np.zeros(self.action_space.shape)


        image_space_definition = {}
        if camera_mode in ["rgb", "grey", "both"]:
            channel = 1 if camera_mode == "grey" else 3
            if "wrist" in config.REALSENSE_CAMERAS.keys():
                image_space_definition["wrist"] = gym.spaces.Box(
                    0, 255, shape=(128, 128, channel), dtype=np.uint8
                )
            if "wrist_2" in config.REALSENSE_CAMERAS.keys():
                image_space_definition["wrist_2"] = gym.spaces.Box(
                    0, 255, shape=(128, 128, channel), dtype=np.uint8
                )

        if camera_mode in ["depth", "both"]:
            if "wrist" in config.REALSENSE_CAMERAS.keys():
                image_space_definition["wrist_depth"] = gym.spaces.Box(
                    0, 255, shape=(128, 128, 1), dtype=np.uint8
                )
            if "wrist_2" in config.REALSENSE_CAMERAS.keys():
                image_space_definition["wrist_2_depth"] = gym.spaces.Box(
                    0, 255, shape=(128, 128, 1), dtype=np.uint8
                )

        if camera_mode is not None and camera_mode not in ["rgb", "both", "depth", "grey"]:
            raise NotImplementedError(f"camera mode {camera_mode} not implemented")

        state_space = gym.spaces.Dict(
            {
                "tcp_pose": gym.spaces.Box(
                    -np.inf, np.inf, shape=(7,)
                ),  # xyz + quat
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_state": gym.spaces.Box(-1., 1., shape=(2,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "action": gym.spaces.Box(-1., 1., shape=self.action_space.shape),
                "time_diff": gym.spaces.Box(0., np.inf, shape=(1,)),
                "ema_force": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "ema_tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
            }
        )

        obs_space_definition = gym.spaces.Dict(
            {"state": state_space}
        )
        if self.camera_mode in ["rgb", "both", "depth", "grey"]:
            obs_space_definition["images"] = gym.spaces.Dict(
                image_space_definition
            )

        self.observation_space = gym.spaces.Dict(obs_space_definition)

        self.cycle_count = 0
        self.controller = None
        self.cap = None

        if fake_env:
            print("[UR30Env] is fake!")
            return

        self.controller = ControllerClientWithGripper(
            robot_ip=config.ROBOT_IP,
            config=config
        )
        self.controller.start()  # start Thread

        if self.camera_mode is not None:
            self.init_cameras(config.REALSENSE_CAMERAS)
            self.img_queue = queue.Queue()
            if self.visualize_camera_mode:
                self.displayer = ImageDisplayer(self.img_queue)
                self.displayer.start()

            print("[CAM] Cameras are ready!")

        while not self.controller.is_ready():  # wait for controller
            time.sleep(0.1)
        print("[RIC] Controller has started and is ready!")


    def clip_safety_box(self, next_pos: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        next_pos[:3] = np.clip(
            next_pos[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        orientation_diff = (R.from_quat(next_pos[3:]) * R.from_quat(self.curr_reset_pose[3:]).inv()).as_mrp()
        orientation_diff = np.clip(
            orientation_diff, self.mrp_bounding_box.low, self.mrp_bounding_box.high
        )
        next_pos[3:] = (R.from_mrp(orientation_diff) * R.from_quat(self.curr_reset_pose[3:])).as_quat()

        return next_pos

    def get_cost_infos(self, done):
        if not done:
            return {}
        cost_infos = self.cost_infos.copy()
        self.cost_infos = {}
        return cost_infos

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # position
        next_pos = self.curr_pos.copy()
        next_pos[:3] += action[:3] * self.action_scale[0]
        next_pos[3:] = (
                R.from_mrp(action[3:6] * self.action_scale[1] / 4.) * R.from_quat(next_pos[3:])
        ).as_quat()             # c * r  --> applies c after r
        gripper_action = action[6] * self.action_scale[2]

        safe_pos = self.clip_safety_box(next_pos)
        self.send_pos_command(safe_pos)
        self.send_gripper_command(gripper_action)
        # print(f"sent pose: {safe_pos}  with action {action}    actual pose: {self.curr_pos}")
        self.curr_path_length += 1

        # wait
        dt = time.time() - start_time
        to_sleep = max(0., (1. / self.hz) - dt)
        time.sleep(to_sleep)

        # get next observation
        obs = self._get_obs(action)

        current_force6 = np.concatenate((self.curr_force, self.curr_torque)).astype(np.float32)
        self.ema_force = (1.0 - self.ema_alpha) * self.ema_force + self.ema_alpha * current_force6
        self.ema_tcp_vel = (1.0 - self.ema_alpha) * self.ema_tcp_vel + self.ema_alpha * self.curr_vel
        obs["state"]["ema_force"] = self.ema_force.copy()
        obs["state"]["ema_tcp_vel"] = self.ema_tcp_vel.copy()

        reward = self.compute_reward(obs, action)
        truncated = self._is_truncated()
        reward = reward if not truncated else reward - 200.  # truncation penalty
        done = self.curr_path_length >= self.max_episode_length or self.reached_goal_state(obs) or truncated

        return obs, reward, done, truncated, self.get_cost_infos(done)

    def compute_reward(self, obs, action) -> float:
        return 0.   # overwrite for each task

    def reached_goal_state(self, obs) -> bool:
        return False  # overwrite for each task

    def go_to_rest(self, deactiveate_gripper: bool = True):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform Carteasian reset
        reset_Q = np.zeros((6))
        if self.resetQ.shape == (1, 6):
            reset_Q[:] = self.resetQ.copy()
        elif self.resetQ.shape[1] == 6 and self.resetQ.shape[0] > 1:
            reset_Q[:] = self.resetQ[0, :].copy()
            self.resetQ[:] = np.roll(self.resetQ, -1, axis=0)  # roll one (not random)
        else:
            raise ValueError(f"invalid resetQ dimension: {self.resetQ.shape}")

        self.send_reset_command(reset_Q)

        while not self.controller.is_reset():
            time.sleep(0.1)  # wait for the reset operation

        self.update_currpos()
        reset_pose = np.asarray(self.controller.get_state()["pos"])

        if self.random_reset:  # randomize reset position in xy plane
            reset_shift = np.random.uniform(np.negative(self.random_position_range), self.random_position_range, (3,))
            reset_pose[:3] += reset_shift

            if self.random_rot_range[0] > 0.:
                random_rot = np.random.triangular(np.negative(self.random_rot_range), 0., self.random_rot_range, size=(3,))
            else:
                random_rot = np.zeros((3,))
            reset_pose[3:][:] = (R.from_quat(reset_pose[3:]) * R.from_mrp(random_rot)).as_quat()

            self.curr_reset_pose[:] = reset_pose

            self.controller.set_target_pose(reset_pose)  # random movement after resetting
            time.sleep(0.1)
            while self.controller.is_moving():
                time.sleep(0.1)
            return reset_shift
        else:
            self.curr_reset_pose[:] = reset_pose
            return np.zeros((2,))

    # def go_to_detected_box(self):
    #     """"
    #     function for the demo
    #     """
    #     if self.gripper_state[0] > 0.01:
    #         reset_Q = self.curr_Q.copy()
    #         reset_Q[:4] = [0., -np.pi / 2., np.pi / 2., -np.pi / 2.]
    #         self.send_reset_command(reset_Q)
    #         while not self.controller.is_reset():
    #             time.sleep(0.1)  # wait for the reset operation

    #         reset_Q[:4] = [np.pi / 2, -np.pi / 2., np.pi / 2., -np.pi / 2.]
    #         self.send_reset_command(reset_Q)
    #         while not self.controller.is_reset():
    #             time.sleep(0.1)  # wait for the reset operation

    #         # release the box
    #         self.send_gripper_command(np.array(-1))
    #         time.sleep(0.1)

    #     # go back on top
    #     reset_Q = [0., -np.pi / 2., np.pi / 2., -np.pi / 2., -np.pi / 2., 0.]
    #     self.send_reset_command(reset_Q)
    #     while not self.controller.is_reset():
    #         time.sleep(0.1)  # wait for the reset operation
    #     time.sleep(0.5)

    #     def get_request(i=10):
    #         if i == 0:
    #             raise Exception("err")
    #         try:
    #             r = requests.get('http://192.168.1.204:5000/api/data')
    #             r.raise_for_status()
    #             boxes = r.json()
    #             if len(boxes) == 0:
    #                 time.sleep(0.1)
    #                 return get_request(i)
    #             else:
    #                 return boxes

    #         except (json.decoder.JSONDecodeError, requests.exceptions.HTTPError):
    #             return get_request(i=i - 1)

    #     boxes = get_request()

    #     highest = list(boxes.keys())[np.argmax([b["world2box"]["pos"][1] for b in boxes.values()])]
    #     box = boxes[highest]["world2box"]
    #     print(f"pose: {[round(b, 2) for b in box['pos']]} {[round(b, 2) for b in box['rot']]}")

    #     t = R.from_euler("xyz", [-np.pi / 2., np.pi, 0.])
    #     pos = t.apply(np.array(box["pos"]) + np.array([0., 0.1 + boxes[highest]["size"][1] / 2., 0.]))
    #     rot = (R.from_euler("xyz", t.apply(box["rot"])) * R.from_euler("xyz", [np.pi, 0., 0.])).as_rotvec()

    #     init_pose = np.concatenate((pos, rot))

    #     print(f"moving to {init_pose}")
    #     self.send_pos_command(init_pose)
    #     while not self.controller.is_reset():
    #         time.sleep(0.1)  # wait for the reset operation

    #     self.update_currpos()
    #     self.curr_reset_pose[:] = self.curr_pos

    def reset(self, **kwargs):
        self.cycle_count += 1
        if self.save_video:
            self.save_video_recording()

        shift = self.go_to_rest()
        self.curr_path_length = 0
        self.last_state_timestamp = None

        obs = self._get_obs(np.zeros_like(self.last_action))

        current_force = np.concatenate((self.curr_force, self.curr_torque)).astype(np.float32)
        self.ema_force[:] = current_force
        self.ema_tcp_vel[:] = self.curr_vel
        obs["state"]["ema_force"] = self.ema_force.copy()
        obs["state"]["ema_tcp_vel"] = self.ema_tcp_vel.copy()
        
        return obs, {"reset_shift": shift}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    f'./videos/{datetime.now().strftime("%m-%d_%H-%M")}.mp4',
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    self.recording_frames[0].shape[:2][::-1],
                )
                for frame in self.recording_frames:
                    video_writer.write(frame)
                video_writer.release()
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init both cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, cam_serial in name_serial_dict.items():
            print(f"cam serial: {cam_serial}")
            rgb = self.camera_mode in ["rgb", "both", "grey"]
            depth = self.camera_mode in ["depth", "both"]
            pointcloud = self.camera_mode in ["pointcloud"]
            rgb_pc = self.camera_mode in ["rgb_pointcloud"]
            cap = VideoCapture(
                RSCapture(name=cam_name, serial_number=cam_serial, fps=30, rgb=rgb, depth=depth, pointcloud=pointcloud, rgb_pointcloud=rgb_pc)
            )
            self.cap[cam_name] = cap

    def crop_image(self, name, image) -> np.ndarray:
        """Crop realsense images to be a square."""
        return image[:, 124:604, :]

    def get_image(self) -> Tuple[Dict[str, np.ndarray], int]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        timestamp = 0
        for key, cap in self.cap.items():
            try:
                image, timestamp = cap.read()
                if self.camera_mode in ["rgb", "both", "grey"]:
                    rgb = image[..., :3].astype(np.uint8)
                    cropped_rgb = self.crop_image(key, rgb)
                    resized = cv2.resize(
                        cropped_rgb, self.observation_space["images"][key].shape[:2][::-1],
                    )
                    # convert to grayscale here
                    if self.camera_mode == "grey":
                        grey = np.array([0.2989, 0.5870, 0.1140])
                        resized = np.dot(resized, grey)[..., None]
                        resized = resized.astype(np.uint8)
                        display_images[key] = np.repeat(resized, 3, axis=-1)
                    else:
                        display_images[key] = resized

                    images[key] = resized[..., ::-1]
                    display_images[key + "_full"] = cropped_rgb

                if self.camera_mode in ["depth", "both"]:
                    depth_key = key + "_depth"
                    depth = image[..., -1:]
                    cropped_depth = self.crop_image(key, depth)

                    resized = cv2.resize(
                        cropped_depth, np.array(self.observation_space["images"][depth_key].shape[:2]) * 3,
                        # (128 * 3, 128 * 3) image
                    )[..., None]

                    resized = resized.reshape((128, 3, 128, 3, 1)).max((1, 3))  # max pool with 3x3

                    images[depth_key] = resized
                    display_images[depth_key] = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
                    display_images[depth_key + "_full"] = cv2.applyColorMap(cropped_depth, cv2.COLORMAP_JET)


            except queue.Empty:
                input(f"{key} camera frozen. Check connect, then press enter to relaunch...")
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_image()


        # self.recording_frames.append(
        #     np.concatenate([image for key, image in display_images.items() if "full" in key], axis=0)
        # )
        self.img_queue.put(display_images)

        return images, timestamp


    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def send_pos_command(self, target_pose: np.ndarray):
        """Internal function to send force command to the robot."""
        self.controller.set_target_pose(target_pose=target_pose)

    def send_gripper_command(self, gripper_pos: np.ndarray):
        if self.neutral_gripper_command:
            gripper_pos = np.array([0.0])
            self.neutral_gripper_command = False
        self.controller.set_gripper_pos(gripper_pos)

    def send_reset_command(self, reset_Q: np.ndarray):
        self.controller.set_reset_angles(reset_Q)

    def update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        state = self.controller.get_state()

        self.curr_pos[:] = state['pos']
        self.curr_vel[:] = state['vel']
        self.curr_force[:] = state['force'][:3]
        self.curr_torque[:] = state['force'][3:]
        self.curr_Q[:] = state['Q']
        self.curr_Qd[:] = state['Qd']
        self.gripper_state[:] = state['gripper']
        self.curr_timestamp = state['timestamp_ms']

    def _is_truncated(self):
        return self.controller.is_truncated()

    def _get_obs(self, action) -> dict:
        # get image before state observation, so they match better in time

        self.update_currpos()
        images = None
        if self.camera_mode is not None:
            images, timestamp = self.get_image()

        self.curr_timestamp_diff[:] = (self.curr_timestamp - self.last_state_timestamp) * 1e-3 if self.last_state_timestamp else 0.
        self.last_state_timestamp = self.curr_timestamp

        state_observation = {
            "tcp_pose": self.curr_pos,
            "tcp_vel": self.curr_vel,
            "gripper_state": self.gripper_state,
            "tcp_force": self.curr_force,
            "tcp_torque": self.curr_torque,
            "action": action,
            "time_diff": self.curr_timestamp_diff
        }

        if images is not None:
            return copy.deepcopy(dict(images=images, state=state_observation))
        else:
            return copy.deepcopy(dict(state=state_observation))

    def close(self):
        if self.controller:
            self.controller.stop()
        super().close()
