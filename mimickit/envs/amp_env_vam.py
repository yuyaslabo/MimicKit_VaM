# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import gymnasium.spaces as spaces
import os
import socket
import struct
import numpy as np
import torch

import engines.engine as engine
import envs.deepmimic_env as deepmimic_env
import util.circular_buffer as circular_buffer
import util.torch_util as torch_util
#import envs.task_location_env as task_location_env
from scipy.spatial.transform import Rotation as R

#from .deepmimic_env import compute_done

from envs.vam_vr_assets import VRAssetsMixin
from envs.vam_vr_body import VRBodyMixin
from envs.vam_vr_physics import VRPhysicsMixin
from envs.vam_vr_tracking import VRTrackingMixin
from envs.vam_udp_sender import VaMUDPMixin
from envs.vam_environment_boxes import EnvironmentBoxesMixin

class AMPEnv(VRAssetsMixin, VRBodyMixin, VRPhysicsMixin, VRTrackingMixin, VaMUDPMixin, EnvironmentBoxesMixin, deepmimic_env.DeepMimicEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        #self._rand_reset = env_config.get("rand_reset", "random")
        self._num_disc_obs_steps = env_config["num_disc_obs_steps"]
        self._disc_dof_vel_obs = env_config.get("disc_dof_vel_obs", True)
        self._tar_speed = float(env_config.get("tar_speed", 1.0))
        self._tar_change_time_min = float(env_config.get("tar_change_time_min", 5.0))
        self._tar_change_time_max = float(env_config.get("tar_change_time_max", 10.0))
        self._tar_dist_min = float(env_config.get("tar_dist_min", 0.0))
        self._tar_dist_max = float(env_config.get("tar_dist_max", 5.0))
        self._dist_threshold = float(env_config.get("dist_threshold", 0.5))
        self._pos_err_scale = float(env_config.get("pos_err_scale", 0.5))
        self._vel_err_scale = float(env_config.get("vel_err_scale", 4.0))
        self._pos_reward_w = float(env_config.get("pos_reward_w", 0.5))
        self._vel_reward_w = float(env_config.get("vel_reward_w", 0.4))
        self._face_reward_w = float(env_config.get("face_reward_w", 0.1))
        assert self._tar_dist_min <= self._tar_dist_max
        assert self._tar_change_time_min <= self._tar_change_time_max
        self._marker_ids = []
        self._envbox_initialize(env_config)
       
        self._vrhand_enabled = bool(env_config.get("vrhand_udp_enabled", visualize))
        self._vrhand_ip = env_config.get("vrhand_udp_ip", "0.0.0.0")
        self._vrhand_port = int(env_config.get("vrhand_udp_port", 9999))
        self._vrhand_packet = struct.Struct("<23f")
        # VRhand offset is independent of the existing VaM retarget correction.
        self._vrhand_position_offset = np.asarray(
            env_config.get("vrhand_position_offset", [0.0, 0.15, 0.0]), dtype=np.float64
        )                 
        self._prev_vrhand_pos = {}
        self._vrhand_triggers = {"left": 0.0, "right": 0.0}
        self._vrhand_grabbed_targets = {"left": None, "right": None}
        # Double trigger while no nearby bone is locked: set/clear a distant facing point.
        self._vrhand_double_trigger_window = float(env_config.get("vrhand_double_trigger_window", 0.65))
        self._vrhand_ground_point_threshold = float(env_config.get("vrhand_ground_point_threshold", -0.65))
        self._vrhand_remote_player_distance = float(env_config.get("vrhand_remote_player_distance", 5.0))
        self._vrhand_trigger_pressed = {"left": False, "right": False}
        self._vrhand_last_unlocked_press = {"left": None, "right": None}
        self._vrhand_remote_player_world_pos = None  # None: live HMD; otherwise fixed world-space facing point
        self._vrhand_target_frozen = False  # Left-hand double-trigger toggles target hold
        self._vrhand_debug = bool(env_config.get("vrhand_debug", True))
        self._vrhand_packet_count = 0
        self._vrhand_update_count = 0
        self._vrhand_bad_packet_count = 0
        self._vrhand_socket = None
        self._vrhand_box_ids = []
        self._vrhand_latest = None
        # Delay VR hand/HMD/body physics until the simulation has started.
        self._vrhand_start_delay = max(0.0, float(env_config.get("vrhand_start_delay", 3.0)))
        self._vrhand_start_time = None  # Set on first simulation update, not during asset loading.
        self._vrhand_delay_finished = False
        self._vrhand_asset = os.path.join("data", "assets", "objects", "vrhand_collision_box.xml")
        self._vrhmd_asset = os.path.join("data", "assets", "objects", "vrhmd_collision_box.xml")
        # Player collision capsules: torso, left/right upper arms and forearms.
        self._vrbody_enabled = self._vrhand_enabled and bool(env_config.get("vrbody_enabled", True))
        self._vrbody_shoulder_width = float(env_config.get("vrbody_shoulder_width", 0.25))
        self._vrbody_upper_arm_length = float(env_config.get("vrbody_upper_arm_length", 0.25))
        self._vrbody_forearm_length = float(env_config.get("vrbody_forearm_length", 0.25))
        self._vrbody_torso_length = float(env_config.get("vrbody_torso_length", 0.25))
        self._vrbody_neck_drop = float(env_config.get("vrbody_neck_drop", 0.16))
        self._vrbody_chest_back = float(env_config.get("vrbody_chest_back", 0.02))
        self._vrbody_torso_radius = float(env_config.get("vrbody_torso_radius", 0.08))
        self._vrbody_upper_arm_radius = float(env_config.get("vrbody_upper_arm_radius", 0.02))
        self._vrbody_forearm_radius = float(env_config.get("vrbody_forearm_radius", 0.012))
        self._vrbody_elbow_smoothing = float(env_config.get("vrbody_elbow_smoothing", 0.35))
        # Physical capsule clearances: separate actual capsule surfaces, not just joint centers.
        self._vrbody_upper_shoulder_trim = float(env_config.get("vrbody_upper_shoulder_trim", 0.03))
        self._vrbody_upper_elbow_trim = float(env_config.get("vrbody_upper_elbow_trim", 0.025))
        self._vrbody_forearm_elbow_trim = float(env_config.get("vrbody_forearm_elbow_trim", 0.025))
        self._vrbody_wrist_clearance = float(env_config.get("vrbody_wrist_clearance", 0.03))
        self._vrbody_shoulder_outset = float(env_config.get("vrbody_shoulder_outset", 0.035))
        if self._vrbody_upper_shoulder_trim < 0 or self._vrbody_upper_elbow_trim < 0 or self._vrbody_forearm_elbow_trim < 0 or self._vrbody_wrist_clearance < 0:
            raise ValueError("VR body capsule trims must be nonnegative")
        if self._vrbody_upper_arm_length <= self._vrbody_upper_shoulder_trim + self._vrbody_upper_elbow_trim:
            raise ValueError("Upper arm capsule must have positive length after trims")
        if self._vrbody_forearm_length <= self._vrbody_forearm_elbow_trim:
            raise ValueError("Forearm capsule must have positive length after trim")
        # Wrist orientation biases the IK bend plane; 0.0 restores the original behavior.
        self._vrbody_wrist_pole_weight = float(env_config.get("vrbody_wrist_pole_weight", 0.50))
        # Controller-local direction from wrist toward elbow (calibrate for each controller).
        self._vrbody_left_wrist_axis = np.asarray(env_config.get("vrbody_left_wrist_axis", [0.0, 0.0, -1.0]), dtype=np.float64)
        self._vrbody_right_wrist_axis = np.asarray(env_config.get("vrbody_right_wrist_axis", [0.0, 0.0, -1.0]), dtype=np.float64)
        self._vrbody_box_ids = []
        self._vrbody_asset_files = {}
        self._vrbody_prev_elbows = {}
        self._vrbody_prev_pos = {}
        # Translation PD: hands, forearms, and upper arms; upper arms use gentler gains.
        self._vrpd_enabled = bool(env_config.get("vrpd_enabled", True))
        self._vrpd_position_kp = float(env_config.get("vrpd_position_kp", 1000.0))
        self._vrpd_position_kd = float(env_config.get("vrpd_position_kd", 40.0))
        self._vrpd_max_force = float(env_config.get("vrpd_max_force", 300.0))  # maximum PD correction, excluding gravity support
        self._vrpd_max_error = float(env_config.get("vrpd_max_error", 0.20))
        self._vrpd_gravity_comp = bool(env_config.get("vrpd_gravity_comp", True))
        self._vrpd_forearm_lock_rotation = bool(env_config.get("vrpd_forearm_lock_rotation", True))
        self._vrpd_upper_arm_kp = float(env_config.get("vrpd_upper_arm_kp", 450.0))
        self._vrpd_upper_arm_kd = float(env_config.get("vrpd_upper_arm_kd", 35.0))
        self._vrpd_upper_arm_max_force = float(env_config.get("vrpd_upper_arm_max_force", 150.0))
        self._vrpd_upper_arm_lock_rotation = bool(env_config.get("vrpd_upper_arm_lock_rotation", True))
        self._vrpd_debug = bool(env_config.get("vrpd_debug", False))
        self._vrpd_targets = {}
        self._vrpd_initialized = set()
        self._vrpd_masses = {"left_hand": 2.0, "right_hand": 2.0, "left_forearm": 1.7, "right_forearm": 1.7, "left_upper_arm": 1.5, "right_upper_arm": 1.5}
        self._vrhand_left_pd_asset = os.path.join("data", "assets", "objects", "vrhand_left_pd_collision_box.xml")
        self._vrhand_right_pd_asset = os.path.join("data", "assets", "objects", "vrhand_right_pd_collision_box.xml")
        if self._vrhand_enabled:
            self._write_vrhand_asset()
            if self._vrpd_enabled:
                self._write_vrhand_left_pd_asset()
                self._write_vrhand_right_pd_asset()
            self._write_vrhmd_asset()
            if self._vrbody_enabled:
                self._write_vrbody_assets()
            
            self._vrhand_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._vrhand_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._vrhand_socket.bind((self._vrhand_ip, self._vrhand_port))
            self._vrhand_socket.setblocking(False)
            if self._vrhand_debug:
                print(f"[VRhand] UDP listening on {self._vrhand_ip}:{self._vrhand_port}; packet={self._vrhand_packet.size} bytes; offset={self._vrhand_position_offset.tolist()}", flush=True)                 
        super().__init__(env_config=env_config, engine_config=engine_config, num_envs=num_envs, device=device,
                         visualize=visualize, record_video=record_video)                 
        self._vam_enabled = env_config.get("vam_udp_enabled", visualize)
        self._vam_ip = env_config.get("vam_udp_ip", "192.168.2.100")
        self._vam_port = int(env_config.get("vam_udp_port", 9998))
        self._vam_packet = struct.Struct("<133f")
        self._vam_position_offset = np.asarray(env_config.get("vam_position_offset", [0.0, 0.06, 0.0]), dtype=np.float32)
        self._vam_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self._vam_enabled else None
        self._latest_hmd_pos = torch.zeros((1, 3), device=self._device)
        self._latest_hmd_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device) # [x, y, z, w]
        return

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        self._envbox_build_objects(env_id)

        if self._vrhand_enabled:
            print("start")
            left_id = self._engine.create_obj(
                env_id=env_id, obj_type=engine.ObjType.rigid,
                asset_file=self._vrhand_left_pd_asset if self._vrpd_enabled else self._vrhand_asset, name="vrhand_left_collision_box",
                is_visual=False, fix_root=False, color=[0.2, 0.65, 1.0] # False に変更
            )
            right_id = self._engine.create_obj(
                env_id=env_id, obj_type=engine.ObjType.rigid,
                asset_file=self._vrhand_right_pd_asset if self._vrpd_enabled else self._vrhand_asset, name="vrhand_right_collision_box",
                is_visual=False, fix_root=False, color=[1.0, 0.4, 0.2] # False に変更
            )
            hmd_id = self._engine.create_obj(
                env_id=env_id, obj_type=engine.ObjType.rigid,
                asset_file=self._vrhmd_asset, name="vrhmd_collision_box",
                is_visual=False, fix_root=False, color=[0.3, 0.8, 0.8] # False に変更
            )
            if self._vrbody_enabled:
                body_ids = []
                for name in ("torso", "left_forearm", "right_forearm", "left_upper_arm", "right_upper_arm"):
                    body_ids.append(self._engine.create_obj(
                        env_id=env_id, obj_type=engine.ObjType.rigid,
                        asset_file=self._vrbody_asset_files[name], name="vrbody_" + name,
                        is_visual=False, fix_root=False, color=[0.3, 0.8, 0.8]
                    ))
                if env_id == 0:
                    self._vrbody_box_ids = body_ids
                else:
                    assert self._vrbody_box_ids == body_ids
            if env_id == 0:
                self._vrhand_box_ids = [left_id, right_id, hmd_id]
                if self._vrhand_debug:
                    print(f"[VRhand] collision boxes created: left_id={left_id}, right_id={right_id}, hmd_id={hmd_id} env_id={env_id}, asset={self._vrhand_asset}", flush=True)
            else:
                assert self._vrhand_box_ids == [left_id, right_id, hmd_id]
        return

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        self._envbox_park_all()
        
        if self._vrhand_enabled and len(self._vrhand_box_ids) == 3:
            num_envs = self.get_num_envs()
            env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)
            
            # 左手の退避位置: 空中左側 (x = -2.0, y = 0.0, z = 10.0)
            left_parked_pos = torch.zeros((num_envs, 3), device=self._device, dtype=torch.float32)
            left_parked_pos[:, 0] = 2.0
            left_parked_pos[:, 1] = 1.0
            left_parked_pos[:, 2] = 10.0
            
            # 右手の退避位置: 空中右側 (x = 2.0, y = 0.0, z = 10.0)
            right_parked_pos = torch.zeros((num_envs, 3), device=self._device, dtype=torch.float32)
            right_parked_pos[:, 0] = 2.0
            right_parked_pos[:, 1] = -1.0
            right_parked_pos[:, 2] = 10.0
            
            hmd_parked_pos = torch.zeros((num_envs, 3), device=self._device, dtype=torch.float32)
            hmd_parked_pos[:, 0] = 2.0
            hmd_parked_pos[:, 1] = 0.0
            hmd_parked_pos[:, 2] = 10.0
            
            # それぞれ別の座標に配置
            left_id, right_id, hmd_id = self._vrhand_box_ids[0], self._vrhand_box_ids[1], self._vrhand_box_ids[2]
            
            self._engine.set_root_pos(env_ids, left_id, left_parked_pos)
            self._engine.set_root_vel(env_ids, left_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, left_id, 0.0)
            
            self._engine.set_root_pos(env_ids, right_id, right_parked_pos)
            self._engine.set_root_vel(env_ids, right_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, right_id, 0.0)
            
            self._engine.set_root_pos(env_ids, hmd_id, hmd_parked_pos)
            self._engine.set_root_vel(env_ids, hmd_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, hmd_id, 0.0)
            
            
            
            if self._vrbody_enabled and len(self._vrbody_box_ids) == 5:
                for index, obj_id in enumerate(self._vrbody_box_ids):
                    parked_pos = torch.zeros((num_envs, 3), device=self._device, dtype=torch.float32)
                    parked_pos[:, 0] = 4.0 + index
                    parked_pos[:, 2] = 10.0
                    self._engine.set_root_pos(env_ids, obj_id, parked_pos)
                    self._engine.set_root_vel(env_ids, obj_id, 0.0)
                    self._engine.set_root_ang_vel(env_ids, obj_id, 0.0)
            if self._vrhand_debug:
                print(f"[VRhand] simulation tensors ready; left parked at (-2, 0, 10), right parked at (2, 0, 10) for {num_envs} env(s)", flush=True)
        
        num_envs = self.get_num_envs()
        self._tar_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._tar_change_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        return

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        char_id = self._get_char_id()
        self._prev_root_pos[:] = self._engine.get_root_pos(char_id)
        return

    def _compute_obs(self, env_ids=None):
        # ベースとなる環境の観測を取得
        obs = super()._compute_obs(env_ids)

        return obs

    def get_disc_obs_space(self):
        disc_obs = self.fetch_disc_obs_demo(1)
        disc_obs_shape = list(disc_obs.shape[1:])
        disc_obs_dtype = torch_util.torch_dtype_to_numpy(disc_obs.dtype)
        disc_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=disc_obs_shape,
            dtype=disc_obs_dtype,
        )
        return disc_obs_space

    def get_num_disc_obs_steps(self):
        return self._num_disc_obs_steps

    def fetch_disc_obs_demo(self, num_samples):
        motion_ids, motion_times0 = self._sample_motion_times(num_samples)
        disc_obs = self._compute_disc_obs_demo(motion_ids, motion_times0)
        return disc_obs

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = self._fetch_disc_demo_data(motion_ids, motion_times0)
        
        if (self._track_global_root()):
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]
            
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)
        
        disc_obs = compute_disc_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=root_pos,
                                  root_rot=root_rot, 
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  joint_rot=joint_rot,
                                  dof_vel=dof_vel,
                                  key_pos=key_pos,
                                  global_obs=self._global_obs,
                                  root_height_obs=self._root_height_obs,
                                  dof_vel_obs=self._disc_dof_vel_obs)
        return disc_obs

    def _fetch_disc_demo_data(self, motion_ids, motion_times0):
        num_samples = motion_ids.shape[0]

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_disc_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -self._engine.get_timestep() * torch.arange(0, self._num_disc_obs_steps, device=self._device)
        time_steps = torch.flip(time_steps, dims=[0])
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        
        body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)

        root_pos = torch.reshape(root_pos, shape=[num_samples, self._num_disc_obs_steps, root_pos.shape[-1]])
        root_rot = torch.reshape(root_rot, shape=[num_samples, self._num_disc_obs_steps, root_rot.shape[-1]])
        root_vel = torch.reshape(root_vel, shape=[num_samples, self._num_disc_obs_steps, root_vel.shape[-1]])
        root_ang_vel = torch.reshape(root_ang_vel, shape=[num_samples, self._num_disc_obs_steps, root_ang_vel.shape[-1]])
        joint_rot = torch.reshape(joint_rot, shape=[num_samples, self._num_disc_obs_steps, joint_rot.shape[-2], joint_rot.shape[-1]])
        dof_vel = torch.reshape(dof_vel, shape=[num_samples, self._num_disc_obs_steps, dof_vel.shape[-1]])
        body_pos = torch.reshape(body_pos, shape=[num_samples, self._num_disc_obs_steps, body_pos.shape[-2], body_pos.shape[-1]])
        
        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._build_disc_obs_buffers()
        self._info["disc_obs"] = self._disc_obs_buf
        return

    def _build_disc_obs_buffers(self):
        num_envs = self.get_num_envs()
        n = self._num_disc_obs_steps
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        self._disc_hist_root_pos = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n, 
                                                                 shape=root_pos.shape[1:],
                                                                 dtype=root_pos.dtype,
                                                                 device=self._device)

        self._disc_hist_root_rot = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n,
                                                                 shape=root_rot.shape[1:],
                                                                 dtype=root_rot.dtype,
                                                                 device=self._device)

        self._disc_hist_root_vel = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n,
                                                                 shape=root_vel.shape[1:],
                                                                 dtype=root_vel.dtype,
                                                                 device=self._device)
        
        self._disc_hist_root_ang_vel = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                     buffer_len=n,
                                                                     shape=root_ang_vel.shape[1:],
                                                                     dtype=root_ang_vel.dtype,
                                                                     device=self._device)
        
        self._disc_hist_joint_rot = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                  buffer_len=n,
                                                                  shape=joint_rot.shape[1:],
                                                                  dtype=joint_rot.dtype,
                                                                  device=self._device)

        self._disc_hist_dof_vel = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                buffer_len=n,
                                                                shape=dof_vel.shape[1:],
                                                                dtype=dof_vel.dtype,
                                                                device=self._device)
        
        self._disc_hist_body_pos = circular_buffer.CircularBuffer(batch_size=num_envs,
                                                                 buffer_len=n,
                                                                 shape=body_pos.shape[1:],
                                                                 dtype=body_pos.dtype,
                                                                 device=self._device)

        disc_obs_space = self.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_obs_buf = torch.zeros([num_envs] + list(disc_obs_space.shape), device=self._device, dtype=disc_obs_dtype)
        return

    def _update_misc(self):
        super()._update_misc()
        self._update_disc_hist()
        self._update_task()
        self._send_vam_udp()
        self._update_vrhand_boxes()
        self._envbox_sync()
        return

    def _update_task(self):

        return

    def _update_disc_hist(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        self._disc_hist_root_pos.push(root_pos)
        self._disc_hist_root_rot.push(root_rot)
        self._disc_hist_root_vel.push(root_vel)
        self._disc_hist_root_ang_vel.push(root_ang_vel)
        self._disc_hist_dof_vel.push(dof_vel)
        self._disc_hist_body_pos.push(body_pos)
        
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        self._disc_hist_joint_rot.push(joint_rot)
        return

    def _update_ref_motion(self):
        if (self._enable_ref_char()):
            super()._update_ref_motion()
        return

    def _update_observations(self, env_ids=None):
        super()._update_observations(env_ids)

        if (env_ids is None or len(env_ids) > 0):
            self._update_disc_obs(env_ids)
        return

    def _update_disc_obs(self, env_ids=None):
        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        joint_rot = self._disc_hist_joint_rot.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()
        body_pos = self._disc_hist_body_pos.get_all()
        
        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            joint_rot = joint_rot[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]
        
        if (self._track_global_root()):
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]
            
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        disc_obs = compute_disc_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=root_pos,
                                  root_rot=root_rot, 
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  joint_rot=joint_rot,
                                  dof_vel=dof_vel,
                                  key_pos=key_pos,
                                  global_obs=self._global_obs,
                                  root_height_obs=self._root_height_obs,
                                  dof_vel_obs=self._disc_dof_vel_obs)

        if (env_ids is None):
            self._disc_obs_buf[:] = disc_obs
        else:
            self._disc_obs_buf[env_ids] = disc_obs

        return

    def _update_reward(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self._reset_disc_hist(env_ids)
            #self._reset_location_task(env_ids)
        return

    def _reset_disc_hist(self, env_ids):
        motion_ids = self._motion_ids[env_ids]
        motion_times0 = self._get_motion_times(env_ids)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = self._fetch_disc_demo_data(motion_ids, motion_times0)
        
        self._disc_hist_root_pos.fill(env_ids, root_pos)
        self._disc_hist_root_rot.fill(env_ids, root_rot)
        self._disc_hist_root_vel.fill(env_ids, root_vel)
        self._disc_hist_root_ang_vel.fill(env_ids, root_ang_vel)
        self._disc_hist_joint_rot.fill(env_ids, joint_rot)
        self._disc_hist_dof_vel.fill(env_ids, dof_vel)
        self._disc_hist_body_pos.fill(env_ids, body_pos)
        return

    def _reset_ref_motion(self, env_ids):
        n = len(env_ids)
        motion_ids, motion_times = self._sample_motion_times(n)
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        # 1. まずは既存のロジックで全対象環境にデフォルト（通常リセット）の値をセット
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel
        
        # 2. 最初は全環境を通常のFKで計算（これで全環境のベースラインが保証される）
        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                             self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot

        dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
        self._ref_dof_pos[env_ids] = dof_pos

        # --- 💥 ここから：条件付き引き継ぎの割り込み処理 💥 ---
        char_id = self._get_char_id()
        
        # エンジンから全環境の生位置を取得 [num_envs, 3]
        actual_root_pos_np = self._vam_to_numpy(self._engine.get_root_pos(char_id))
        actual_root_pos_torch = torch.from_numpy(actual_root_pos_np).to(device=self._device, dtype=torch.float)
        
        # 💡 【選別ロジック】引き継ぎを適用したい「有効な環境（inherit_ids）」を絞り込む
        inherit_ids = []
        for idx in env_ids:
            idx_item = idx.item() if hasattr(idx, 'item') else idx
            
            # ① 初回判定: progress_buf が 0 の場合は初回なのでスキップ
            is_initial = False
            if hasattr(self, 'progress_buf'):
                if self.progress_buf[idx_item] <= 0:
                    is_initial = True
                    
            # ② 転倒判定: キャラクターの現在の高さ（Z座標）が 0.5m 未満なら転倒とみなしてスキップ
            # (※タスク内に self._is_fall_down() や self.die_buf があればそちらに差し替えてもOKです)
            current_z = actual_root_pos_torch[idx_item, 2].item()
            is_fallen = (current_z < 0.5) 
            
            # 初回でも転倒でもない場合のみ、姿勢引き継ぎの対象にする
            if not is_initial and not is_fallen:
                inherit_ids.append(idx_item)
                

        if len(inherit_ids) > 0:
            inherit_ids = torch.tensor(inherit_ids, dtype=torch.long, device=self._device)
            
            # 該当環境（inherit_ids）の位置・回転を取得して上書き
            actual_root_rot_np = self._vam_to_numpy(self._engine.get_root_rot(char_id))
            actual_root_rot_torch = torch.from_numpy(actual_root_rot_np).to(device=self._device, dtype=torch.float)
            
            self._ref_root_pos[inherit_ids] = actual_root_pos_torch[inherit_ids]
            self._ref_root_rot[inherit_ids] = actual_root_rot_torch[inherit_ids]

            # # ルート速度・角速度の同期
            # actual_root_vel_np = None
            # for method_name in ("get_root_vel", "get_root_velocity", "get_root_linear_velocity"):
                # method = getattr(self._engine, method_name, None)
                # if callable(method):
                    # try: actual_root_vel_np = self._vam_to_numpy(method(char_id)); break
                    # except Exception: pass
            # if actual_root_vel_np is not None:
                # actual_root_vel_torch = torch.from_numpy(actual_root_vel_np).to(device=self._device, dtype=torch.float)
                # self._ref_root_vel[inherit_ids] = actual_root_vel_torch[inherit_ids]


        
        return

import envs.task_location_env_vam as task_location_env

#3. run.py が「amp_env.AMPEnv」を呼び出すため、名前をタスク用に差し替える
AMPEnv = task_location_env.TaskLocationEnv
    
    
@torch.jit.script
def compute_disc_vel_obs(ref_root_rot, root_vel, root_ang_vel, dof_vel, global_obs, dof_vel_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_vel.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                               heading_inv_rot_expand.shape[2]))
        
        root_vel_obs_flat = torch.reshape(root_vel, [root_vel.shape[0] * root_vel.shape[1], root_vel.shape[2]])
        root_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_vel_obs_flat)
        root_vel_obs = torch.reshape(root_vel_obs_flat, root_vel.shape)
        
        root_ang_vel_obs_flat = torch.reshape(root_ang_vel, [root_ang_vel.shape[0] * root_ang_vel.shape[1], root_ang_vel.shape[2]])
        root_ang_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_ang_vel_obs_flat)
        root_ang_vel_obs = torch.reshape(root_ang_vel_obs_flat, root_ang_vel.shape)
    else:
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel

    obs = [root_vel_obs, root_ang_vel_obs]

    if (dof_vel_obs):
        obs = obs + [dof_vel]

    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_disc_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs, dof_vel_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool) -> Tensor

    pos_obs = deepmimic_env.compute_tar_obs(ref_root_pos=ref_root_pos, 
                                               ref_root_rot=ref_root_rot, 
                                               root_pos=root_pos, 
                                               root_rot=root_rot, 
                                               joint_rot=joint_rot, 
                                               key_pos=key_pos,
                                               global_obs=global_obs, 
                                               root_height_obs=root_height_obs)

    vel_obs = compute_disc_vel_obs(ref_root_rot=ref_root_rot, 
                                  root_vel=root_vel, 
                                  root_ang_vel=root_ang_vel, 
                                  dof_vel=dof_vel,
                                  global_obs=global_obs,
                                  dof_vel_obs=dof_vel_obs)

    disc_obs = torch.cat([pos_obs, vel_obs], dim=-1)
    disc_obs = torch.reshape(disc_obs, [disc_obs.shape[0], -1])

    return disc_obs
    
    

