# Inference-only version: AMP discriminator observations removed.
# Actor observation and DeepMimic reference-motion path are unchanged.
# Preserve Apache-2.0 original attribution below.
# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import os
import socket
import struct
import numpy as np
import torch

import engines.engine as engine
import envs.deepmimic_env as deepmimic_env
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
        # Analog trigger hysteresis: a shallow squeeze counts as a press.
        self._vrhand_trigger_press_threshold = float(env_config.get("vrhand_trigger_press_threshold", 0.25))
        self._vrhand_trigger_release_threshold = float(env_config.get("vrhand_trigger_release_threshold", 0.10))
        if not (0.0 <= self._vrhand_trigger_release_threshold < self._vrhand_trigger_press_threshold <= 1.0):
            raise ValueError("VR trigger thresholds must satisfy 0 <= release < press <= 1")
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
        # Lightweight test: torso capsule only; retain HMD UDP tracking without HMD physics body.
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
        self._vrpd_masses = {"left_hand": 5.0, "right_hand": 5.0, "left_forearm": 1.7, "right_forearm": 1.7, "left_upper_arm": 1.5, "right_upper_arm": 1.5}
        self._vrhand_left_pd_asset = os.path.join("data", "assets", "objects", "vrhand_left_pd_collision_box.xml")
        self._vrhand_right_pd_asset = os.path.join("data", "assets", "objects", "vrhand_right_pd_collision_box.xml")
        if self._vrhand_enabled:
            self._write_vrhand_asset()
            if self._vrpd_enabled:
                self._write_vrhand_left_pd_asset()
                self._write_vrhand_right_pd_asset()
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
        # Test-only: select rotation mapping for foot/shin/thigh on both sides.
        self._vam_leg_rotation_mode = str(env_config.get("vam_leg_rotation_mode", "legacy")).strip().lower()
        if self._vam_leg_rotation_mode not in ("legacy", "matrix", "matrix_xz_pos", "matrix_xz_neg"):
            raise ValueError("vam_leg_rotation_mode must be one of: legacy, matrix, matrix_xz_pos, matrix_xz_neg")
        if self._vam_enabled:
            print(f"[VaM UDP] leg rotation mode: {self._vam_leg_rotation_mode}", flush=True)
        # Test switch for all 12 limb rotations. Torso/pelvis/head always legacy.
        self._vam_limb_rotation_mode = str(env_config.get("vam_limb_rotation_mode", "legacy")).strip().lower()
        if self._vam_limb_rotation_mode not in ("legacy", "matrix_bind"):
            raise ValueError("vam_limb_rotation_mode must be legacy or matrix_bind")
        self._vam_limb_bind_euler_deg = env_config.get("vam_limb_bind_euler_deg", {}) or {}
        self._vam_limb_bind_quat_xyzw = env_config.get("vam_limb_bind_quat_xyzw", {}) or {}
        if not isinstance(self._vam_limb_bind_euler_deg, dict) or not isinstance(self._vam_limb_bind_quat_xyzw, dict):
            raise ValueError("vam_limb_bind_euler_deg and vam_limb_bind_quat_xyzw must be mappings")
        valid_limb_keys = self._vam_limb_parts
        for key, angles in self._vam_limb_bind_euler_deg.items():
            if key not in valid_limb_keys or len(angles) != 3 or not np.all(np.isfinite(np.asarray(angles, dtype=float))):
                raise ValueError("Invalid vam_limb_bind_euler_deg entry: " + str(key))
        for key, quat in self._vam_limb_bind_quat_xyzw.items():
            if key not in valid_limb_keys or len(quat) != 4 or not np.all(np.isfinite(np.asarray(quat, dtype=float))) or np.linalg.norm(quat) < 1e-8:
                raise ValueError("Invalid vam_limb_bind_quat_xyzw entry: " + str(key))
        if self._vam_enabled:
            print(f"[VaM UDP] limb rotation mode: {self._vam_limb_rotation_mode} "
                  f"(bind euler={len(self._vam_limb_bind_euler_deg)}, "
                  f"bind quat={len(self._vam_limb_bind_quat_xyzw)})", flush=True)
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
            if self._vrbody_enabled:
                body_ids = []
                for name in ("torso",):
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
                self._vrhand_box_ids = [left_id, right_id]
                if self._vrhand_debug:
                    print(f"[VRhand] collision boxes created: left_id={left_id}, right_id={right_id}, hmd=tracking_only env_id={env_id}, asset={self._vrhand_asset}", flush=True)
            else:
                assert self._vrhand_box_ids == [left_id, right_id]
        return

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        self._envbox_park_all()
        
        if self._vrhand_enabled and len(self._vrhand_box_ids) == 2:
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
            
            # Only left and right hand rigid bodies exist. HMD is tracking-only.
            left_id, right_id = self._vrhand_box_ids
            self._engine.set_root_pos(env_ids, left_id, left_parked_pos)
            self._engine.set_root_vel(env_ids, left_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, left_id, 0.0)
            self._engine.set_root_pos(env_ids, right_id, right_parked_pos)
            self._engine.set_root_vel(env_ids, right_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, right_id, 0.0)

            if self._vrbody_enabled and len(self._vrbody_box_ids) == 1:
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


    def _update_misc(self):
        super()._update_misc()
        self._update_task()
        self._send_vam_udp()
        self._update_vrhand_boxes()
        self._envbox_sync()
        return

    def _update_task(self):

        return


    def _update_ref_motion(self):
        if (self._enable_ref_char()):
            super()._update_ref_motion()
        return


    def _update_reward(self):
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
    
    

    
    

