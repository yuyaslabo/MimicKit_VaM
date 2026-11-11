# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.
import numpy as np
import torch
import engines.engine as engine
#import envs.smp_env as smp_env
import envs.amp_env_vam as amp_env
import util.torch_util as torch_util
from util.logger import Logger
from engines.newton_engine import SimState

class TaskLocationEnv(amp_env.AMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._tar_speed = env_config["tar_speed"]
        # HMD -> target XY distance bounds (independent from task tar_dist_min/max).
        self._hmd_target_min_distance = float(env_config.get("hmd_target_min_distance", 0.2))
        self._hmd_target_max_distance = float(env_config.get("hmd_target_max_distance", 0.55))
        if not (0.0 <= self._hmd_target_min_distance < self._hmd_target_max_distance):
            raise ValueError("hmd_target_min_distance must be >= 0 and less than hmd_target_max_distance")
        # XY movement of the target point that restores the full available distance.
        self._hmd_target_recover_distance = float(env_config.get("hmd_target_recover_distance", 1.0))
        if self._hmd_target_recover_distance <= 0.0:
            raise ValueError("hmd_target_recover_distance must be positive")
        # Ignore tiny per-step target displacement when restoring distance.
        self._hmd_target_recover_deadzone = float(env_config.get("hmd_target_recover_deadzone", 0.005))
        if self._hmd_target_recover_deadzone < 0.0:
            raise ValueError("hmd_target_recover_deadzone must be nonnegative")
        # Hysteresis for continuous HMD forward movement (m/s).
        # Time constant of the forward-velocity EMA (seconds).
        self._hmd_forward_speed_smoothing = float(env_config.get("hmd_forward_speed_smoothing", 0.12))
        if self._hmd_forward_speed_smoothing < 0.0:
            raise ValueError("hmd_forward_speed_smoothing must be nonnegative")
        self._hmd_forward_speed_on = float(env_config.get("hmd_forward_speed_on", 0.35))
        self._hmd_forward_speed_off = float(env_config.get("hmd_forward_speed_off", 0.2))
        if not (0.0 <= self._hmd_forward_speed_off < self._hmd_forward_speed_on):
            raise ValueError("hmd_forward_speed_off must be >= 0 and less than hmd_forward_speed_on")
        self._tar_dist_min = env_config.get("tar_dist_min", 0.0)
        self._tar_dist_max = env_config["tar_dist_max"]
        assert(self._tar_dist_max >= self._tar_dist_min)

        self._dist_threshold = env_config.get("dist_threshold", 0.5)

        self._pos_err_scale = env_config.get("pos_err_scale", 0.5)
        self._vel_err_scale = env_config.get("vel_err_scale", 4.0)
        self._pos_reward_w = env_config.get("pos_reward_w", 0.5)
        self._vel_reward_w = env_config.get("vel_reward_w", 0.4)
        self._face_reward_w = env_config.get("face_reward_w", 0.1)
        self._vrhand_position_offset = np.array([0.0, 0.18, 0.0], dtype=np.float32)
        self._vrhand_triggers = {"left": 0.0, "right": 0.0}
        self._vrhand_grabbed_targets = {"left": None, "right": None}
        
        

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        if hasattr(self, "_engine") and hasattr(self._engine, "_sim_state"):
            self._sim_state = self._engine._sim_state
        # if hasattr(self, "_engine") and hasattr(self._engine, "_sim_model"):
            # self._sim_state = SimState(self._engine._sim_model, num_envs)
        # else:
            # # 万が一、engine側にもない場合は親クラスがすでに _sim_state を持っていないか確認
            # if hasattr(self, "_engine") and hasattr(self._engine, "_sim_state"):
                # self._sim_state = self._engine._sim_state
        # if hasattr(self._sim_state, '_wp_body_force') and self._sim_state._wp_body_force is not None:
            # if hasattr(self._sim_state._wp_body_force, "zero_"):
                # self._sim_state._wp_body_force.zero_()
        return
    
    def _build_envs(self, config, num_envs):
        if (self._visualize):
            self._marker_ids = []

        super()._build_envs(config, num_envs)
        return
    
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        
        if (self._visualize):
            marker_id = self._build_marker(env_id)

            if (env_id == 0):
                self._marker_ids.append(marker_id)
            else:
                marker_id0 = self._marker_ids[0]
                assert(marker_id0 == marker_id)
        return

    def _build_marker(self, env_id):
        asset_file = "data/assets/objects/location_marker.xml"

        marker_id = self._engine.create_obj(env_id=env_id, 
                                            obj_type=engine.ObjType.rigid,
                                            asset_file=asset_file, 
                                            name="marker",
                                            is_visual=True,
                                            fix_root=True,
                                            color=[0.5, 0.5, 0.5])
        
        return marker_id

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        
        num_envs = self.get_num_envs()
        self._tar_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._hmd_target_initialized = torch.zeros(num_envs, device=self._device, dtype=torch.bool)
        self._hmd_target_prev_pos = torch.zeros((num_envs, 2), device=self._device, dtype=torch.float)
        self._hmd_target_prev_facing = torch.zeros((num_envs, 2), device=self._device, dtype=torch.float)
        self._hmd_forward_continuous = torch.zeros(num_envs, device=self._device, dtype=torch.bool)
        self._hmd_forward_speed_filtered = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        return
    
    def _get_marker_id(self):
        return self._marker_ids[0]

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._record_char_root()
        return

    def _record_char_root(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        self._prev_root_pos[:] = root_pos
        return

    def _update_marker(self, env_ids):
        marker_id = self._get_marker_id()
        rot = torch.tensor([0, 0, 0, 1], device=self._device, dtype=torch.float32)

        self._engine.set_root_pos(env_ids, marker_id, self._tar_pos[env_ids])
        self._engine.set_root_rot(env_ids, marker_id, rot)
        self._engine.set_root_vel(env_ids, marker_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, marker_id, 0.0)
        return
    
    def _update_task(self):
        super()._update_task()
        # HMDの最新位置・向きを使って毎制御ステップ更新する。
        # tar_change_time_min/maxによる周期制御は使用しない。
        env_ids = torch.arange(self.get_num_envs(), device=self._device, dtype=torch.long)
        self._reset_tar(env_ids)
        if self._visualize:
            self._update_marker(env_ids)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        if getattr(self, "_vrhand_target_frozen", False):
            return  # Keep the same target even if the agent resets
        # Reset should spawn a fresh target 0.5m in front of the HMD.
        self._hmd_target_initialized[env_ids] = False
        self._hmd_forward_continuous[env_ids] = False
        self._hmd_forward_speed_filtered[env_ids] = 0.0
        self._reset_tar(env_ids)
        if (self._visualize):
            self._update_marker(env_ids)
        return
        
    def _reset_tar(self, env_ids):
        if getattr(self, "_vrhand_target_frozen", False):
            return  # Hold the current world-space target, independent of HMD motion
        # The target is ALWAYS directly in front of the HMD.
        # Only its distance is allowed to vary between 0.1 and 0.5 metres.
        if env_ids.numel() == 0:
            return

        # Do not initialize against the placeholder (0,0,0) HMD pose.
        # The first valid HMD packet will initialize the target at max distance.
        if not getattr(self, "_hmd_pose_received", False):
            self._hmd_target_initialized[env_ids] = False
            self._hmd_forward_continuous[env_ids] = False
            self._hmd_forward_speed_filtered[env_ids] = 0.0
            return

        # On the first valid HMD sample, freeze the existing startup target
        # without moving it to a position in front of the HMD.
        if not getattr(self, "_vrhand_initial_target_locked_once", False):
            self._vrhand_initial_target_locked_once = True
            self._vrhand_target_frozen = True
            if getattr(self, "_vrhand_debug", False):
                print(f"[VRhand] first HMD pose: startup target FROZEN at {self._tar_pos[env_ids].detach().cpu().tolist()}", flush=True)
            return

        hmd_pos = self._latest_hmd_pos
        hmd_quat = self._latest_hmd_quat
        if hmd_pos.dim() == 1:
            hmd_pos = hmd_pos.unsqueeze(0)
        if hmd_quat.dim() == 1:
            hmd_quat = hmd_quat.unsqueeze(0)

        # One tracked HMD may be shared across several environments.
        if hmd_pos.shape[0] == 1:
            hmd_pos = hmd_pos.expand(self.get_num_envs(), -1)
        if hmd_quat.shape[0] == 1:
            hmd_quat = hmd_quat.expand(self.get_num_envs(), -1)

        player_xy = hmd_pos[env_ids, :2]
        quat = hmd_quat[env_ids]
        z, w = quat[:, 2], quat[:, 3]
        facing = torch.stack((1.0 - 2.0 * z.square(), 2.0 * w * z), dim=-1)
        facing = torch.nn.functional.normalize(facing, dim=-1, eps=1e-6)

        initialized = self._hmd_target_initialized[env_ids]
        previous_player_xy = self._hmd_target_prev_pos[env_ids]
        previous_distance = torch.linalg.vector_norm(
            self._tar_pos[env_ids, :2] - previous_player_xy, dim=-1
        )

        # As before, only movement along the local forward axis changes
        # the HMD-to-target distance (forward shortens, backward extends).
        displacement = player_xy - previous_player_xy
        previous_facing = self._hmd_target_prev_facing[env_ids]
        mid_facing = torch.nn.functional.normalize(previous_facing + facing, dim=-1, eps=1e-6)
        mid_facing = torch.where(initialized.unsqueeze(-1), mid_facing, facing)
        forward_movement = torch.sum(displacement * mid_facing, dim=-1)
        dt = float(self._engine.get_timestep())
        forward_speed = forward_movement / max(dt, 1e-6)
        # First valid HMD frame starts at zero velocity, not at a velocity
        # computed from the previous placeholder HMD position.
        forward_speed = torch.where(initialized, forward_speed, torch.zeros_like(forward_speed))
        alpha = dt / (self._hmd_forward_speed_smoothing + dt) if dt > 0.0 else 1.0
        filtered_speed = self._hmd_forward_speed_filtered[env_ids] + alpha * (
            forward_speed - self._hmd_forward_speed_filtered[env_ids]
        )
        self._hmd_forward_speed_filtered[env_ids] = filtered_speed
        forward_continuous = self._hmd_forward_continuous[env_ids]
        # Switch ON at >= 0.5 m/s; remain ON until <= 0.3 m/s.
        forward_continuous = torch.where(
            forward_continuous,
            filtered_speed > self._hmd_forward_speed_off,
            filtered_speed >= self._hmd_forward_speed_on,
        ) & initialized
        self._hmd_forward_continuous[env_ids] = forward_continuous
        distance = (previous_distance - forward_movement).clamp(
            self._hmd_target_min_distance, self._hmd_target_max_distance
        )
        distance = torch.where(
            initialized,
            distance,
            torch.full_like(distance, self._hmd_target_max_distance)
        )

        # Recover continuously in proportion to the target point's XY travel.
        # Measure motion BEFORE applying recovery. Previous frame's final target
        # is the reference, so recovery-induced displacement is not counted on
        # the next frame when the HMD is stationary.
        previous_target_xy = self._tar_pos[env_ids, :2].clone()
        candidate_xy = player_xy + facing * distance.unsqueeze(-1)
        target_movement = torch.linalg.vector_norm(candidate_xy - previous_target_xy, dim=-1)
        recovery_range = self._hmd_target_max_distance - self._hmd_target_min_distance
        effective_movement = (target_movement - self._hmd_target_recover_deadzone).clamp_min(0.0)
        recovery = recovery_range * (effective_movement / self._hmd_target_recover_distance)
        recovery = torch.where(initialized, recovery, torch.zeros_like(recovery))
        distance = (distance + recovery).clamp(
            self._hmd_target_min_distance, self._hmd_target_max_distance
        )
        # Continuous forward movement overrides shortening/recovery entirely.
        distance = torch.where(
            forward_continuous,
            torch.full_like(distance, self._hmd_target_max_distance),
            distance,
        )

        # Always place the target directly in front of the current HMD yaw.
        target_xy = player_xy + facing * distance.unsqueeze(-1)
        self._tar_pos[env_ids, :2] = target_xy
        self._hmd_target_prev_pos[env_ids] = player_xy
        self._hmd_target_prev_facing[env_ids] = facing
        self._hmd_target_initialized[env_ids] = True
        return

    def _get_facing_player_pos(self):
        # Shared by task observation and reward: never train on different player positions.
        if hasattr(self, '_latest_hmd_pos'):
            player_pos = self._latest_hmd_pos.clone()
        else:
            player_pos = self._tar_pos.clone()
        world_pos = getattr(self, '_vrhand_remote_player_world_pos', None)
        if world_pos is not None:
            # Absolute world-space point: ignore all later HMD translation and yaw.
            world_pos = world_pos.to(device=player_pos.device, dtype=player_pos.dtype)
            player_pos = world_pos.reshape(1, 3).clone()
        if player_pos.dim() == 1:
            player_pos = player_pos.unsqueeze(0)
        if player_pos.shape[0] == 1 and self.get_num_envs() > 1:
            player_pos = player_pos.expand(self.get_num_envs(), -1)
        return player_pos

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_root_rot = self._engine.get_root_rot(char_id)
        tar_pos = self._tar_pos
        
        player_pos = self._get_facing_player_pos()
        


        if (env_ids is not None):
            char_root_pos = char_root_pos[env_ids]
            char_root_rot = char_root_rot[env_ids]
            tar_pos = tar_pos[env_ids]
            player_pos = player_pos[env_ids]
            
        # 単一環境の場合の次元補正（[3] ->）
        if player_pos.dim() == 1:
            player_pos = player_pos.unsqueeze(0)
        
        # ターゲットとプレイヤー（HMD）の両方のローカル座標を計算して結合 (2次元 + 2次元 = 4次元追加)
        task_obs = compute_location_observations(char_root_pos, char_root_rot, tar_pos, player_pos)
        obs = torch.cat([obs, task_obs], dim=-1)

        return obs

    def _update_reward(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_root_rot = self._engine.get_root_rot(char_id)
        
        player_pos = self._get_facing_player_pos()

        self._reward_buf[:] = compute_location_reward(root_pos=char_root_pos,
                                                      prev_root_pos=self._prev_root_pos,
                                                      root_rot=char_root_rot,
                                                      tar_pos=self._tar_pos, 
                                                      player_pos=player_pos,  # 👈 ここに新しい引数を追加
                                                      tar_speed=self._tar_speed,
                                                      dt=self._engine.get_timestep(),
                                                      dist_threshold=self._dist_threshold,
                                                      pos_err_scale=self._pos_err_scale,
                                                      vel_err_scale=self._vel_err_scale,
                                                      pos_reward_w=self._pos_reward_w,
                                                      vel_reward_w=self._vel_reward_w,
                                                      face_reward_w=self._face_reward_w
                                                      )
        
        #Logger.print(f"reward:{self._reward_buf[0].item():.4f}",end='\r')
        # 💡 Logger.print に player_pos と self._tar_pos のXYZ座標を追加（第0環境のデータを抽出）
        # 💡 【ログ表示の修正】1次元・2次元どちらのテンソル形状でも安全にXYZを抽出できるよう修正
        flat_p = player_pos.squeeze()
        flat_t = self._tar_pos.squeeze()
        
        p_x, p_y, p_z = flat_p[0].item(), flat_p[1].item(), flat_p[2].item()
        t_x, t_y, t_z = flat_t[0].item(), flat_t[1].item(), flat_t[2].item()
        
        Logger.print(
            f"reward:{self._reward_buf[0].item():.4f} | "
            f"Player:[{p_x:.2f}, {p_y:.2f}, {p_z:.2f}] | "
            f"Target:[{t_x:.2f}, {t_y:.2f}, {t_z:.2f}]", 
            end='\r'
        )
        return

    def _update_misc(self):
        super()._update_misc()

        self._update_task()
        return

    def _render_scene(self):
        super()._render_scene()
        self._render_location_lines()
        return

    def _render_location_lines(self):
        line_width = 2.0
        col = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)

        starts = char_root_pos.cpu().numpy()
        ends = self._tar_pos.cpu().numpy()
        
        num_envs = self.get_num_envs()
        for i in range(num_envs):
            curr_start = starts[i:i + 1]
            curr_end = ends[i:i + 1]
            self._engine.draw_lines(i, curr_start, curr_end, col, line_width)

        return
        
    def _vam_pos_to_engine(self, pos_vam):
        p = np.asarray(pos_vam, dtype=np.float32) - self._vrhand_position_offset
        return np.array([p[2], -p[0], p[1]], dtype=np.float32)
        


    def _vam_quat_to_engine(self, quat_vam_xyzw):
        qx, qy, qz, qw = np.asarray(quat_vam_xyzw, dtype=np.float64)
        q = np.asarray([-qx, -qz, -qy, qw], dtype=np.float64)
        norm = np.linalg.norm(q)
        if not np.isfinite(q).all() or norm < 1e-8:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return (q / norm).astype(np.float32)
    

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_location_observations(root_pos, root_rot, tar_pos, player_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

    heading_rot = torch_util.calc_heading_quat_inv(root_rot)
    local_tar_pos = torch_util.quat_rotate(heading_rot, tar_pos - root_pos)
    local_player_pos = torch_util.quat_rotate(heading_rot, player_pos - root_pos)
    player_dir = torch.nn.functional.normalize(local_player_pos[..., 0:2],dim=-1,eps=1e-6)
    obs = torch.cat([local_tar_pos[..., 0:2],player_dir], dim=-1)
    return obs

@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, tar_pos, player_pos, tar_speed, dt, dist_threshold, pos_err_scale, vel_err_scale, pos_reward_w, vel_reward_w, face_reward_w):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float) -> Tensor
    # 💡 引数に「Tensor型のplayer_pos」を追加したため、上記アノテーションの4番目と5番目を「Tensor, Tensor」へと正しく拡張しています
    
    pos_diff = tar_pos - root_pos
    pos_err = pos_diff[..., 0] * pos_diff[..., 0] + pos_diff[..., 1] * pos_diff[..., 1]
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    tar_dir = tar_pos - root_pos
    tar_dir = tar_dir[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))

    wrong_dir = tar_dir_speed <= 0
    vel_reward[wrong_dir] = 0

    # エージェントの現在の正面方向ベクトルを計算（実績のあるX軸プラス基準）
    heading_rot = torch_util.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = torch_util.quat_rotate(heading_rot, facing_dir)
    facing_dir_2d = facing_dir[..., 0:2]

    # 通常時：ターゲットの方を向いているかどうかの facing_err
    facing_err_tar = torch.sum(tar_dir * facing_dir_2d, dim=-1)
    facing_reward = (facing_err_tar + 1.0) * 0.5

    # プレイヤー方向の単位ベクトルを計算
    player_dir = player_pos - root_pos
    player_dir = player_dir[..., 0:2]
    player_dir = torch.nn.functional.normalize(player_dir, dim=-1)
    
    # プレイヤーの方を向いているかどうかの facing_err
    facing_err_player = torch.sum(player_dir * facing_dir_2d, dim=-1)
    facing_reward_player = (facing_err_player + 1.0) * 0.5

    # 条件分岐：pos_err < dist_threshold のマスク作成
    dist_mask = pos_err < dist_threshold
    
    # ターゲットに十分近づいた（dist_mask == True）時は、プレイヤーの方を向いていれば報酬を与える仕様に変更
    facing_reward = torch.where(dist_mask, facing_reward_player, facing_reward)
    vel_reward[dist_mask] = 1.0  # 速度報酬のマスク処理は元のロジックを維持

    reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    return reward