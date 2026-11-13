import numpy as np
import torch

import engines.engine as engine
#import envs.smp_env as smp_env
import envs.amp_env as amp_env
import util.torch_util as torch_util
from util.logger import Logger

class TaskLocationEnv(amp_env.AMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._tar_change_time_min = env_config["tar_change_time_min"]
        self._tar_change_time_max = env_config["tar_change_time_max"]
        self._tar_dist_min = env_config.get("tar_dist_min", 0.0)
        self._tar_dist_max = env_config["tar_dist_max"]
        assert(self._tar_dist_max >= self._tar_dist_min)

        self._dist_threshold = env_config.get("dist_threshold", 0.5)
                                                                  
                                                                                                    

        self._pos_err_scale = env_config.get("pos_err_scale", 0.5)
        self._vel_err_scale = env_config.get("vel_err_scale", 4.0)
        self._pos_reward_w = env_config.get("pos_reward_w", 0.4) #0.5
        self._vel_reward_w = env_config.get("vel_reward_w", 0.3) #0.4
        self._face_reward_w = env_config.get("face_reward_w", 0.3) #0.1
        # Weights used only when the character is sufficiently close to the target.
        self._near_pos_reward_w = float(env_config.get("near_pos_reward_w", 0.4))
        self._near_vel_reward_w = float(env_config.get("near_vel_reward_w", 0.2))
        self._near_face_reward_w = float(env_config.get("near_face_reward_w", 0.4))

        # Virtual player: independent polar offset around the moving target.
        self._player_radius_min = float(env_config.get("player_radius_min", 1.0))
        self._player_radius_max = float(env_config.get("player_radius_max", 1.0))
        self._player_angular_speed = float(env_config.get("player_angular_speed", 0.5)) # rad/s
        self._player_radial_speed = float(env_config.get("player_radial_speed", 0.01)) # m/s
        self._player_change_time_min = float(env_config.get("player_change_time_min", 2.0))
        self._player_change_time_max = float(env_config.get("player_change_time_max", 5.0))
        self._player_velocity_smoothing = float(env_config.get("player_velocity_smoothing", 1.0)) # seconds
        self._player_dt = 1.0 / float(engine_config.get("control_freq", 30))
        assert 0.0 < self._player_radius_min <= self._player_radius_max
        assert self._player_angular_speed >= 0.0 and self._player_radial_speed >= 0.0
        assert 0.0 < self._player_change_time_min <= self._player_change_time_max
        assert self._player_velocity_smoothing > 0.0

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return
    
    def _build_envs(self, config, num_envs):
        if (self._visualize):
            self._marker_ids = []
            self._player_marker_ids = []

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

            player_marker_id = self._build_player_marker(env_id)
            if env_id == 0:
                self._player_marker_ids.append(player_marker_id)
            else:
                assert self._player_marker_ids[0] == player_marker_id
        return

    def _build_player_marker(self, env_id):
        return self._engine.create_obj(env_id=env_id,
                                       obj_type=engine.ObjType.rigid,
                                       asset_file="data/assets/objects/location_marker.xml",
                                       name="player_marker",
                                       is_visual=True,
                                       fix_root=True,
                                       color=[0.1, 0.45, 1.0])

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
        self._tar_speed = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._tar_change_times = torch.zeros([num_envs], device=self._device, dtype=torch.float)
        self._prev_root_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._player_pos = torch.zeros([num_envs, 3], device=self._device, dtype=torch.float)
        self._player_radius = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._player_angle = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._player_ang_vel = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._player_rad_vel = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._player_target_ang_vel = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._player_target_rad_vel = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._player_change_times = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        return
    
    def _get_marker_id(self):
        return self._marker_ids[0]

    def _update_player_marker(self, env_ids=None):
        marker_id = self._player_marker_ids[0]
        rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device, dtype=torch.float32)
        pos = self._player_pos if env_ids is None else self._player_pos[env_ids]
        self._engine.set_root_pos(env_ids, marker_id, pos)
        self._engine.set_root_rot(env_ids, marker_id, rot)
        self._engine.set_root_vel(env_ids, marker_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, marker_id, 0.0)
        return

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

        if env_ids is None:
            self._engine.set_root_pos(None, marker_id, self._tar_pos)
            self._engine.set_root_rot(None, marker_id, rot)
        else:
            self._engine.set_root_pos(env_ids, marker_id, self._tar_pos[env_ids])
            self._engine.set_root_rot(env_ids, marker_id, rot)
            
        self._engine.set_root_vel(env_ids, marker_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, marker_id, 0.0)
        return
    
    def _update_task(self):
        super()._update_task()
        # reset_task_mask = self._time_buf >= self._tar_change_times
        # rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()

        # if len(rest_env_ids) > 0:
            # self._reset_task(rest_env_ids)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_task(env_ids)
            self._reset_pull_task(env_ids)
        return

    def _reset_task(self, env_ids):
        # 1. リセットされた環境に対して、その時点の「数秒後」のターゲットを設定
        self._reset_tar(env_ids)
        self._reset_player(env_ids)
        
        # ※ ランダム時間でのターゲット切り替え（_tar_change_times）は、
        # モーション追従型にする場合は不要になるため、計算だけ残すか無視して問題ありません。
        rand_dt = torch.rand(env_ids.shape[0], device=self._device, dtype=torch.float)
        rand_dt = (self._tar_change_time_max - self._tar_change_time_min) * rand_dt + self._tar_change_time_min
        self._tar_change_times[env_ids] = self._time_buf[env_ids] + rand_dt
        
        if (self._visualize):
            self._update_marker(env_ids)
            self._update_player_marker(env_ids)
        return
        
    
    # def _update_tar(self):
        # # すべての環境（またはアクティブな環境）の現在のモーション進行時間を取得
        # motion_ids = self._motion_ids
        # motion_times = self._get_motion_times() # 既存の便利関数を活用

        # # 現在の時間から2秒先を計算
        # lookahead_time = 2.0
        # tar_times = motion_times + lookahead_time

        # # 各モーションの最大長でクランプ
        # motion_lengths = self._motion_lib._motion_lengths[motion_ids]
        # clamped_tar_times = torch.min(tar_times, motion_lengths)

        # # 未来のフレームのルート座標を計算
        # tar_root_pos, _, _, _, _, _ = self._motion_lib.calc_motion_frame(motion_ids, clamped_tar_times)

        # # ターゲット位置を毎フレーム動的に更新
        # self._tar_pos[..., 0:2] = tar_root_pos[..., 0:2]


        # return
        
    def _sync_player_pos(self, env_ids=None):
        # Rebuild the shared world-space position from the moving target + polar offset.
        if env_ids is None:
            self._player_pos[:, :] = self._tar_pos
            self._player_pos[:, 0] += self._player_radius * torch.cos(self._player_angle)
            self._player_pos[:, 1] += self._player_radius * torch.sin(self._player_angle)
        else:
            self._player_pos[env_ids] = self._tar_pos[env_ids]
            self._player_pos[env_ids, 0] += self._player_radius[env_ids] * torch.cos(self._player_angle[env_ids])
            self._player_pos[env_ids, 1] += self._player_radius[env_ids] * torch.sin(self._player_angle[env_ids])
        return

    def _sample_player_motion(self, env_ids):
        n = env_ids.numel()
        if n == 0:
            return
        self._player_target_ang_vel[env_ids] = (2.0 * torch.rand(n, device=self._device) - 1.0) * self._player_angular_speed
        self._player_target_rad_vel[env_ids] = (2.0 * torch.rand(n, device=self._device) - 1.0) * self._player_radial_speed
        interval = self._player_change_time_min + (
            self._player_change_time_max - self._player_change_time_min
        ) * torch.rand(n, device=self._device)
        self._player_change_times[env_ids] = self._time_buf[env_ids] + interval
        return

    def _reset_player(self, env_ids):
        n = env_ids.numel()
        if n == 0:
            return
        self._player_radius[env_ids] = self._player_radius_min + (
            self._player_radius_max - self._player_radius_min
        ) * torch.rand(n, device=self._device)
        self._player_angle[env_ids] = 2.0 * np.pi * torch.rand(n, device=self._device)
        self._sample_player_motion(env_ids)
        self._player_ang_vel[env_ids] = self._player_target_ang_vel[env_ids]
        self._player_rad_vel[env_ids] = self._player_target_rad_vel[env_ids]
        self._sync_player_pos(env_ids)
        return

    def _update_player_motion(self):
        # Called once per control step, after the moving reference target is updated.
        change_ids = (self._time_buf >= self._player_change_times).nonzero(as_tuple=False).flatten()
        self._sample_player_motion(change_ids)
        alpha = 1.0 - np.exp(-self._player_dt / self._player_velocity_smoothing)
        self._player_ang_vel += alpha * (self._player_target_ang_vel - self._player_ang_vel)
        self._player_rad_vel += alpha * (self._player_target_rad_vel - self._player_rad_vel)

        self._player_angle[:] = torch.remainder(
            self._player_angle + self._player_ang_vel * self._player_dt, 2.0 * np.pi
        )
        self._player_radius[:] = torch.clamp(
            self._player_radius + self._player_rad_vel * self._player_dt,
            min=self._player_radius_min, max=self._player_radius_max
        )
        # At the radius boundary reverse any outward command, not just the current velocity.
        outer = (self._player_radius >= self._player_radius_max) & (self._player_target_rad_vel > 0)
        inner = (self._player_radius <= self._player_radius_min) & (self._player_target_rad_vel < 0)
        self._player_target_rad_vel[outer] = -self._player_target_rad_vel[outer]
        self._player_target_rad_vel[inner] = -self._player_target_rad_vel[inner]
        self._sync_player_pos()
        return

    def _reset_tar(self, env_ids):
        if len(env_ids) == 0:
            return

        # 実際にこの環境で選択された参照モーションを使う（再抽選しない）。
        motion_ids = self._motion_ids[env_ids]
        motion_times = self._get_motion_times(env_ids)
        tar_times = motion_times + 2.0
        motion_lengths = self._motion_lib._motion_lengths[motion_ids]

        safe_lengths = torch.clamp(motion_lengths, min=1e-5)
        loop_count = torch.floor(tar_times / safe_lengths)
        tar_times_wrapped = torch.remainder(tar_times, safe_lengths)
        nonwrap_motions = self._motion_lib._motion_loop_modes[motion_ids] != 1
        loop_count[nonwrap_motions] = 0.0
        tar_times_wrapped[nonwrap_motions] = torch.minimum(
            tar_times[nonwrap_motions], motion_lengths[nonwrap_motions]
        )

        tar_root_pos, _, tar_root_vel, _, _, _ = self._motion_lib.calc_motion_frame(
            motion_ids, tar_times_wrapped
        )
        tar_root_pos = tar_root_pos + loop_count.unsqueeze(-1) * self._motion_lib._motion_wrap_delta[motion_ids]
        self._tar_pos[env_ids, 0:2] = tar_root_pos[..., 0:2]
        self._tar_pos[env_ids, 2] = 0.0
        self._tar_speed[env_ids] = torch.linalg.vector_norm(tar_root_vel[..., 0:2], dim=-1)
        ended = nonwrap_motions & (tar_times >= motion_lengths)
        self._tar_speed[env_ids[ended]] = 0.0
        return


    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_root_rot = self._engine.get_root_rot(char_id)
        tar_pos = self._tar_pos

        if (env_ids is not None):
            char_root_pos = char_root_pos[env_ids]
            char_root_rot = char_root_rot[env_ids]
            tar_pos = tar_pos[env_ids]
        
        # Use the same shared virtual player state as the reward function.
        player_pos = self._player_pos if env_ids is None else self._player_pos[env_ids]
        
        # ターゲットとプレイヤーの両方のローカル座標を計算して結合 (2次元 + 2次元 = 4次元追加)
        task_obs = compute_location_observations(char_root_pos, char_root_rot, tar_pos, player_pos)
        obs = torch.cat([obs, task_obs], dim=-1)

        return obs

    # def _compute_obs(self, env_ids=None):
        # obs = super()._compute_obs(env_ids)
        
        # char_id = self._get_char_id()
        # char_root_pos = self._engine.get_root_pos(char_id)
        # char_root_rot = self._engine.get_root_rot(char_id)
        # tar_pos = self._tar_pos

        # if (env_ids is not None):
            # char_root_pos = char_root_pos[env_ids]
            # char_root_rot = char_root_rot[env_ids]
            # tar_pos = tar_pos[env_ids]
        
        # task_obs = compute_location_observations(char_root_pos, char_root_rot, tar_pos)
        # obs = torch.cat([obs, task_obs], dim=-1)

        # return obs
    
    def _update_reward(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_root_rot = self._engine.get_root_rot(char_id)
        
        player_pos = self._player_pos
            
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
                                                      face_reward_w=self._face_reward_w,
                                                      near_pos_reward_w=self._near_pos_reward_w,
                                                      near_vel_reward_w=self._near_vel_reward_w,
                                                      near_face_reward_w=self._near_face_reward_w
                                                      )
        # Logger.print(f"reward:{self._reward_buf[0].item():.4f}",end='\r')
        # flat_p = player_pos.squeeze()
        # flat_t = self._tar_pos.squeeze()
        
        # # 💡 【修正】0番目の環境 (インデックス0) の X(0), Y(1), Z(2) 軸をそれぞれピンポイントで抽出
        # p_x, p_y, p_z = player_pos[0, 0].item(), player_pos[0, 1].item(), player_pos[0, 2].item()
        # t_x, t_y, t_z = self._tar_pos[0, 0].item(), self._tar_pos[0, 1].item(), self._tar_pos[0, 2].item()
        
        # Logger.print(
            # f"reward:{self._reward_buf[0].item():.4f} | "
            # f"Player:[{p_x:.2f}, {p_y:.2f}, {p_z:.2f}] | "
            # f"Target:[{t_x:.2f}, {t_y:.2f}, {t_z:.2f}]", 
            # end='\r'
        # )
        return

    def _update_misc(self):
        super()._update_misc()
        # 1. 既存の参考モーションデータの更新
        self._update_ref_motion()

        if (self._enable_ref_char()):
            self._update_ref_char()
            
        # 2. 【追加】毎フレーム、全環境のターゲット位置を「現在のモーション時間 + 数秒先」に追従させる
        self._update_tar_dynamic()
        self._update_player_motion()
        if self._visualize:
            self._update_player_marker(None)
        return

    def _update_tar_dynamic(self):
        motion_ids = self._motion_ids
        # 現在の全環境の正確なモーション進行時間を取得
        motion_times = self._get_motion_times() 

        # 現在時間から2秒先を計算
        lookahead_time = 2.0
        tar_times = motion_times + lookahead_time
        
        # モーションの最大長を取得
        motion_lengths = self._motion_lib._motion_lengths[motion_ids]

        # ----------------------------------------------------
        # 【Warp対応】ループ回数と、ループ内の時間を計算する
        # ----------------------------------------------------
        # 1. 2秒後の時間が、何周ループしているかを計算 (0周目, 1周目, 2周目...)
        # 分母が0になるのを防ぐため、安全に除算します
        loop_count = torch.floor(tar_times / torch.clamp(motion_lengths, min=1e-5))
        
        # 2. ループ内の「0 ～ 最大長」に収まる時間に変換する (剰余算)
        # ただし、Warpしない（単発）モーションの場合は元のクランプ処理を適用する
        tar_times_wrapped = torch.fmod(tar_times, motion_lengths)
        
        # モーションのループモード（LoopMode）が WRAP ではない場合のマスクを作成
        # ※ self._motion_lib._motion_loop_modes が利用可能な想定です
        # もし利用できない場合は、一律で loop_count を使っても WRAP 以外は delta が 0 になっているので安全です
        nonwrap_motions = (self._motion_lib._motion_loop_modes[motion_ids] != 1) # 1: WRAP (motion.LoopMode.WRAP.value)
        loop_count[nonwrap_motions] = 0.0
        tar_times_wrapped[nonwrap_motions] = torch.min(tar_times[nonwrap_motions], motion_lengths[nonwrap_motions])

        # 3. ループ内に収まった時間を使って、ベースとなるルート座標を計算
        tar_root_pos, _, tar_root_vel, _, _, _ = self._motion_lib.calc_motion_frame(motion_ids, tar_times_wrapped)
        self._tar_speed[:] = torch.linalg.vector_norm(tar_root_vel[..., 0:2], dim=-1)
        # CLAMP 終端ではターゲットは停止するため、目標速度もゼロにする。
        ended = nonwrap_motions & (tar_times >= motion_lengths)
        self._tar_speed[ended] = 0.0

        # 4. ループ回数分の移動量（_motion_wrap_delta）をオフセットとして加算する
        wrap_deltas = self._motion_lib._motion_wrap_delta[motion_ids]
        tar_root_pos += loop_count.unsqueeze(-1) * wrap_deltas
        # ----------------------------------------------------

        # ターゲットの X, Y 座標を更新
        self._tar_pos[..., 0:2] = tar_root_pos[..., 0:2]
        
        # 高さを地面に固定（フラットな床を想定）
        self._tar_pos[..., 2] = 0.0 
        # 手のpullターゲット更新
        self._pull_target_pos[..., 0:2] = self._tar_pos[..., 0:2]

        if (self._visualize):
            self._update_marker(None)

        # Logger.print(
            # f"DynTar | CurTime:{motion_times[0].item():.2f}s -> "
            # f"TarTime:{tar_times_wrapped[0].item():.2f}s (Loop:{int(loop_count[0].item())}) | "
            # f"TarPos:({self._tar_pos[0, 0].item():.2f}, {self._tar_pos[0, 1].item():.2f})", 
            # end='\r'
        # )
        return
        
    # def _update_misc(self):
        # super()._update_misc()

        # self._update_tar()
        # self._update_task()
        # return

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
    
# @torch.jit.script
# def compute_location_observations(root_pos, root_rot, tar_pos, player_pos):
    # # (Tensor, Tensor, Tensor, Tensor) -> Tensor
    # heading_rot = torch_util.calc_heading_quat_inv(root_rot)
    
    # # ターゲットのローカル座標 (XY)
    # local_tar_pos = torch_util.quat_rotate(heading_rot, tar_pos - root_pos)
    
    # # プレイヤーのローカル座標 (XY)
    # local_player_pos = torch_util.quat_rotate(heading_rot, player_pos - root_pos)
    
    # # 結合して4次元の観測とする
    # obs = torch.cat([local_tar_pos[..., 0:2], local_player_pos[..., 0:2]], dim=-1)
    # return obs
# @torch.jit.script
# def compute_location_observations(root_pos, root_rot, tar_pos):
    # # (Tensor, Tensor, Tensor) -> Tensor
    # heading_rot = torch_util.calc_heading_quat_inv(root_rot)
    # local_tar_pos = torch_util.quat_rotate(heading_rot, tar_pos - root_pos)
    # obs = local_tar_pos[..., 0:2]
    # return obs

@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, tar_pos, player_pos, tar_speed, dt, dist_threshold, pos_err_scale, vel_err_scale, pos_reward_w, vel_reward_w, face_reward_w, near_pos_reward_w, near_vel_reward_w, near_face_reward_w):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float) -> Tensor
    
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

    far_reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward
    near_reward = near_pos_reward_w * pos_reward + near_vel_reward_w * vel_reward + near_face_reward_w * facing_reward
    reward = torch.where(dist_mask, near_reward, far_reward)

    return reward
# @torch.jit.script
# def compute_location_reward(root_pos, prev_root_pos, root_rot, tar_pos, tar_speed, dt, dist_threshold, pos_err_scale, vel_err_scale, pos_reward_w, vel_reward_w, face_reward_w):
    # # (Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float) -> Tensor
    
    # pos_diff = tar_pos - root_pos
    # pos_err = pos_diff[..., 0] * pos_diff[..., 0] + pos_diff[..., 1] * pos_diff[..., 1]
    # pos_reward = torch.exp(-pos_err_scale * pos_err)

    # tar_dir = tar_pos - root_pos
    # tar_dir = tar_dir[..., 0:2]
    # tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    # delta_root_pos = root_pos - prev_root_pos
    # root_vel = delta_root_pos / dt
    # tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    # tar_vel_err = tar_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    # vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))

    # wrong_dir = tar_dir_speed <= 0
    # vel_reward[wrong_dir] = 0

    # heading_rot = torch_util.calc_heading_quat(root_rot)
    # facing_dir = torch.zeros_like(root_pos)
    # facing_dir[..., 0] = 1.0
    # facing_dir = torch_util.quat_rotate(heading_rot, facing_dir)
    # facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    # #facing_reward = torch.clamp_min(facing_err, 0.0)
    # facing_reward = (facing_err + 1.0) * 0.5

    # dist_mask = pos_err < dist_threshold
    # facing_reward[dist_mask] = 1.0
    # vel_reward[dist_mask] = 1.0

    # reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    # return reward