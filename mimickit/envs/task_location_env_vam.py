# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.
import numpy as np
import heapq
import time
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull, QhullError
import torch
import engines.engine as engine
import envs.amp_env_vam as amp_env
import util.torch_util as torch_util

class TaskLocationEnv(amp_env.AMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
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
        # Dead zone for normal target placement AND target-travel distance recovery.
        # HMD movement / speed hysteresis still use the raw HMD facing.
        self._hmd_target_yaw_deadzone_deg = float(env_config.get("hmd_target_yaw_deadzone_deg", 10.0))
        if not (0.0 <= self._hmd_target_yaw_deadzone_deg <= 180.0):
            raise ValueError("hmd_target_yaw_deadzone_deg must be between 0 and 180")
        self._tar_dist_min = env_config.get("tar_dist_min", 0.0)
        self._tar_dist_max = env_config["tar_dist_max"]
        assert(self._tar_dist_max >= self._tar_dist_min)

        # Trial: closest collision-free target within the agent-centered XY disk.
        self._avoid_radius = max(0.05, float(env_config.get("tar_avoid_radius", 1.0)))
        self._avoid_grid = max(0.025, float(env_config.get("tar_avoid_grid", 0.10)))
        self._avoid_clearance = max(0.0, float(env_config.get("tar_avoid_clearance", 0.15)))
        # Conservative fallback only when a root lies INSIDE an existing
        # furniture clearance shell; ordinary LOS and route planner are unchanged.
        self._avoid_escape_enabled = bool(env_config.get("tar_avoid_escape_enabled", True))
        self._avoid_escape_sample_spacing = max(0.025, float(env_config.get("tar_avoid_escape_sample_spacing", 0.05)))
        # Dynamic HMD obstacle: LOCAL REAR semicircle with total XY radius 0.4m.
        self._avoid_hmd_enabled = bool(env_config.get("tar_avoid_hmd_enabled", True))
        self._avoid_hmd_radius = max(0.0, float(env_config.get("tar_avoid_hmd_radius", 0.40)))
        rear_angles = np.linspace(np.pi/2.0, 3.0*np.pi/2.0, 11)
        rear_arc = np.stack((np.cos(rear_angles), np.sin(rear_angles)), axis=1)
        self._avoid_hmd_unit_rear = np.vstack((np.zeros((1, 2)), rear_arc))
        # Optional diagnostic for the HMD forward-distance regression.
        # Off by default to avoid extra GPU synchronizations at runtime.
        self._hmd_distance_diag = bool(env_config.get("hmd_distance_diag", False))
        self._hmd_distance_diag_last = -1e20
        self._hmd_distance_diag_prev_mode = None
        # Hysteresis ON: basic target 0.5m from HMD toward the agent.
        self._hmd_moving_agent_target_distance = max(0.0, float(
            env_config.get("hmd_moving_agent_target_distance", 0.50)))
        # Only for avoidance fallback: keep the safe target away from the physical HMD.
        self._avoid_player_radius = max(0.0, float(env_config.get("tar_avoid_player_radius", 0.50)))
        self._avoid_height_below = max(0.0, float(env_config.get("tar_avoid_height_below", 0.80)))
        self._avoid_height_above = max(0.0, float(env_config.get("tar_avoid_height_above", 0.70)))
        self._avoid_geom_cache = {}
        self._avoid_corner_signs = np.array([(x,y,z) for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
        self._avoid_marker_ids = []
        self._avoid_debug = bool(env_config.get("tar_avoid_debug", False))
        self._avoid_warn_no_free = False
        # Smooth only the final safe target (the route planner itself is unchanged).
        # Both the green marker and Actor observation receive the SAME result.
        self._avoid_smooth_enabled = bool(env_config.get("tar_avoid_smooth_enabled", True))
        self._avoid_smooth_tau = max(0.0, float(env_config.get("tar_avoid_smooth_tau", 0.18)))
        self._avoid_smooth_max_speed = max(0.1, float(env_config.get("tar_avoid_smooth_max_speed", 1.0)))
        # Prevent the old short-waypoint stalling problem during active rescue.
        self._avoid_smooth_rescue_min_reach = max(0.0, float(env_config.get("tar_avoid_smooth_rescue_min_reach", 0.45)))
        self._avoid_smooth_last_time = {}
        self._avoid_smooth_valid = set()
        # Optional rescue planner: do not touch normal nearest-visible selection.
        self._avoid_rescue_enabled = bool(env_config.get("tar_avoid_rescue_enabled", True))
        self._avoid_rescue_stall_time = max(0.1, float(env_config.get("tar_avoid_rescue_stall_time", 1.5)))
        self._avoid_rescue_move_threshold = max(0.01, float(env_config.get("tar_avoid_rescue_move_threshold", 0.12)))
        self._avoid_rescue_goal_change = max(0.05, float(env_config.get("tar_avoid_rescue_goal_change", 0.35)))
        self._avoid_rescue_min_goal_distance = max(0.0, float(env_config.get("tar_avoid_rescue_min_goal_distance", 0.55)))
        self._avoid_route_interval = max(0.1, float(env_config.get("tar_avoid_route_interval", 0.35)))
        self._avoid_route_corner_extra = max(0.01, float(env_config.get("tar_avoid_route_corner_extra", 0.06)))
        # Rescue only: prefer a distant, directly reachable position in the chosen direction.
        self._avoid_rescue_lookahead_min = max(0.0, float(env_config.get("tar_avoid_rescue_lookahead_min", 0.60)))
        self._avoid_rescue_alignment = float(np.clip(env_config.get("tar_avoid_rescue_alignment", 0.55), 0.0, 1.0))
        # Wide-obstacle rescue: follow the blocking face sideways until a corner
        # comes into reach. Keep a WORLD-space heading to avoid direction flips.
        self._avoid_wide_enabled = bool(env_config.get("tar_avoid_wide_enabled", True))
        self._avoid_wide_min_halfspan = max(0.2, float(env_config.get("tar_avoid_wide_min_halfspan", 0.85)))
        self._avoid_wide_min_lateral = max(0.1, float(env_config.get("tar_avoid_wide_min_lateral", 0.55)))
        self._avoid_wide_heading = {}  # env -> (world-space unit vector, side sign)
        self._avoid_route_side = {}
        self._avoid_route_cache = {}
        self._avoid_rescue_state = {}  # env -> (start time, position, desired, active)
        # Cache world-space obstacle polygons (VaM poses are sent at ~0.5 s intervals).
        self._avoid_obstacles_cached = []
        self._avoid_obstacles_at = -1e20
        self._avoid_obstacle_refresh = max(0.01, float(env_config.get("tar_avoid_obstacle_refresh", 0.50)))
        axis = np.arange(-self._avoid_radius, self._avoid_radius + self._avoid_grid * 0.5,
                         self._avoid_grid, dtype=np.float64)
        offsets = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
        self._avoid_offsets = offsets[np.sum(offsets ** 2, axis=1) <= self._avoid_radius ** 2 + 1e-8]
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        if hasattr(self, "_engine") and hasattr(self._engine, "_sim_state"):
            self._sim_state = self._engine._sim_state
        return
    
    def _build_envs(self, config, num_envs):
        if (self._visualize):
            self._marker_ids = []
            self._avoid_marker_ids = []

        super()._build_envs(config, num_envs)
        return
    
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        
        if (self._visualize):
            marker_id = self._build_marker(env_id)

            avoid_marker_id = self._engine.create_obj(
                env_id=env_id, obj_type=engine.ObjType.rigid,
                asset_file="data/assets/objects/location_marker.xml",
                name="safe_target_marker", is_visual=True, fix_root=True,
                color=[0.12, 0.95, 0.30])
            if (env_id == 0):
                self._marker_ids.append(marker_id)
                self._avoid_marker_ids.append(avoid_marker_id)
            else:
                assert(self._marker_ids[0] == marker_id)
                assert(self._avoid_marker_ids[0] == avoid_marker_id)
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
        # The original HMD controller always writes _tar_pos. Actor sees _safe_tar_pos.
        self._safe_tar_pos = self._tar_pos.clone()
        self._avoid_smooth_last_time.clear()
        self._avoid_smooth_valid.clear()
        self._avoid_load_geom_cache()
        self._hmd_target_initialized = torch.zeros(num_envs, device=self._device, dtype=torch.bool)
        self._hmd_target_prev_pos = torch.zeros((num_envs, 2), device=self._device, dtype=torch.float)
        self._hmd_target_prev_facing = torch.zeros((num_envs, 2), device=self._device, dtype=torch.float)
        # Independent placement heading; do not overwrite raw facing history.
        self._hmd_target_placement_facing = torch.zeros((num_envs, 2), device=self._device, dtype=torch.float)
        # Keep nominal HMD-forward target separate from collision-adjusted white marker.
        self._hmd_nominal_target_xy = torch.zeros((num_envs, 2), device=self._device, dtype=torch.float)
        self._hmd_forward_continuous = torch.zeros(num_envs, device=self._device, dtype=torch.bool)
        self._hmd_forward_speed_filtered = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        return
    
    def _get_marker_id(self):
        return self._marker_ids[0]



    def _update_marker(self, env_ids):
        marker_id = self._get_marker_id()
        rot = torch.tensor([0, 0, 0, 1], device=self._device, dtype=torch.float32)

        self._engine.set_root_pos(env_ids, marker_id, self._tar_pos[env_ids])
        self._engine.set_root_rot(env_ids, marker_id, rot)
        self._engine.set_root_vel(env_ids, marker_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, marker_id, 0.0)
        if self._avoid_marker_ids:
            safe_id = self._avoid_marker_ids[0]
            self._engine.set_root_pos(env_ids, safe_id, self._safe_tar_pos[env_ids])
            self._engine.set_root_rot(env_ids, safe_id, rot)
            self._engine.set_root_vel(env_ids, safe_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, safe_id, 0.0)
        return
    
    def _update_task(self):
        super()._update_task()
        # HMDの最新位置・向きを使って毎制御ステップ更新する。
        # tar_change_time_min/maxによる周期制御は使用しない。
        env_ids = torch.arange(self.get_num_envs(), device=self._device, dtype=torch.long)
        self._reset_tar(env_ids)
        # Avoidance is computed only by _compute_obs, after the latest HMD update.
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
        for idx in env_ids.detach().cpu().tolist():
            self._avoid_rescue_reset(int(idx))
            # Reset interpolation history with the agent; never interpolate
            # from the previous episode's target position.
            self._avoid_smooth_last_time.pop(int(idx), None)
            self._avoid_smooth_valid.discard(int(idx))
        self._hmd_target_initialized[env_ids] = False
        self._hmd_nominal_target_xy[env_ids] = self._tar_pos[env_ids, :2]
        self._hmd_target_placement_facing[env_ids] = 0.0
        self._hmd_forward_continuous[env_ids] = False
        self._hmd_forward_speed_filtered[env_ids] = 0.0
        self._reset_tar(env_ids)
        # _compute_obs will update the safe target and both markers.
        return
        
    def _reset_tar(self, env_ids):
        if getattr(self, "_vrhand_target_frozen", False):
            return  # Hold the current world-space target, independent of HMD motion
        # Normal mode: HMD-forward target with the existing variable-distance logic.
        # Continuous-forward mode: place the target at the HMD XY origin.
        if env_ids.numel() == 0:
            return

        # Do not initialize against the placeholder (0,0,0) HMD pose.
        # The first valid HMD packet will initialize the target at max distance.
        if not getattr(self, "_hmd_pose_received", False):
            self._hmd_target_initialized[env_ids] = False
            self._hmd_nominal_target_xy[env_ids] = self._tar_pos[env_ids, :2]
            self._hmd_target_placement_facing[env_ids] = 0.0
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

        # Yaw dead zone (with continuous excess-angle tracking).
        # Keep raw `facing` for forward-motion and speed-hysteresis tests.
        placement_prev = self._hmd_target_placement_facing[env_ids]
        placement_prev = torch.where(initialized.unsqueeze(-1), placement_prev, facing)
        cross = placement_prev[:, 0] * facing[:, 1] - placement_prev[:, 1] * facing[:, 0]
        dot = (placement_prev * facing).sum(dim=-1)
        yaw_error = torch.atan2(cross, dot)
        deadzone_rad = self._hmd_target_yaw_deadzone_deg * (np.pi / 180.0)
        yaw_step = torch.sign(yaw_error) * torch.clamp(torch.abs(yaw_error) - deadzone_rad, min=0.0)
        c = torch.cos(yaw_step)
        sn = torch.sin(yaw_step)
        placement_facing = torch.stack((
            c * placement_prev[:, 0] - sn * placement_prev[:, 1],
            sn * placement_prev[:, 0] + c * placement_prev[:, 1],
        ), dim=-1)
        self._hmd_target_placement_facing[env_ids] = placement_facing

        previous_player_xy = self._hmd_target_prev_pos[env_ids]
        previous_distance = torch.linalg.vector_norm(
            self._hmd_nominal_target_xy[env_ids] - previous_player_xy, dim=-1
        )

        # As before, only movement along the local forward axis changes
        # the HMD-to-target distance (forward shortens, backward extends).
        displacement = player_xy - previous_player_xy
        previous_facing = self._hmd_target_prev_facing[env_ids]
        mid_facing = torch.nn.functional.normalize(previous_facing + facing, dim=-1, eps=1e-6)
        mid_facing = torch.where(initialized.unsqueeze(-1), mid_facing, facing)
        forward_movement = torch.sum(displacement * mid_facing, dim=-1)
        dt = float(self._engine.get_timestep())
        # Hysteresis uses XY speed in ANY direction. Keep forward_movement above
        # unchanged for the original min/max HMD-forward distance controller.
        xy_speed = torch.linalg.vector_norm(displacement, dim=-1) / max(dt, 1e-6)
        xy_speed = torch.where(initialized, xy_speed, torch.zeros_like(xy_speed))
        alpha = dt / (self._hmd_forward_speed_smoothing + dt) if dt > 0.0 else 1.0
        filtered_speed = self._hmd_forward_speed_filtered[env_ids] + alpha * (
            xy_speed - self._hmd_forward_speed_filtered[env_ids]
        )
        self._hmd_forward_speed_filtered[env_ids] = filtered_speed
        was_forward_continuous = self._hmd_forward_continuous[env_ids]
        forward_continuous = was_forward_continuous
        # Switch ON at the configured speed threshold; stay ON until below OFF.
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
        previous_target_xy = self._hmd_nominal_target_xy[env_ids].clone()
        # Measure recovery against the CURRENT yaw-deadzone-adjusted heading,
        # matching the heading used to place the normal target below.
        # Small yaw movements INSIDE the dead zone do not rotate the heading;
        # rotations that EXCEED it contribute to target travel/recovery.
        # Raw HMD facing remains exclusive to movement/speed measurements.
        candidate_xy = player_xy + placement_facing * distance.unsqueeze(-1)
        target_movement = torch.linalg.vector_norm(candidate_xy - previous_target_xy, dim=-1)
        recovery_range = self._hmd_target_max_distance - self._hmd_target_min_distance
        effective_movement = (target_movement - self._hmd_target_recover_deadzone).clamp_min(0.0)
        recovery = recovery_range * (effective_movement / self._hmd_target_recover_distance)
        recovery = torch.where(initialized, recovery, torch.zeros_like(recovery))
        distance = (distance + recovery).clamp(
            self._hmd_target_min_distance, self._hmd_target_max_distance
        )
        # On the ON -> OFF transition, restore the normal target immediately
        # to its configured maximum HMD-forward distance (not the minimum).
        # Other normal-mode distance shortening/recovery remains unchanged.
        just_released = was_forward_continuous & ~forward_continuous
        distance = torch.where(
            just_released,
            torch.full_like(distance, self._hmd_target_max_distance),
            distance,
        )

        # ON: agent-facing target at 0.5m from the HMD. OFF: preserve the
        # original yaw-deadzone forward target and min/max distance behaviour.
        normal_target_xy = player_xy + placement_facing * distance.unsqueeze(-1)
        agent_xy = self._engine.get_root_pos(self._get_char_id())[env_ids, :2]
        agent_delta = agent_xy - player_xy
        agent_distance = torch.linalg.vector_norm(agent_delta, dim=-1, keepdim=True)
        agent_direction = agent_delta / agent_distance.clamp_min(1e-6)
        agent_direction = torch.where(agent_distance > 1e-6, agent_direction, -facing)
        moving_target_xy = player_xy + agent_direction * self._hmd_moving_agent_target_distance
        target_xy = torch.where(forward_continuous.unsqueeze(-1), moving_target_xy, normal_target_xy)
        # Store the *unadjusted* target for the next distance/recovery step;
        # furniture avoidance must not feed back into the HMD min/max controller.
        self._hmd_nominal_target_xy[env_ids] = target_xy
        # White marker avoidance is enabled only during HMD follow mode. The
        # previously implemented target-frozen bypass remains unchanged.
        adjusted_xy = self._hmd_adjust_basic_target(env_ids, player_xy, placement_facing,
                                                    facing, target_xy, forward_continuous)
        self._tar_pos[env_ids, :2] = adjusted_xy
        # Diagnostic ONLY: do not change either target or hysteresis behaviour.
        if self._hmd_distance_diag and env_ids.numel() > 0:
            now_diag = time.monotonic()
            current_mode = bool(forward_continuous[0].item())
            mode_changed = (self._hmd_distance_diag_prev_mode is not None
                            and current_mode != self._hmd_distance_diag_prev_mode)
            if mode_changed or now_diag - self._hmd_distance_diag_last >= 1.0:
                k = 0
                gray_dist = float(torch.linalg.vector_norm(
                    self._tar_pos[env_ids[k], :2] - player_xy[k]).item())
                safe_dist = float(torch.linalg.vector_norm(
                    self._safe_tar_pos[env_ids[k], :2] - player_xy[k]).item())
                print(
                    f"[HMD-DIST] mode={'ON' if current_mode else 'OFF'}"
                    f" xy_speed={float(xy_speed[k].item()):.3f}m/s"
                    f" dt={dt:.5f}s"
                    f" ema={float(filtered_speed[k].item()):.3f}m/s"
                    f" forward_step={float(forward_movement[k].item()):+.4f}m"
                    f" normal_dist={float(distance[k].item()):.3f}m"
                    f" gray_dist={gray_dist:.3f}m"
                    f" safe_dist={safe_dist:.3f}m"
                    f" trigger={'SWITCH' if mode_changed else 'PERIODIC'}",
                    flush=True)
                self._hmd_distance_diag_last = now_diag
            self._hmd_distance_diag_prev_mode = current_mode
        self._hmd_target_prev_pos[env_ids] = player_xy
        self._hmd_target_prev_facing[env_ids] = facing
        self._hmd_target_initialized[env_ids] = True
        return

    def _hmd_adjust_basic_target(self, env_ids, player_xy, placement_facing,
                                 raw_facing, nominal_xy, moving_mode):
        """Keep the WHITE HMD-follow target outside furniture and HMD rear zone.

        Normal non-moving mode: try the SAME yaw-deadzoned forward ray from the
        nominal distance down to hmd_target_min_distance. Only when no point
        on this interval is free, search up to 0.5m around the HMD, preferring
        smallest angular deviation from the forward direction. Moving mode
        retains its agent-facing desired position, using the local 0.5m search
        only when that point is occupied. The green marker planner is unchanged.
        """
        if env_ids.numel() == 0:
            return nominal_xy
        # One batched CPU transfer, once per _reset_tar invocation. Avoid any
        # geometry rebuild: the same cached furniture OBB projections are used.
        hmd = player_xy.detach().cpu().numpy()
        heading = placement_facing.detach().cpu().numpy()
        raw_heading = raw_facing.detach().cpu().numpy()
        nominal = nominal_xy.detach().cpu().numpy()
        moving = moving_mode.detach().cpu().numpy()
        obstacles = self._avoid_collect_obstacles()
        # Avoid an additional GPU state read if there are no furniture objects.
        char_root = (self._engine.get_root_pos(self._get_char_id())[env_ids].detach().cpu().numpy()
                     if obstacles else None)
        output = nominal.copy()
        radius = 0.50
        min_dist = self._hmd_target_min_distance
        # No need for finer sampling than a few centimetres during the simple
        # forward-ray check; fallback is only evaluated when it fails.
        sample_step = 0.025
        angles = np.deg2rad(np.arange(0., 181., 5.))
        for i in range(len(hmd)):
            center = hmd[i, :2].astype(np.float64)
            facing = heading[i].astype(np.float64)
            norm = float(np.linalg.norm(facing))
            if norm < 1e-6 or not np.isfinite(center).all():
                continue
            facing /= norm
            desired = nominal[i, :2].astype(np.float64)
            relevant = []
            for poly, bottom, top, low, high in obstacles:
                zmin = char_root[i, 2] - self._avoid_height_below
                zmax = char_root[i, 2] + self._avoid_height_above
                if top < zmin or bottom > zmax:
                    continue
                if np.linalg.norm(np.maximum(np.maximum(low-center, center-high), 0.0)) > radius + self._avoid_clearance:
                    continue
                relevant.append((poly, low, high))
            # HMD's rear semicircle is also excluded; its front remains free
            # apart from the existing shared clearance around the diameter.
            hmd_obstacle = self._avoid_hmd_obstacle(center, raw_heading[i])
            if hmd_obstacle is not None:
                relevant.append(hmd_obstacle)
            if not relevant:
                continue

            def free_points(points):
                valid = np.ones(len(points), dtype=bool)
                for poly, low, high in relevant:
                    close = valid & np.all((points >= low) & (points <= high), axis=1)
                    if np.any(close):
                        idx = np.flatnonzero(close)
                        valid[idx] = self._avoid_free_points(points[idx], poly, self._avoid_clearance)
                    if not valid.any():
                        break
                return valid

            if free_points(desired.reshape(1, 2))[0]:
                continue
            # Straight-ahead contraction is only for stationary/low-speed mode.
            if not moving[i]:
                start = max(float(np.linalg.norm(desired-center)), min_dist)
                if start > min_dist + 1e-8:
                    distances = np.linspace(start, min_dist,
                                            max(2, int(np.ceil((start-min_dist)/sample_step))+1))
                else:
                    distances = np.array([min_dist])
                direct = center[None, :] + distances[:, None] * facing[None, :]
                valid = free_points(direct)
                if valid.any():
                    output[i, :2] = direct[np.flatnonzero(valid)[0]]
                    continue
            # Fallback: HMD-centered 0.5m disk; prioritize front-facing angle,
            # then nominal-target proximity. Start at min distance so the
            # configured minimum remains respected in the lateral search.
            if min_dist > radius:
                continue  # No admissible fallback annulus.
            radii = np.linspace(max(0., min_dist), radius,
                                max(2, int(np.ceil((radius-min_dist)/sample_step))+1))
            all_points = []
            all_scores = []
            base_angle = float(np.arctan2(facing[1], facing[0]))
            for abs_angle in angles:
                offsets = [0.0] if abs_angle == 0.0 else [-abs_angle, abs_angle]
                for angle in offsets:
                    theta = base_angle + angle
                    direction = np.array([np.cos(theta), np.sin(theta)])
                    pts = center[None, :] + radii[:, None] * direction[None, :]
                    all_points.append(pts)
                    all_scores.append(np.column_stack((
                        np.full(len(radii), abs_angle),
                        np.sum((pts-desired)**2, axis=1))))
            candidates = np.vstack(all_points)
            scores = np.vstack(all_scores)
            valid = free_points(candidates)
            if valid.any():
                indexes = np.flatnonzero(valid)
                score = scores[indexes]
                best = indexes[np.lexsort((score[:, 1], score[:, 0]))[0]]
                output[i, :2] = candidates[best]
            # If literally no admissible point exists, do not invent a safe
            # destination. Keep nominal and optionally emit a diagnostics log.
            elif self._avoid_debug:
                print(f"[HMD-TAR] no free white target in 0.5m, env={int(env_ids[i])}", flush=True)
        return torch.as_tensor(output, device=self._device, dtype=nominal_xy.dtype)

    def _avoid_load_geom_cache(self):
        """Read the actual persisted MJCF boxes, not the gross Atom bounding box."""
        self._avoid_geom_cache = {}
        if not getattr(self, "_envbox_enabled", False):
            return
        for atom_id, path in getattr(self, "_envbox_assets", {}).items():
            if atom_id not in getattr(self, "_envbox_objects", {}):
                continue
            try:
                root = ET.parse(path).getroot()
                geoms = []
                for geom in root.findall(".//geom"):
                    if geom.get("type") != "box":
                        continue
                    offset = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
                    half = np.fromstring(geom.get("size", ""), sep=" ")
                    q = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
                    if offset.size != 3 or half.size != 3 or q.size != 4 or np.any(half <= 0):
                        continue
                    mat = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                    geoms.append((offset, half, mat))
                self._avoid_geom_cache[atom_id] = geoms
            except (OSError, ET.ParseError, ValueError) as exc:
                print(f"[TarAvoid] Could not parse atom_{atom_id} MJCF: {exc}", flush=True)

    def _avoid_collect_obstacles(self):
        """Recreate EnvironmentBoxesMixin's root pose transformation in CPU memory.

        This avoids reading every obstacle pose back from the GPU. Only registered
        MJCF objects present in the latest VaM UDP pose packet are considered.
        """
        if not getattr(self, "_envbox_enabled", False):
            return []
        now = time.monotonic()
        if now - self._avoid_obstacles_at < self._avoid_obstacle_refresh:
            return self._avoid_obstacles_cached
        obstacles = []
        C = self._envbox_axis
        offset = self._envbox_position_offset
        for atom_id, (pos, _size, quat) in self._envbox_latest.items():
            geoms = self._avoid_geom_cache.get(atom_id)
            if not geoms or atom_id not in self._envbox_objects:
                continue
            root_pos = C @ (pos - offset)
            root_pos[2] *= self._envbox_height_scale
            root_pos[2] += 0.22  # Same fixed offset as _envbox_sync().
            root_mat = C @ Rotation.from_quat(quat).as_matrix() @ C.T
            for local_pos, half, local_mat in geoms:
                world_center = root_pos + root_mat @ local_pos
                world_mat = root_mat @ local_mat
                # Eight OBB corners: conservative ground-plane projection.
                signs = self._avoid_corner_signs
                corners = world_center + (signs * half) @ world_mat.T
                try:
                    hull = ConvexHull(corners[:, :2])
                    poly = corners[hull.vertices, :2]
                except QhullError:
                    continue
                obstacles.append((poly, float(corners[:, 2].min()), float(corners[:, 2].max()),
                                  np.min(poly, axis=0) - self._avoid_clearance,
                                  np.max(poly, axis=0) + self._avoid_clearance))
        self._avoid_obstacles_cached = obstacles
        self._avoid_obstacles_at = now
        return obstacles

    @staticmethod
    def _avoid_free_points(points, poly, clearance):
        """For CCW convex polygon, outside-and-near-edge checks of point-sized robot."""
        edges = np.roll(poly, -1, axis=0) - poly
        rel = points[:, None, :] - poly[None, :, :]
        cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
        inside = np.all(cross >= -1e-9, axis=1)
        edge_sq = np.maximum(np.sum(edges * edges, axis=1), 1e-12)
        t = np.clip(np.sum(rel * edges[None, :, :], axis=2) / edge_sq[None, :], 0, 1)
        closest = poly[None, :, :] + t[:, :, None] * edges[None, :, :]
        d2 = np.min(np.sum((points[:, None, :] - closest) ** 2, axis=2), axis=1)
        return ~(inside | (d2 < clearance * clearance))

    def _avoid_free_paths(self, origin, destinations, relevant):
        """Check straight XY segments against cached convex obstacle polygons.

        Collision includes the configured point clearance. The test is entirely
        on CPU NumPy arrays already used by the cached obstacle search; it does
        not call the Newton engine, parse XML, or rebuild a ConvexHull.
        """
        destinations = np.asarray(destinations, dtype=np.float64).reshape(-1, 2)
        origin = np.asarray(origin, dtype=np.float64).reshape(2)
        free = np.ones(len(destinations), dtype=bool)
        if len(destinations) == 0 or not relevant:
            return free
        clearance2 = self._avoid_clearance ** 2
        eps = 1e-12
        for poly, low, high in relevant:
            if not np.any(free):
                break
            # First remove paths whose expanded segment AABB misses this obstacle.
            active = np.flatnonzero(free)
            end = destinations[active]
            seg_low = np.minimum(origin, end)
            seg_high = np.maximum(origin, end)
            near = np.all(seg_high >= low - eps, axis=1) & np.all(seg_low <= high + eps, axis=1)
            if not np.any(near):
                continue
            indices = active[near]
            end = destinations[indices]
            direction = end - origin
            sq_len = np.sum(direction * direction, axis=1)
            edges_end = np.roll(poly, -1, axis=0)
            edges = edges_end - poly
            # Segment intersection test, including endpoint / collinear contact.
            ao = poly[None, :, :] - origin
            cross_dir_edge = direction[:, None, 0] * edges[None, :, 1] - direction[:, None, 1] * edges[None, :, 0]
            cross_ao_edge = ao[:, :, 0] * edges[None, :, 1] - ao[:, :, 1] * edges[None, :, 0]
            cross_ao_dir = ao[:, :, 0] * direction[:, None, 1] - ao[:, :, 1] * direction[:, None, 0]
            nonparallel = np.abs(cross_dir_edge) > eps
            sign = np.where(cross_dir_edge >= 0.0, 1.0, -1.0)
            hit = nonparallel & (cross_ao_edge * sign >= -eps) & (cross_ao_edge * sign <= np.abs(cross_dir_edge) + eps) & (cross_ao_dir * sign >= -eps) & (cross_ao_dir * sign <= np.abs(cross_dir_edge) + eps)
            blocked = np.any(hit, axis=1)
            # Point inside polygon (including the origin inside) must be blocked.
            if not self._avoid_free_points(origin.reshape(1, 2), poly, 0.0)[0]:
                free[indices] = False
                continue
            if self._avoid_clearance > 0.0:
                # Distance to each polygon edge from both path endpoints.
                edge_sq = np.maximum(np.sum(edges * edges, axis=1), eps)
                for points in (np.broadcast_to(origin, end.shape), end):
                    delta = points[:, None, :] - poly[None, :, :]
                    t = np.clip(np.sum(delta * edges[None, :, :], axis=2) / edge_sq[None, :], 0.0, 1.0)
                    closest = poly[None, :, :] + t[:, :, None] * edges[None, :, :]
                    blocked |= np.any(np.sum((points[:, None, :] - closest) ** 2, axis=2) < clearance2, axis=1)
                # Distance from polygon vertices to the moving segment.
                delta = poly[None, :, :] - origin
                t = np.clip(np.sum(delta * direction[:, None, :], axis=2) / np.maximum(sq_len[:, None], eps), 0.0, 1.0)
                closest = origin + t[:, :, None] * direction[:, None, :]
                blocked |= np.any(np.sum((poly[None, :, :] - closest) ** 2, axis=2) < clearance2, axis=1)
            # For zero-clearance segments fully contained in a polygon, also
            # reject the endpoint (the origin was checked above).
            if self._avoid_clearance == 0.0:
                blocked |= ~self._avoid_free_points(end, poly, 0.0)
            free[indices[blocked]] = False
        return free

    def _avoid_route_corners(self, poly):
        """Offset the convex CCW polygon vertices outward beyond clearance.

        The bisector formula yields true edge-normal clearance at acute corners.
        Every resulting point and segment is still checked against ALL obstacles.
        """
        verts = np.asarray(poly, dtype=np.float64)
        prev = verts - np.roll(verts, 1, axis=0)
        nxt = np.roll(verts, -1, axis=0) - verts
        lp = np.maximum(np.linalg.norm(prev, axis=1, keepdims=True), 1e-12)
        ln = np.maximum(np.linalg.norm(nxt, axis=1, keepdims=True), 1e-12)
        # CCW polygon: outward normal is the right-hand perpendicular.
        npv = np.column_stack((prev[:, 1], -prev[:, 0])) / lp
        nnv = np.column_stack((nxt[:, 1], -nxt[:, 0])) / ln
        bis = npv + nnv
        norm = np.maximum(np.linalg.norm(bis, axis=1, keepdims=True), 1e-12)
        bis /= norm
        dot = np.maximum(np.sum(bis * npv, axis=1), 0.15)
        distance = (self._avoid_clearance + self._avoid_route_corner_extra) / dot
        return verts + bis * distance[:, None]

    def _avoid_plan_route(self, center, want, player, relevant):
        """Small visibility graph around the first obstructing OBB polygons.

        Returns (safe local waypoint, side), or (None, 0). Every chosen
        waypoint is within the 1-m agent disk and has collision-free line of sight.
        The planner is limited to two blocking obstacle polygons for bounded cost.
        """
        if not relevant:
            return None, 0
        direction = want - center
        length = float(np.linalg.norm(direction))
        if length < 1e-5:
            return None, 0
        blockers = []
        for obs in relevant:
            if not self._avoid_free_paths(center, want[None, :], [obs])[0]:
                poly = obs[0]
                blockers.append((float(np.linalg.norm(poly.mean(axis=0) - center)), obs))
        if not blockers:
            return None, 0
        blockers.sort(key=lambda x: x[0])
        corners = []
        for _, obs in blockers[:2]:
            corners.extend(self._avoid_route_corners(obs[0]))
        # In pathological complex scenes cap graph size; do not create more
        # geometry or extra physics bodies for navigation.
        corners = np.asarray(corners[:12], dtype=np.float64).reshape(-1, 2)
        if corners.size == 0:
            return None, 0
        # The actual desired target may be inside a collider or within the
        # physical-HMD exclusion disk, so consider a few nearby local goals.
        disk_points = center[None, :] + self._avoid_offsets
        disk_dist = np.sum((disk_points - want) ** 2, axis=1)
        order = np.argsort(disk_dist)
        goals = []
        clipped = center + direction * min(1.0, self._avoid_radius / length)
        candidates = np.vstack((clipped[None, :], disk_points[order[:120]]))
        for goal in candidates:
            if player is not None and np.linalg.norm(goal - player) < self._avoid_player_radius:
                continue
            if any(not self._avoid_free_points(goal[None, :], poly, self._avoid_clearance)[0]
                   for poly, _, _ in relevant):
                continue
            if not any(np.linalg.norm(goal - existing) < self._avoid_grid * 0.8 for existing in goals):
                goals.append(goal.copy())
            if len(goals) >= 6:
                break
        if not goals:
            return None, 0
        # Filter out corners that are occupied by some other overlapping Box.
        corner_valid = [corner for corner in corners if all(
            self._avoid_free_points(corner[None, :], poly, self._avoid_clearance)[0]
            for poly, _, _ in relevant)]
        if not corner_valid:
            return None, 0
        corners = np.asarray(corner_valid, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)
        nodes = np.vstack((center[None, :], corners, goals))
        n = len(nodes)
        goal_start = 1 + len(corners)
        # Build a compact visibility graph. Calls are batched by source node.
        adjacency = [[] for _ in range(n)]
        for u in range(n):
            if u >= goal_start:
                continue  # goals are terminal
            indexes = np.arange(u + 1, n)
            if not indexes.size:
                continue
            visible = self._avoid_free_paths(nodes[u], nodes[indexes], relevant)
            for v in indexes[visible]:
                d = float(np.linalg.norm(nodes[v] - nodes[u]))
                adjacency[u].append((int(v), d))
                adjacency[v].append((u, d))
        # Dijkstra, with first-turn side attached to state. This permits a
        # deliberate left/right choice instead of oscillating between corners.
        initial = (0.0, 0, 0, (0,))  # cost, vertex, side, path
        queue = [initial]
        seen = {}
        routes = []
        while queue:
            cost, u, side, path = heapq.heappop(queue)
            key = (u, side)
            if cost > seen.get(key, float('inf')) + 1e-8:
                continue
            if u >= goal_start:
                goal = nodes[u]
                # Remaining distance to real desired location matters more
                # than a marginal difference in route length.
                score = cost + 2.5 * float(np.linalg.norm(goal - want))
                routes.append((score, side, path))
                continue
            for v, d in adjacency[u]:
                if u != 0 and v == 0:
                    continue
                if v in path:
                    continue
                new_side = side
                if u == 0:
                    cross = direction[0] * (nodes[v, 1] - center[1]) - direction[1] * (nodes[v, 0] - center[0])
                    new_side = 1 if cross > 1e-5 else (-1 if cross < -1e-5 else 0)
                nc = cost + d
                state = (int(v), new_side)
                if nc >= seen.get(state, float('inf')):
                    continue
                seen[state] = nc
                heapq.heappush(queue, (nc, int(v), new_side, path + (int(v),)))
        if not routes:
            return None, 0
        return routes, nodes

    def _avoid_rescue_lookahead(self, center, want, player, relevant, initial_direction, side):
        """Select an extended *visible* target near the route's first heading.

        The graph may pick a corner only 0.2-0.3m away. Instead of commanding
        that tiny waypoint repeatedly, find a far local point in the same
        corridor. No global path/normal target logic is changed.
        """
        direction = np.asarray(initial_direction, dtype=np.float64) - center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        unit = direction / norm
        offsets = self._avoid_offsets
        distances = np.linalg.norm(offsets, axis=1)
        minimum = min(self._avoid_rescue_lookahead_min, self._avoid_radius * 0.95)
        mask = (distances >= minimum) & (distances <= self._avoid_radius + 1e-6)
        # Keep targets broadly aligned with the chosen path; no abrupt
        # jump to the opposite side of the obstruction.
        alignment = offsets @ unit / np.maximum(distances, 1e-9)
        mask &= alignment >= self._avoid_rescue_alignment
        desired = want - center
        if side and float(np.linalg.norm(desired)) > 1e-6:
            cross = desired[0] * offsets[:, 1] - desired[1] * offsets[:, 0]
            mask &= cross * side >= -1e-5
        candidates = center + offsets[mask]
        if not len(candidates):
            return None
        if player is not None and self._avoid_player_radius > 0:
            candidates = candidates[np.sum((candidates-player)**2, axis=1) >= self._avoid_player_radius**2]
        if not len(candidates):
            return None
        # Check only local candidates; batched LOS avoids per-candidate Python calls.
        for poly, low, high in relevant:
            near = np.all((candidates >= low) & (candidates <= high), axis=1)
            if np.any(near):
                good = np.ones(len(candidates), dtype=bool)
                good[near] = self._avoid_free_points(candidates[near], poly, self._avoid_clearance)
                candidates = candidates[good]
                if not len(candidates):
                    return None
        visible = self._avoid_free_paths(center, candidates, relevant)
        candidates = candidates[visible]
        if not len(candidates):
            return None
        disp = candidates - center
        # Favor forward reach along the chosen route, then a longer lookahead;
        # penalize sideways deviation from its initial heading.
        progress = disp @ unit
        radial = np.linalg.norm(disp, axis=1)
        score = progress + 0.35 * radial - 0.35 * (radial - progress)
        return candidates[int(np.argmax(score))].copy()

    def _avoid_wide_wall_follow(self, env_index, center, want, player, relevant):
        """Fallback only: advance sideways along a wide obstructing face.

        A 3-m-wide wall can have both corners outside the agent's 1-m disk.
        Local visibility graphs then cannot consistently choose a useful goal.
        Instead, issue a fresh, directly visible ~1-m waypoint along the
        wall. The world-space travel direction is locked for this rescue.
        Once the corner approaches, yield to the original graph planner.
        """
        if not self._avoid_wide_enabled or not relevant:
            return None, 0
        delta = want - center
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            return None, 0
        forward = delta / length
        left = np.array([-forward[1], forward[0]], dtype=np.float64)
        # Find the first obstructing polygon. Its lateral projection indicates
        # whether BOTH ends are out of reach (wide-face case).
        blockers = []
        for obs in relevant:
            if not self._avoid_free_paths(center, want[None, :], [obs])[0]:
                poly = obs[0]
                projected = (poly - center) @ left
                left_extent = float(projected.max())
                right_extent = float(-projected.min())
                nearest = float(np.linalg.norm(poly.mean(axis=0) - center))
                blockers.append((nearest, left_extent, right_extent))
        if not blockers:
            return None, 0
        blockers.sort(key=lambda item: item[0])
        _, left_extent, right_extent = blockers[0]

        stored = self._avoid_wide_heading.get(env_index)
        if stored is None:
            if min(left_extent, right_extent) < self._avoid_wide_min_halfspan:
                return None, 0
            # Favor the shorter route to the edge, then keep it in world space.
            side = 1 if left_extent <= right_extent else -1
            heading = left * side
            self._avoid_wide_heading[env_index] = (heading.copy(), side)
        else:
            heading, side = stored
            # Stop wall following when the chosen edge is near enough for the
            # original 1-m corner planner to take over. Side lock remains.
            remaining = left_extent if side > 0 else right_extent
            if remaining < self._avoid_wide_min_halfspan:
                self._avoid_wide_heading.pop(env_index, None)
                return None, 0

        offsets = self._avoid_offsets
        lateral = offsets @ heading
        radial = np.linalg.norm(offsets, axis=1)
        mask = (lateral >= self._avoid_wide_min_lateral) & (radial <= self._avoid_radius + 1e-7)
        candidates = center[None, :] + offsets[mask]
        if len(candidates) == 0:
            return None, 0
        if player is not None and self._avoid_player_radius > 0:
            candidates = candidates[np.sum((candidates - player) ** 2, axis=1) >= self._avoid_player_radius ** 2]
        if not len(candidates):
            return None, 0
        for poly, low, high in relevant:
            near = np.all((candidates >= low) & (candidates <= high), axis=1)
            if np.any(near):
                keep = np.ones(len(candidates), dtype=bool)
                keep[near] = self._avoid_free_points(candidates[near], poly, self._avoid_clearance)
                candidates = candidates[keep]
                if len(candidates) == 0:
                    return None, 0
        visible = self._avoid_free_paths(center, candidates, relevant)
        candidates = candidates[visible]
        if len(candidates) == 0:
            return None, 0
        steps = candidates - center
        along = steps @ heading
        # Primary objective: lateral progress, not proximity to the player.
        # Secondary: modest progress toward the actual target when safe.
        progress = steps @ forward
        scores = along + 0.12 * progress - 0.25 * np.abs(progress)
        return candidates[int(np.argmax(scores))].copy(), side

    def _avoid_choose_route(self, env_index, center, want, player, relevant):
        planned = self._avoid_plan_route(center, want, player, relevant)
        if planned[0] is None:
            return None, 0
        routes, nodes = planned
        locked = self._avoid_route_side.get(env_index, 0)
        # Hard preference for the current side while it remains reachable.
        # Waypoints themselves are recalculated as the agent moves; this avoids
        # the near-root stalling caused by locking a world-space target.
        routes.sort(key=lambda item: (0 if locked == 0 or item[1] == locked else 1, item[0]))
        for _, side, path in routes:
            # Look ahead along the planned route until reaching the maximum
            # directly visible point within the local search disk.
            path_points = nodes[list(path[1:])]
            chosen = None
            for pt in path_points:
                direction = pt - center
                dist = float(np.linalg.norm(direction))
                if dist < 1e-8:
                    continue
                pt_local = center + direction * min(1.0, self._avoid_radius / dist)
                if not self._avoid_free_paths(center, pt_local[None, :], relevant)[0]:
                    break
                if player is not None and np.linalg.norm(pt_local - player) < self._avoid_player_radius:
                    # Don't select points in the player exclusion radius.
                    continue
                chosen = pt_local
                # Prefer a 1-m lookahead along the SAME visible ray where safe.
                # At an obstacle corner the ray may not extend to 1 m; in that
                # case keep the shorter reachable corner waypoint.
                if dist < self._avoid_radius:
                    extended = center + direction / dist * self._avoid_radius
                    if (self._avoid_free_paths(center, extended[None, :], relevant)[0]
                            and (player is None or np.linalg.norm(extended-player) >= self._avoid_player_radius)):
                        chosen = extended
                        break
                if dist >= self._avoid_radius:
                    break
            if chosen is not None:
                # The graph's first corner can be within 30cm of the agent;
                # keep the selected side, but look for a farther directly
                # visible target in that SAME direction before accepting it.
                first_heading = nodes[path[1]] if len(path) > 1 else chosen
                extended = self._avoid_rescue_lookahead(center, want, player, relevant,
                                                        first_heading, side)
                if extended is not None:
                    return extended, side
                # Geometry may leave no visible 0.6m+ target; retain the
                # original safe route point instead of inventing a route.
                return chosen, side
        return None, 0

    def _avoid_rescue_reset(self, index):
        self._avoid_rescue_state.pop(index, None)
        self._avoid_route_side.pop(index, None)
        self._avoid_route_cache.pop(index, None)
        self._avoid_wide_heading.pop(index, None)

    def _avoid_rescue_try(self, index, center, want, player, relevant, now, current_target):
        """Return rescue waypoint only when progress has stalled behind an obstacle.

        The caller first computes the original nearest-visible point. This helper
        never replaces that result unless the agent has been nearly stationary
        for the configured interval. An active rescue continues until the direct
        route clears; this prevents alternating between both targeting policies.
        """
        # When HMD target following is OFF, the desired target is locked in
        # world space. HMD movement must not trigger or sustain Rescue.
        # Keep normal safe-target collision/LOS avoidance active.
        if getattr(self, "_vrhand_target_frozen", False):
            self._avoid_rescue_reset(index)
            return None
        if not self._avoid_rescue_enabled or not relevant:
            self._avoid_rescue_reset(index)
            return None
        if np.linalg.norm(want - center) < self._avoid_rescue_min_goal_distance:
            self._avoid_rescue_reset(index)
            return None
        # Check the FULL desired goal, not only the clipped 1-m local target.
        # Otherwise a clear first metre resets rescue even while a farther
        # obstacle still separates the agent from the player/desired point.
        if self._avoid_free_paths(center, want[None, :], relevant)[0] and all(
                self._avoid_free_points(want[None, :], poly, self._avoid_clearance)[0]
                for poly, _, _ in relevant):
            self._avoid_rescue_reset(index)
            return None
        previous = self._avoid_rescue_state.get(index)
        if previous is None:
            self._avoid_rescue_state[index] = (now, center.copy(), want.copy(), False)
            return None
        started, anchor, old_want, active = previous
        # A substantial HMD target relocation restarts the stalled timer, except
        # during an active rescue where re-planning should follow the new goal.
        if not active and np.linalg.norm(want-old_want) >= self._avoid_rescue_goal_change:
            self._avoid_rescue_state[index] = (now, center.copy(), want.copy(), False)
            return None
        if not active and np.linalg.norm(center-anchor) >= self._avoid_rescue_move_threshold:
            self._avoid_rescue_state[index] = (now, center.copy(), want.copy(), False)
            return None
        if not active and now-started < self._avoid_rescue_stall_time:
            return None
        if not active:
            self._avoid_rescue_state[index] = (now, center.copy(), want.copy(), True)
            if self._avoid_debug:
                print(f"[TarAvoid] env={index} STALLED: fallback route enabled", flush=True)
        cached = self._avoid_route_cache.get(index)
        if cached is not None:
            stamp, old_root, old_goal, point = cached
            if point is None and now-stamp < self._avoid_route_interval:
                return None
            if (point is not None and now-stamp < self._avoid_route_interval
                    and np.linalg.norm(center-old_root) < 0.15
                    and np.linalg.norm(want-old_goal) < 0.18
                    and np.linalg.norm(point-center) <= self._avoid_radius + 1e-6
                    and (player is None or np.linalg.norm(point-player) >= self._avoid_player_radius)
                    and self._avoid_free_paths(center, point[None, :], relevant)[0]):
                return point
        # Wide-face fallback first. This runs only after the original stall
        # detector activates and keeps the same 1-m local LOS constraint.
        point, side = self._avoid_wide_wall_follow(index, center, want, player, relevant)
        wide_mode = point is not None
        if point is None:
            point, side = self._avoid_choose_route(index, center, want, player, relevant)
        if point is None:
            # No feasible waypoint: let the original nearest-visible algorithm
            # run; retry at a limited frequency rather than every observation.
            self._avoid_route_cache[index] = (now, center.copy(), want.copy(), None)
            return None
        if side:
            self._avoid_route_side[index] = side
        self._avoid_route_cache[index] = (now, center.copy(), want.copy(), point.copy())
        if self._avoid_debug:
            direction = 'LEFT' if side > 0 else 'RIGHT' if side < 0 else 'STRAIGHT'
            mode = "WALL" if wide_mode else "CORNER"
            print(f"[TarAvoid] env={index} rescue={direction} mode={mode} waypoint={point.round(2)} distance={np.linalg.norm(point-center):.2f}m", flush=True)
        return point

    def _avoid_smooth_targets(self, env_ids, root, raw_targets, contexts):
        """Smooth XY output while preserving disk, line-of-sight and player safety.

        Rescue planning and nearest-visible candidate selection remain untouched.
        A rejected interpolation is replaced by the original validated target;
        it is never held at an obsolete near-root waypoint.
        """
        if not self._avoid_smooth_enabled:
            return raw_targets

        now = time.monotonic()
        # One GPU read for the whole batch, not one synchronization per candidate.
        old_targets = self._safe_tar_pos[env_ids, :2].detach().cpu().numpy()
        output = raw_targets.copy()
        for i, (center, player, relevant, avoidance_mode) in enumerate(contexts):
            idx = int(env_ids[i])
            raw = np.asarray(raw_targets[i, :2], dtype=np.float64)
            prev = np.asarray(old_targets[i], dtype=np.float64)
            stamp = self._avoid_smooth_last_time.get(idx)
            self._avoid_smooth_last_time[idx] = now
            if (idx not in self._avoid_smooth_valid or stamp is None
                    or not np.isfinite(prev).all() or not np.isfinite(raw).all()):
                output[i, :2] = raw
                self._avoid_smooth_valid.add(idx)
                continue

            # Real elapsed time allows low-FPS operation without artificially
            # slowing the target when observations arrive at irregular intervals.
            dt = float(np.clip(now - stamp, 0.001, 0.15))
            displacement = raw - prev
            distance = float(np.linalg.norm(displacement))
            if distance <= 1e-6:
                continue
            alpha = 1.0 if self._avoid_smooth_tau <= 0.0 else -np.expm1(-dt / self._avoid_smooth_tau)
            step = min(distance * alpha, self._avoid_smooth_max_speed * dt)
            point = prev + displacement * (step / distance)

            # A moving agent can overtake a filtered waypoint. Do not allow
            # smoothing to turn a valid ~1m rescue into a near-root command.
            rescue = self._avoid_rescue_state.get(idx)
            rescue_active = rescue is not None and rescue[3]
            if (rescue_active and self._avoid_smooth_rescue_min_reach > 0.0
                    and np.linalg.norm(raw - center) >= self._avoid_smooth_rescue_min_reach
                    and np.linalg.norm(point - center) < self._avoid_smooth_rescue_min_reach):
                point = raw.copy()

            # The allowed target disk moves with the agent.
            offset = point - center
            radial = float(np.linalg.norm(offset))
            if radial > self._avoid_radius and radial > 1e-8:
                point = center + offset * (self._avoid_radius / radial)

            valid = np.isfinite(point).all()
            if valid and avoidance_mode and player is not None and self._avoid_player_radius > 0.0:
                valid = float(np.sum((point - player) ** 2)) >= self._avoid_player_radius ** 2
            # Interpolation between two valid endpoints is not necessarily safe
            # around a corner: check both endpoint occupancy and direct LOS.
            if valid and relevant:
                for poly, low, high in relevant:
                    if np.all(point >= low) and np.all(point <= high):
                        if not self._avoid_free_points(point[None, :], poly, self._avoid_clearance)[0]:
                            valid = False
                            break
                if valid:
                    valid = bool(self._avoid_free_paths(center, point[None, :], relevant)[0])
            # Never freeze the actor at a previous waypoint if interpolation fails.
            output[i, :2] = point if valid else raw
        return output

    def _avoid_clearance_escape_path(self, origin, destination, relevant):
        """Conservative fallback when ROOT starts inside furniture clearance.

        Standard LOS rejects all directions because its start point violates the
        requested clearance, even when moving straight AWAY from an obstacle.
        Here only, sample the candidate ray. Never cross the actual polygon;
        allow clearance overlap solely if it decreases monotonically until
        the full clearance is recovered. No changes to the normal planner.
        Called only after normal candidates fail, not per grid point.
        """
        origin = np.asarray(origin, dtype=np.float64)
        end = np.asarray(destination, dtype=np.float64)
        span = float(np.linalg.norm(end - origin))
        if span < 0.10 or not np.isfinite(span):
            return False
        count = min(65, max(3, int(np.ceil(span / self._avoid_escape_sample_spacing)) + 1))
        samples = origin[None, :] + np.linspace(0.0, 1.0, count)[:, None] * (end - origin)
        clearance2 = self._avoid_clearance ** 2
        for poly, low, high in relevant:
            if np.any(np.maximum(np.minimum(origin, end), low) > np.minimum(np.maximum(origin, end), high)):
                continue
            # Never allow escape logic to travel THROUGH a solid polygon.
            if not np.all(self._avoid_free_points(samples, poly, 0.0)):
                return False
            if self._avoid_clearance <= 0:
                continue
            edges = np.roll(poly, -1, axis=0) - poly
            rel = samples[:, None, :] - poly[None, :, :]
            edge_sq = np.maximum(np.sum(edges * edges, axis=1), 1e-12)
            fraction = np.clip(np.sum(rel * edges[None, :, :], axis=2) / edge_sq[None, :], 0., 1.)
            closest = poly[None, :, :] + fraction[:, :, None] * edges[None, :, :]
            d2 = np.min(np.sum((samples[:, None, :] - closest) ** 2, axis=2), axis=1)
            near = d2 < clearance2 - 1e-10
            if not near[0]:
                if np.any(near):
                    return False  # Started outside clearance: keep normal safety.
            else:
                # Stay outside actual collision and keep increasing separation.
                # Once outside the clearance shell, never re-enter it.
                recovered = np.flatnonzero(~near)
                if not len(recovered):
                    return False  # Cannot escape full clearance within this ray.
                first_clear = int(recovered[0])
                if np.any(np.diff(d2[:first_clear+1]) < -1e-8):
                    return False
                if np.any(near[first_clear:]):
                    return False
        return True

    def _avoid_hmd_obstacle(self, player, facing):
        """HMD-local REAR half-disc, rotated by current RAW HMD yaw.

        Uses the existing furniture/rescue path machinery, without changing
        rescue activation. The reported radius includes furniture clearance.
        Like other convex polygons, edge clearance expands the diameter line
        slightly into HMD's front half (up to _avoid_clearance metres).
        """
        if (not self._avoid_hmd_enabled or self._avoid_hmd_radius <= 0.0
                or player is None or facing is None
                or not np.isfinite(player).all() or not np.isfinite(facing).all()):
            return None
        length = float(np.linalg.norm(facing))
        if length < 1e-8:
            return None
        fx, fy = facing / length
        # Compensate circular polygon's chord sagitta; clearance is then added
        # by the existing point/path checks. Avoid double-counting radius.
        core = max(0.001, self._avoid_hmd_radius - self._avoid_clearance)
        core /= np.cos(np.pi / 20.0)
        xy = self._avoid_hmd_unit_rear * core
        # Local X points forwards, local Y is left; rigid rotation about HMD.
        poly = player[None, :] + np.column_stack((
            fx * xy[:, 0] - fy * xy[:, 1],
            fy * xy[:, 0] + fx * xy[:, 1],
        ))
        low = np.min(poly, axis=0) - self._avoid_clearance
        high = np.max(poly, axis=0) + self._avoid_clearance
        return poly, low, high

    def _avoid_update_target(self, env_ids):
        """Find nearest free position; bypass ALL avoidance when target is frozen."""
        if env_ids is None or env_ids.numel() == 0:
            return
        if getattr(self, "_vrhand_target_frozen", False):
            # HMD follow OFF: Actor and green marker use the exact fixed target.
            # Skip furniture/HMD obstacle queries, rescue planning and smoothing.
            self._safe_tar_pos[env_ids] = self._tar_pos[env_ids]
            # Clear cached rescue/waypoints and smoothing history so follow ON
            # starts a fresh search, rather than inheriting a stale waypoint.
            for env_index in env_ids.detach().cpu().tolist():
                env_index = int(env_index)
                self._avoid_rescue_reset(env_index)
                self._avoid_smooth_last_time.pop(env_index, None)
                self._avoid_smooth_valid.discard(env_index)
            return
        char_id = self._get_char_id()
        root = self._engine.get_root_pos(char_id)[env_ids].detach().cpu().numpy()
        desired = self._tar_pos[env_ids].detach().cpu().numpy()
        # Physical player location, not the remote facing target controlled by the
        # right trigger.  Read once per observation, not once per candidate point.
        player_xy = None
        hmd_facing_xy = None
        if getattr(self, "_hmd_pose_received", False) and hasattr(self, "_latest_hmd_pos"):
            hmd_pos = self._latest_hmd_pos
            if hmd_pos.dim() == 1:
                hmd_pos = hmd_pos.unsqueeze(0)
            if hmd_pos.shape[0] == 1 and self.get_num_envs() > 1:
                hmd_pos = hmd_pos.expand(self.get_num_envs(), -1)
            player_xy = hmd_pos[env_ids, :2].detach().cpu().numpy()
            if hasattr(self, "_latest_hmd_quat"):
                hmd_quat = self._latest_hmd_quat
                if hmd_quat.dim() == 1:
                    hmd_quat = hmd_quat.unsqueeze(0)
                if hmd_quat.shape[0] == 1 and self.get_num_envs() > 1:
                    hmd_quat = hmd_quat.expand(self.get_num_envs(), -1)
                q = hmd_quat[env_ids]
                z, w = q[:, 2], q[:, 3]
                facing = torch.stack((1.0 - 2.0*z.square(), 2.0*w*z), dim=-1)
                facing = torch.nn.functional.normalize(facing, dim=-1, eps=1e-6)
                hmd_facing_xy = facing.detach().cpu().numpy()
        obstacles = self._avoid_collect_obstacles()
        result = desired.copy()
        r = self._avoid_radius
        smooth_contexts = []
        for i in range(len(root)):
            center = root[i, :2].astype(np.float64)
            want = desired[i, :2].astype(np.float64)
            vec = want - center
            length = np.linalg.norm(vec)
            closest_disk = center + vec * min(1.0, r / max(length, 1e-12))
            zmin = root[i, 2] - self._avoid_height_below
            zmax = root[i, 2] + self._avoid_height_above
            relevant = []
            for poly, bottom, top, bbox_min, bbox_max in obstacles:
                if top < zmin or bottom > zmax:
                    continue
                # If obstacle AABB lies outside search disk, its polygon cannot block it.
                if np.linalg.norm(np.maximum(np.maximum(bbox_min - center, center - bbox_max), 0.0)) > r:
                    continue
                relevant.append((poly, bbox_min, bbox_max))
            # Fast path: use exact nearest point within disk when collision-free.
            # Even if the first metre is clear, a farther obstacle may block
            # the FULL desired goal. Rescue must be allowed to stay active
            # across observations instead of resetting to the near-root point.
            player = player_xy[i] if player_xy is not None else None
            hmd_facing = hmd_facing_xy[i] if hmd_facing_xy is not None else None
            hmd_obstacle = self._avoid_hmd_obstacle(player, hmd_facing)
            if hmd_obstacle is not None:
                poly, low, high = hmd_obstacle
                if np.linalg.norm(np.maximum(np.maximum(low-center, center-high), 0.0)) <= r:
                    relevant.append(hmd_obstacle)
            first = closest_disk.reshape(1, 2)
            first_free = True
            for poly, low, high in relevant:
                if np.any(closest_disk < low) or np.any(closest_disk > high):
                    continue
                if not self._avoid_free_points(first, poly, self._avoid_clearance)[0]:
                    first_free = False
                    break
            fast_path = first_free and self._avoid_free_paths(center, first, relevant)[0]
            # Existing path selection is unaffected; context is only for the
            # final output interpolation and its safety checks.
            smooth_contexts.append((center, player, relevant, not fast_path))
            if fast_path:
                result[i, :2] = closest_disk
                # Only invoke the planner's cheap stall/visibility gate here.
                # The expensive route graph is built only once stalled and
                # its result remains cached for the configured interval.
                rescue = self._avoid_rescue_try(int(env_ids[i]), center, want, player,
                                                relevant, time.monotonic(), result[i, :2])
                if rescue is not None:
                    result[i, :2] = rescue
                continue
            # Grid search if desired location is occupied OR the straight
            # route from the agent would cross an obstacle.
            points = center[None, :] + self._avoid_offsets
            valid = np.ones(len(points), dtype=bool)
            # Only when obstacle avoidance is needed.  The regular HMD target
            # must retain its original distance shortening behaviour unchanged.
            if player is not None and self._avoid_player_radius > 0.0:
                valid &= np.sum((points - player) ** 2, axis=1) >= self._avoid_player_radius ** 2
            for poly, low, high in relevant:
                mask = valid & np.all((points >= low) & (points <= high), axis=1)
                if not np.any(mask):
                    continue
                idx = np.flatnonzero(mask)
                valid[idx] = self._avoid_free_points(points[idx], poly, self._avoid_clearance)
                if not np.any(valid):
                    break
            if np.any(valid):
                free_points = points[valid]
                reachable = self._avoid_free_paths(center, free_points, relevant)
                if np.any(reachable):
                    free_points = free_points[reachable]
                    result[i, :2] = free_points[np.argmin(np.sum((free_points - want) ** 2, axis=1))]
                    # If all normally reachable points remain near the root,
                    # try a direct outward clearance escape only as a fallback.
                    if (self._avoid_escape_enabled and length > 0.35
                            and np.linalg.norm(result[i, :2] - center) < 0.20
                            and (player is None or self._avoid_player_radius <= 0.0
                                 or np.linalg.norm(closest_disk - player) >= self._avoid_player_radius)
                            and self._avoid_clearance_escape_path(center, closest_disk, relevant)):
                        result[i, :2] = closest_disk
                    rescue = self._avoid_rescue_try(int(env_ids[i]), center, want, player,
                                                    relevant, time.monotonic(), result[i, :2])
                    if rescue is not None:
                        result[i, :2] = rescue
                    continue
            # Normal LOS can reject an outward ray if its origin is already
            # inside furniture/HMD safety clearance. Only accept an escape
            # that crosses no polygon and monotonically exits the shell.
            if (self._avoid_escape_enabled and length >= 0.10
                    and (player is None or self._avoid_player_radius <= 0.0
                         or np.linalg.norm(closest_disk - player) >= self._avoid_player_radius)
                    and self._avoid_clearance_escape_path(center, closest_disk, relevant)):
                result[i, :2] = closest_disk
                continue
            # A reachable point may not exist when the agent starts inside an
            # obstacle. Keep the previous target only if its path is still free;
            # otherwise fall back to the root (no fictitious safe route).
            previous = self._safe_tar_pos[env_ids[i]].detach().cpu().numpy()[:2]
            prev_in_range = np.linalg.norm(previous - center) <= r
            prev_free = prev_in_range and self._avoid_free_paths(center, previous.reshape(1, 2), relevant)[0]
            if player is not None and self._avoid_player_radius > 0.0:
                prev_free = prev_free and np.sum((previous - player) ** 2) >= self._avoid_player_radius ** 2
            # No qualifying safe destination exists.  Retain the original
            # root fallback rather than inventing a collision-free destination.
            result[i, :2] = previous if prev_free else center
            rescue = self._avoid_rescue_try(int(env_ids[i]), center, want, player,
                                            relevant, time.monotonic(), result[i, :2])
            if rescue is not None:
                result[i, :2] = rescue
            if self._avoid_debug and not self._avoid_warn_no_free:
                print("[TarAvoid] No directly reachable target in search disk; using fallback", flush=True)
            self._avoid_warn_no_free = True
        # Rescue may have overridden a clear 1m path; apply HMD exclusion there
        # as well, without changing ordinary unobstructed HMD targeting.
        for i, (center, player, relevant, avoid_mode) in enumerate(smooth_contexts):
            state = self._avoid_rescue_state.get(int(env_ids[i]))
            if state is not None and state[3]:
                smooth_contexts[i] = (center, player, relevant, True)
        result = self._avoid_smooth_targets(env_ids, root, result, smooth_contexts)
        self._safe_tar_pos[env_ids] = torch.as_tensor(result, device=self._device,
                                                     dtype=self._safe_tar_pos.dtype)

    def _get_facing_player_pos(self):
        # Shared world-space facing point for the Actor task observation.
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
        # Refresh the candidate using current root state, even after an environment reset.
        refresh_ids = (torch.arange(self.get_num_envs(), device=self._device, dtype=torch.long)
                       if env_ids is None else env_ids)
        self._avoid_update_target(refresh_ids)
        if self._visualize and refresh_ids.numel() > 0:
            self._update_marker(refresh_ids)
        tar_pos = self._safe_tar_pos
        
        player_pos = self._get_facing_player_pos()
        


        if (env_ids is not None):
            char_root_pos = char_root_pos[env_ids]
            char_root_rot = char_root_rot[env_ids]
            tar_pos = tar_pos[env_ids]
            player_pos = player_pos[env_ids]
            
        # 単一環境の場合の次元補正（[3] ->）
        if player_pos.dim() == 1:
            player_pos = player_pos.unsqueeze(0)
        
        # Preserve the trained Actor task observation: target XY + facing direction XY (4D).
        task_obs = compute_location_observations(char_root_pos, char_root_rot, tar_pos, player_pos)
        obs = torch.cat([obs, task_obs], dim=-1)

        return obs

    def _update_reward(self):
        # Inference-only: retain the env.step() reward buffer contract.
        # No reward computation is needed to select Actor actions.
        self._reward_buf.zero_()
        return

    def _update_misc(self):
        # Preserve the original two target updates per control step.
        # Removing the duplicate changes target motion and is a separate experiment.
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
        ends = self._safe_tar_pos.detach().cpu().numpy()
        
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

