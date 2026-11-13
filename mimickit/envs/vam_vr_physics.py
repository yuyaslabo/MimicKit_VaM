# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import numpy as np
import time
import torch


class VRPhysicsMixin:
    def _vrpd_target(self, name, obj_id, pos, quat, env_ids):
        # Hand PD only. Arm capsules are updated kinematically by VRBodyMixin.
        if name not in ("left_hand", "right_hand"):
            return
        p = torch.as_tensor(pos, dtype=torch.float32, device=self._device).reshape(1, 3)
        q = torch.as_tensor(quat, dtype=torch.float32, device=self._device).reshape(4)
        # VaM UDP inputs were validated before entering this function.
        self._vrpd_targets[name] = (obj_id, p)
        if name not in self._vrpd_initialized:
            self._engine.set_root_pos(env_ids, obj_id, p)
            self._engine.set_root_rot(env_ids, obj_id, q)
            self._engine.set_root_vel(env_ids, obj_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, obj_id, 0.0)
            self._vrpd_initialized.add(name)
            if self._vrpd_debug:
                print(f"[VRPD] initialized {name} obj={obj_id}", flush=True)

    def _vrpd_apply_forces(self, env_ids):
        if not self._vrpd_enabled or not callable(getattr(self._engine, "set_body_forces", None)):
            return
        # Keep all PD operations on the Newton tensor device; no per-step .cpu()/.item().
        for name in ("left_hand", "right_hand"):
            if name not in self._vrpd_initialized or name not in self._vrpd_targets:
                continue
            obj_id, target = self._vrpd_targets[name]
            pos = self._engine.get_root_pos(obj_id).reshape(-1, 3)[:1]
            vel = self._engine.get_root_vel(obj_id).reshape(-1, 3)[:1]
            error = target - pos
            if self._vrpd_max_error > 0.0:
                error = error * (self._vrpd_max_error / torch.linalg.vector_norm(error, dim=-1, keepdim=True).clamp_min(self._vrpd_max_error)).clamp(max=1.0)
            correction = self._vrpd_position_kp * error - self._vrpd_position_kd * vel
            if self._vrpd_max_force > 0.0:
                correction = correction * (self._vrpd_max_force / torch.linalg.vector_norm(correction, dim=-1, keepdim=True).clamp_min(self._vrpd_max_force)).clamp(max=1.0)
            force = correction
            if self._vrpd_gravity_comp:
                gravity = torch.tensor([[0.0, 0.0, 9.81 * self._vrpd_masses[name]]], device=self._device, dtype=force.dtype)
                force = force + gravity
            # Invalid tracking data is already filtered on UDP reception.
            force = torch.where(torch.isfinite(force), force, torch.zeros_like(force))
            self._engine.set_body_forces(env_ids, obj_id, 0, force)
            self._has_body_forces = True
            if self._vrpd_debug and self._vrhand_update_count % 30 == 0:
                # Diagnostics intentionally synchronize; leave disabled for FPS testing.
                print(f"[VRPD] {name} error={torch.linalg.vector_norm(error).item():.3f}m applied={torch.linalg.vector_norm(force).item():.1f}N", flush=True)

    def _vrhand_double_trigger_action(self, hand_name, hand_pos):
        """Right double-trigger: set/reset facing; left: toggle target freeze."""
        if hand_name == "left":
            # The left-hand gesture does not depend on controller direction.
            self._vrhand_target_frozen = not self._vrhand_target_frozen
            if not self._vrhand_target_frozen and hasattr(self, "_hmd_target_initialized"):
                # Resume HMD target tracking without counting accumulated motion
                # during the frozen interval as a single displacement.
                self._hmd_target_initialized[:] = False
                self._hmd_forward_continuous[:] = False
                self._hmd_forward_speed_filtered[:] = 0.0
                # Snap directly to the maximum HMD-forward distance on release.
                # _reset_tar treats uninitialized tracking as max distance and
                # skips displacement/recovery from the previously frozen point.
                if getattr(self, "_hmd_pose_received", False):
                    env_ids = torch.arange(self.get_num_envs(), device=self._device, dtype=torch.long)
                    self._reset_tar(env_ids)
                    if self._visualize:
                        self._update_marker(env_ids)
            if self._vrhand_debug:
                print(f"[VRhand] left double trigger: target {'FROZEN' if self._vrhand_target_frozen else 'LIVE'}", flush=True)
            return

        if hand_name != "right":
            return
        if not getattr(self, "_hmd_pose_received", False):
            return
        packet = getattr(self, "_vrhand_latest", None)
        if packet is None:
            return

        # Right controller +X pointing axis, with the same quaternion conversion
        # as vam_vr_tracking.py.
        quat_vam = np.asarray(packet[11:15], dtype=np.float64)
        if not np.isfinite(quat_vam).all() or np.linalg.norm(quat_vam) < 1e-8:
            return
        raw = self._vam_quat_to_engine(quat_vam).astype(np.float64)
        q_hand = np.array([raw[1], -raw[0], raw[2], raw[3]], dtype=np.float64)
        q_hand /= max(np.linalg.norm(q_hand), 1e-8)
        v = q_hand[:3]
        local_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        world_dir = local_axis + 2.0 * (
            q_hand[3] * np.cross(v, local_axis) + np.cross(v, np.cross(v, local_axis)))
        world_dir /= max(np.linalg.norm(world_dir), 1e-8)
        if world_dir[2] <= self._vrhand_ground_point_threshold:
            self._vrhand_remote_player_world_pos = None
            if self._vrhand_debug:
                print("[VRhand] right downward double trigger: facing point reset to HMD", flush=True)
            return
        xy = world_dir[:2]
        xy_length = np.linalg.norm(xy)
        if xy_length < 0.10:
            return
        world_xy = torch.as_tensor(xy / xy_length, device=self._device, dtype=torch.float32)
        hmd_pos = self._latest_hmd_pos.reshape(-1, 3)[0]
        # Store the absolute world-space point once. Subsequent HMD movement
        # and yaw rotation must not affect the chosen facing point.
        world_pos = hmd_pos.detach().clone()
        world_pos[:2] += world_xy * self._vrhand_remote_player_distance
        self._vrhand_remote_player_world_pos = world_pos
        if self._vrhand_debug:
            print(f"[VRhand] right double trigger: WORLD facing point XYZ=({float(world_pos[0]):.2f}, {float(world_pos[1]):.2f}, {float(world_pos[2]):.2f})", flush=True)

    def _apply_vrhand_drag(self, env_ids, tracked=None):
        """Preserve double-trigger and grab semantics with lazy tensor reads."""
        # This buffer must still be cleared before both drag and hand PD forces.
        self._has_body_forces = False
        if hasattr(self._sim_state, '_wp_body_force'):
            if hasattr(self._sim_state._wp_body_force, "zero_"):
                self._sim_state._wp_body_force.zero_()
            elif isinstance(self._sim_state._wp_body_force, np.ndarray):
                self._sim_state._wp_body_force.fill(0.0)

        # Hysteresis lets partial pulls count, while requiring a genuine release
        # before the next press edge. Keep existing drag-force scaling unchanged.
        press_threshold = self._vrhand_trigger_press_threshold
        release_threshold = self._vrhand_trigger_release_threshold
        for name in ("left", "right"):
            if self._vrhand_triggers[name] <= release_threshold:
                self._vrhand_grabbed_targets[name] = None
                self._vrhand_trigger_pressed[name] = False

        active = [(name, hand_id, offset) for name, hand_id, offset in (
            ("left", self._vrhand_box_ids[0], 0),
            ("right", self._vrhand_box_ids[1], 8),
        ) if self._vrhand_triggers[name] >= press_threshold or self._vrhand_trigger_pressed[name]]
        if not active:
            return

        char_id = self._get_char_id()
        body_pos = None
        body_vel = None
        for hand_name, hand_id, packet_offset in active:
            trigger_val = self._vrhand_triggers[hand_name]
            if trigger_val < press_threshold and not self._vrhand_trigger_pressed[hand_name]:
                continue
            hand_pos = self._engine.get_root_pos(hand_id).reshape(-1, 3)[0]
            pointing_pos = hand_pos
            if tracked is not None and packet_offset in tracked:
                pointing_pos = torch.as_tensor(tracked[packet_offset][0], device=self._device, dtype=hand_pos.dtype)
            rising = not self._vrhand_trigger_pressed[hand_name]
            if rising:
                self._vrhand_trigger_pressed[hand_name] = True
            if rising and self._vrhand_grabbed_targets[hand_name] is None:
                if body_pos is None:
                    body_pos_tensor = self._engine.get_body_pos(char_id)
                    body_pos = body_pos_tensor[0] if body_pos_tensor.ndim == 3 else body_pos_tensor
                distances = torch.linalg.vector_norm(body_pos - hand_pos, dim=-1)
                closest_idx = int(torch.argmin(distances).item())  # only once per trigger press
                if float(distances[closest_idx]) < 0.2:
                    self._vrhand_grabbed_targets[hand_name] = closest_idx
                    self._vrhand_last_unlocked_press[hand_name] = None
                    if self._vrhand_debug:
                        print(f"[VRhand] {hand_name} hand LOCKED onto bone index: {closest_idx}", flush=True)
                else:
                    now = time.monotonic()
                    previous = self._vrhand_last_unlocked_press[hand_name]
                    if previous is not None and 0.0 < now - previous <= self._vrhand_double_trigger_window:
                        self._vrhand_last_unlocked_press[hand_name] = None
                        self._vrhand_double_trigger_action(hand_name, pointing_pos)
                    else:
                        self._vrhand_last_unlocked_press[hand_name] = now
            target_bone_idx = self._vrhand_grabbed_targets[hand_name]
            if target_bone_idx is None:
                continue
            if body_pos is None:
                body_pos_tensor = self._engine.get_body_pos(char_id)
                body_pos = body_pos_tensor[0] if body_pos_tensor.ndim == 3 else body_pos_tensor
            if body_vel is None:
                body_vel_tensor = self._engine.get_body_vel(char_id)
                body_vel = body_vel_tensor[0] if body_vel_tensor.ndim == 3 else body_vel_tensor
            scaled_trigger = (max(0.3, min(1.0, trigger_val)) - 0.3) / 0.7
            pull_force = (250.0 * (hand_pos - body_pos[target_bone_idx]) - 15.0 * body_vel[target_bone_idx]) * scaled_trigger
            force_norm = torch.linalg.vector_norm(pull_force).clamp_min(500.0)
            pull_force = pull_force * (500.0 / force_norm)
            if hasattr(self._engine, "set_body_forces"):
                self._engine.set_body_forces(env_ids, char_id, target_bone_idx, pull_force.reshape(1, 3))
        return
