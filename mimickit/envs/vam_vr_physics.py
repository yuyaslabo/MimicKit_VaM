# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import numpy as np
import time
import torch


class VRPhysicsMixin:
    def _vrpd_target(self, name, obj_id, pos, quat, env_ids):
        target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        target_quat = np.asarray(quat, dtype=np.float64).reshape(4)
        if not np.isfinite(target_pos).all() or not np.isfinite(target_quat).all():
            return
        self._vrpd_targets[name] = (obj_id, target_pos.copy())
        if name not in self._vrpd_initialized:
            p = torch.as_tensor(target_pos, device=self._device, dtype=torch.float32).reshape(1, 3)
            q = torch.as_tensor(target_quat, device=self._device, dtype=torch.float32)
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
        for name in ("left_hand", "right_hand", "left_forearm", "right_forearm", "left_upper_arm", "right_upper_arm"):
            if name not in self._vrpd_initialized or name not in self._vrpd_targets:
                continue
            obj_id, target = self._vrpd_targets[name]
            current = self._engine.get_root_pos(obj_id)
            velocity = self._engine.get_root_vel(obj_id)
            pos = np.asarray(self._vam_to_numpy(current), dtype=np.float64).reshape(-1, 3)[0]
            vel = np.asarray(self._vam_to_numpy(velocity), dtype=np.float64).reshape(-1, 3)[0]
            if not np.isfinite(pos).all() or not np.isfinite(vel).all():
                continue
            error = target - pos
            err_norm = float(np.linalg.norm(error))
            if err_norm > self._vrpd_max_error > 0.0:
                error *= self._vrpd_max_error / err_norm
            # Clamp only the PD correction: gravity compensation must not be clipped away.
            upper_arm = name.endswith("_upper_arm")
            kp = self._vrpd_upper_arm_kp if upper_arm else self._vrpd_position_kp
            kd = self._vrpd_upper_arm_kd if upper_arm else self._vrpd_position_kd
            max_force = self._vrpd_upper_arm_max_force if upper_arm else self._vrpd_max_force
            correction = kp * error - kd * vel
            norm = float(np.linalg.norm(correction))
            raw_force_norm = norm
            if norm > max_force > 0.0:
                correction *= max_force / norm
            force = correction
            if self._vrpd_gravity_comp:
                force[2] += 9.81 * self._vrpd_masses[name]
            if not np.isfinite(force).all():
                continue
            force_tensor = torch.as_tensor(force, device=self._device, dtype=torch.float32).reshape(1, 3)
            self._engine.set_body_forces(env_ids, obj_id, 0, force_tensor)
            self._has_body_forces = True
            if self._vrpd_debug and self._vrhand_update_count % 30 == 0:
                print(f"[VRPD] {name} error={err_norm:.3f}m rawPD={raw_force_norm:.1f}N applied={np.linalg.norm(force):.1f}N vel={np.linalg.norm(vel):.2f}m/s", flush=True)

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
        raw = torch.as_tensor(self._vam_quat_to_engine(quat_vam),
                              device=self._device, dtype=torch.float32)
        q_hand = torch.stack((raw[1], -raw[0], raw[2], raw[3]))
        q_hand = q_hand / torch.linalg.vector_norm(q_hand).clamp_min(1e-8)
        local_axis = torch.tensor([1.0, 0.0, 0.0],
                                  device=self._device, dtype=torch.float32)
        v = q_hand[:3]
        world_dir = local_axis + 2.0 * (
            q_hand[3] * torch.cross(v, local_axis, dim=0)
            + torch.cross(v, torch.cross(v, local_axis, dim=0), dim=0)
        )
        world_dir = world_dir / torch.linalg.vector_norm(world_dir).clamp_min(1e-8)

        if float(world_dir[2]) <= self._vrhand_ground_point_threshold:
            self._vrhand_remote_player_world_pos = None
            if self._vrhand_debug:
                print("[VRhand] right downward double trigger: facing point reset to HMD", flush=True)
            return

        xy = world_dir[:2]
        xy_length = float(torch.linalg.vector_norm(xy))
        if xy_length < 0.10:
            return
        world_xy = xy / xy_length
        hmd_pos = self._latest_hmd_pos.reshape(-1, 3)[0]
        # Store the absolute world-space point once. Subsequent HMD movement
        # and yaw rotation must not affect the chosen facing point.
        world_pos = hmd_pos.detach().clone()
        world_pos[:2] += world_xy * self._vrhand_remote_player_distance
        self._vrhand_remote_player_world_pos = world_pos
        if self._vrhand_debug:
            print(f"[VRhand] right double trigger: WORLD facing point XYZ=({float(world_pos[0]):.2f}, {float(world_pos[1]):.2f}, {float(world_pos[2]):.2f})", flush=True)

    def _apply_vrhand_drag(self, env_ids, tracked=None):
        """Nearby bone grabbing and double-trigger facing-point selection."""
        char_id = self._get_char_id()
        body_pos_tensor = self._engine.get_body_pos(char_id)
        body_vel_tensor = self._engine.get_body_vel(char_id)
        body_pos = body_pos_tensor[0] if body_pos_tensor.ndim == 3 else body_pos_tensor
        body_vel = body_vel_tensor[0] if body_vel_tensor.ndim == 3 else body_vel_tensor

        # Preserve existing buffer clearing semantics; VRPD forces follow this call.
        self._has_body_forces = False
        if hasattr(self._sim_state, '_wp_body_force'):
            if hasattr(self._sim_state._wp_body_force, "zero_"):
                self._sim_state._wp_body_force.zero_()
            elif isinstance(self._sim_state._wp_body_force, np.ndarray):
                self._sim_state._wp_body_force.fill(0.0)

        for hand_name, hand_id, packet_offset in (("left", self._vrhand_box_ids[0], 0), ("right", self._vrhand_box_ids[1], 8)):
            trigger_val = self._vrhand_triggers[hand_name]
            hand_pos = self._engine.get_root_pos(hand_id)
            if hand_pos.ndim > 1:
                hand_pos = hand_pos[0]
            # Use tracking coordinates for gesture direction, not the PD-delayed collider.
            pointing_pos = hand_pos
            if tracked is not None and packet_offset in tracked:
                pointing_pos = torch.as_tensor(tracked[packet_offset][0], device=self._device, dtype=hand_pos.dtype)

            if trigger_val < 0.3:
                self._vrhand_grabbed_targets[hand_name] = None
                self._vrhand_trigger_pressed[hand_name] = False
                continue
            if trigger_val < 0.5:
                continue

            rising = not self._vrhand_trigger_pressed[hand_name]
            if rising:
                self._vrhand_trigger_pressed[hand_name] = True

            # Search only on an initial press; never fall back to remote grabbing.
            if rising and self._vrhand_grabbed_targets[hand_name] is None:
                distances = torch.linalg.vector_norm(body_pos - hand_pos, dim=-1)
                closest_idx = int(torch.argmin(distances).item())
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
            bone_pos = body_pos[target_bone_idx]
            bone_vel = body_vel[target_bone_idx]
            clamped_trigger = max(0.3, min(1.0, trigger_val))
            scaled_trigger = (clamped_trigger - 0.3) / (1.0 - 0.3)
            kp_force = 250.0
            kd_force = 15.0
            pull_force = (kp_force * (hand_pos - bone_pos) - kd_force * bone_vel) * scaled_trigger
            force_speed = torch.norm(pull_force)
            max_grab_force = 500.0
            if force_speed > max_grab_force:
                pull_force = (pull_force / force_speed) * max_grab_force
            if hasattr(self._engine, "set_body_forces"):
                self._engine.set_body_forces(env_ids, char_id, target_bone_idx, pull_force.reshape(1, 3))
        return
