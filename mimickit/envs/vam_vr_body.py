# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class VRBodyMixin:
    def _vrbody_solve_elbow(self, shoulder, wrist, side, facing, wrist_quat=None):
        upper = self._vrbody_upper_arm_length
        lower = self._vrbody_forearm_length
        delta = wrist - shoulder
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            delta = facing.apply(np.array([0.0, side, -0.05]))
            distance = float(np.linalg.norm(delta))
        direction = delta / distance
        reach = np.clip(distance, abs(upper - lower) + 1e-5, upper + lower - 1e-5)
        along = (upper * upper - lower * lower + reach * reach) / (2.0 * reach)
        height = np.sqrt(max(0.0, upper * upper - along * along))
        # Original bend direction remains the fallback and stabilizing reference.
        pole = facing.apply(np.array([-0.35, side * 0.8, -0.25]))
        pole = pole - np.dot(pole, direction) * direction
        norm = float(np.linalg.norm(pole))
        if norm < 1e-6:
            pole = np.cross(direction, np.array([0.0, 0.0, 1.0]))
            norm = float(np.linalg.norm(pole))
        pole = pole / max(norm, 1e-6)
        # The wrist-to-elbow direction inferred from the controller orientation
        # specifies a preferred IK bend plane, not an exact elbow position.
        weight = float(np.clip(self._vrbody_wrist_pole_weight, 0.0, 1.0))
        if wrist_quat is not None and weight > 0.0:
            key = "left" if side > 0 else "right"
            local_axis = self._vrbody_left_wrist_axis if side > 0 else self._vrbody_right_wrist_axis
            axis_norm = float(np.linalg.norm(local_axis))
            if axis_norm > 1e-6 and np.isfinite(wrist_quat).all():
                wrist_to_elbow = R.from_quat(wrist_quat).apply(local_axis / axis_norm)
                wrist_pole = wrist_to_elbow - np.dot(wrist_to_elbow, direction) * direction
                wrist_pole_norm = float(np.linalg.norm(wrist_pole))
                # A wrist axis parallel to shoulder->wrist gives no bend-plane information.
                if wrist_pole_norm > 0.15:
                    wrist_pole /= wrist_pole_norm
                    # Reduce influence when the projected axis is poorly conditioned.
                    effective_weight = weight * min(1.0, wrist_pole_norm / 0.35)
                    blended = (1.0 - effective_weight) * pole + effective_weight * wrist_pole
                    blended_norm = float(np.linalg.norm(blended))
                    if blended_norm > 1e-6:
                        pole = blended / blended_norm
        elbow = shoulder + along * direction + height * pole
        key = "left" if side > 0 else "right"
        previous = self._vrbody_prev_elbows.get(key)
        if previous is not None:
            alpha = np.clip(self._vrbody_elbow_smoothing, 0.0, 1.0)
            elbow = previous * (1.0 - alpha) + elbow * alpha
        self._vrbody_prev_elbows[key] = elbow.copy()
        return elbow

    def _vrbody_update_capsule(self, env_ids, obj_id, start, end, dt):
        midpoint = np.asarray((start + end) * 0.5, dtype=np.float32)
        axis = np.asarray(end - start, dtype=np.float64)
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            return
        axis /= length
        rot, _ = R.align_vectors([axis], [[0.0, 0.0, 1.0]])
        pos = torch.as_tensor(midpoint, device=self._device, dtype=torch.float32).reshape(1, 3)
        quat = torch.as_tensor(rot.as_quat(), device=self._device, dtype=torch.float32)
        previous = self._vrbody_prev_pos.get(obj_id)
        if previous is None:
            vel = torch.zeros((1, 3), device=self._device, dtype=torch.float32)
        else:
            vel = (pos - previous) / dt
            speed = float(torch.linalg.vector_norm(vel))
            if speed < 0.03:
                vel.zero_()
            elif speed > 3.0:
                vel *= (3.0 / speed)
        self._vrbody_prev_pos[obj_id] = pos.clone()
        if self._vrpd_enabled and len(self._vrbody_box_ids) == 5:
            pd_names = ("left_forearm", "right_forearm", "left_upper_arm", "right_upper_arm")
            for i, name in enumerate(pd_names, start=1):
                if obj_id != self._vrbody_box_ids[i]:
                    continue
                self._vrpd_target(name, obj_id, midpoint, quat.detach().cpu().numpy(), env_ids)
                lock_rotation = self._vrpd_upper_arm_lock_rotation if name.endswith("_upper_arm") else self._vrpd_forearm_lock_rotation
                if lock_rotation:
                    self._engine.set_root_rot(env_ids, obj_id, quat)
                    self._engine.set_root_ang_vel(env_ids, obj_id, torch.zeros((1, 3), device=self._device))
                return
        self._engine.set_root_pos(env_ids, obj_id, pos)
        self._engine.set_root_rot(env_ids, obj_id, quat)
        self._engine.set_root_vel(env_ids, obj_id, vel)
        self._engine.set_root_ang_vel(env_ids, obj_id, torch.zeros((1, 3), device=self._device))

    def _update_vrbody_capsules(self, env_ids, tracked, dt):
        if not self._vrbody_enabled or len(self._vrbody_box_ids) != 5:
            return
        if not all(key in tracked for key in (0, 8, 16)):
            return
        left_hand, left_quat = tracked[0]
        right_hand, right_quat = tracked[8]
        head = tracked[16][0]
        hmd_quat = tracked[16][1]
        # Only HMD yaw is used, keeping the torso upright even when looking down.
        facing_forward = R.from_quat(hmd_quat).apply([1.0, 0.0, 0.0])
        yaw = np.arctan2(facing_forward[1], facing_forward[0])
        facing = R.from_euler("z", yaw)
        chest = head + facing.apply([-self._vrbody_chest_back, 0.0, -self._vrbody_neck_drop])
        waist = chest + np.array([0.0, 0.0, -self._vrbody_torso_length])
        shoulder_center = chest + np.array([0.0, 0.0, 0.0])
        left_shoulder = shoulder_center + facing.apply([0.0, self._vrbody_shoulder_width * 0.5, 0.0])
        right_shoulder = shoulder_center + facing.apply([0.0, -self._vrbody_shoulder_width * 0.5, 0.0])
        left_elbow = self._vrbody_solve_elbow(left_shoulder, left_hand, 1.0, facing, left_quat)
        right_elbow = self._vrbody_solve_elbow(right_shoulder, right_hand, -1.0, facing, right_quat)
        self._vrbody_update_capsule(env_ids, self._vrbody_box_ids[0], chest, waist, dt)
        for side, shoulder, elbow, wrist, forearm_id, upper_id in (
            (1.0, left_shoulder, left_elbow, left_hand, self._vrbody_box_ids[1], self._vrbody_box_ids[3]),
            (-1.0, right_shoulder, right_elbow, right_hand, self._vrbody_box_ids[2], self._vrbody_box_ids[4]),
        ):
            # Wrist-first placement: keep the hand-ball/forearm gap constant in
            # the target geometry, and absorb excess reach at the shoulder.
            forearm_dir = wrist - elbow
            forearm_dist = float(np.linalg.norm(forearm_dir))
            if forearm_dist < 1e-6:
                continue
            forearm_dir /= forearm_dist
            forearm_length = self._vrbody_forearm_length - self._vrbody_forearm_elbow_trim
            forearm_end = wrist - forearm_dir * self._vrbody_wrist_clearance
            forearm_start = forearm_end - forearm_dir * forearm_length
            self._vrbody_update_capsule(env_ids, forearm_id, forearm_start, forearm_end, dt)
            # The elbow anchor follows the wrist rather than the fixed shoulder.
            # Preserve the existing elbow trim between the two capsules.
            elbow_anchor = forearm_start - forearm_dir * self._vrbody_forearm_elbow_trim
            upper_dir = elbow_anchor - shoulder
            upper_dist = float(np.linalg.norm(upper_dir))
            if upper_dist < 1e-6:
                upper_dir = -forearm_dir
            else:
                upper_dir /= upper_dist
            upper_end = elbow_anchor - upper_dir * self._vrbody_upper_elbow_trim
            outward = facing.apply([0.0, side * self._vrbody_shoulder_outset, 0.0])
            shoulder_guide = shoulder + upper_dir * self._vrbody_upper_shoulder_trim + outward
            capsule_axis = upper_end - shoulder_guide
            capsule_dist = float(np.linalg.norm(capsule_axis))
            if capsule_dist < 1e-6:
                capsule_axis = upper_dir
            else:
                capsule_axis /= capsule_dist
            capsule_length = self._vrbody_upper_arm_length - self._vrbody_upper_shoulder_trim - self._vrbody_upper_elbow_trim
            upper_start = upper_end - capsule_axis * capsule_length
            self._vrbody_update_capsule(env_ids, upper_id, upper_start, upper_end, dt)
