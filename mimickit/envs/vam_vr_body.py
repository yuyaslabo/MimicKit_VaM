# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class VRBodyMixin:
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
            speed = torch.linalg.vector_norm(vel)
            vel = torch.where(speed < 0.03, torch.zeros_like(vel), vel * (3.0 / speed.clamp_min(3.0)))
        self._vrbody_prev_pos[obj_id] = pos.clone()
        self._engine.set_root_pos(env_ids, obj_id, pos)
        self._engine.set_root_rot(env_ids, obj_id, quat)
        self._engine.set_root_vel(env_ids, obj_id, vel)
        self._engine.set_root_ang_vel(env_ids, obj_id, torch.zeros((1, 3), device=self._device))

    def _update_vrbody_capsules(self, env_ids, tracked, dt):
        # Lightweight VR body: one torso capsule; no elbows, forearms or upper arms.
        if not self._vrbody_enabled or len(self._vrbody_box_ids) != 1:
            return
        if 16 not in tracked:
            return
        head, hmd_quat = tracked[16]
        # Only HMD yaw drives torso orientation; leave torso upright.
        facing_forward = R.from_quat(hmd_quat).apply([1.0, 0.0, 0.0])
        yaw = np.arctan2(facing_forward[1], facing_forward[0])
        facing = R.from_euler("z", yaw)
        chest = head + facing.apply([-self._vrbody_chest_back, 0.0, -self._vrbody_neck_drop])
        waist = chest + np.array([0.0, 0.0, -self._vrbody_torso_length])
        self._vrbody_update_capsule(env_ids, self._vrbody_box_ids[0], chest, waist, dt)
