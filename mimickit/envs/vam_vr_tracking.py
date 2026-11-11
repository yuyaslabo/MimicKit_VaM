# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import numpy as np
import time
import torch


class VRTrackingMixin:
    def _vrhand_park_during_startup(self):
        # Dynamic collision bodies must not fall into the scene during the delay.
        ids = list(self._vrhand_box_ids)
        if self._vrbody_enabled:
            ids.extend(self._vrbody_box_ids)
        if not ids:
            return
        num_envs = self.get_num_envs()
        env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)
        for idx, obj_id in enumerate(ids):
            parked = torch.zeros((num_envs, 3), device=self._device, dtype=torch.float32)
            parked[:, 0] = 100.0 + float(idx) * 2.0
            parked[:, 2] = 1000.0
            self._engine.set_root_pos(env_ids, obj_id, parked)
            self._engine.set_root_vel(env_ids, obj_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, obj_id, 0.0)

    def _vam_pos_to_engine(self, pos_vam):
        p = np.asarray(pos_vam, dtype=np.float64) - self._vrhand_position_offset
        return np.asarray([p[2], -p[0], p[1]], dtype=np.float32)

    def _vam_quat_to_engine(self, quat_vam_xyzw):
        qx, qy, qz, qw = np.asarray(quat_vam_xyzw, dtype=np.float64)
        q = np.asarray([-qx, -qz, -qy, qw], dtype=np.float64)
        norm = np.linalg.norm(q)
        if not np.isfinite(q).all() or norm < 1e-8:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return (q / norm).astype(np.float32)

    def _update_vrhand_boxes(self):
        if not self._vrhand_enabled or self._vrhand_socket is None:
            return
        # 💡 self._vrhand_box_ids を [左手ID, 右手ID, HMD_ID] の3つの要素にしてください
        if len(self._vrhand_box_ids) != 3:
            return

        if not hasattr(self, '_prev_vrhand_pos'):
            self._prev_vrhand_pos = {}
            self._vrhand_triggers = {"left": 0.0, "right": 0.0}
            self._vrhand_grabbed_targets = {"left": None, "right": None}

        # 💡 パケットサイズを 92 バイト（float 23個分：左8 + 右8 + HMD7）に変更
        # ※ self._vrhand_packet（struct）のフォーマットも '23f' に書き換えてください
        EXPECTED_SIZE = 92 

        latest_packet = None
        while True:
            try:
                packet, addr = self._vrhand_socket.recvfrom(1024)
            except BlockingIOError:
                break
            if len(packet) == EXPECTED_SIZE:
                latest_packet = packet
                self._vrhand_packet_count += 1
            else:
                # 💡 想定外のサイズが来ている場合の警告デバッグ
                print(f"[VRHand Debug] Warning: Received invalid packet size: {len(packet)} bytes (Expected: {EXPECTED_SIZE})")

        # Discard incoming samples during startup. Do not set target positions,
        # apply PD forces, drag the agent, or activate hand/HMD/body colliders.
        # The timer starts on the first physics update after construction.
        now = time.monotonic()
        if self._vrhand_start_time is None:
            self._vrhand_start_time = now
            if self._vrhand_start_delay > 0.0:
                print(f"[VRHand] Startup synchronization delay: {self._vrhand_start_delay:.1f}s", flush=True)
        if now - self._vrhand_start_time < self._vrhand_start_delay:
            self._vrhand_latest = None
            self._vrhand_park_during_startup()
            return
        if not self._vrhand_delay_finished:
            self._vrhand_delay_finished = True
            self._vrhand_latest = None
            self._prev_vrhand_pos.clear()
            self._vrbody_prev_pos.clear()
            self._vrbody_prev_elbows.clear()
            self._vrpd_targets.clear()
            self._vrpd_initialized.clear()
            self._vrhand_grabbed_targets = {"left": None, "right": None}
            self._vrhand_triggers = {"left": 0.0, "right": 0.0}
            self._vrhand_trigger_pressed = {"left": False, "right": False}
            self._vrhand_last_unlocked_press = {"left": None, "right": None}
            print("[VRHand] Startup delay finished; hand/HMD/body synchronization enabled", flush=True)

        if latest_packet is not None:
            # 23個のfloatを展開
            self._vrhand_latest = self._vrhand_packet.unpack(latest_packet)
            
            # トリガー値を取得 (左: 7, 右: 15)
            self._vrhand_triggers["left"] = float(self._vrhand_latest[7])
            self._vrhand_triggers["right"] = float(self._vrhand_latest[15])

        if self._vrhand_latest is None:
            return

        # # 💡 --- HMD同期＆データ検証用デバッグ出力 (約1秒=30回に1回出力) ---
        # if self._vrhand_update_count % 15 == 0:
            # print(f"\n================ [VRHand Update Debug #{self._vrhand_update_count}] ================")
            # print(f"Total Floats Unpacked: {len(self._vrhand_latest)}")
            # print(f"Triggers -> Left: {self._vrhand_triggers['left']:.2f} | Right: {self._vrhand_triggers['right']:.2f}")
            
            # names = ["Left Hand", "Right Hand", "HMD"]
            # offsets = [0, 8, 16]
            # for name, offset in zip(names, offsets):
                # raw_pos = self._vrhand_latest[offset:offset+3]
                # raw_quat = self._vrhand_latest[offset+3:offset+7]
                # print(f"[{name}] Raw UDP Pos: {raw_pos} | Quat: {raw_quat}")
            # print("========================================================================\n")
        # # ------------------------------------------------------------------

        env_ids = torch.tensor([0], device=self._device, dtype=torch.long)
        dt = 1.0 / 30.0
        gravity = 9.81

        # 各データのオフセットと、対応するオブジェクトIDのペア
        # 左手(0番目から7個+トリガー1), 右手(8番目から7個+トリガー1), HMD(16番目から7個)
        targets = (
            (0, self._vrhand_box_ids[0]), # 左手
            (8, self._vrhand_box_ids[1]), # 右手
            (16, self._vrhand_box_ids[2]) # HMD
        )

        tracked = {}
        for offset, obj_id in targets:
            values = self._vrhand_latest
            pos_vam = np.asarray(values[offset:offset + 3], dtype=np.float64)
            quat_vam = np.asarray(values[offset + 3:offset + 7], dtype=np.float64)
            #quat_vam = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            
            
            if not np.isfinite(pos_vam).all() or not np.isfinite(quat_vam).all() or np.linalg.norm(quat_vam) < 1e-8:
                # 💡 データが不正でスキップされた場合のデバッグログ
                if self._vrhand_update_count % 30 == 0:
                    target_name = "LeftHand" if offset == 0 else "RightHand" if offset == 8 else "HMD"
                    print(f"[VRHand Debug] {target_name} skipped due to invalid data (Finite: {np.isfinite(pos_vam).all()}, Norm: {np.linalg.norm(quat_vam)})")
                continue
            
            pos_np = self._vam_pos_to_engine(pos_vam)
            quat_np = self._vam_quat_to_engine(quat_vam)
            
            # 手・HMD共通でZ軸のスケール補正を適用
            pos_np[2] *= 0.8  # [0]=X, [1]=Y(高さ), [2]=Z
            
            # # 💡 変換後の最終座標の確認用デバッグ
            # if self._vrhand_update_count % 30 == 0:
                # target_name = "LeftHand" if offset == 0 else "RightHand" if offset == 8 else "HMD"
                # print(f"[VRHand Debug] {target_name} -> Final Engine Pos: {pos_np}")

            pos = torch.as_tensor(pos_np, device=self._device, dtype=torch.float32).reshape(1, 3)
            raw_quat = torch.as_tensor(quat_np, device=self._device, dtype=torch.float32)
            quat = torch.zeros(4, device=self._device, dtype=torch.float32)
            quat[0] = raw_quat[1]  # 左右の傾きを前後の傾きへ
            quat[1] = -raw_quat[0]  # 前後の傾きを左右の傾きへ
            quat[2] = raw_quat[2]  # ヨー回転（合っている回転）
            quat[3] = raw_quat[3]  # W成分
            
            if offset == 16:
                quat[0] = 0.0
                quat[1] = 0.0
                #pos_np[2] = pos_np[2] - 0.1
            
            mag = torch.norm(quat)
            if mag > 1e-6:
                quat = quat / mag
            else:
                quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device, dtype=torch.float32)
                
            if offset == 16:
                self._latest_hmd_pos = pos.clone()
                self._latest_hmd_quat = quat.clone()
                # Set only after a valid HMD pose has actually been applied.
                # Unlike packet_count this cannot be triggered by hand-only/invalid data.
                self._hmd_pose_received = True
                
            # Keep the same transformed coordinates as the original hand/HMD colliders.
            tracked[offset] = (pos_np.astype(np.float64).copy(), quat.detach().cpu().numpy().copy())
            # 速度計算
            if obj_id in self._prev_vrhand_pos:
                prev_pos = self._prev_vrhand_pos[obj_id]
                vel = (pos - prev_pos) / dt
                speed = torch.norm(vel)
                if speed < 0.03:
                    vel = torch.zeros_like(vel)
                elif speed > 3.0:
                    vel = (vel / speed) * 3.0
            else:
                vel = torch.zeros((1, 3), device=self._device, dtype=torch.float32)

            self._prev_vrhand_pos[obj_id] = pos

            # 重力沈み込み防止補正
            compensated_pos = pos.clone()
            compensated_pos[:, 2] += 0.5 * gravity * (dt ** 2)

            if self._vrpd_enabled and offset in (0, 8):
                name = "left_hand" if offset == 0 else "right_hand"
                self._vrpd_target(name, obj_id, compensated_pos.detach().cpu().numpy().reshape(3), quat.detach().cpu().numpy(), env_ids)
                continue
            self._engine.set_root_pos(env_ids, obj_id, compensated_pos)
            self._engine.set_root_rot(env_ids, obj_id, quat)
            self._engine.set_root_vel(env_ids, obj_id, vel)
            self._engine.set_root_ang_vel(env_ids, obj_id, torch.zeros((1, 3), device=self._device))

        # Additional torso/arm colliders use the same UDP frame.
        self._update_vrbody_capsules(env_ids, tracked, dt)
        # ドラッグ＆ドロップ物理の適用
        self._apply_vrhand_drag(env_ids, tracked)
        # Apply PD after drag because drag clears the shared external-force buffer.
        self._vrpd_apply_forces(env_ids)
        
        self._vrhand_update_count += 1
        return
