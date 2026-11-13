# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class VaMUDPMixin:
    # Existing body conversion retained as fallback only.
    # Legs and arms override rotations via MJCF joint FK.
    _vam_limb_parts = frozenset((
        "rhand", "lhand", "relbow", "lelbow", "rarm", "larm",
        "rfoot", "lfoot", "rshin", "lshin", "rthigh", "lthigh",
    ))
    _vam_limb_basis = np.array([
        [0., -1., 0.],
        [0.,  0., 1.],
        [1.,  0., 0.],
    ], dtype=np.float64)

    @staticmethod
    def _vam_newton_rotation(quat):
        """MimicKit body quaternion [w,x,y,z] -> scipy Rotation."""
        q = np.asarray(quat, dtype=np.float64).reshape(-1)
        if q.size != 4 or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-8:
            return R.identity()
        return R.from_quat([q[1], q[2], q[3], q[0]] / np.linalg.norm(q))

    def _vam_limb_bind(self, key):
        """Fixed CP rest-orientation only; no dynamic axis permutations."""
        bind_quats = getattr(self, "_vam_limb_bind_quat_xyzw", {})
        if key in bind_quats:
            q = np.asarray(bind_quats[key], dtype=np.float64).reshape(-1)
            if q.size != 4 or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-8:
                raise ValueError("Invalid vam_limb_bind_quat_xyzw for " + key)
            return R.from_quat(q / np.linalg.norm(q))
        # 左右で独立して調整
        arm_offsets = {
            "rhand":  [-90.0, -90.0, 0.0],
            "lhand":  [-90.0, 90.0, 0.0],
            "relbow": [-90.0, -90.0, 0.0],
            "lelbow": [-90.0, 90.0, 0.0],
            "rarm": [-90.0, -90.0, 0.0],
            "larm": [-90.0, 90.0, 0.0],
        }

        default = arm_offsets.get(
            key,
            [(-80.0 if key in ("rfoot", "lfoot") else -90.0), 0., 0.]
        )
        #default = [(-70.0 if key in ("rfoot", "lfoot") else -90.0), 0., 0.]
        euler = getattr(self, "_vam_limb_bind_euler_deg", {}).get(key, default)
        return R.from_euler("xyz", euler, degrees=True)

    def _vam_limb_pelvis_relative_rotation(self, quat, pelvis_quat, pelvis_vam_world, key):
        """Single matrix basis for all 12 limbs. Never mix Euler remapping."""
        relative_n = self._vam_newton_rotation(pelvis_quat).inv() * self._vam_newton_rotation(quat)
        m = self._vam_limb_basis
        relative_v = R.from_matrix(m @ relative_n.as_matrix() @ m.T)
        return pelvis_vam_world * relative_v * self._vam_limb_bind(key)

    # Bilateral G1 leg FK. DoF indices assume MimicKit's 29-DoF order
    # follows the supplied G1 actuator list (not guaranteed by MJCF).
    # Left 0..5, right 6..11. Validate angles from the logs.
    _vam_leg_fk_indices = {"L": (0, 1, 2, 3, 4, 5),
                           "R": (6, 7, 8, 9, 10, 11)}
    _vam_leg_fk_diag_enabled = False
    _vam_leg_fk_diag_interval_sec = 1.0

    @staticmethod
    def _vam_leg_fk_mjcf(dof6, side):
        """Pelvis-local G1 orientations for thigh, shin, foot (MJCF).

        Both provided left/right XML excerpts use the SAME body-frame
        fixed rotations on hip_roll_link and knee_link. Their inertial
        quaternions are deliberately NOT applied (inertial frame only).
        """
        hp, hr, hy, kn, ap, ar = [float(v) for v in dof6]
        if side not in ("L", "R"):
            raise ValueError("Invalid FK side: %s" % side)
        hip_roll_bind = R.from_quat([0., -0.0873386, 0., 0.996179])
        knee_bind = R.from_quat([0., +0.0873386, 0., 0.996179])
        q_thigh = (R.from_rotvec([0., hp, 0.])
                   * hip_roll_bind
                   * R.from_rotvec([hr, 0., 0.])
                   * R.from_rotvec([0., 0., hy]))
        q_shin = q_thigh * knee_bind * R.from_rotvec([0., kn, 0.])
        q_foot = (q_shin * R.from_rotvec([0., ap, 0.])
                  * R.from_rotvec([ar, 0., 0.]))
        prefix = 'l' if side == 'L' else 'r'
        return {prefix + 'thigh': q_thigh,
                prefix + 'shin': q_shin,
                prefix + 'foot': q_foot}

    def _vam_apply_both_legs_6dof_fk(self, char_id, pelvis_vam_rot, data):
        """Apply MJCF FK to both legs, leaving ALL positions untouched.

        Body orientations are not referenced here. Both legs share the
        proven right-leg basis conversion and per-control bind logic.
        Read DoFs once; calculate both before committing either leg.
        """
        try:
            dofs = np.asarray(self._vam_to_numpy(
                self._engine.get_dof_pos(char_id)), dtype=np.float64)
            if dofs.ndim == 2:
                dofs = dofs[0]
            if dofs.ndim != 1:
                raise ValueError('get_dof_pos shape=%s' % (dofs.shape,))
            if dofs.size <= 11:
                raise ValueError('get_dof_pos size=%d (need >=12)' % dofs.size)
            basis = self._vam_limb_basis
            pending = {}
            logs = []
            for side in ('L', 'R'):
                indices = self._vam_leg_fk_indices[side]
                angles = dofs[list(indices)]
                if not np.all(np.isfinite(angles)):
                    raise ValueError('nonfinite %s leg DoFs' % side)
                leg_fk = self._vam_leg_fk_mjcf(angles, side)
                for label, rel in leg_fk.items():
                    rel_v = R.from_matrix(basis @ rel.as_matrix() @ basis.T)
                    pending[label] = (pelvis_vam_rot * rel_v
                                      * self._vam_limb_bind(label))
                logs.append((side, angles))
            for label, rotation in pending.items():
                data[label]['rot'] = rotation
        except Exception as exc:
            now = time.monotonic()
            if now - getattr(self, '_vam_leg_fk_error_last', -1e10) >= 2.0:
                self._vam_leg_fk_error_last = now
                print('[LEG-FK] fallback to Body rotation: %s' % exc,
                      flush=True)
            return False

        if self._vam_leg_fk_diag_enabled:
            now = time.monotonic()
            if now - getattr(self, '_vam_leg_fk_diag_last', -1e10) >= self._vam_leg_fk_diag_interval_sec:
                self._vam_leg_fk_diag_last = now
                names = ('hipP', 'hipR', 'hipY', 'knee', 'ankP', 'ankR')
                for side, angles in logs:
                    fields = ' '.join('%s=%+6.1f' % (name, np.rad2deg(val))
                                      for name, val in zip(names, angles))
                    print('[LEG-FK %s] %s' % (side, fields), flush=True)
        return True

    # Bilateral G1 arm FK: left indices 15..21, right 22..28.
    # Anchor is the torso (waist-pitch) body, NOT pelvis: preserves waist rotation.
    # The user-supplied MJCF has opposite shoulder static quaternions L/R.
    _vam_arm_fk_indices = {"L": (15, 16, 17, 18, 19, 20, 21),
                           "R": (22, 23, 24, 25, 26, 27, 28)}
    _vam_arm_fk_diag_enabled = False
    _vam_arm_fk_diag_interval_sec = 1.0

    @staticmethod
    def _vam_arm_fk_mjcf(dof7, side):
        """Torso-local G1 shoulder-yaw, elbow, wrist-yaw orientations.

        MJCF fixed body quaternions are wxyz; SciPy accepts xyzw.
        Note inertial quats are NOT body-frame transforms.
        Each hinge axis belongs to that body's local coordinate system.
        """
        sp, sr, sy, el, wr, wp, wy = [float(v) for v in dof7]
        if side == 'L':
            shoulder_pitch_bind_wxyz = (0.990264, 0.139201, 0.0000138722, -0.0000986868)
            shoulder_roll_bind_wxyz = (0.990268, -0.139172, 0.0, 0.0)
            prefix = 'l'
        elif side == 'R':
            shoulder_pitch_bind_wxyz = (0.990264, -0.139201, 0.0000138722, 0.0000986868)
            shoulder_roll_bind_wxyz = (0.990268, 0.139172, 0.0, 0.0)
            prefix = 'r'
        else:
            raise ValueError('Invalid FK side: %s' % side)

        def from_wxyz(q):
            w, x, y, z = q
            return R.from_quat([x, y, z, w])

        q_pitch = from_wxyz(shoulder_pitch_bind_wxyz) * R.from_rotvec([0., sp, 0.])
        q_roll = q_pitch * from_wxyz(shoulder_roll_bind_wxyz) * R.from_rotvec([sr, 0., 0.])
        q_arm = q_roll * R.from_rotvec([0., 0., sy])
        q_elbow = q_arm * R.from_rotvec([0., el, 0.])
        q_wrist_roll = q_elbow * R.from_rotvec([wr, 0., 0.])
        q_wrist_pitch = q_wrist_roll * R.from_rotvec([0., wp, 0.])
        q_hand = q_wrist_pitch * R.from_rotvec([0., 0., wy])
        # rarm/larm = shoulder_yaw_link, elbow = elbow_link,
        # hand = wrist_yaw_link (per original Body index mapping).
        return {prefix + 'arm': q_arm,
                prefix + 'elbow': q_elbow,
                prefix + 'hand': q_hand}

    def _vam_apply_both_arms_7dof_fk(self, char_id, torso_vam_rot, data):
        """Change only six arm control rotations; positions remain unchanged.

        FK calculated in the MJCF torso frame, then mapped to the same
        torso-local VaM basis. With no body rotations, fallback remains intact.
        """
        try:
            dofs = np.asarray(self._vam_to_numpy(
                self._engine.get_dof_pos(char_id)), dtype=np.float64)
            if dofs.ndim == 2:
                dofs = dofs[0]
            if dofs.ndim != 1 or dofs.size < 29:
                raise ValueError('get_dof_pos needs 29 values, got %s' % (dofs.shape,))
            basis = self._vam_limb_basis
            pending = {}
            logs = []
            for side in ('L', 'R'):
                angles = dofs[list(self._vam_arm_fk_indices[side])]
                if not np.all(np.isfinite(angles)):
                    raise ValueError('nonfinite %s arm DoFs' % side)
                fk = self._vam_arm_fk_mjcf(angles, side)
                for label, rel in fk.items():
                    rel_v = R.from_matrix(basis @ rel.as_matrix() @ basis.T)
                    pending[label] = (torso_vam_rot * rel_v
                                      * self._vam_limb_bind(label))
                logs.append((side, angles))
            for label, rotation in pending.items():
                data[label]['rot'] = rotation
        except Exception as exc:
            now = time.monotonic()
            if now - getattr(self, '_vam_arm_fk_error_last', -1e10) >= 2.0:
                self._vam_arm_fk_error_last = now
                print('[ARM-FK] fallback to Body rotation: %s' % exc,
                      flush=True)
            return False

        if self._vam_arm_fk_diag_enabled:
            now = time.monotonic()
            if now - getattr(self, '_vam_arm_fk_diag_last', -1e10) >= self._vam_arm_fk_diag_interval_sec:
                self._vam_arm_fk_diag_last = now
                names = ('shP', 'shR', 'shY', 'elbow', 'wrR', 'wrP', 'wrY')
                for side, angles in logs:
                    fields = ' '.join('%s=%+6.1f' % (name, np.rad2deg(val))
                                      for name, val in zip(names, angles))
                    print('[ARM-FK %s] %s' % (side, fields), flush=True)
        return True

    def _vam_to_numpy(self, value):
        if torch.is_tensor(value):
            return value.detach().to("cpu").numpy()
        return np.asarray(value)

    def _vam_pos(self, pos):
        # MuJoCo [x, y, z] -> VaM [-y, z, x] （既存の成功ロジックを維持）
        p = np.asarray(pos, dtype=np.float64)
        rotated_x = -p[1]
        rotated_y = p[2]
        rotated_z = p[0]
        return np.asarray([rotated_x, rotated_y, rotated_z], dtype=np.float32) + self._vam_position_offset

    def _vam_quat_xyzw(self, quat, key=None, pelvis_quat=None):
        q = np.asarray(quat, dtype=np.float64).reshape(-1)
        if q.size != 4 or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-8:
            return R.from_quat([0.0, 0.0, 0.0, 1.0])
        q = q / np.linalg.norm(q)
        
        # DeepMimic [w, x, y, z] -> scipy [x, y, z, w]
        current_world_rot = R.from_quat([q[1], q[2], q[3], q[0]])

        if pelvis_quat is not None and key != "pelvis":
            qp = np.asarray(pelvis_quat, dtype=np.float64).reshape(-1)
            qp = qp / np.linalg.norm(qp)
            pelvis_world_rot = R.from_quat([qp[1], qp[2], qp[3], qp[0]])
        else:
            pelvis_world_rot = R.from_quat([0.0, 0.0, 0.0, 1.0])

        local_rot = pelvis_world_rot.inv() * current_world_rot

        # --- 2. ローカル空間のオイラー角で軸の入れ替えを行う ---
        euler = local_rot.as_euler('xyz', degrees=True)
        e_x, e_y, e_z = float(euler[0]), float(euler[1]), float(euler[2])
        
        if key in ("rfoot", "lfoot"):
            # # 入れ替え後の [e_y, -e_z, -e_x] の第3引数である「-e_x」がプラスマイナス反転して「e_x」になります
            unity_local_rot = R.from_euler('zxy', [e_x, e_z, e_y], degrees=True)
        elif key in ("relbow", "rhand"):
            unity_local_rot = R.from_euler('zxy', [e_z+90, e_y, e_x], degrees=True)
        elif key in ("lelbow","lhand"):
            unity_local_rot = R.from_euler('zxy', [e_z+90, -e_y, -e_x], degrees=True)
        else:
            # # それ以外の部位は通常通り
            unity_local_rot = R.from_euler('zxy', [e_y, e_z, -e_x], degrees=True)

        # --- 3. 左手系変換に伴うクォータニオンの反転 ---
        qx, qy, qz, qw = unity_local_rot.as_quat()
        unity_local_rot = R.from_quat([-qx, -qy, -qz, qw])

        # --- 4. Tポーズにおける「右半身」のローカル軸反転 ---
        if key in ("rhand", "relbow", "rshin", "rarm", "rthigh"):
            mqx, mqy, mqz, mqw = unity_local_rot.as_quat()
            unity_local_rot = R.from_quat([-mqx, mqy, -mqz, mqw])

        # 🟢 🟢 🟢 【下を向いたZ軸矢印を正面に起こす補正】 🟢 🟢 🟢
        # パケットが正しく送れる状態であれば、ここの角度（90度 または -90度）で
        # 完全にコントロールポイントの矢印の向きを制御できます。
        # まずは -90度（または 90度）を試して、矢印が水平に起き上がるか確認してください。
        if key in ("rfoot", "lfoot"):
            unity_local_rot = unity_local_rot * R.from_euler('x', -70, degrees=True)
        # elif key in ("relbow", "lelbow"):
            # unity_local_rot = unity_local_rot * R.from_euler('x', -90, degrees=True)
        else:
            unity_local_rot = unity_local_rot * R.from_euler('x', -90, degrees=True)
        return unity_local_rot

    def _send_vam_udp(self):
        if self._vam_socket is None:
            return            
        try:
            char_id = self._get_char_id()
            
            # 1. MimiC(MuJoCo)エンジンから生データの取得
            body_pos = self._vam_to_numpy(self._engine.get_body_pos(char_id))
            if body_pos.ndim == 3:
                body_pos = body_pos[0]  # 安全に次元を絞る (1, 31, 3) -> (31, 3)
            
            body_count = body_pos.shape[0]  # 31

            root_pos = self._vam_to_numpy(self._engine.get_root_pos(char_id))
            root_rot = self._vam_to_numpy(self._engine.get_root_rot(char_id))
            if root_pos.ndim > 1: root_pos = root_pos[0]
            if root_rot.ndim > 1: root_rot = root_rot[0]

            body_rot = None
            for method_name in ("get_body_rot", "get_body_rotation", "get_body_quat"):
                method = getattr(self._engine, method_name, None)
                if callable(method):
                    try:
                        body_rot = self._vam_to_numpy(method(char_id))
                        if body_rot.ndim == 3:
                            body_rot = body_rot[0]
                        break
                    except Exception:
                        body_rot = None

            # 2. XMLの定義順序に基づく、MimiC(31ボーン) -> VaMへのダイレクトインデックス対応表
            MIMIC_INDEX_MAP = {
                "pelvis": 0,
                "torso": 15,
                "head": 16,
                "rhand": 30,
                "lhand": 23,
                "rfoot": 12,
                "lfoot": 6,
                "relbow": 27,
                "lelbow": 20,
                "rshin": 10,
                "lshin": 4,
                "rarm": 26,
                "larm": 19,
                "rthigh": 9,
                "lthigh": 3,
            }

            # 骨盤（pelvis）の「変換前の生クォータニオン」を抽出
            pelvis_idx = MIMIC_INDEX_MAP["pelvis"]
            if body_rot is not None and pelvis_idx < len(body_rot):
                pelvis_q_base = body_rot[pelvis_idx]
            else:
                pelvis_q_base = root_rot

            # 骨盤のベース位置を取得
            if pelvis_idx < body_count:
                p_base = body_pos[pelvis_idx]
            else:
                p_base = root_pos
            pelvis_converted = self._vam_pos(p_base)

            # 🟢 骨盤自身の変換済み「Unityワールド回転」を真っ先に確定させる
            # (pelvis自身は基準なので pelvis_quat=None で呼び出す)
            pelvis_rot_converted = self._vam_quat_xyzw(pelvis_q_base, "pelvis", pelvis_quat=None)
            
            # 足伸ばし計算で使用する骨盤回転とその逆回転（Inverse）
            R_pelvis = pelvis_rot_converted
            R_pelvis_inv = pelvis_rot_converted.inv()

            # 3. 各ボーンの位置の基本変換、および「ローカル軸反転・マッピング」の実行とワールド復元
            raw_spatial_data = {}
            for label in MIMIC_INDEX_MAP.keys():
                idx = MIMIC_INDEX_MAP[label]
                
                # 位置の取得と Unity 空間への基本変換
                if idx < body_count:
                    p_vec = body_pos[idx]
                else:
                    p_vec = root_pos
                pos_conv = self._vam_pos(p_vec)

                # 回転の取得
                if body_rot is not None and idx < len(body_rot):
                    q_vec = body_rot[idx]
                else:
                    q_vec = root_rot
                
                # 12 limbs share ONE pelvis-relative matrix mapping.
                # Ankle axis correction is NOT applied here (Newton-side
                # correction previously cancelled with this basis mapping).
                if label in self._vam_limb_parts:
                    rot_conv = self._vam_limb_pelvis_relative_rotation(
                        q_vec, pelvis_q_base, pelvis_rot_converted, label
                    )
                else:
                    # Preserve proven torso / head / pelvis implementation.
                    local_unity_rot = self._vam_quat_xyzw(
                        q_vec, label, pelvis_quat=pelvis_q_base
                    )
                    rot_conv = (local_unity_rot if label == "pelvis"
                                else pelvis_rot_converted * local_unity_rot)

                raw_spatial_data[label] = {"pos": pos_conv.copy(), "rot": rot_conv}

            # Both legs: MJCF joint FK from independent 6-DoF chains.
            # No additional output-space ankle correction for either leg.
            # This deliberately matches the previously successful RIGHT leg.
            self._vam_apply_both_legs_6dof_fk(
                char_id, pelvis_rot_converted, raw_spatial_data
            )

            # Both arms: 7DoF FK, parent = torso (waist-pitch link).
            # Use the EXISTING correctly converted torso rotation; do not
            # change torso/head/pelvis, body positions, or leg FK.
            self._vam_apply_both_arms_7dof_fk(
                char_id, raw_spatial_data['torso']['rot'], raw_spatial_data
            )

            # 🟢 【STEP 1: 骨盤回転を考慮したローカル空間での足伸ばし】
            extended_positions = {}
            leg_scale = 1.32  # 足伸ばし倍率

            # 引き伸ばす前の「最も低い足（床に接している部分）」のY座標（高さ）を記録しておく
            original_lowest_y = 999.0
            for label in ("rfoot", "lfoot"):
                if label in raw_spatial_data:
                    original_lowest_y = min(original_lowest_y, float(raw_spatial_data[label]["pos"][1]))

            for label, data in raw_spatial_data.items():
                pos_conv = data["pos"]
                
                if label in ("rshin", "lshin", "rfoot", "lfoot"):
                    # 骨盤から対象ボーンへの世界座標（Unity空間）ベクトル
                    world_vec = pos_conv - pelvis_converted
                    
                    # 💡 骨盤の逆回転を適用し、ロボット自身のローカル座標ベクトルに変換
                    local_vec = R_pelvis_inv.apply(world_vec)
                    
                    # 🛠️ ロボットのローカルな「股関節から足先への方向」を引き伸ばす
                    local_vec[2] *= leg_scale

                    
                    # 💡 引き伸ばしたローカルベクトルを、現在の骨盤の回転を使って世界座標系に逆変換
                    world_vec_extended = R_pelvis.apply(local_vec)
                    
                    # 骨盤の現在位置をベースに世界座標を確定
                    extended_positions[label] = pelvis_converted + world_vec_extended
                else:
                    # 足以外のボーン（骨盤自身、上半身、頭、腕など）は、この段階では元の位置をキープ
                    extended_positions[label] = pos_conv.copy()

            # 🟢 【STEP 2: 地面基準（床に倒れた時も浮かないようにする）の高さ補正値を算出】
            extended_lowest_y = 999.0
            for label in ("rfoot", "lfoot"):
                if label in extended_positions:
                    extended_lowest_y = min(extended_lowest_y, float(extended_positions[label][1]))

            # 💡 足が伸びたことによって床を突き抜けた（あるいは倒れて位置がずれた）距離を計算
            lift_offset_y = original_lowest_y - extended_lowest_y

            # 🟢 【STEP 3: 全身のボーンに床基準のオフセットを一括適用して最終確定】
            local_offset = np.array([0.0, -0.0, 0.0])
            world_offset = R_pelvis.apply(local_offset)
            final_positions = {}
            for label in raw_spatial_data.keys():
                final_positions[label] = extended_positions[label].copy()
                if label in ("relbow", "lelbow", "rhand", "lhand","head","pelvis"):
                    final_positions[label][1] += (lift_offset_y * 0.5)
                else:
                    final_positions[label][1] += lift_offset_y
                final_positions[label] += world_offset

            # 4. 全15部位の最終変換済みデータを抽出
            pelvis_p, pelvis_q = final_positions["pelvis"], raw_spatial_data["pelvis"]["rot"]
            torso_p,  torso_q  = final_positions["torso"],  raw_spatial_data["torso"]["rot"]
            head_p,   head_q   = final_positions["head"],   raw_spatial_data["head"]["rot"]
            

            
            
            rhand_p,  rhand_q  = final_positions["rhand"],  raw_spatial_data["rhand"]["rot"]
            lhand_p,  lhand_q  = final_positions["lhand"],  raw_spatial_data["lhand"]["rot"]
            
            rfoot_p,  rfoot_q  = final_positions["rfoot"],  raw_spatial_data["rfoot"]["rot"]
            lfoot_p,  lfoot_q  = final_positions["lfoot"],  raw_spatial_data["lfoot"]["rot"]
            
            relbow_p, relbow_q = final_positions["relbow"], raw_spatial_data["relbow"]["rot"]
            lelbow_p, lelbow_q = final_positions["lelbow"], raw_spatial_data["lelbow"]["rot"]
            
            rshin_p,  rshin_q  = final_positions["rshin"],  raw_spatial_data["rshin"]["rot"]
            lshin_p,  lshin_q  = final_positions["lshin"],  raw_spatial_data["lshin"]["rot"]
            
            rarm_p,   rarm_q   = final_positions["rarm"],   raw_spatial_data["rarm"]["rot"]
            larm_p,   larm_q   = final_positions["larm"],   raw_spatial_data["larm"]["rot"]
            
            rthigh_p, rthigh_q = final_positions["rthigh"], raw_spatial_data["rthigh"]["rot"]
            lthigh_p, lthigh_q = final_positions["lthigh"], raw_spatial_data["lthigh"]["rot"]
            
            offset_pelvis_angle_deg = -90.0 
            pelvis_local_offset = R.from_euler('x', offset_pelvis_angle_deg, degrees=True)
            pelvis_q = pelvis_q * pelvis_local_offset
            
            offset_head_angle_deg = 3.0 
            
            # 首ローカルのX軸回転クォータニオンを作成
            head_local_offset = R.from_euler('x', offset_head_angle_deg, degrees=True)
            
            # 💡 現在の首の世界回転(head_q)に対して、ローカル基準で回転を掛け合わせる（右から掛ける）
            head_q = head_q * head_local_offset

            # 5. VaMの133fパケット仕様（全19個のコントロールポイント）に完全に一致させる
            points_config = [
                ("pelvis", pelvis_p, pelvis_q),
                ("torso", torso_p, torso_q),
                ("head", head_p, head_q),
                ("rhand", rhand_p, rhand_q),
                ("lhand", lhand_p, lhand_q),
                ("rfoot", rfoot_p, rfoot_q),
                ("lfoot", lfoot_p, lfoot_q),
                ("relbow", relbow_p, relbow_q),
                ("lelbow", lelbow_p, lelbow_q),
                ("rshin", rshin_p, rshin_q),
                ("lshin", lshin_p, lshin_q),
                ("rarm", rarm_p, rarm_q),
                ("larm", larm_p, larm_q),
                ("rthigh", rthigh_p, rthigh_q),
                ("lthigh", lthigh_p, lthigh_q),
                ("rarm_dup", rarm_p, rarm_q),  
                ("larm_dup", larm_p, larm_q),  
                ("rtoe", rfoot_p, rfoot_q),    
                ("ltoe", lfoot_p, lfoot_q)     
            ]

            # 6. パケットのシリアライズと送信
            packet = []
            for idx, (label, pos_vec, rot_obj) in enumerate(points_config):
                qx, qy, qz, qw = rot_obj.as_quat()
                packet.extend([float(pos_vec[0]), float(pos_vec[1]), float(pos_vec[2]), 
                               float(qx), float(qy), float(qz), float(qw)])
            
            packet = np.asarray(packet, dtype="<f4")
            
            if packet.size == 133:
                self._vam_socket.sendto(self._vam_packet.pack(*packet.tolist()), (self._vam_ip, self._vam_port))
            else:
                print(f"[ERROR] Packet size is incorrect: {packet.size} (Expected: 133)")
                
        except Exception as e:
            print(f"UDP Send Error: {e}")
