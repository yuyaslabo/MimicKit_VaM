# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class VaMUDPMixin:
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
            unity_local_rot = R.from_euler('zxy', [e_y, e_z, e_x], degrees=True)
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
                
                # 🟢 【今回の核心】
                # すべてのボーンに「生の骨盤クォータニオン」を渡し、一度ローカルに戻して処理をさせる
                local_unity_rot = self._vam_quat_xyzw(q_vec, label, pelvis_quat=pelvis_q_base)
                
                # 🟢 最後にUnity側の骨盤ワールド回転と掛け合わせることで、破綻のない世界座標を生成
                if label == "pelvis":
                    rot_conv = local_unity_rot
                else:
                    rot_conv = pelvis_rot_converted * local_unity_rot
                
                raw_spatial_data[label] = {"pos": pos_conv.copy(), "rot": rot_conv}

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
            local_offset = np.array([0.0, -0.05, 0.0])
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
            
            # 💡 角度を設定（例：+2度。下を向くか上を向くかはVaMのボーン仕様によります）
            # 数値をマイナス（-2.0など）にすれば逆方向に回転します
            
            
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
