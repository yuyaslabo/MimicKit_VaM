"""VaM compound collider MJCF assets + pose sync for MimicKit/Newton.

Port 9001: original 0.5s Atom poses, unchanged.
Port 9002: initial/on-add one-shot per-Atom collider definitions, 12 floats/geom.
Only shape packets create MJCF assets; an unknown shape received after model build
is saved for loading at the next Newton restart.
"""
import os
import re
import socket
import struct
import time
import select

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import engines.engine as engine


class EnvironmentBoxesMixin:
    # Unity world (LH) -> Newton world (RH); same position basis as VR tracking.
    _envbox_axis = np.array([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.]])
    _envbox_filename = re.compile(r'^atom_(\d+)\.xml$')

    def _envbox_initialize(self, config):
        self._envbox_enabled = bool(config.get('envbox_udp_enabled', False))
        self._envbox_socket = None
        self._envbox_shape_socket = None
        self._envbox_latest = {}  # Atom ID -> (world center, world full size, world quaternion)
        self._envbox_shapes = {}  # Atom ID -> validated list of collider records
        self._envbox_pending_pose_ids = set()
        self._envbox_assets = {}  # Atom ID -> persisted MJCF path
        self._envbox_objects = {}  # Atom ID -> Newton obj_id (only populated for env 0)
        self._envbox_active = set()
        self._envbox_last_warning = 0.0
        self._envbox_height_scale = float(config.get('envbox_height_scale', 0.8))
        self._envbox_position_offset = np.asarray(
            config.get('vrhand_position_offset', [0.0, 0.15, 0.0]), dtype=np.float64)
        self._envbox_asset_dir = os.path.abspath(config.get(
            'envbox_asset_dir', os.path.join('data', 'assets', 'objects', 'vam_env_boxes')))
        # Far enough from the active scene, and spread out to prevent self-collision.
        self._envbox_parking_height = float(config.get('envbox_parking_height', 1000.0))
        if not self._envbox_enabled:
            return
        if not np.isfinite(self._envbox_height_scale) or self._envbox_height_scale <= 0:
            raise ValueError('envbox_height_scale must be finite and positive')
        if not np.isfinite(self._envbox_position_offset).all() or self._envbox_position_offset.shape != (3,):
            raise ValueError('vrhand_position_offset must have 3 finite elements')
        if not np.isfinite(self._envbox_parking_height):
            raise ValueError('envbox_parking_height must be finite')
        os.makedirs(self._envbox_asset_dir, exist_ok=True)
        self._envbox_ip = str(config.get('envbox_udp_ip', '0.0.0.0'))
        self._envbox_port = int(config.get('envbox_udp_port', 9001))
        self._envbox_shape_port = int(config.get('envbox_shape_udp_port', 9002))
        timeout = max(0.0, float(config.get('envbox_startup_timeout', 5.0)))
        self._envbox_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._envbox_shape_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for sock, port in ((self._envbox_socket, self._envbox_port),
                               (self._envbox_shape_socket, self._envbox_shape_port)):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self._envbox_ip, port))
                sock.setblocking(False)
            # Both sockets MUST be bound before waiting, to avoid losing the one-shot
            # shape packet when the first pose packet arrives later.
            self._envbox_load_assets()
            if timeout:
                print(f'[EnvBox] Awaiting VaM shapes UDP:{self._envbox_shape_port} '
                      f'and poses UDP:{self._envbox_port} for up to {timeout:g}s', flush=True)
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    ready, _, _ = select.select(
                        [self._envbox_socket, self._envbox_shape_socket], [], [],
                        max(0.0, deadline - time.monotonic()))
                    for sock in ready:
                        self._envbox_read_socket(sock)
                    # If the geometry already arrived, allow time for its paired pose;
                    # once both are present, proceed to model construction.
                    if self._envbox_shapes and set(self._envbox_latest).intersection(self._envbox_shapes):
                        break
            self._envbox_read_socket(self._envbox_shape_socket)
            self._envbox_read_socket(self._envbox_socket)
            print(f'[EnvBox] Preloaded {len(self._envbox_assets)} MJCF assets; '
                  f'received shapes={len(self._envbox_shapes)}; '
                  f'initial active pose IDs={len(self._envbox_latest)}', flush=True)
            if not self._envbox_assets:
                print('[EnvBox] No compound collider assets available. '
                      'Start Newton receiver BEFORE enabling/reloading the VaM C# plugin '
                      'so its one-shot UDP:9002 shape packet is received.', flush=True)
        except Exception:
            self._envbox_socket.close()
            self._envbox_shape_socket.close()
            self._envbox_socket = None
            self._envbox_shape_socket = None
            raise

    @staticmethod
    def _envbox_decode_shape(data):
        # 4 header floats: magic, version, Atom ID, collider count
        # 12 floats per record: index, kind, relative offset xyz, full size xyz,
        #                     relative quaternion xyzw.
        if len(data) < 16 or len(data) % 4:
            raise ValueError('Invalid shape packet size')
        magic, version, id_float, count_float = struct.unpack_from('<4f', data, 0)
        if magic != 424242.0 or version != 1.0:
            raise ValueError('Unknown shape magic/version')
        if not np.isfinite(id_float) or id_float != int(id_float) or not 0 <= id_float <= 0xFFFFFF:
            raise ValueError('Invalid shape Atom ID')
        if not np.isfinite(count_float) or count_float != int(count_float) or not 1 <= count_float <= 1000:
            raise ValueError('Invalid shape collider count')
        count = int(count_float)
        if len(data) != 4 * (4 + 12 * count):
            raise ValueError(f'Shape packet length mismatch: count={count}, bytes={len(data)}')
        shapes = []
        used_indices = set()
        for n in range(count):
            rec = np.asarray(struct.unpack_from('<12f', data, 16 + n * 48), dtype=np.float64)
            if not np.isfinite(rec).all():
                raise ValueError(f'Nonfinite shape record {n}')
            idx_f, kind_f = rec[:2]
            if idx_f != int(idx_f) or int(idx_f) < 0 or int(idx_f) in used_indices:
                raise ValueError(f'Invalid/duplicate collider index {idx_f}')
            if kind_f != int(kind_f) or int(kind_f) not in (1, 2, 3, 4):
                raise ValueError(f'Unknown collider type {kind_f}')
            if np.any(rec[5:8] <= 0) or np.linalg.norm(rec[8:12]) < 1e-8:
                raise ValueError(f'Invalid collider size/rotation at index {idx_f}')
            used_indices.add(int(idx_f))
            shapes.append((int(idx_f), int(kind_f), rec[2:5], rec[5:8],
                           rec[8:12] / np.linalg.norm(rec[8:12])))
        return int(id_float), shapes

    def _envbox_read_socket(self, sock):
        if sock is None:
            return
        while True:
            try:
                packet, _ = sock.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                if sock is self._envbox_shape_socket:
                    atom_id, shapes = self._envbox_decode_shape(packet)
                    self._envbox_shapes[atom_id] = shapes
                    if atom_id not in self._envbox_assets:
                        self._envbox_register_shape_asset(atom_id, shapes)
                        # Dynamic MJCF insertion after model finalization is not supported.
                        if self._envbox_objects:
                            self._envbox_warn(f'ID={atom_id} compound asset saved; '
                                              'restart Newton to load its colliders')
                else:
                    self._envbox_latest = self._envbox_decode(packet)
                    self._envbox_pending_pose_ids.update(
                        set(self._envbox_latest).difference(self._envbox_assets))
            except (ValueError, OverflowError) as exc:
                self._envbox_warn(f'Ignoring invalid UDP:{self._envbox_shape_port if sock is self._envbox_shape_socket else self._envbox_port} packet: {exc}')

    @staticmethod
    def _envbox_decode(data):
        if len(data) < 4 or len(data) % 4:
            raise ValueError('Invalid UDP payload size')
        (count_float,) = struct.unpack_from('<f', data, 0)
        if not np.isfinite(count_float) or count_float < 0 or count_float != int(count_float):
            raise ValueError('Invalid object count')
        count = int(count_float)
        if len(data) != 4 * (1 + count * 11):
            raise ValueError(f'Packet length mismatch: count={count}, bytes={len(data)}')
        result = {}
        for index in range(count):
            values = np.asarray(struct.unpack_from('<11f', data, 4 + index * 44), dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f'Entry {index} contains a nonfinite value')
            id_float = values[0]
            if id_float != int(id_float) or not 0 <= id_float <= 0xFFFFFF:
                raise ValueError(f'Entry {index} has invalid 24-bit Atom ID')
            atom_id = int(id_float)
            if atom_id in result:
                raise ValueError(f'Duplicate Atom ID {atom_id}; possible FNV hash collision')
            center, size, quat = values[1:4], values[4:7], values[7:11]
            if np.any(size <= 0) or np.linalg.norm(quat) < 1e-8:
                raise ValueError(f'Entry {index} has invalid size or quaternion')
            result[atom_id] = (center, size, quat / np.linalg.norm(quat))
        return result

    def _envbox_load_assets(self):
        for filename in sorted(os.listdir(self._envbox_asset_dir)):
            match = self._envbox_filename.fullmatch(filename)
            if match is None:
                continue
            atom_id = int(match.group(1))
            if atom_id > 0xFFFFFF:
                self._envbox_warn(f'Ignoring out-of-range asset: {filename}')
                continue
            if atom_id in self._envbox_assets:
                raise ValueError(f'Duplicate MJCF asset ID {atom_id} in {self._envbox_asset_dir}')
            self._envbox_assets[atom_id] = os.path.join(self._envbox_asset_dir, filename)

    def _envbox_register_shape_asset(self, atom_id, shapes):
        """Build one rigid body with multiple box geoms, using C# local shapes.

        MeshCollider records are their individual mesh-bounds OBBs, not triangle meshes.
        The C# offset is relative to the *9001 combined bounds center*, so unchanged
        original root pose sync positions this compound correctly.
        """
        if atom_id in self._envbox_assets:
            return False
        path = os.path.join(self._envbox_asset_dir, f'atom_{atom_id}.xml')
        C = self._envbox_axis
        lines = [f'<mujoco model="vam_atom_{atom_id}">',
                 '  <worldbody>',
                 f'    <body name="vam_atom_{atom_id}" pos="10 10 10">',
                 '      <freejoint name="root"/>']
        # Mass is distributed over the separate geoms (prior single geom: 10000).
        # The engine's per-geom inertia calculation stays valid.
        mass_per_geom = 10000.0 / len(shapes)
        for idx, kind, offset, full_size, quat in sorted(shapes, key=lambda r: r[0]):
            # Same axis mapping as the existing single-box generation.
            local_pos = C @ offset
            local_pos[2] *= self._envbox_height_scale
            half = np.array([full_size[2], full_size[0], full_size[1]], dtype=np.float64) * 0.5
            half[2] *= self._envbox_height_scale
            # Rotation is relative to the 9001 root; change basis, do not touch
            # runtime root rotation implementation.
            mat = C @ Rotation.from_quat(quat).as_matrix() @ C.T
            local_quat = Rotation.from_matrix(mat).as_quat()  # xyzw
            # The existing XML is box-based. Every individual Collider is currently
            # represented by a separate oriented box (including sphere/capsule).
            # MuJoCo/MJCF quat attributes are wxyz, not Unity/Python xyzw.
            pos_s = ' '.join(f'{x:.7f}' for x in local_pos)
            size_s = ' '.join(f'{x:.7f}' for x in half)
            quat_s = ' '.join(f'{x:.8f}' for x in
                              [local_quat[3], local_quat[0], local_quat[1], local_quat[2]])
            lines.append(f'      <geom name="collider_{idx}" type="box" '
                         f'pos="{pos_s}" quat="{quat_s}" size="{size_s}" '
                         f'mass="{mass_per_geom:.7f}" rgba="0.6 0.7 0.3 0.5" '
                         'contype="1" conaffinity="1"/>')
        lines.extend(['    </body>', '  </worldbody>', '</mujoco>', ''])
        try:
            with open(path, 'x', encoding='utf-8') as file:
                file.write('\n'.join(lines))
            print(f'[EnvBox] Created compound MJCF ID={atom_id} '
                  f'geoms={len(shapes)} -> {path}', flush=True)
        except FileExistsError:
            print(f'[EnvBox] Existing MJCF preserved ID={atom_id} -> {path}', flush=True)
        self._envbox_assets[atom_id] = path
        return True

    def _envbox_build_objects(self, env_id):
        if not self._envbox_enabled:
            return
        ids = {}
        # Keep creation order deterministic for all environments.
        for atom_id, asset in sorted(self._envbox_assets.items()):
            obj_id = self._engine.create_obj(
                env_id=env_id, obj_type=engine.ObjType.rigid,
                asset_file=asset, name=f'vam_atom_{atom_id}',
                is_visual=False, fix_root=False, color=[0.6, 0.7, 0.3])
            ids[atom_id] = obj_id
        if env_id == 0:
            self._envbox_objects = ids
        elif ids != self._envbox_objects:
            raise RuntimeError('EnvBox object mapping differs across Newton environments')

    def _envbox_park_position(self, atom_id):
        # Isolated parking coordinates; the same ID always gets the same spot.
        return np.array([float(atom_id % 256) * 20.,
                         float(atom_id // 256 % 256) * 20.,
                         self._envbox_parking_height], dtype=np.float32)

    def _envbox_park_all(self):
        """Call after Newton tensors/model initialization, before stepping."""
        if not self._envbox_enabled or not self._envbox_objects:
            return
        env_ids = torch.arange(self.get_num_envs(), device=self._device, dtype=torch.long)
        for atom_id, obj_id in self._envbox_objects.items():
            self._envbox_set_pose(env_ids, obj_id, self._envbox_park_position(atom_id),
                                  np.array([0., 0., 0., 1.], dtype=np.float32))
        self._envbox_active.clear()

    def _envbox_set_pose(self, env_ids, obj_id, pos, quat):
        n = len(env_ids)
        p = torch.as_tensor(pos, device=self._device, dtype=torch.float32).reshape(1, 3).expand(n, -1)
        q = torch.as_tensor(quat, device=self._device, dtype=torch.float32)
        self._engine.set_root_pos(env_ids, obj_id, p)
        self._engine.set_root_rot(env_ids, obj_id, q)
        self._engine.set_root_vel(env_ids, obj_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, obj_id, 0.0)

    def _envbox_sync(self):
        if not self._envbox_enabled or self._envbox_socket is None:
            return
        # UDP:9002 geometry arrives only on plugin start / new Atom. Receive it
        # independently of the periodic UDP:9001 pose updates.
        self._envbox_read_socket(self._envbox_shape_socket)
        self._envbox_read_socket(self._envbox_socket)
        if self._envbox_pending_pose_ids:
            missing = self._envbox_pending_pose_ids.difference(self._envbox_assets)
            if missing:
                self._envbox_warn(f'No shape for Atom IDs {sorted(missing)}; '
                                  'awaiting UDP:9002, NOT creating a giant fallback box')
            self._envbox_pending_pose_ids = missing
        if not self._envbox_objects:
            return
        # Keep inactive dynamic objects parked every frame. Otherwise gravity
        # eventually brings the freejoint bodies back into the active scene.
        # Other environments are also kept parked: only env 0 receives VaM poses.
        all_env_ids = torch.arange(self.get_num_envs(), device=self._device, dtype=torch.long)
        env_ids = torch.tensor([0], device=self._device, dtype=torch.long)
        other_env_ids = all_env_ids[1:]
        available = set(self._envbox_latest).intersection(self._envbox_objects)
        for atom_id, obj_id in self._envbox_objects.items():
            parked = self._envbox_park_position(atom_id)
            if atom_id not in available:
                self._envbox_set_pose(all_env_ids, obj_id, parked, [0., 0., 0., 1.])
                if atom_id in self._envbox_active:
                    print(f'[EnvBox] Deactivated ID={atom_id}', flush=True)
                    self._envbox_active.remove(atom_id)
            elif len(other_env_ids):
                self._envbox_set_pose(other_env_ids, obj_id, parked, [0., 0., 0., 1.])
        C = self._envbox_axis
        for atom_id in sorted(available):
            pos, _size_ignored, quat = self._envbox_latest[atom_id]
            p = C @ (pos - self._envbox_position_offset)
            p[2] *= self._envbox_height_scale
            p[0] -= 0.00
            p[1] += 0.00
            p[2] += 0.22
            rotation = C @ Rotation.from_quat(quat).as_matrix() @ C.T
            q = Rotation.from_matrix(rotation).as_quat()
            self._envbox_set_pose(env_ids, self._envbox_objects[atom_id], p, q)
            if atom_id not in self._envbox_active:
                print(f'[EnvBox] Activated ID={atom_id}', flush=True)
                self._envbox_active.add(atom_id)

    def _envbox_warn(self, message):
        now = time.monotonic()
        if now - self._envbox_last_warning >= 2.0:
            self._envbox_last_warning = now
            print('[EnvBox] ' + message, flush=True)
