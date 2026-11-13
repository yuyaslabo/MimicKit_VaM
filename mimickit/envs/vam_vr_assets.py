# Original code authored by Xue Bin Peng (MimicKit)
# Modified by [LocalAIVRLab] in 2026 for integration with Virt-A-Mate (VaM).
# Licensed under the Apache License, Version 2.0.

import os


class VRAssetsMixin:
    def _write_vrhand_asset(self):
        os.makedirs(os.path.dirname(self._vrhand_asset), exist_ok=True)
        xml = """<mujoco model="vrhand_collision_box">
  <worldbody>
    <body name="vrhand_collision_box" pos="10 10 10">
      <freejoint name="root"/>                        
      <geom name="vrhand_box_geom" type="sphere" size="0.008"
            rgba="0.2 0.65 1.0 0.65"
            mass="70.0"
            solref="0.15 1.6" solimp="0.5 0.99 0.01 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
"""
        with open(self._vrhand_asset, "w", encoding="utf-8") as f:
            f.write(xml)

    def _write_vrhand_left_pd_asset(self):
        os.makedirs(os.path.dirname(self._vrhand_left_pd_asset), exist_ok=True)
        xml = """<mujoco model="vrhand_left_pd_collision_box">
  <worldbody>
    <body name="vrhand_left_pd_collision_box" pos="10 10 10">
      <freejoint name="root"/>
      <geom name="vrhand_left_pd_geom" type="sphere" size="0.03"
            rgba="0.2 0.65 1.0 0.65" mass="1.0"
            solref="0.15 1.6" solimp="0.5 0.99 0.01 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
"""
        with open(self._vrhand_left_pd_asset, "w", encoding="utf-8") as f:
            f.write(xml)

    def _write_vrhand_right_pd_asset(self):
        os.makedirs(os.path.dirname(self._vrhand_right_pd_asset), exist_ok=True)
        xml = """<mujoco model="vrhand_right_pd_collision_box">
  <worldbody>
    <body name="vrhand_right_pd_collision_box" pos="10 10 10">
      <freejoint name="root"/>
      <geom name="vrhand_right_pd_geom" type="sphere" size="0.03"
            rgba="1.0 0.4 0.2 0.65" mass="1.0"
            solref="0.15 1.6" solimp="0.5 0.99 0.01 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
"""
        with open(self._vrhand_right_pd_asset, "w", encoding="utf-8") as f:
            f.write(xml)

    def _write_vrhmd_asset(self):
        os.makedirs(os.path.dirname(self._vrhmd_asset), exist_ok=True)
        xml = """<mujoco model="vrhmd_collision_box">
  <worldbody>
    <body name="vrhmd_collision_box" pos="10 10 10">
      <freejoint name="root"/>                        
      <geom name="vrhmd_box_geom" type="sphere" size="0.068 " pos="0 0 0"
            rgba="0.3 0.8 0.8 0.5"
            mass="10.0"
            solref="0.15 1.6" solimp="0.5 0.99 0.01 0.5 2"/>
      
    </body>
  </worldbody>
</mujoco>
"""
        with open(self._vrhmd_asset, "w", encoding="utf-8") as f:
            f.write(xml)

    def _write_vrbody_assets(self):
        # Lightweight test: write only the torso MJCF asset.
        name = "torso"
        radius = self._vrbody_torso_radius
        length = self._vrbody_torso_length
        mass = 8.0
        asset_file = os.path.join("data", "assets", "objects", "vrbody_torso.xml")
        xml = f'''<mujoco model="vrbody_torso">
  <worldbody>
    <body name="vrbody_torso" pos="10 10 10">
      <freejoint name="root"/>
      <geom name="vrbody_torso_geom" type="capsule" size="{radius:.6f} {length * 0.5:.6f}"
            rgba="0.3 0.8 0.8 0.5" mass="{mass:.6f}"
            contype="1" conaffinity="2"
            solref="0.15 1.6" solimp="0.5 0.99 0.01 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
'''
        os.makedirs(os.path.dirname(asset_file), exist_ok=True)
        with open(asset_file, "w", encoding="utf-8") as f:
            f.write(xml)
        self._vrbody_asset_files[name] = asset_file
