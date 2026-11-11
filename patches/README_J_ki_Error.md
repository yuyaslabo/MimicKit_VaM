### Troubleshooting: `J_ki` Error on Certain GPUs

Depending on your GPU architecture, you may encounter a `J_ki`-related Warp compilation error after installing the dependencies (e.g., it works fine on an RTX 4070 Ti, while an RTX 3090 required the following fix).

To resolve this, replace the patched file in your Conda environment using the file provided in the `patches/` folder, or manually edit `solver.py` located at `<your_conda_env_path>\Lib\site-packages\mujoco_warp\_src\solver.py` as shown below:

```python
      # AD: leaving bounds-check disabled here because I'm not entirely sure that
      # everything always hits the fast path. The padding takes care of any
      # potential OOB accesses.
      J_ki = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(k, offset_i), bounds_check=False)
      J_kj = J_ki
      if offset_i != offset_j:
        J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(k, offset_j), bounds_check=False)
      # --- End of modification ---
      # if offset_i != offset_j:
        # J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(k, offset_j), bounds_check=False)
      # else:
        # #wp.tile_assign(J_kj, J_ki, (0, 0))
        # J_kj[:] = J_ki[:]
```
