# Voxel connectivity runtime

The eight algorithm modules in this directory come from `room_instance_seg/backend/` at commit
`2b9eb0b60588f214dcad525d02e60967c53771e3` and implement `voxel-connectivity-v8`.
The Boxer adaptation in `supervoxel_anchor.py` imports the shared `room_color` function from
`utils.potential_field_runtime.pointcloud`. The other algorithm modules are copied verbatim.

The array entry point is `voxel_connectivity.run_voxel_connectivity`. It takes XYZ coordinates
in metres with +Z up, and returns point labels in input order and the native XY map artifacts.
