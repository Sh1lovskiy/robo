# vision.cloud subpackage

Utilities for manipulating point clouds captured from the camera.

- `generator.py` – convert depth images to XYZ points using the pinhole camera model. For a pixel at $(u, v)$ with depth $d$ and intrinsics $(f_x, f_y, c_x, c_y)$ the 3‑D coordinates are 

$$
(d(u-c_x)/f_x, \quad d(v-c_y)/f_y, \quad d).
$$

- `aggregator.py` – merge multiple frames into a single cloud.  It can optionally run ICP to refine alignment between successive captures.
- `pipeline.py` – common filters (voxel down‑sample, pass‑through) and simple analysis helpers for trajectories.

The classes here are small and focus on a single task in accordance with the Single Responsibility Principle. They are orchestrated by higher level workflows in `vision/tools.py`.
