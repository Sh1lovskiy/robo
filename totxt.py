import open3d as o3d
import numpy as np

from farm import BBOX_POINTS, HAND_EYE_R, TARGET_POSE, HAND_EYE_t
from farm1 import filter_bbox, load_point_cloud
from save_rotate_clouds import transform_cloud_to_tcp

pcd_path = ".data_clouds/farm_20250730_143651.ply"
pcd = load_point_cloud(pcd_path)
pcd_down = pcd.voxel_down_sample(0.005)
pcd = transform_cloud_to_tcp(pcd_down, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
pcd_filtered = filter_bbox(pcd, BBOX_POINTS)
pcd_filtered.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
)
points = np.asarray(pcd_filtered.points)
normals = np.asarray(pcd_filtered.normals)

labels = np.zeros((points.shape[0], 1))
out = np.hstack([points, normals, labels])
np.savetxt("cloud_for_pc2beam.txt", out, fmt="%.6f")
