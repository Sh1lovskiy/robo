import open3d as o3d


pcd = o3d.io.read_point_cloud("/home/sha/Documents/work/robohand_v2/colmap/data1/dense1/fused.ply")
o3d.visualization.draw_geometries([pcd])
