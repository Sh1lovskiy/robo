# uv pip install git+https://github.com/Tencent/Hunyuan3D-2.git
import open3d as o3d
import numpy as np
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import torch


pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")

mesh = pipeline(image="/home/sha/Downloads/images_jpg/IMG_2018.jpg")[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
mesh = pipeline(mesh, image="/home/sha/Downloads/images_jpg/IMG_2038.jpg")

temp_obj_path = "/tmp/temp_mesh.obj"
mesh.export(temp_obj_path)

o3d_mesh = o3d.io.read_triangle_mesh(temp_obj_path)

o3d_mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([o3d_mesh])
