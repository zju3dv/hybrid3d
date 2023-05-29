import open3d as o3d
import numpy as np
import utils.sampler as sampler

fps_sampler = sampler.FarthestSampler()
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def load_pointcloud(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd

def mesh_voxelize_to_points(mesh_path, voxel_size):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=20000000)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)

def fps_sample_points(points, fps_num):
    return fps_sampler.sample(points, fps_num)

def save_pointcloud(pcd_np, pcd_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    o3d.io.write_point_cloud(pcd_path, pcd)
