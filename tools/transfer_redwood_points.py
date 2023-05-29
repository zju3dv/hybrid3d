import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())  # noqa
import utils.vis_3d as vis_3d
import utils.utils_algo as utils_algo
import utils.util as util
import utils.clustering as clustering

def colorize_open3d_pcd(pcd):
    # use z axis to map color
    pts = np.asarray(pcd.points)
    pts_z_min = np.min(pts[:, 2])
    pts_z_max = np.max(pts[:, 2])
    colors = plt.cm.viridis(
        (pts[:, 2] - pts_z_min) / (pts_z_max - pts_z_min))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def decompose_to_sRT(Trans):
    t = Trans[:3, 3]
    R = Trans[:3, :3]
    # assume x y z have the same scale
    scale = np.linalg.norm(R[:3, 0])
    R = R / scale
    print(scale, R, t)
    return scale, R, t


def transform_points(transformation_file, points, inverse=False):
    trans_info = util.read_json(transformation_file)
    Trans = np.transpose(np.asarray(trans_info['transformation']).reshape(4, 4))
    scale, R, t = decompose_to_sRT(Trans)
    if inverse:
        # X_old = R * (X_new * s) + t
        points = np.transpose(np.matmul(R, np.transpose(points * scale))) + t
    else:
        # X_new = R^T * ((x_old - t)/s)
        points = np.matmul(np.transpose(R), np.transpose((points - t) / scale))
        points = np.transpose(points)
    return points


if __name__ == '__main__':
    # scene = sys.argv[1]
    # kpt_file = sys.argv[2]
    # pcd_orig_path = 'data/redwood_lidar/lidar_resampled/aligned_low_%s/%s_no_ceiling.ply' % (scene, scene)
    # pcd_orig_o3d = o3d.io.read_point_cloud(pcd_orig_path, print_progress=True)
    # pcd_orig_o3d = pcd_orig_o3d.voxel_down_sample(voxel_size=0.08)
    # pcd_orig_np = np.asarray(pcd_orig_o3d.points)
    scene = 'apartment'
    xyz = utils_algo.load_keypints_from_file(f'{Path.home()}/Downloads/log_hybrid_feature/log_hf_2d_usip_cluster_0611_113040/kpts-076.json')[scene]
    xyz = transform_points('data/redwood_lidar/ours_registration/%s_alignment.json' % (scene), xyz, True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud('kpt-{}.ply'.format(scene), pcd)
    

