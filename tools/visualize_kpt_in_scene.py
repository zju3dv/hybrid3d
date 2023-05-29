import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
import os

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


def transform_points(transformation_file, points):
    trans_info = util.read_json(transformation_file)
    Trans = np.transpose(np.asarray(trans_info['transformation']).reshape(4, 4))
    scale, R, t = decompose_to_sRT(Trans)
    points = np.matmul(np.transpose(R), np.transpose((points - t) / scale))
    points = np.transpose(points)
    return points


if __name__ == '__main__':
    scene = sys.argv[1]
    # kpt_file = sys.argv[2]
    pcd_orig_path = 'data/redwood_lidar/lidar_resampled/aligned_low_%s/%s_no_ceiling.ply' % (scene, scene)
    pcd_orig_o3d = o3d.io.read_point_cloud(pcd_orig_path, print_progress=True)
    pcd_orig_o3d = pcd_orig_o3d.voxel_down_sample(voxel_size=0.08)
    pcd_orig_np = np.asarray(pcd_orig_o3d.points)

    pcd_orig_np = transform_points('data/redwood_lidar/ours_registration/%s_alignment.json' % (scene), pcd_orig_np)

    # keypoints = np.asarray(util.read_json('data/redwood_lidar/initial_keypoints/usip/apartment.json'))
    # keypoints = clustering.fps_clustering(keypoints[:, :3], 0.1, 0.1, 1000)

    kpt_raw = utils_algo.load_keypints_from_file(f'{Path.home()}/Downloads/log_hybrid_feature/log_hf_0622_131845_redwood_full_subset_0.1_self_bootstrap_cluster_subset_0.3/kpts/kpts-raw-047.json')[scene]
    kpt_raw = kpt_raw[np.where(kpt_raw[:,3] > 0.4), ...]
    kpt_raw = np.squeeze(kpt_raw)
    print('kpt_raw.shape after selection', kpt_raw.shape)
    kpt_raw = kpt_raw[:100000, ...]
    kpt_clustered = utils_algo.load_keypints_from_file(f'{Path.home()}/Downloads/log_hybrid_feature/log_hf_0617_231847_redwood_subset_0.3/kpts/kpts-028.json')[scene]

    kpt_offset = utils_algo.load_keypints_from_file(f'{Path.home()}/Downloads/log_hybrid_feature/log_hf_0622_131845_redwood_full_subset_0.1_self_bootstrap_cluster_subset_0.3/kpts/kpts-offset-046.json')[scene]

    kpt_usip = np.asarray(o3d.io.read_point_cloud('data/redwood_lidar/initial_keypoints/usip/apartment_no_ceiling.ply').points)
    kpt_usip = transform_points('data/redwood_lidar/ours_registration/%s_alignment.json' % (scene), kpt_usip)
    # print(kpt_usip.shape)
    kpt_usip = clustering.fps_clustering(kpt_usip, 0.1, 0.1, 1000)
    # kpt_raw = clustering.fps_clustering(kpt_raw[:, :3], 0.1, 0.1, 1000)
    # kpt_raw = clustering.fps_weighted_clustering(kpt_raw[:, :3], kpt_raw[:, 3], 0.1, 0.05, 1000, 0.6)
    # print(kpt_raw.shape)
    offsets = kpt_offset[:, 3]
    offsets = np.sort(offsets)
    # print(offsets)
    # print(np.mean(offsets), np.max(offsets), np.median(offsets), np.percentile(offsets, 75))
    
    ax = vis_3d.plot_pc(pcd_orig_np, size=3)
    # show space heatmap
    if 0:
        ax = vis_3d.plot_pc(kpt_raw, ax=ax, size=1, color=kpt_raw[:, 3], cmap='Reds', vmin=0.4, vmax=1)
    # show offset
    if 1:
        ax = vis_3d.plot_pc(kpt_offset, ax=ax, size=30, color=kpt_offset[:, 3], cmap='coolwarm', alpha=1, vmax=0.4)
    # show kpt
    if 0:
        ax = vis_3d.plot_pc(kpt_clustered, ax=ax, size=30, color=np.asarray([1, 0, 0]).reshape((1, 3)))
        ax = vis_3d.plot_pc(kpt_usip, ax=ax, size=30, color=np.asarray([0, 0, 1]).reshape((1, 3)))
    plt.show()
