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
from utils.vis_3d_o3d import map_to_color, O3dVisualizer

def decompose_to_sRT(Trans):
    t = Trans[:3, 3]
    R = Trans[:3, :3]
    # assume x y z have the same scale
    scale = np.linalg.norm(R[:3, 0])
    R = R / scale
    return scale, R, t

def transform_points_redwood(transformation_file, points):
    trans_info = util.read_json(transformation_file)
    Trans = np.transpose(np.asarray(trans_info['transformation']).reshape(4, 4))
    scale, R, t = decompose_to_sRT(Trans)
    points = np.matmul(np.transpose(R), np.transpose((points - t) / scale))
    points = np.transpose(points)
    return points

def read_fragment_pose(filepath):
    with open(filepath, 'r') as f:
        metastr = f.readline()
        mat = np.empty((4, 4))
        for i in range(4):
            matstr = f.readline()
            mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
        return mat
    assert(False)

def transform_points_3dmatch_fragment(transformation_file, points):
    mat = np.linalg.inv(read_fragment_pose(transformation_file))
    R = mat[:3, :3]
    t = mat[:3, 3]
    points = np.matmul(np.transpose(R), np.transpose(points - t))
    points = np.transpose(points)
    return points

def transform_points(points, mat):
    R = mat[:3, :3]
    t = mat[:3, 3]
    points = np.matmul(np.transpose(R), np.transpose(points - t))
    points = np.transpose(points)
    return points

if __name__ == '__main__':
    scene = sys.argv[1]
    # kpt_file = sys.argv[2]
    fragment = False
    pcd_orig_path = 'data/3dmatch/reconstruction/%s/mesh.ply' % (scene)
    if not os.path.isfile(pcd_orig_path):
        pcd_orig_path = 'data/redwood_lidar/lidar_resampled/aligned_low_%s/%s.ply' % (scene, scene)
    if not os.path.isfile(pcd_orig_path):
        fragment = True
        area, fragment_idx = scene.rsplit('_', 1)
        pcd_orig_path = 'data/3dmatch/fragments_adaptive/%s/cloud_bin_%s.ply' % (area, fragment_idx)
    pcd_orig_o3d = o3d.io.read_point_cloud(pcd_orig_path, print_progress=True)
    pcd_orig_o3d = pcd_orig_o3d.voxel_down_sample(voxel_size=0.01)
    pcd_orig_np = np.asarray(pcd_orig_o3d.points)

    if 'redwood' in pcd_orig_path:
        pcd_orig_np = transform_points_redwood('data/redwood_lidar/ours_registration/%s_alignment.json' % (scene), pcd_orig_np)
        pcd_orig_o3d.points = o3d.utility.Vector3dVector(pcd_orig_np)
    if fragment:
        # fragment_pose_file = 'data/3dmatch/fragments_adaptive/%s/cloud_bin_%s.info.txt' % (area, fragment_idx)
        base_frag_pose_file = 'data/3dmatch/fragments_adaptive/%s/cloud_bin_%s.info.txt' % (area, 55)
        curr_frag_pose_file = 'data/3dmatch/fragments_adaptive/%s/cloud_bin_%s.info.txt' % (area, fragment_idx)
        base_Twc = read_fragment_pose(base_frag_pose_file)
        curr_Twc = read_fragment_pose(curr_frag_pose_file)
        trans = np.matmul(np.linalg.inv(base_Twc), curr_Twc)
        trans = np.linalg.inv(trans)
        pcd_orig_np = transform_points(pcd_orig_np, trans)
        # pcd_orig_np = transform_points_3dmatch_fragment(fragment_pose_file, pcd_orig_np)
        pcd_base_np = np.asarray(o3d.io.read_point_cloud('data/3dmatch/fragments_adaptive/%s/cloud_bin_55.ply' % (area)).voxel_down_sample(voxel_size=0.01).points)
        pcd_orig_np = np.concatenate([pcd_orig_np, pcd_base_np], axis=0)
        pcd_orig_o3d.points = o3d.utility.Vector3dVector(pcd_orig_np)

    # keypoints = np.asarray(util.read_json('data/redwood_lidar/initial_keypoints/usip/apartment.json'))
    # keypoints = clustering.fps_clustering(keypoints[:, :3], 0.1, 0.1, 1000)
    base_dir = '/home/ybbbbt/Downloads/log_hybrid_feature/kpt_tmp/'
    idx = 1
    scene = 'sun3d-mit_32_d507-d507_2_55_sun3d-mit_32_d507-d507_2_57'
    kpt_raw = utils_algo.load_keypints_from_file('{}/kpts/kpts-raw-{:03d}.json'.format(base_dir, idx))[scene]
    # kpt_raw = kpt_raw[np.where(kpt_raw[:,3] > 0.4), ...]
    if 0:
        # test voxelize
        pcd = o3d.geometry.PointCloud()
        kpts = kpt_raw[:, :3]
        pcd.points = o3d.utility.Vector3dVector(kpts)
        pcd, cubic_id, indices = pcd.voxel_down_sample_and_trace(0.01, np.min(kpts, axis=0), np.max(kpts, axis=0))
        indices = [x[0] for x in indices]
        indices = np.asarray(indices)
        kpt_raw = kpt_raw[indices, ...]
        # weights = weights[indices, ...]

    kpt_raw = np.squeeze(kpt_raw)[:,:4]
    print(kpt_raw)
    print('kpt_raw.shape after selection', kpt_raw.shape)
    # kpt_raw = kpt_raw[:100000, ...]
    kpt_clustered = utils_algo.load_keypints_from_file('{}/kpts/kpts-{:03d}.json'.format(base_dir, idx))[scene]

    kpt_offset = utils_algo.load_keypints_from_file('{}/kpts/kpts-offset-{:03d}.json'.format(base_dir, idx))[scene]

    # kpt_clustered = clustering.weighted_clustering(kpt_raw[:, :3], kpt_raw[:, 3], 0.1, 0.1, 500, 0, 'nms', 10)

    # kpt_usip = np.asarray(o3d.io.read_point_cloud('data/redwood_lidar/initial_keypoints/usip/apartment_no_ceiling.ply').points)
    # kpt_usip = transform_points_redwood('data/redwood_lidar/ours_registration/%s_alignment.json' % (scene), kpt_usip)
    # print(kpt_usip.shape)
    # kpt_usip = clustering.fps_clustering(kpt_usip, 0.1, 0.1, 1000)
    # kpt_raw = clustering.fps_clustering(kpt_raw[:, :3], 0.1, 0.1, 1000)
    # kpt_raw = clustering.fps_weighted_clustering(kpt_raw[:, :3], kpt_raw[:, 3], 0.1, 0.05, 1000, 0.6)
    # kpt_clustered = clustering.fps_clustering(kpt_raw[:, :3], 0.1, 0.1, 1200)
    # kpt_clustered = clustering.weighted_clustering(kpt_raw[:,:3], kpt_raw[:,3], 0.1, 0.1, 1200, 0.2, 'nms', 10)

    # kpt_clustered = clustering.weighted_clustering(kpt_raw[:,:3], kpt_raw[:,3], 0.1, 0.1, 1200, 0.2, 'nms', 10)

    # kpt_clustered, distinctive_score, _ = clustering.weighted_clustering_distinctive(kpt_raw[:,:3], kpt_raw[:,3], 0.15, 0.15, 1200, 0.2, 'nms', 10)
    # kpt_clustered = clustering.K_means_clustering(kpt_raw, 1000, 10)

    # print('kpt_clustered.shape', kpt_clustered.shape)
    # print(kpt_raw.shape)
    # kpt_offset = kpt_offset[np.where(kpt_offset[:,3] < 0.25), ...]
    # kpt_offset = np.squeeze(kpt_offset)
    offsets = kpt_offset[:, 3]
    offsets = np.sort(offsets)
    print(offsets)
    print(np.mean(offsets), np.max(offsets), np.median(offsets), np.percentile(offsets, 75))
    
    visualizer = O3dVisualizer()
    visualizer.add_o3d_geometry(pcd_orig_o3d)

    # show space heatmap
    if 1:
        visualizer.add_np_points(kpt_raw, size=0.005, color=map_to_color(kpt_raw[:, 3], cmap='Reds', vmin=0.2, vmax=1.0))
    # show offset
    if 0:
        visualizer.add_np_points(kpt_offset, size=0.05, color=map_to_color(kpt_offset[:, 3], vmin=0.0, vmax=0.4))
    # print(np.mean(distinctive_score), np.max(distinctive_score), np.min(distinctive_score))
    # visualizer.add_np_points(kpt_clustered, size=0.05, color=map_to_color(distinctive_score, vmin=1, vmax=5))
    # show kpt
    if 1:
        visualizer.add_np_points(kpt_clustered, size=0.02, color=np.asarray([0, 0.8, 0]))

    visualizer.run_visualize()
