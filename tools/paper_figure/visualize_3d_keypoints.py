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
from utils.vis_3d_o3d import map_to_color, O3dVisualizer
from utils.sampler import nms_3D
import torch

if __name__ == '__main__':
    # 3dmatch
    log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/0_paper_selected_3dmatch/3d_final/log_hf_1105_185250_H3DNetCoordVote_X3DMatchFragmentDataLoader_pos_neg_256_256_601_subset_round_2_epoch99_36dccd'
    # log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/0_paper_selected_3dmatch/kp2d/log_hf_1107_225538_H3DNetCoordVote_X3DMatchFragmentDataLoader_kp2d'
    # log_dir = '/home/ybbbbt/Developer/D3Feat/geometric_registration/3dmatch_final_eval/D3Feat_contralo-54-pred'
    # pcd_data_base_dir = '/home/ybbbbt/Data-2T/redwood_lidar_3dmatch_format/fragments_adaptive'
    pcd_data_base_dir = 'data/3dmatch/fragments_color'

    scene = 'sun3d-hotel_uc-scan3'
    fragment_idx = 105

    base_pt = o3d.io.read_point_cloud('data/3dmatch/fragments_color/{}/cloud_bin_{}.ply'.format(scene, fragment_idx))

    # # redwood
    # # log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/log_hf_1108_001836_H3DNetCoordVote_RedwoodLidarFragDataLoader_pos_neg_256_256_601_subset_round_2_epoch99_redwood_8f20f0'
    # # log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/1_paper_select_redwood/kp2d/log_hf_1107_225913_H3DNetCoordVote_RedwoodLidarFragDataLoader_kp2d_redwood'
    # log_dir = '/home/ybbbbt/Developer/D3Feat/geometric_registration/D3Feat_contralo-54-pred'
    # pcd_data_base_dir = '/mnt/ssd_disk/redwood_lidar_3dmatch_format/fragments_color'

    # scene = 'apartment'
    # fragment_idx = 240

    # base_pt = o3d.io.read_point_cloud('data/redwood_lidar/fragments_color/{}/cloud_bin_{}.ply'.format(scene, fragment_idx))
    
    reference_pc_keypoints = np.load('{}/keypoints/{}/cloud_bin_{}.npy'.format(log_dir, scene, fragment_idx))

    src_scores = np.load('{}/scores/{}/cloud_bin_{}.npy'.format(log_dir, scene, fragment_idx))

    is_d3feat = 'D3Feat' in log_dir

    if is_d3feat:
        src_scores = np.squeeze(src_scores)
        reference_pc_keypoints = reference_pc_keypoints.astype(np.float64)

    new_order = np.argsort(-src_scores)
    reference_pc_keypoints = reference_pc_keypoints[new_order]
    src_scores = src_scores[new_order]

    max_kpt_num = 500
    max_kpt_num = min(reference_pc_keypoints.shape[0], max_kpt_num)
    reference_pc_keypoints = reference_pc_keypoints[:max_kpt_num]
    src_scores = src_scores[:max_kpt_num]

    visualizer = O3dVisualizer()

    visualizer.add_o3d_geometry(base_pt)

    # base_pt = np.asarray(base_pt.points)
    # visualizer.add_np_points(base_pt, color=map_to_color(base_pt[:, 2], cmap='Spectral'), size=0.004, resolution=2, with_normal=True)
    

    # color = map_to_color(src_scores, cmap='Reds', vmin=0, vmax=1)
    # color = map_to_color(src_scores, cmap='Reds')

    color = np.array([255/255.0, 69/255.0, 0/255.0])

    visualizer.add_np_points(reference_pc_keypoints, color=color, size=0.015, resolution=20, with_normal=True)
    
    visualizer.run_visualize()
