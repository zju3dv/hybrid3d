import copy
import numpy as np
import os
import subprocess
from open3d import *
import open3d
import open3d as o3d


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([50/255.0, 132/255.0, 186/255.0])
    # target_temp.paint_uniform_color([228/255.0, 86/255.0, 39/255.0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def execute_global_registration(
        source_down, target_down, reference_desc, target_desc, distance_threshold):
    result = open3d.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, reference_desc, target_desc,
    distance_threshold,
    open3d.registration.TransformationEstimationPointToPoint(False), 3,
    [open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        open3d.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
    open3d.registration.RANSACConvergenceCriteria(50000, 1000))
    return result

# # redwood
# # log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/log_hf_1108_001836_H3DNetCoordVote_RedwoodLidarFragDataLoader_pos_neg_256_256_601_subset_round_2_epoch99_redwood_8f20f0'
# # log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/1_paper_select_redwood/2d_rgb_only/log_hf_1108_112305_H3DCoordRGB_RedwoodLidarFragDataLoader_coord_rgb_only_epoch99_redwood_5e4a91'
# # log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/1_paper_select_redwood/kp2d/log_hf_1107_225913_H3DNetCoordVote_RedwoodLidarFragDataLoader_kp2d_redwood'
# # log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/1_paper_select_redwood/redwood_fcgf_0.02_rand_score/log_hf_1026_165515_H3DNetCoordVote_RedwoodLidarFragDataLoader_pos_neg_256_1024_601_subset_epoch51_redwood_fcgf_rand_score_6882d2'
# log_dir = '/home/ybbbbt/Developer/D3Feat/geometric_registration/D3Feat_contralo-54-pred'
# # log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/1_paper_select_redwood/perfectmatch/log_3dsmoothnet_redwood'
# # pcd_data_base_dir = '/home/ybbbbt/Data-2T/redwood_lidar_3dmatch_format/fragments_adaptive'
# pcd_data_base_dir = '/mnt/ssd_disk/redwood_lidar_3dmatch_format/fragments_color'

# scene = 'loft'
# src_id = 28
# tgt_id = 37

# 3dmatch
# log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/0_paper_selected_3dmatch/3d_final/log_hf_1105_185250_H3DNetCoordVote_X3DMatchFragmentDataLoader_pos_neg_256_256_601_subset_round_2_epoch99_36dccd'
# log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/0_paper_selected_3dmatch/kp2d/log_hf_1107_225538_H3DNetCoordVote_X3DMatchFragmentDataLoader_kp2d'
# log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/0_paper_selected_3dmatch/fcgf_0.02_rand_score/log_hf_1024_150751_H3DNetCoordVote_X3DMatchFragmentDataLoader_pos_neg_256_1024_symmetric_contras_epoch99_fcgf_torch_points_voxel_0.02_rand_score_1a3091'
log_dir = '/home/ybbbbt/Developer/D3Feat/geometric_registration/3dmatch_final_eval/D3Feat_contralo-54-pred'
# log_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/0_paper_selected_3dmatch/perfectmatch/log_3dsmoothnet_3dmatch'
# pcd_data_base_dir = '/home/ybbbbt/Data-2T/redwood_lidar_3dmatch_format/fragments_adaptive'
pcd_data_base_dir = 'data/3dmatch/fragments_color'


scene = 'sun3d-hotel_uc-scan3'
src_id = 105
tgt_id = 116


is_d3feat = 'D3Feat' in log_dir



reference_pc_keypoints = np.load('{}/keypoints/{}/cloud_bin_{}.npy'.format(log_dir, scene, src_id))
test_pc_keypoints = np.load('{}/keypoints/{}/cloud_bin_{}.npy'.format(log_dir, scene, tgt_id))
src_desc = np.load('{}/descriptors/{}/cloud_bin_{}{}.npy'.format(log_dir, scene, src_id, '.D3Feat' if is_d3feat else ''))
tgt_desc = np.load('{}/descriptors/{}/cloud_bin_{}{}.npy'.format(log_dir, scene, tgt_id, '.D3Feat' if is_d3feat else ''))

# print(src_desc.shape)

# ordered by scores
src_scores = np.load('{}/scores/{}/cloud_bin_{}.npy'.format(log_dir, scene, src_id))
tgt_scores = np.load('{}/scores/{}/cloud_bin_{}.npy'.format(log_dir, scene, tgt_id))

if is_d3feat:
    src_scores = np.squeeze(src_scores)
    tgt_scores = np.squeeze(tgt_scores)
    src_desc = src_desc.astype(np.float64)
    tgt_desc = tgt_desc.astype(np.float64)
    reference_pc_keypoints = reference_pc_keypoints.astype(np.float64)
    test_pc_keypoints = test_pc_keypoints.astype(np.float64)
    print(src_desc.dtype)

new_order_s, new_order_t = np.argsort(-src_scores), np.argsort(-tgt_scores)
reference_pc_keypoints, test_pc_keypoints = reference_pc_keypoints[new_order_s], test_pc_keypoints[new_order_t]
src_desc, tgt_desc = src_desc[new_order_s], tgt_desc[new_order_t]

max_kpt_num = 5000
max_kpt_num = min(min(src_desc.shape[0], tgt_desc.shape[0]), max_kpt_num)

src_desc = src_desc[:max_kpt_num]
tgt_desc = tgt_desc[:max_kpt_num]
reference_pc_keypoints = reference_pc_keypoints[:max_kpt_num]
test_pc_keypoints = test_pc_keypoints[:max_kpt_num]

# Save as open3d feature 
ref = open3d.registration.Feature()
ref.data = src_desc.T

test = open3d.registration.Feature()
test.data = tgt_desc.T


reference_pc = o3d.io.read_point_cloud('{}/{}/cloud_bin_{}.ply'.format(pcd_data_base_dir, scene, src_id))
test_pc = o3d.io.read_point_cloud('{}/{}/cloud_bin_{}.ply'.format(pcd_data_base_dir, scene, tgt_id))
reference_pc.estimate_normals()
test_pc.estimate_normals()

# Save ad open3d point clouds
ref_key = o3d.geometry.PointCloud()
ref_key.points = o3d.utility.Vector3dVector(reference_pc_keypoints)

test_key = o3d.geometry.PointCloud()
test_key.points = o3d.utility.Vector3dVector(test_pc_keypoints)

result_ransac = execute_global_registration(ref_key, test_key,
            ref, test, 0.05)


# # First plot the original state of the point clouds
# draw_registration_result(reference_pc, test_pc, np.identity(4))

# src_pose = np.array([
# -5.88374000e-01, 4.55734000e-01,-6.67921000e-01, 1.30842100e+00	,
#  2.84090000e-02,-8.13874000e-01,-5.80347000e-01, 8.49796000e-01	,
# -8.08089000e-01,-3.60435000e-01, 4.65915000e-01, 1.17514300e+00	,
#  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00	,
#  ]).reshape(4, 4)

# dst_pose = np.array([
# -9.98617000e-01,-4.69570000e-02,-2.35370000e-02, 1.43053800e+00	,
#  5.19500000e-02,-8.16876000e-01,-5.74468000e-01, 8.19226000e-01	,
#  7.74700000e-03,-5.74896000e-01, 8.18189000e-01, 1.22721900e+00	,
#  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00	,
#  ]).reshape(4, 4)

# result_ransac.transformation = np.linalg.inv( np.linalg.inv(src_pose) @ dst_pose)

# result_ransac.transformation = np.linalg.inv(np.asarray([
#     0.17652595593287168,	 0.5713145064650954	, -0.8015225022321764,	 1.9363734592188313	 ,
# -0.45749738394567213,	 0.7686281916600183,	 0.4471094347789082,	 -0.4767643650479027	, 
# 0.8715128975321317,	 0.28776802756385716,	 0.3970577184077332	, 1.2729953687911049	 ,
# 0.0	, 0.0,	 0.0,	 1.0	
# ]).reshape(4, 4))

print(result_ransac.transformation)

# Plot point clouds after registration
print(result_ransac)
draw_registration_result(reference_pc, test_pc,
            result_ransac.transformation)


