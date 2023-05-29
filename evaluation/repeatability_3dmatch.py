import sys
import open3d
import numpy as np
import time
import os
sys.path.append(os.getcwd())  # noqa
from evaluation.eval_utils import get_pcd, get_keypts, get_desc, loadlog
from utils.utils_algo import load_keypints_from_file
from utils.util import read_pkl
from utils.sampler import nms_3D
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

geometric_registration_dir = '' # e.g. data/3dmatch/geometric_registration_adaptive/
keypoints_dict = {}
need_compensate_transformation = False

def transform_points_3dmatch_fragment(transformation_file, points, direction):
    if not need_compensate_transformation:
        return points
    with open(transformation_file, 'r') as f:
        metastr = f.readline()
        mat = np.empty((4, 4))
        for i in range(4):
            matstr = f.readline()
            mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
    if direction == 'frag2world':
        mat = np.linalg.inv(mat)
    elif direction == 'world2frag':
        pass
    else:
        print('Unknown direction ', direction)
        exit(0)
    R = mat[:3, :3]
    t = mat[:3, 3]
    points = np.matmul(np.transpose(R), np.transpose(points - t))
    points = np.transpose(points)
    return points

nms_set = set()

def get_keypts_3dmatch(scene, id):
    fragment_key = '{}_{}'.format(scene, id)
    if keypoints_dict[fragment_key].size == 0:
        return None
    pcd_np = keypoints_dict[fragment_key][:, :3]
    scores_np = keypoints_dict[fragment_key][:, 3]
    # if not fragment_key in nms_set:
    #     # print('A', pcd_np.shape)
    #     pcd_np, scores_np, _ = nms_3D(pcd_np, scores_np, 0.01, method='max')
    #     # print('B', pcd_np.shape)
    #     keypoints_dict[fragment_key] = np.concatenate([pcd_np, scores_np[:, None]], axis=1)
    #     nms_set.add(fragment_key)
    fragment_pose_file = '%s/%s/cloud_bin_%s.info.txt' % (geometric_registration_dir, scene, id)
    return transform_points_3dmatch_fragment(fragment_pose_file, pcd_np, 'world2frag'), scores_np

def deal_with_one_scene(scene, num_keypts):
    """
    calculate the relative repeatability under {num_keypts} settings for {scene}
    """
    # pcdpath = f"../data/3DMatch/fragments/{scene}/"
    pcdpath = '{}/{}'.format(geometric_registration_dir, scene)
    # keyptspath = f"../geometric_registration/{desc_name}_{timestr}/keypoints/{scene}"
    # gtpath = f'../geometric_registration/gt_result/{scene}-evaluation/'
    gtpath = '{}/{}-evaluation/'.format(geometric_registration_dir, scene)
    gtLog = loadlog(gtpath)

    # register each pair
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    num_repeat_list = []
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
        # for id2 in range(id1 + 1, id1 + 2):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f'{cloud_bin_s.split("_")[-1]}_{cloud_bin_t.split("_")[-1]}'
            # print(scene, key)
            if key not in gtLog.keys():
                continue

            source_keypts, source_scores = get_keypts_3dmatch(scene, id1)
            target_keypts, target_scores = get_keypts_3dmatch(scene, id2)
            if source_keypts is None or target_keypts is None:
                id = id1 if source_keypts is None else id2
                # print('Warning None pts found at {} {}'.format(scene, id))
                continue

            num_keypts = min(min(source_keypts.shape[0], target_keypts.shape[0]), num_keypts)

            source_keypts = source_keypts[:num_keypts, :3]
            target_keypts = target_keypts[:num_keypts, :3]
            # print('s t shape', source_keypts.shape, target_keypts.shape)
            
            gtTrans = gtLog[key]
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(target_keypts)
            pcd.transform(gtTrans)
            target_keypts = np.asarray(pcd.points)

            distance = cdist(source_keypts, target_keypts, metric='euclidean')
            num_repeat = np.sum(distance.min(axis=0) < 0.1)
            # print(num_repeat)
            num_repeat_list.append(num_repeat * 1.0 / num_keypts)
    # print(f"Scene {scene} repeatability: {sum(num_repeat_list) / len(num_repeat_list)}")
    return sum(num_repeat_list) / len(num_repeat_list)

scene_list = [
    '7-scenes-redkitchen',
    'sun3d-home_at-home_at_scan1_2013_jan_1',
    'sun3d-home_md-home_md_scan9_2012_sep_30',
    'sun3d-hotel_uc-scan3',
    'sun3d-hotel_umd-maryland_hotel1',
    'sun3d-hotel_umd-maryland_hotel3',
    'sun3d-mit_76_studyroom-76-1studyroom2',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
]

def calculate_repeatability(num_keypts):
    """
    calculate the relative repeatability of {desc_name}_{timestr} under {num_keypts} setting.
    """

    scene_repeatability_list = []
    for scene in scene_list:
        repeatability = deal_with_one_scene(scene, num_keypts=num_keypts)
        scene_repeatability_list.append(repeatability)
    ave_repeatability = sum(scene_repeatability_list) / len(scene_list)
    print(f"Average Repeatability at num_keypts = {num_keypts}: {ave_repeatability}")
    return ave_repeatability


if __name__ == '__main__':
    # desc_name = sys.argv[1]
    # timestr = sys.argv[2]
    geometric_registration_dir = sys.argv[1]
    clustered_kpts_file = sys.argv[2]
    if 'redwood' in geometric_registration_dir:
        scene_list = ['apartment', 'bedroom', 'boardroom', 'lobby', 'loft']
    if '.json' in clustered_kpts_file:
        keypoints_dict = load_keypints_from_file(clustered_kpts_file)
    elif '.pkl' in clustered_kpts_file:
        keypoints_dict = read_pkl(clustered_kpts_file)
        for k, v in keypoints_dict.items():
            keypoints_dict[k] = np.array(v)
    else:
        for scene in scene_list:
            for i in range(999999):
                if not os.path.isfile('{}/keypoints/{}/cloud_bin_{}.npy'.format(clustered_kpts_file, scene, i)):
                    break
                kpts = np.load('{}/keypoints/{}/cloud_bin_{}.npy'.format(clustered_kpts_file, scene, i))
                scores = np.load('{}/scores/{}/cloud_bin_{}.npy'.format(clustered_kpts_file, scene, i))
                keypoints_dict['{}_{}'.format(scene, i)] = np.concatenate([kpts, scores[:, None]], axis=1)

    num_list = [4, 8, 16, 32, 64, 128, 256, 512]
    rep_list = []
    for i in num_list:
        ave_repeatability = calculate_repeatability(i)
        rep_list.append(ave_repeatability)
