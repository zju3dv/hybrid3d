import sys
sys.path.append('.')

import utils.util as util
import os
import numpy as np
import open3d as o3d
import torch_points_kernels as tp
import torch

scenes = [
    '7-scenes-chess',
    '7-scenes-redkitchen',
    '7-scenes-fire',
    '7-scenes-heads',
    '7-scenes-office',
    '7-scenes-pumpkin',
    '7-scenes-stairs',
    'analysis-by-synthesis-apt1-kitchen',
    'analysis-by-synthesis-apt1-living',
    'analysis-by-synthesis-apt2-bed',
    'analysis-by-synthesis-apt2-kitchen',
    'analysis-by-synthesis-apt2-living',
    'analysis-by-synthesis-apt2-luke',
    'analysis-by-synthesis-office2-5a',
    'analysis-by-synthesis-office2-5b',
    'bundlefusion-apt0',
    'bundlefusion-apt1',
    'bundlefusion-apt2',
    'bundlefusion-copyroom',
    'bundlefusion-office0',
    'bundlefusion-office1',
    'bundlefusion-office2',
    'bundlefusion-office3',
    'rgbd-scenes-v2-scene_01',
    'rgbd-scenes-v2-scene_02',
    'rgbd-scenes-v2-scene_03',
    'rgbd-scenes-v2-scene_04',
    'rgbd-scenes-v2-scene_05',
    'rgbd-scenes-v2-scene_06',
    'rgbd-scenes-v2-scene_07',
    'rgbd-scenes-v2-scene_08',
    'rgbd-scenes-v2-scene_09',
    'rgbd-scenes-v2-scene_10',
    'rgbd-scenes-v2-scene_11',
    'rgbd-scenes-v2-scene_12',
    'rgbd-scenes-v2-scene_13',
    'rgbd-scenes-v2-scene_14',
    'sun3d-brown_bm_1-brown_bm_1',
    'sun3d-brown_bm_4-brown_bm_4',
    'sun3d-brown_cogsci_1-brown_cogsci_1',
    'sun3d-brown_cs_2-brown_cs2',
    'sun3d-brown_cs_3-brown_cs3',
    'sun3d-harvard_c11-hv_c11_2',
    'sun3d-harvard_c3-hv_c3_1',
    'sun3d-harvard_c5-hv_c5_1',
    'sun3d-harvard_c6-hv_c6_1',
    'sun3d-harvard_c8-hv_c8_3',
    'sun3d-home_at-home_at_scan1_2013_jan_1',
    'sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika',
    'sun3d-home_md-home_md_scan9_2012_sep_30',
    'sun3d-hotel_nips2012-nips_4',
    'sun3d-hotel_sf-scan1',
    'sun3d-hotel_uc-scan3',
    'sun3d-hotel_umd-maryland_hotel1',
    'sun3d-hotel_umd-maryland_hotel3',
    'sun3d-mit_32_d507-d507_2',
    'sun3d-mit_46_ted_lab1-ted_lab_2',
    'sun3d-mit_76_417-76-417b',
    'sun3d-mit_76_studyroom-76-1studyroom2',
    'sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
    'sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika'
]


def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = []
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result.append({'trans':trans, 'pair': (int(line[0]), int(line[1]))})

    return result

def find_fragment_info(fragment_list, scene, idx):
    for x in fragment_list:
        if x[0] == scene and x[1] == idx:
            return x
    return None

def load_pcd(filename, voxel_size=0.03):
    pcd = o3d.io.read_point_cloud(filename)
    pcd = pcd.voxel_down_sample(voxel_size)
    return pcd

def save_pcd(filename, pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    o3d.io.write_point_cloud(filename, pcd)

if __name__ == "__main__":
    fragment_base_dir = 'data/3dmatch/fragments_adaptive'
    fragment_overlap_save_dir = 'data/3dmatch/fragments_adaptive_overlap'
    fragment_list = util.read_json(f'{fragment_base_dir}/fragment_list_v2.json')
    fragment_pair_list_full = []
    for scene in scenes:
        gt_path_dir = f'{fragment_base_dir}/{scene}-evaluation/'
        curr_save_dir = f'{fragment_overlap_save_dir}/{scene}'
        util.ensure_dir(curr_save_dir)
        log = loadlog(gt_path_dir)
        # print(log)
        for item in log:
            idx1, idx2 = item['pair']
            pcd1 = load_pcd(f'{fragment_base_dir}/{scene}/cloud_bin_{idx1}.ply')
            pcd1 = np.asarray(pcd1.points)
            pcd2 = load_pcd(f'{fragment_base_dir}/{scene}/cloud_bin_{idx2}.ply')
            pcd2.transform(item['trans'])
            pcd2 = np.asarray(pcd2.points)
            idx, dist = tp.ball_query(0.05, 1, torch.from_numpy(pcd2).unsqueeze(0), torch.from_numpy(pcd1).unsqueeze(0), mode="dense", sort=True)
            dist = dist.squeeze()
            mask = dist >= 0
            pcd_overlap = pcd1[mask]
            overlap_ratio = float(torch.sum(mask)) / dist.shape[0]
            if overlap_ratio < 0.3:
                print('overlap ratio {} in {} {} {} '.format(overlap_ratio, scene, idx1, idx2))
            save_pcd(f'{curr_save_dir}/overlap_{idx1}_{idx2}.ply', pcd_overlap)
            print('\r', scene, idx1, idx2, '{:4f}'.format(overlap_ratio), end='')
        print('')
    # util.write_json(fragment_pair_list_full, f'{fragment_base_dir}/fragment_pair_list_v2.json')
