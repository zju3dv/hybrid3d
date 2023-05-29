import sys
import os

sys.path.append(os.getcwd())  # noqa
from utils import write_json



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

full_list = []
base_dir = 'data/3dmatch/'
for scene in scenes:
    print(scene)
    for seq_idx in range(1, 99):
        for i in range(99999):
            pose_file = '{}/rgbd/{}/seq-{:02d}/frame-{:06d}.pose.txt'.format(base_dir, scene, seq_idx, i)
            if os.path.isfile(pose_file):
                full_list.append([scene, 'seq-{:02d}'.format(seq_idx), '{:06d}'.format(i)])
write_json(full_list, '{}/full_list.json'.format(base_dir))
