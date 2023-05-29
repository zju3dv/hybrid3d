import numpy as np
import os

def load_log(path, item_lines):
    with open(os.path.join(path)) as f:
        content = f.readlines()
    data = []
    i = 0
    while i < len(content):
        line = content[i]
        data.append([content[i+j] for j in range(item_lines)])
        i = i + item_lines
    return data

def save_log(path, data):
    with open(os.path.join(path), 'w') as f:
        for d in data:
            f.writelines(d)

scene_list = [
    'apartment',
    'bedroom',
    'boardroom',
    'lobby',
    'loft'
]

if __name__ == '__main__':
    max_pairs_per_scene = 200
    for scene in scene_list:
        gt_log_dir = '/mnt/ssd_disk/redwood_lidar_3dmatch_format/fragments_adaptive/{}-evaluation/gt_orig.log'.format(scene)
        gt_info_dir = '/mnt/ssd_disk/redwood_lidar_3dmatch_format/fragments_adaptive/{}-evaluation/gt_orig.info'.format(scene)
        gt_log_save_dir = '/mnt/ssd_disk/redwood_lidar_3dmatch_format/fragments_adaptive/{}-evaluation/gt.log'.format(scene)
        gt_info_save_dir = '/mnt/ssd_disk/redwood_lidar_3dmatch_format/fragments_adaptive/{}-evaluation/gt.info'.format(scene)
        gt_logs = load_log(gt_log_dir, 5)
        gt_infos = load_log(gt_info_dir, 7)
        print(len(gt_logs), len(gt_infos))
        curr_length = len(gt_logs)
        if curr_length > max_pairs_per_scene:
            rand_idx = list(range(curr_length))
            import random
            random.shuffle(rand_idx)
            rand_idx = rand_idx[:max_pairs_per_scene]
            gt_logs = [gt_logs[i] for i in rand_idx]
            gt_infos = [gt_infos[i] for i in rand_idx]
            save_log(gt_log_save_dir, gt_logs)
            save_log(gt_info_save_dir, gt_infos)
