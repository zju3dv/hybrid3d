import sys
sys.path.append('..')

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from base import BaseDataLoader
import torchvision
from torchvision import datasets
import cv2
import numpy as np
import matplotlib.pyplot as plt
import data_loader.transforms as transforms
import PIL
import utils
from data_loader.dense_to_sparse import dense_to_sparse
from data_loader.pcd_preprocess import *
from random import randrange

class X3DMatchDataLoader(BaseDataLoader):
    def __init__(self, config, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = X3DMatchDataset(config, self.data_dir, training=training, augmentation=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class X3DMatchDataset(Dataset):
    def __init__(self, config, data_dir, training=True, augmentation=True):
        self.data_dir = data_dir
        self.config = config
        self.augmentation = augmentation
        self.train_area = ['sun3d-brown_bm_1-brown_bm_1', 'sun3d-brown_bm_4-brown_bm_4', 'sun3d-brown_cogsci_1-brown_cogsci_1', 'sun3d-brown_cs_2-brown_cs2', 'sun3d-brown_cs_3-brown_cs3', 'sun3d-harvard_c3-hv_c3_1', 'sun3d-harvard_c5-hv_c5_1', 'sun3d-harvard_c6-hv_c6_1', 'sun3d-harvard_c8-hv_c8_3', 'sun3d-harvard_c11-hv_c11_2', 'sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika', 'sun3d-hotel_nips2012-nips_4', 'sun3d-hotel_sf-scan1', 'sun3d-mit_32_d507-d507_2', 'sun3d-mit_46_ted_lab1-ted_lab_2', 'sun3d-mit_76_417-76-417b', 'sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika', 'sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika', '7-scenes-chess', '7-scenes-fire', '7-scenes-heads', '7-scenes-office', '7-scenes-pumpkin', '7-scenes-stairs', 'rgbd-scenes-v2-scene_01', 'rgbd-scenes-v2-scene_02', 'rgbd-scenes-v2-scene_03', 'rgbd-scenes-v2-scene_04', 'rgbd-scenes-v2-scene_05', 'rgbd-scenes-v2-scene_06', 'rgbd-scenes-v2-scene_07', 'rgbd-scenes-v2-scene_08', 'rgbd-scenes-v2-scene_09', 'rgbd-scenes-v2-scene_10', 'rgbd-scenes-v2-scene_11', 'rgbd-scenes-v2-scene_12', 'rgbd-scenes-v2-scene_13', 'rgbd-scenes-v2-scene_14', 'bundlefusion-apt0', 'bundlefusion-apt1', 'bundlefusion-apt2', 'bundlefusion-copyroom', 'bundlefusion-office0', 'bundlefusion-office1', 'bundlefusion-office2', 'bundlefusion-office3', 'analysis-by-synthesis-apt1-kitchen', 'analysis-by-synthesis-apt1-living', 'analysis-by-synthesis-apt2-bed', 'analysis-by-synthesis-apt2-kitchen', 'analysis-by-synthesis-apt2-living', 'analysis-by-synthesis-apt2-luke', 'analysis-by-synthesis-office2-5a', 'analysis-by-synthesis-office2-5b']
        # self.train_area = ['7-scenes-chess', '7-scenes-fire', '7-scenes-heads', '7-scenes-office', '7-scenes-pumpkin', '7-scenes-stairs']
        self.test_area = ['7-scenes-redkitchen', 'sun3d-home_at-home_at_scan1_2013_jan_1', 'sun3d-home_md-home_md_scan9_2012_sep_30', 'sun3d-hotel_uc-scan3', 'sun3d-hotel_umd-maryland_hotel1', 'sun3d-hotel_umd-maryland_hotel3', 'sun3d-mit_76_studyroom-76-1studyroom2', 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']
        with open(os.path.join(data_dir, 'full_list.json'), 'r') as f:
            self.dataset_list = json.load(f)
        if training:
            self.areas = self.train_area
        else:
            self.areas = self.test_area
        # remove items not in areas
        self.dataset_list = [item for item in self.dataset_list if item[0] in self.areas]
        self.corner_size = self.config['data_loader']['corner_size']
        self.init_kpt_method = self.config['data_loader']['initial_keypoints_method']
        self.depth_trunc = self.config['data_loader']['depth_trunc']
        # read initial keypoints
        self.keypoints = {}
        self.keypoints_max_K = {}
        self.keypoints_chamfer_offset = {}
        if self.init_kpt_method != 'fps':
            keypoint_file_path = "%s/initial_keypoints/%s/full.json" % (self.data_dir, self.init_kpt_method)
            keypoints_full = utils.utils_algo.load_keypints_from_file(keypoint_file_path)
        for area in self.areas:
            self.keypoints_chamfer_offset[area] = None
            if self.init_kpt_method == 'fps':
                mesh_filepath = "%s/reconstruction/%s/mesh_clean.ply" % (self.data_dir, area)
                scene_pts = mesh_voxelize_to_points(mesh_filepath, 0.03)
                max_K = int(scene_pts.shape[0] / 200)
                self.keypoints_max_K[area] = max_K
                print('initial keypoints', area, max_K)
                self.keypoints[area] = fps_sample_points(scene_pts, max_K)
                
            else:
                self.keypoints[area] = keypoints_full[area]

    def __len__(self):
        return len(self.dataset_list)

    def update_keypoints(self, new_keypoints):
        for key, value in self.keypoints.items():
            if key in new_keypoints:
                self.keypoints[key] = new_keypoints[key]

    def update_keypoints_chamfer_offset(self, new_chamfer_offset):
        for key, value in self.keypoints.items():
            if key in new_chamfer_offset:
                self.keypoints_chamfer_offset[key] = new_chamfer_offset[key]

    def data_transform(self, rgb, depth, keypoint_depth, sparse_depth, chamfer_offset_map, heatmap):
        s = np.random.uniform(0.5, 1.5)  # random scaling
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        output_size = (480, 640)

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            # transforms.Rotate(angle),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        color_jitter = torchvision.transforms.ColorJitter(0.4, 0.4, 0.4)
        rgb_np = PIL.Image.fromarray(rgb_np)
        rgb_np = color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        # Scipy affine_transform produced RuntimeError when the depth map was
        # given as a 'numpy.ndarray'

        depth_np = depth / s
        depth_np = transform(depth_np)

        keypoint_depth_np = keypoint_depth / s
        keypoint_depth_np = transform(keypoint_depth_np)

        sparse_depth_np = sparse_depth / s
        sparse_depth_np = transform(sparse_depth_np)

        chamfer_offset_map = transform(chamfer_offset_map)

        heatmap_np = transform(heatmap)

        return rgb_np, depth_np.copy(), keypoint_depth_np.copy(), sparse_depth_np.copy(), chamfer_offset_map.copy(), heatmap_np.copy()


    def __getitem__(self, idx):
        item = self.dataset_list[idx]
        area, seq_name, img_idx = item
        camera_intrinsics_file = os.path.join(self.data_dir, 'rgbd', area, 'camera-intrinsics.txt')
        rgb_path = '{}/rgbd/{}/{}/frame-{}.color.png'.format(self.data_dir, area, seq_name, img_idx)
        # depth_path = '{}/depth_from_reconstruction/{}/{}/{}.png'.format(self.data_dir, area, seq_name, img_idx)
        depth_path = '{}/rgbd/{}/{}/frame-{}.depth.png'.format(self.data_dir, area, seq_name, img_idx)
        pose_path = '{}/rgbd/{}/{}/frame-{}.pose.txt'.format(self.data_dir, area, seq_name, img_idx)

        K_mat = np.loadtxt(camera_intrinsics_file)
        camera_intrinsics = [K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]]
        
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) * 1e-3
        depth[depth > self.depth_trunc] = 0
        
        sparse_depth = dense_to_sparse(depth, randrange(100, 1000), 20.0)

        camera_pose_Twc = np.loadtxt(pose_path)
        back_proj_dict = utils.utils_algo.generate_back_projection(self.keypoints[area], depth,
            camera_pose_Twc, camera_intrinsics, self.corner_size, False, self.depth_trunc, self.keypoints_chamfer_offset[area])
        heatmap, keypoint_depth, chamfer_offset_map = back_proj_dict['heatmap'], back_proj_dict['keypoint_depth'], back_proj_dict['chamfer_offset_map']
        if self.augmentation:
            rgb, depth, keypoint_depth, sparse_depth, chamfer_offset_map, heatmap =\
                self.data_transform(rgb, depth, keypoint_depth, sparse_depth, chamfer_offset_map, heatmap)
        else:
            rgb = rgb / 255

        # convert to [channel, H, W]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        keypoint_depth = torch.from_numpy(keypoint_depth).float().unsqueeze(0)
        sparse_depth = torch.from_numpy(sparse_depth).float().unsqueeze(0)
        heatmap = torch.from_numpy(heatmap).float().unsqueeze(0)
        chamfer_offset_map = torch.from_numpy(chamfer_offset_map).float().unsqueeze(0)

        sample = {'rgb': rgb,
                  'sparse_depth': sparse_depth,
                  'depth': depth,
                  'keypoint_depth': keypoint_depth,
                  'chamfer_offset_map': chamfer_offset_map,
                  'heatmap': heatmap,
                  'area': area,
                  'idx': img_idx,
                  'camera_intrinsics': np.asarray(camera_intrinsics),
                  'camera_pose_Twc': camera_pose_Twc
                  }
        return sample


# only for dataloader testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()
    dataset = X3DMatchDataset(data_dir=args.data_dir)

    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    for i_batch, sample_batched in enumerate(dataloader):
        print('\r%d / %d' % (i_batch * batch_size, len(dataset)), end='')
        # observe 4th batch and stop.
        if i_batch == 4:
            # print(sample_batched['area'])
            plt.imshow(torch.squeeze(sample_batched['depth'], 0)[0, :], cmap='Reds')
            plt.figure()
            plt.imshow(torch.squeeze(sample_batched['rgb'], 0)[0, :])
            plt.figure()
            plt.imshow(torch.squeeze(sample_batched['sparse_depth'], 0)[0, :], cmap='Reds')
            plt.figure()
            plt.imshow(torch.squeeze(sample_batched['heatmap'], 0)[0, :], cmap='Reds')
            plt.show()
            break
