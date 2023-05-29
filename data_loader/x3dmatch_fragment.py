import sys
sys.path.append('.')

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
import open3d as o3d
import data_loader.transforms as transforms
import PIL
import utils
from data_loader.dense_to_sparse import dense_to_sparse
from data_loader.pcd_preprocess import *
from random import randrange
import random
from torch_geometric.data import Data
from torch_points3d.datasets.multiscale_data import MultiScaleBatch, MultiScaleData
from torch_points3d.core.data_transform import MultiScaleTransform, GridSampling3D, RandomNoise, Random3AxisRotation, RandomScaleAnisotropic, AddOnes, AddFeatByKey, Jitter, SaveOriginalPosId
from torch_geometric.transforms import Compose, FixedPoints
import torch_points_kernels as tp
import time
from multiprocessing import Manager
from utils.utils_algo import *
import MinkowskiEngine as ME


class X3DMatchFragmentDataLoader(BaseDataLoader):
    def __init__(self, config, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = X3DMatchFragmentDataset(config, self.data_dir, training=training)
        super().__init__(self.dataset, None, shuffle, validation_split, num_workers, collate_fn=None)

class X3DMatchFragmentDataset(Dataset):
    def __init__(self, config, data_dir, training=True):
        self.data_dir = data_dir
        self.config = config
        self.train_area = ['sun3d-brown_bm_1-brown_bm_1', 'sun3d-brown_bm_4-brown_bm_4', 'sun3d-brown_cogsci_1-brown_cogsci_1', 'sun3d-brown_cs_2-brown_cs2', 'sun3d-brown_cs_3-brown_cs3', 'sun3d-harvard_c3-hv_c3_1', 'sun3d-harvard_c5-hv_c5_1', 'sun3d-harvard_c6-hv_c6_1', 'sun3d-harvard_c8-hv_c8_3', 'sun3d-harvard_c11-hv_c11_2', 'sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika', 'sun3d-hotel_nips2012-nips_4', 'sun3d-hotel_sf-scan1', 'sun3d-mit_32_d507-d507_2', 'sun3d-mit_46_ted_lab1-ted_lab_2', 'sun3d-mit_76_417-76-417b', 'sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika', 'sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika', '7-scenes-chess', '7-scenes-fire', '7-scenes-heads', '7-scenes-office', '7-scenes-pumpkin', '7-scenes-stairs', 'rgbd-scenes-v2-scene_01', 'rgbd-scenes-v2-scene_02', 'rgbd-scenes-v2-scene_03', 'rgbd-scenes-v2-scene_04', 'rgbd-scenes-v2-scene_05', 'rgbd-scenes-v2-scene_06', 'rgbd-scenes-v2-scene_07', 'rgbd-scenes-v2-scene_08', 'rgbd-scenes-v2-scene_09', 'rgbd-scenes-v2-scene_10', 'rgbd-scenes-v2-scene_11', 'rgbd-scenes-v2-scene_12', 'rgbd-scenes-v2-scene_13', 'rgbd-scenes-v2-scene_14', 'bundlefusion-apt0', 'bundlefusion-apt1', 'bundlefusion-apt2', 'bundlefusion-copyroom', 'bundlefusion-office0', 'bundlefusion-office1', 'bundlefusion-office2', 'bundlefusion-office3', 'analysis-by-synthesis-apt1-kitchen', 'analysis-by-synthesis-apt1-living', 'analysis-by-synthesis-apt2-bed', 'analysis-by-synthesis-apt2-kitchen', 'analysis-by-synthesis-apt2-living', 'analysis-by-synthesis-apt2-luke', 'analysis-by-synthesis-office2-5a', 'analysis-by-synthesis-office2-5b']
        # self.train_area = ['7-scenes-chess', '7-scenes-fire', '7-scenes-heads', '7-scenes-office', '7-scenes-pumpkin', '7-scenes-stairs']

        self.test_area = ['7-scenes-redkitchen', 'sun3d-home_at-home_at_scan1_2013_jan_1', 'sun3d-home_md-home_md_scan9_2012_sep_30', 'sun3d-hotel_uc-scan3', 'sun3d-hotel_umd-maryland_hotel1', 'sun3d-hotel_umd-maryland_hotel3', 'sun3d-mit_76_studyroom-76-1studyroom2', 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']

        # for generate pretrained candidates
        # self.test_area += self.train_area

        self.training = training
        if training:
            self.areas = self.train_area
        else:
            self.areas = self.test_area


        # single fragment mode, or fragment pair mode
        self.mode = self.config['data_loader']['mode'] # 'single' or 'pair'
        if self.mode == 'single':
            with open('{}/fragments_adaptive/fragment_list_v2.json'.format(data_dir), 'r') as f:
                self.dataset_list = json.load(f)
            # remove items not in areas
            self.dataset_list = [item for item in self.dataset_list if item[0] in self.areas]
        elif self.mode == 'pair':
            with open('{}/fragments_adaptive/fragment_pair_list_v2.json'.format(data_dir), 'r') as f:
                self.dataset_list = json.load(f)
            # remove items not in areas
            self.dataset_list = [item for item in self.dataset_list if item[0][0] in self.areas]
        self.depth_trunc = self.config['data_loader']['depth_trunc']
        self.pcd_voxel_size = self.config['data_loader']['pcd_voxel_size']
        self.max_K_downsample_voxel_size = self.config['data_loader']['max_K_downsample_voxel_size']
        self.max_K_div = self.config['data_loader']['max_K_div']
        self.min_K = self.config['data_loader']['min_K']
        self.frame_skip_step = self.config['data_loader']['frame_skip_step']
        self.covisible_mask = self.config['data_loader']['covisible_mask']
        self.read_frame_data = self.config['data_loader'].get('read_frame_data', True)
        self.read_pretrained_2d_candidates = self.config['data_loader'].get('read_pretrained_2d_candidates', False)
        self.manager = Manager()
        if self.read_pretrained_2d_candidates:
            self.cache_2d_pts = self.manager.dict()
            self.cache_2d_desc = self.manager.dict()
        self.cache_pcd = shared_dict = self.manager.dict()
        self.cache_base_Twc = shared_dict = self.manager.dict()
        self.keypoints = {}
        self.keypoints_max_K = {}
        self.keypoints_chamfer_offset = {}

        # downsample dataset if dataset_subset_ratio < 1
        dataset_subset_ratio = self.config['data_loader']['dataset_subset_ratio']
        if dataset_subset_ratio < 1.0 and dataset_subset_ratio > 0:
            import random
            random.shuffle(self.dataset_list)
            self.dataset_list = self.dataset_list[:int(len(self.dataset_list) * dataset_subset_ratio)]

        self.flip_transform = transforms.Compose([transforms.HorizontalFlip(True)])

    def __len__(self):
        return len(self.dataset_list)

    def set_spatial_ops(self, spatial_ops):
        if spatial_ops != None:
            self.ms_transform = MultiScaleTransform(spatial_ops)

    def update_keypoints(self, new_keypoints):
        self.keypoints.update(new_keypoints)

    def update_keypoints_chamfer_offset(self, new_chamfer_offset):
        self.keypoints_chamfer_offset.update(new_chamfer_offset)

    def update_dynamic_subset(self, curr_epoch, full_idx_list, max_length):
        self.use_dynamic_subset = True
        rng = random.Random(curr_epoch)
        self.curr_subset_indices = rng.sample(full_idx_list, max_length)
        rng = random.Random(curr_epoch + 1)
        self.next_subset_indices = rng.sample(full_idx_list, max_length)

    def data_transform(self, rgb, depth, sparse_depth):
        # scale_factor = 0.2
        scale_factor = 0.0
        s = np.random.uniform(1 - scale_factor, 1 + scale_factor)  # random scaling
        # do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        # TODO: currently, descriptor branch supervision does not support flip augmentation
        do_flip = False

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            # transforms.Rotate(angle),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)

        # denormalize for augmentation
        color_jitter_factor = 0.2
        color_jitter = torchvision.transforms.ColorJitter(color_jitter_factor, color_jitter_factor, color_jitter_factor)
        rgb_np = PIL.Image.fromarray(rgb_np)
        rgb_np = color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255

        depth_np = np.squeeze(depth) / s
        depth_np = transform(depth_np)

        sparse_depth_np = np.squeeze(sparse_depth) / s
        sparse_depth_np = transform(sparse_depth_np)

        return rgb_np, depth_np.copy(), sparse_depth_np.copy(), do_flip

    def get_single_frame_data(self, area, fragment_idx, seq_name, img_idx):
        camera_intrinsics_file = os.path.join(self.data_dir, 'rgbd', area, 'camera-intrinsics.txt')
        rgb_path = '{}/rgbd/{}/{}/frame-{:06d}.color.png'.format(self.data_dir, area, seq_name, img_idx)
        depth_path = '{}/rgbd/{}/{}/frame-{:06d}.depth.png'.format(self.data_dir, area, seq_name, img_idx)
        pose_path = '{}/rgbd/{}/{}/frame-{:06d}.pose.txt'.format(self.data_dir, area, seq_name, img_idx)

        K_mat = np.loadtxt(camera_intrinsics_file)
        # dirty fix for 3DMatch dataset
        camera_intrinsics = [K_mat[0, 0], K_mat[1, 1], K_mat[0, 2] - 0.5, K_mat[1, 2] - 0.5]
        
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) * 1e-3
        depth[depth > self.depth_trunc] = 0
        
        sparse_depth = dense_to_sparse(depth, randrange(100, 1000), 20.0)

        camera_pose_Twc = np.loadtxt(pose_path)

        # data augmentation
        if self.training:
            rgb_aug, depth_aug, sparse_depth_aug, do_flip = self.data_transform(rgb, depth, sparse_depth)

        rgb = rgb / 255
        # normalize according to ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = (rgb - mean.reshape([1, 1, 3])) / std.reshape([1, 1, 3])

        # convert to [channel, H, W]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        sparse_depth = torch.from_numpy(sparse_depth).float().unsqueeze(0)

        sample = {'rgb': rgb,
                  'sparse_depth': sparse_depth,
                  'depth': depth,
                  'area': area,
                  'idx': img_idx,
                  'camera_intrinsics': np.asarray(camera_intrinsics),
                  'camera_pose_Twc': camera_pose_Twc
                  }
        # data augmentation
        if self.training:
            rgb_aug = (rgb_aug - mean.reshape([1, 1, 3])) / std.reshape([1, 1, 3])
            sample['rgb_aug'] = torch.from_numpy(rgb_aug.transpose((2, 0, 1))).float()
            sample['sparse_depth_aug'] = torch.from_numpy(sparse_depth_aug).float().unsqueeze(0)
            sample['depth_aug'] = torch.from_numpy(depth_aug).float().unsqueeze(0)
            sample['do_flip'] = do_flip
        return sample

    def transform_pcd(self, pcd_data):
        if self.config['arch']['vote_pcd_conv_model'] == 'fcgf_official':
            # we use official quantize for fcgf_official implementation
            coords = (pcd_data.pos / self.pcd_voxel_size).floor()
            _, inds = ME.utils.sparse_quantize(coords, return_index=True)
            # coords = coords[inds]
            pcd_data.pos = pcd_data.pos[inds]
            # MAX_NUM_PTS = 25000
            # if pcd_data.pos.shape[0] > MAX_NUM_PTS:
            #     rand_idx = torch.randperm(pcd_data.pos.shape[0])[:MAX_NUM_PTS]
            #     pcd_data.pos = pcd_data.pos[rand_idx]
            return pcd_data, pcd_data.clone()
        else:
            transform_preprocess = Compose([
                # SaveOriginalPosId(),
                GridSampling3D(self.pcd_voxel_size, quantize_coords=True, mode='last'),
                FixedPoints(20000, replace=False, allow_duplicates=False),
                AddOnes(),
                AddFeatByKey(add_to_x=True, feat_name='ones')
            ])
            pcd_data = transform_preprocess(pcd_data)
            if self.config['arch']['vote_pcd_conv_model'] == 'fcgf':
                # RandomNoise may break coord in fcgf
                return pcd_data, pcd_data.clone()
            if self.training:
                # fusion training mode does not support random rotation
                transform_add_noise = Compose([
                    # Random3AxisRotation(apply_rotation=True, rot_x=180, rot_y=180, rot_z=180),
                    # Random3AxisRotation(apply_rotation=True, rot_x=30, rot_y=30, rot_z=30),
                    # RandomScaleAnisotropic(scales=[0.9, 1.1]),
                    RandomNoise(sigma=0.005, clip=0.05)
                ])
                pcd_data_noise = transform_add_noise(pcd_data.clone())
                return pcd_data, pcd_data_noise
            else:
                return pcd_data, None


    def update_max_keypoint_of_fragment(self, idx, given_max_K=0):
        fragment_info = self.dataset_list[idx]
        if len(fragment_info) != 2: # single fragment
            area, fragment_idx, seq_name, frame_idx_begin, frame_idx_end = fragment_info
            fragment_key = '{}_{}'.format(area, fragment_idx)
            fragment_key_swap = None
            fragment_info = [fragment_info]
        else: # fragment pair
            fragment_key = '{}_{}_{}_{}'.format(
                fragment_info[0][0], fragment_info[0][1], fragment_info[1][0], fragment_info[1][1])
            fragment_key_swap = '{}_{}_{}_{}'.format(
                fragment_info[1][0], fragment_info[1][1], fragment_info[0][0], fragment_info[0][1])

        if given_max_K > 0: # check if given max_K
            self.keypoints_max_K[fragment_key] = given_max_K
        elif self.keypoints_max_K.get(fragment_key) is None: # check if we have cache this
            if self.covisible_mask:
                area, frag_idx1, frag_idx2 = fragment_info[0][0], fragment_info[0][1], fragment_info[1][1]
                pcd_all = load_pointcloud('{}/fragments_adaptive_overlap/{}/overlap_{}_{}.ply'.format(self.data_dir, area, frag_idx1, frag_idx2))
            else:
                pcd_all_np = np.empty((0, 3))
                # concat pointclouds
                for frag in fragment_info:
                    # read pointcloud fragment
                    area, fragment_idx, seq_name, frame_idx_begin, frame_idx_end = frag
                    pose_path = '{}/rgbd/{}/{}/frame-{:06d}.pose.txt'.format(self.data_dir, area, seq_name, frame_idx_begin)
                    camera_pose_Twc = np.loadtxt(pose_path)
                    pcd = load_pointcloud('{}/fragments_adaptive/{}/cloud_bin_{}.ply'.format(self.data_dir, area, fragment_idx))
                    pcd.transform(camera_pose_Twc)
                    pcd_all_np = np.concatenate([pcd_all_np, np.asarray(pcd.points)], axis=0)
                pcd_all = o3d.geometry.PointCloud()
                pcd_all.points = o3d.utility.Vector3dVector(pcd_all_np)
            # voxel downsample to normalize
            pcd_all = pcd_all.voxel_down_sample(self.max_K_downsample_voxel_size)
            # initialize max_K
            max_K = np.asarray(pcd_all.points).shape[0] // self.max_K_div
            # self.keypoints_max_K[fragment_key] = max_K
            self.keypoints_max_K[fragment_key] = max(max_K, self.min_K)
            if fragment_key_swap:
                self.keypoints_max_K[fragment_key_swap] = self.keypoints_max_K[fragment_key]

    def construct_2D_3D_correspondence(self, pcd, depth, camera_intrinsics, Twc):
        depth = np.squeeze(depth.numpy())
        H, W = depth.shape
        depth = dense_to_sparse(depth, randrange(10000, 20000), 50.0)
        ys, xs = np.where(depth > 0)
        pts = np.empty((len(xs), 3))
        pts[:, 0] = (xs - camera_intrinsics[2]) / camera_intrinsics[0]
        pts[:, 1] = (ys - camera_intrinsics[3]) / camera_intrinsics[1]
        pts[:, 2] = 1.0
        pts = pts * np.repeat(depth[ys, xs].reshape(-1, 1), 3, axis=1)
        # lift to space
        pts_w = np.transpose(np.matmul(Twc[:3, :3], np.transpose(pts))) + Twc[:3, 3]
        # find closest pts in pcd
        pts_w = torch.from_numpy(pts_w).float()
        idx, dist = tp.ball_query(0.05, 1, pcd.unsqueeze(0).contiguous(), pts_w.unsqueeze(0).contiguous(), mode="dense", sort=True)
        idx = idx.squeeze()
        dist = dist.squeeze()
        mask = dist >= 0
        # print('construct correspondence ratio', float(torch.sum(mask)) / pts_w.shape[0])
        # TODO: check projection error ?
        crsp = np.full(depth.shape, -1, dtype=np.long)
        crsp[ys[mask], xs[mask]] = idx[mask]
        crsp = torch.from_numpy(crsp).long()
        return crsp

    def get_fragment_data(self, idx, fragment_info, inference_only):
        ret_dict = {}
        area, fragment_idx, seq_name, frame_idx_begin, frame_idx_end = fragment_info
        fragment_key = '{}_{}'.format(area, fragment_idx)
        # read pointcloud fragment
        # pcd = load_pointcloud('{}/fragments_adaptive/{}/cloud_bin_{}.ply'.format(self.data_dir, area, fragment_idx))
        pcd = self.cache_pcd.get(fragment_key, None)
        if pcd is None:
            pcd_path = '{}/fragments_adaptive/{}/cloud_bin_{}.ply'.format(self.data_dir, area, fragment_idx)
            pcd = load_pointcloud(pcd_path)#.voxel_down_sample(self.pcd_voxel_size)
            pcd = torch.from_numpy(np.asarray(pcd.points)).to(torch.float)
            # do not perform duplicated downsample
            # self.cache_pcd[fragment_key] = pcd
        pcd_data = Data(pos=pcd)
        # del pcd
        # pointcloud data augmentation
        pcd_data, pcd_data_aug = self.transform_pcd(pcd_data)

        # get pretrained 2d candidates
        if self.read_pretrained_2d_candidates:
            if not fragment_key in self.cache_2d_pts:
                candidate_2d_base_dir = self.config['data_loader']['pretrained_2d_candidates_path']
                pt_np_path = '{}/keypoints/{}/cloud_bin_{}.npy'.format(candidate_2d_base_dir, area, fragment_idx)
                desc_np_path = '{}/descriptors/{}/cloud_bin_{}.npy'.format(candidate_2d_base_dir, area, fragment_idx)
                pts = np.load(pt_np_path)
                desc = np.load(desc_np_path)
                self.cache_2d_pts[fragment_key] = pts
                self.cache_2d_desc[fragment_key] = desc
            ret_dict['2d_candidate_pts'] = self.cache_2d_pts[fragment_key]
            ret_dict['2d_candidate_desc'] = self.cache_2d_desc[fragment_key]

        # get base_Twc from first frame
        base_Twc = self.cache_base_Twc.get(fragment_key, None)
        if base_Twc is None:
            first_frame = self.get_single_frame_data(area, fragment_idx, seq_name, frame_idx_begin)
            base_Twc = first_frame['camera_pose_Twc']
            self.cache_base_Twc[fragment_key] = base_Twc

        frame_data = None
        if self.read_frame_data:
            # read frames
            frame_list = list(range(random.randint(frame_idx_begin, frame_idx_begin+self.frame_skip_step), frame_idx_end + 1, self.frame_skip_step))
            frame_data = [self.get_single_frame_data(area, fragment_idx, seq_name, frame_idx) for frame_idx in frame_list]
            # construct 2D 3D correspondence (indices of pcd to image frames)
            for i, frame in enumerate(frame_data):
                frame['camera_pose_Twc'] = np.matmul(np.linalg.inv(base_Twc), frame['camera_pose_Twc'])
                frame['pcd_crsp_idx'] = self.construct_2D_3D_correspondence(pcd_data.pos, frame['depth'], frame['camera_intrinsics'], frame['camera_pose_Twc'])
                if self.training:
                    frame['pcd_crsp_idx_aug'] = torch.from_numpy(self.flip_transform(frame['pcd_crsp_idx'].numpy()).copy()) \
                        if frame['do_flip'] else frame['pcd_crsp_idx']
                frame_data[i] = frame

        # multiscale accelerate
        if hasattr(self, 'ms_transform'):
            pcd_data = self.ms_transform(pcd_data)
            pcd_data = MultiScaleBatch.from_data_list([pcd_data])
            if pcd_data_aug != None:
                pcd_data_aug = self.ms_transform(pcd_data_aug)
                pcd_data_aug = MultiScaleBatch.from_data_list([pcd_data_aug])

        ret_dict.update({'idx': idx,
                'fragment_info': fragment_info,
                'fragment_key': fragment_key,
                'frame_data': frame_data,
                'pcd_data': pcd_data,
                'pcd_data_aug': pcd_data_aug,
                'base_Twc': base_Twc,
                'inference_only': inference_only})
        return ret_dict

    def apply_covisible_mask(self, data, covisible_pts_np):
        dilate_kernel = np.ones((50, 50), dtype=np.uint8)
        if data['frame_data'] is None:
            return
        for i, frame in enumerate(data['frame_data']):
            Twc = frame['camera_pose_Twc']
            camera_intrinsics = frame['camera_intrinsics']
            # sample some depth
            depth = np.squeeze(frame['depth'].numpy())
            depth = dense_to_sparse(depth, 5000, 50.0)
            ys, xs = np.where(depth > 0)
            pts = np.empty((len(xs), 3))
            pts[:, 0] = (xs - camera_intrinsics[2]) / camera_intrinsics[0]
            pts[:, 1] = (ys - camera_intrinsics[3]) / camera_intrinsics[1]
            pts[:, 2] = 1.0
            pts = pts * np.repeat(depth[ys, xs].reshape(-1, 1), 3, axis=1)
            # lift to space
            pts_w = np.transpose(np.matmul(Twc[:3, :3], np.transpose(pts))) + Twc[:3, 3]
            # find closest pts in pcd
            pts_w = torch.from_numpy(pts_w).float()
            _, dist = tp.ball_query(0.05, 1, torch.from_numpy(covisible_pts_np).unsqueeze(0).contiguous(),
                pts_w.unsqueeze(0).contiguous(), mode="dense", sort=True)
            dist = dist.squeeze()
            pts_mask = dist >= 0
            # construct covisible mask
            covisible_mask = np.full(depth.shape, 0, dtype=np.uint8)
            covisible_mask[ys[pts_mask], xs[pts_mask]] = 1
            # mark neighbors as covisible
            covisible_mask = cv2.dilate(covisible_mask, dilate_kernel, iterations=1)
            # apply mask to depth
            masked_depth = frame['depth']
            ys_not_visible, xs_not_visible = np.where(covisible_mask == 0)
            masked_depth[:, ys_not_visible, xs_not_visible] = 0
            data['frame_data'][i]['depth'] = masked_depth
            if self.training:
                masked_depth = frame['depth_aug']
                masked_depth[:, ys_not_visible, xs_not_visible] = 0
                data['frame_data'][i]['depth_aug'] = masked_depth

    def __getitem__(self, idx):
        inference_only = False
        # if hasattr(self, 'use_dynamic_subset'):
        #     if idx not in self.curr_subset_indices and idx not in self.next_subset_indices:
        #         return None
        #     if idx not in self.curr_subset_indices and idx in self.next_subset_indices:
        #         inference_only = True
        #         # try not using chamfer offset map
        #         return None
        if self.mode == 'single':
            fragment_info = self.dataset_list[idx]
            return self.get_fragment_data(idx, fragment_info, inference_only)
        elif self.mode == 'pair':
            data_0 = self.get_fragment_data(idx, self.dataset_list[idx][0], inference_only)
            data_1 = self.get_fragment_data(idx, self.dataset_list[idx][1], inference_only)
            data_0['pair_tag'] = 0
            data_1['pair_tag'] = 1
            # convert data_1 to local coordinate of data_0
            pose_to_local = np.matmul(np.linalg.inv(data_0['base_Twc']), data_1['base_Twc'])
            if data_1['frame_data']:
                for i, f in enumerate(data_1['frame_data']):
                    data_1['frame_data'][i]['camera_pose_Twc'] = np.matmul(pose_to_local, f['camera_pose_Twc'])
            # data_1['base_Twc'] = np.matmul(pose_to_local, data_0['base_Twc'])
            # read overlap points
            area, frag_idx1, frag_idx2 = data_0['fragment_info'][0], data_0['fragment_info'][1], data_1['fragment_info'][1]
            pcd_overlap_np = np.asarray(
                    load_pointcloud('{}/fragments_adaptive_overlap/{}/overlap_{}_{}.ply'.format(
                        self.data_dir, area, frag_idx1, frag_idx2)).points).astype(np.float32)
            data_0['pcd_overlap_np'] = data_1['pcd_overlap_np'] = pcd_overlap_np
            # apply covisible mask to depth
            if self.covisible_mask:
                assert(pcd_overlap_np.size > 0)
                self.apply_covisible_mask(data_0, pcd_overlap_np)
                self.apply_covisible_mask(data_1, pcd_overlap_np)

            # transform pcd to base pcd coordinate
            data_1['pcd_data'].pos = \
                transform_pcd_pose(data_1['pcd_data'].pos, np.matmul(np.linalg.inv(data_0['base_Twc']), data_1['base_Twc']))
            data_1['pcd_data_aug'].pos = \
                transform_pcd_pose(data_1['pcd_data_aug'].pos, np.matmul(np.linalg.inv(data_0['base_Twc']), data_1['base_Twc']))

            ret_list = [data_0, data_1]
            if 'random_swap_pair' in self.config['data_loader'] and self.config['data_loader']['random_swap_pair']:
                random.shuffle(ret_list)
            return ret_list

# only for dataloader testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config = utils.read_yaml(args.config)
    dataset = X3DMatchFragmentDataset(config=config, data_dir=config['data_loader']['args']['data_dir'])

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
