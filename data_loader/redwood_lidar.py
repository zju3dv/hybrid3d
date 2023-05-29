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

class RedwoodLidarDataLoader(BaseDataLoader):
    def __init__(self, config, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = RedwoodLidarDataset(config, self.data_dir, training=training, augmentation=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class RedwoodLidarDataset(Dataset):
    def __init__(self, config, data_dir, training=True, augmentation=True):
        self.data_dir = data_dir
        self.config = config
        self.augmentation = augmentation
        self.train_area = ['apartment', 'bedroom', 'boardroom', 'lobby']
        # self.train_area = ['apartment']
        self.test_area = ['loft']
        with open(os.path.join(data_dir, 'full_list.json'), 'r') as f:
            self.dataset_list = json.load(f)
        if training:
            self.areas = self.train_area
        else:
            self.areas = self.test_area
        # remove items not in areas
        self.dataset_list = [item for item in self.dataset_list if item[0] in self.areas]
        # read trajectories and initial keypoints
        self.pose_dict = {}
        self.keypoints = {}
        self.keypoints_max_K = {}
        self.keypoints_chamfer_offset = {}
        self.corner_size = self.config['data_loader']['corner_size']
        self.init_kpt_method = self.config['data_loader']['initial_keypoints_method']
        self.camera_intrinsics = self.config['data_loader']['camera_intrinsics']
        self.depth_trunc = self.config['data_loader']['depth_trunc']
        for area in self.areas:
            pose_path = "%s/pose/pose_%s/%s.log" % (self.data_dir, area, area)
            self.pose_dict[area] = self.read_trajectory(pose_path)
            self.keypoints_chamfer_offset[area] = None
            if self.init_kpt_method == 'fps':
                mesh_filepath = "%s/ours_reconstruction/ours_%s/%s_clean.ply" % (self.data_dir, area, area)
                scene_pts = mesh_voxelize_to_points(mesh_filepath, 0.03)
                max_K = int(scene_pts.shape[0] / 300)
                self.keypoints_max_K[area] = max_K
                print('initial keypoints', area, max_K)
                self.keypoints[area] = fps_sample_points(scene_pts, max_K)
            else:
                keypoint_file_path = "%s/initial_keypoints/%s/%s.json" % (self.data_dir, self.init_kpt_method, area)
                self.keypoints[area] = np.asarray(utils.read_json(keypoint_file_path))

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

    # adopt from Redwood Lidar
    class CameraPose:
        def __init__(self, meta, mat):
            self.metadata = meta
            self.pose = mat
        def __str__(self):
            return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
                "Pose : " + "\n" + np.array_str(self.pose)

    # adopt from Redwood Lidar
    def read_trajectory(self, filename):
        traj = []
        with open(filename, 'r') as f:
            metastr = f.readline();
            while metastr:
                metadata = map(int, metastr.split())
                mat = np.zeros(shape = (4, 4))
                for i in range(4):
                    matstr = f.readline();
                    mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
                traj.append(self.CameraPose(metadata, mat))
                metastr = f.readline()
        return traj

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
        area = item[0]
        img_idx = item[1]
        rgb_path = "%s/rgbd/rgbd_%s/%s/image/%s.jpg" % (self.data_dir, area, area, img_idx)
        # depth_path = "%s/depth_from_reconstruction/%s/%s.png" % (self.data_dir, area, img_idx)
        depth_path = "%s/rgbd/rgbd_%s/%s/depth/%s.png" % (self.data_dir, area, area, img_idx)
        slam_kpt_proj_path = "%s/slam_projection/%s/%s.png" % (self.data_dir, area, img_idx)
        
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) * 1e-3
        depth[depth > self.depth_trunc] = 0
        
        use_depth_from_slam = False
        if use_depth_from_slam:
            sparse_depth = cv2.imread(slam_kpt_proj_path, cv2.IMREAD_ANYDEPTH)  * 1e-3
        else:
            sparse_depth = dense_to_sparse(depth, randrange(100, 1000), 20.0)

        camera_pose_Twc = self.pose_dict[area][int(img_idx)].pose
        back_proj_dict = utils.utils_algo.generate_back_projection(self.keypoints[area], depth,
            camera_pose_Twc, self.camera_intrinsics, self.corner_size, False, self.depth_trunc, self.keypoints_chamfer_offset[area])
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
                  'camera_intrinsics': np.asarray(self.camera_intrinsics),
                  'camera_pose_Twc': camera_pose_Twc
                  }
        return sample


# only for dataloader testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()
    dataset = RedwoodLidarDataset(data_dir=args.data_dir)

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
