import numpy as np
import PIL
import data_loader.transforms as transforms
import torchvision
from torchvision import datasets
import torch
import utils
import utils.utils_algo as utils_algo
import cv2
import random
import torch_points_kernels as tp

flip_transform = transforms.Compose([transforms.HorizontalFlip(True)])


def prepare_batch_frame_data(choices, data, supervision, frame_batch_size, augmentation):
    # generate batch list
    batch_list = []
    for i in range(len(choices) // frame_batch_size):
        batch_list.append(choices[i*frame_batch_size:(i+1)*frame_batch_size])
    batch_frame_data = []

    for indices in batch_list:
        batch_frame_data.append({
            'rgb': torch.stack([(data[i]['rgb_aug'] if augmentation else data[i]['rgb']) for i in indices]),
            'sparse_depth' : torch.stack([(data[i]['sparse_depth_aug'] if augmentation else data[i]['sparse_depth']) for i in indices]),
            'depth' : torch.stack([(data[i]['depth_aug'] if augmentation else data[i]['depth']) for i in indices]),
            'pcd_crsp_idx': torch.stack([(data[i]['pcd_crsp_idx_aug'] if augmentation else data[i]['pcd_crsp_idx']) for i in indices]),
            'area' : [data[i]['area'] for i in indices],
            'idx' : [data[i]['idx'] for i in indices],
            'camera_intrinsics' : np.stack([data[i]['camera_intrinsics'] for i in indices]),
            'camera_pose_Twc' : np.stack([data[i]['camera_pose_Twc'] for i in indices]),
            'heatmap': torch.stack([
                torch.from_numpy(
                    flip_transform(supervision[i]['heatmap']).copy() if (augmentation and data[i]['do_flip']) else supervision[i]['heatmap']
                    ).unsqueeze(0) for i in indices]),
            'chamfer_offset_map': torch.stack([
                torch.from_numpy(
                    flip_transform(supervision[i]['chamfer_offset_map']).copy() if (augmentation and data[i]['do_flip']) else supervision[i]['chamfer_offset_map']
                    ).unsqueeze(0) for i in indices]),
            'projection_pts_info': [supervision[i]['projection_pts_info'] for i in indices]
        })
    return batch_frame_data

def prepare_supervision(config, dataset, fragment_key, frame_data, choice, fragment_idx):
    supervision = []
    corner_size = config['data_loader']['corner_size']
    divide_heatmap = config['data_loader']['divide_heatmap']
    depth_trunc = config['data_loader']['depth_trunc']
    for idx in range(len(frame_data)):
        if idx in choice:
            target_depth_np = frame_data[idx]['depth'].data.numpy()
            Twc = frame_data[idx]['camera_pose_Twc'].numpy()
            camera_intrinsics = frame_data[idx]['camera_intrinsics'].numpy()
            _, H, W = target_depth_np.shape

            back_proj_dict = utils.utils_algo.generate_back_projection(\
                dataset.keypoints[fragment_key], target_depth_np[0, ...],\
                    Twc, camera_intrinsics, \
                        corner_size, divide_heatmap, depth_trunc,\
                            dataset.keypoints_chamfer_offset[fragment_key],
                            valid_fragment_idx=fragment_idx)
            supervision.append({
                'heatmap': back_proj_dict['heatmap'],
                'chamfer_offset_map': back_proj_dict['chamfer_offset_map'],
                'projection_pts_info': back_proj_dict['projection_pts_info']
            })
        else:
            supervision.append(None)
    return supervision

# generate target heatmap and back projection info
def generate_target_info_by_closest_clusters(config, frame_data, output_heatmap, cluster_kpts, dist_thresh=0.05):
    output_heatmap_np = output_heatmap.cpu().detach().numpy()
    output_heatmap_np = utils_algo.remove_border_for_batch_heatmap(output_heatmap_np)
    B, _, H, W = output_heatmap_np.shape
    target_depth_np = frame_data['depth'].cpu().detach().numpy()
    batch_Twc = frame_data['camera_pose_Twc']
    batch_camera_intrinsics = frame_data['camera_intrinsics']
    lift_conf = config['trainer']['point_lifting']
    heatmap_corner_radius = config['data_loader']['corner_size']

    target_heatmap = []
    for i_frame in range(output_heatmap_np.shape[0]):
        # lift to space
        lift_input = (lift_conf, '', np.squeeze(output_heatmap_np[i_frame]),\
            np.squeeze(target_depth_np[i_frame]), batch_Twc[i_frame], batch_camera_intrinsics[i_frame])
        # conf, area_name, single_heatmap, single_depth, Twc, camera_intrinsics = input
        _, pts_w_weight_depth, pts_2d = utils_algo.lift_heatmap_depth_to_space(lift_input)
        # find if it has nearest cluster keypoints
        valid_coords = []
        valid_coords_label = []
        if cluster_kpts.size > 0:
            for i_pt, pt in enumerate(pts_w_weight_depth):
                dists = np.linalg.norm(cluster_kpts - pt[:3], axis=1, keepdims=False)
                min_idx = np.argmin(dists)
                if dists[min_idx] < dist_thresh:
                    valid_coords.append(pts_2d[i_pt][:2])
                    valid_coords_label.append(min_idx)
        # generate target heatmap
        heatmap = np.zeros((H, W, 1), np.uint8)
        for i_pt, pt in enumerate(valid_coords):
            pt = pt.round()
            heatmap = cv2.circle(heatmap, center=(pt[0], pt[1]),\
                radius=heatmap_corner_radius, color=(255), thickness=-1, lineType=cv2.LINE_AA)
        heatmap = np.squeeze(heatmap).astype(float) / 255.0
        target_heatmap.append(heatmap[None, None, ...])

    target_heatmap = np.concatenate(target_heatmap, axis=0)
    return {
        'target_heatmap': torch.from_numpy(target_heatmap),
        'projection_pts_info': None,
    }

def find_overlap_indices(src_pts_tensor, overlap_pts, dist_thresh=0.1):
    idx, dist = tp.ball_query(
        dist_thresh, 1, src_pts_tensor.detach().cpu().unsqueeze(0), overlap_pts.unsqueeze(0), mode="dense", sort=True)
    dist = dist.squeeze()
    mask = dist >= 0
    idx = idx.squeeze()
    return idx[mask]

def get_matching_indices(source, target, search_radius):
    # source: sampled K points
    # target: full points
    idx, dist = tp.ball_query(
        search_radius, 1, target.detach().cpu().unsqueeze(0).contiguous(), source.detach().cpu().unsqueeze(0).contiguous(), mode="dense", sort=True)
    dist = dist.squeeze()
    mask = dist >= 0
    idx = idx.squeeze()
    assert(idx.shape[0] == source.shape[0])
    src_match_idx = torch.arange(source.shape[0])[mask]
    tgt_match_idx = idx[mask]
    return src_match_idx, tgt_match_idx
