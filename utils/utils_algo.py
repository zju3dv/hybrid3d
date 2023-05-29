import numpy as np
import torch
import cv2
import os
import utils.util as util
import time
import random
from scipy.spatial.distance import pdist, squareform
import open3d as o3d

import sys
sys.path.append('./csrc/build')
import c_utils

def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0]+pad, rc[1]+pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

# single heatmap
def heatmap_to_pts(heatmap, depth=None, conf_thresh=0.5, nms_dist=4):
    if not depth is None:
        heatmap[depth==0] = 0
    # use GPU maxpool to accelerate nms
    # max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # heatmap_gpu = torch.from_numpy(heatmap).float().cuda()[None, ...]
    # maxima = max_filter(heatmap_gpu) == heatmap_gpu
    # maxima = maxima.detach().cpu().numpy()[0]
    # heatmap[maxima==False] = 0

    xs, ys = np.where(heatmap >= conf_thresh)
    if len(xs) == 0:
      return np.zeros((3, 0))
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    H, W = heatmap.shape
    # pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist) # Apply NMS.
    pts, _ = c_utils.nms(np.transpose(pts), H, W, nms_dist)
    pts = np.transpose(pts)
    return pts

def draw_pts_with_confidence(pts, H, W, image=None):
    if not isinstance(pts, (np.ndarray)):
        return image.astype('uint8').copy()
    if image is None:
        image = np.zeros((H, W, 3), np.uint8)
    else:
        image = np.transpose(image, (1, 2, 0)).astype('uint8').copy()
    for i in range(pts.shape[1]):
        cv2.drawMarker(image, (int(round(pts[0, i])), int(round(pts[1, i]))), color=(0, 255, 0), markerSize=10)
    return np.transpose(image, (2, 0, 1))

def generate_back_projection(keypoints, depth, Twc, camera_intrinsics, corner_size=8, 
                     divide_heatmap_generation=False, depth_trunc=-1, chamfer_offset=None, need_kpt_depth=True,
                     valid_fragment_idx=None):
    """
    generate back projection (heatmap, chamfer_offset_map, keypoint_depth, projected_keypoints)
    keypoints: Nx3 numpy array, world keypoints
    Twc: 3x4 numpy array
         Rwc = Twc[:3, :3]
         twc = Twc[:3, 3]
         X_world = Rwc * X_camera + twc
    camera_intrinsics: list [fx, fy, cx, cy, ...]
    depth_trunc: set depth larger than depth_trunc to invalid
    """
    filter_by_fragment_idx = False
    if divide_heatmap_generation and keypoints.shape[1] >= 4:
        fragment_indices = keypoints[:, 3]
        filter_by_fragment_idx = True
    keypoints = keypoints[:,:3]
    # origin input : Twc
    Rwc = Twc[:3, :3]
    twc = Twc[:3, 3]
    fx, fy, cx, cy, *_ = camera_intrinsics
    H, W = depth.shape
    if depth_trunc > 0:
        depth[depth > depth_trunc] = 0
    heatmap = np.zeros((H, W, 1), np.uint8)
    if need_kpt_depth:
        keypoint_depth = np.zeros((H, W), np.float32)
    else:
        keypoint_depth = None
    chamfer_offset_map = np.zeros((H, W), np.float32)
    # the second np.transpose is only for batch compute rotation
    pts_cam = np.matmul(np.transpose(Rwc), np.transpose(keypoints - twc))
    pts_cam = np.transpose(pts_cam)

    projected_pts = np.empty((pts_cam.shape[0], 2))
    projected_pts_label = np.empty((pts_cam.shape[0]), dtype=np.int32)
    count = 0
    for i in range(pts_cam.shape[0]):
        # if this point is only voted by one fragment, skip back projection in another fragment
        # print(fragment_indices[i])
        if filter_by_fragment_idx\
            and (int(round(fragment_indices[i])) != 2)\
            and (int(round(fragment_indices[i])) != valid_fragment_idx):
            continue
        pt = pts_cam[i]
        if pt[2] < 0: continue
        pt_plane = ((pt[0] / pt[2]) * fx + cx, (pt[1] / pt[2]) * fy + cy)
        if pt_plane[0] > W - 1 or pt_plane[0] < 0 or pt_plane[1] > H - 1 or pt_plane[1] < 0: continue
        pt_plane_int = tuple(int(round(x)) for x in pt_plane)
        # pt_plane_int : x, y
        curr_depth = depth[pt_plane_int[1], pt_plane_int[0]]
        if curr_depth == 0:
            continue
        if abs(pt[2] - curr_depth) > 0.1: continue
        cv2.circle(heatmap, center=pt_plane_int, radius=corner_size, color=(255), thickness=-1, lineType=cv2.LINE_AA)
        # keypoint depth
        x, y = pt_plane_int
        x_min = max(0, x - corner_size)
        x_max = min(W, x + corner_size + 1)
        y_min = max(0, y - corner_size)
        y_max = min(H, y + corner_size + 1)
        if need_kpt_depth:
            keypoint_depth[y_min:y_max, x_min:x_max] = pt[2]
        if not chamfer_offset is None:
            chamfer_offset_map[y_min:y_max, x_min:x_max] = chamfer_offset[i, 3]
        # add to projected_pts
        projected_pts[count, :] = pt_plane
        projected_pts_label[count] = i
        count += 1
    projected_pts = projected_pts[:count, ...]
    projected_pts_label = projected_pts_label[:count]
    heatmap = np.squeeze(heatmap).astype(float) / 255.0
    return {
        'heatmap': heatmap,
        'keypoint_depth': keypoint_depth,
        'chamfer_offset_map': chamfer_offset_map,
        'projection_pts_info': (projected_pts, projected_pts_label, H, W)
    }

def lift_points_to_space(pts, depth, Twc, camera_intrinsics):
    """
    lift from image plane to 3D space
    param:
        pts: Nx2 numpy array, 2D points in image plane
        Twc: 3x4 numpy array
            Rwc = Twc[:3, :3]
            twc = Twc[:3, 3]
            X_world = Rwc * X_camera + twc
        camera_intrinsics: list [fx, fy, cx, cy, ...]
    return:
        pts_w: Mx3 numpy array, (M <= N, when corresponding depth is 0)
        pts_depth: M numpy array, depth of points
        indices: Mx1 numpy array, indices of points(some points with invalid depth has been removed)
    """
    if pts.size == 0:
        return np.empty((0, 3)), np.empty((0, 1), dtype=np.int)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    Rwc = Twc[:3, :3]
    twc = Twc[:3, 3]
    fx, fy, cx, cy, *_ = camera_intrinsics
    N = pts.shape[0]
    pts_cam = np.empty((N, 3), dtype=pts.dtype)
    pts_depth = np.empty((N), dtype=pts.dtype)
    indices = np.empty((N, 1), dtype=np.int)
    count = 0
    for i in range(N):
        pt = pts[i]
        d = depth[int(round(pt[1])), int(round(pt[0]))]
        if d <= 0:
            continue
        pt_cam = np.asarray([[(pt[0] - cx) / fx, (pt[1] - cy) / fy, 1]]) * d
        pts_cam[count, :] = pt_cam
        indices[count, :] = i
        pts_depth[count] = d
        count += 1
    pts_cam = pts_cam[:count, :]
    indices = indices[:count, :]
    pts_depth = pts_depth[:count]
    pts_w = np.transpose(np.matmul(Rwc, np.transpose(pts_cam))) + twc
    return pts_w, pts_depth, indices

def lift_heatmap_depth_to_space(input):
    conf, area_name, single_heatmap, single_depth, Twc, camera_intrinsics = input
    pts = heatmap_to_pts(single_heatmap, depth=single_depth, conf_thresh=conf['conf_thresh'], nms_dist=conf['nms_dist'])
    pts = np.transpose(pts) # Nx3 (x, y, weight)
    if pts.shape[0] == 0:
        return (area_name, np.empty((0, 5)), np.empty((0, 2)))
    pts_w, pts_depth, indices = lift_points_to_space(pts[:,:2], single_depth, Twc, camera_intrinsics)
    pts_w_weight_depth = np.concatenate([pts_w, pts[indices, 2], pts_depth[:, None]], axis=1) # Nx5 (x, y, z, weight, depth)
    return (area_name, pts_w_weight_depth, np.squeeze(pts[indices, :2])) # area_name, 3D point weights depths, 2D point coordinates xy

# heatmap and coord
def lift_heatmap_coord_depth_to_space(input):
    conf, area_name, single_heatmap, single_coord, single_depth, Twc, camera_intrinsics = input
    # nearest downsample to heatmap
    H, W = single_heatmap.shape
    single_depth_fit = torch.nn.functional.interpolate(torch.from_numpy(single_depth[None, None, ...]), size=(H, W), mode='nearest').squeeze().numpy()
    pts = heatmap_to_pts(single_heatmap, depth=single_depth_fit, conf_thresh=conf['conf_thresh'], nms_dist=conf['nms_dist'])
    pts = np.transpose(pts) # Nx3 (x in heatmap, y in heatmap, weight)
    if pts.shape[0] == 0:
        return (area_name, np.empty((0, 5)), np.empty((0, 2)))
    # get coordinate from single_coord
    pts_xy = single_coord[:, pts[:, 1].round().astype(int), pts[:, 0].round().astype(int)]
    pts[:, :2] = pts_xy.transpose(1, 0)
    pts_w, pts_depth, indices = lift_points_to_space(pts[:,:2], single_depth, Twc, camera_intrinsics)
    pts_w_weight_depth = np.concatenate([pts_w, pts[indices, 2], pts_depth[:, None]], axis=1) # Nx5 (x, y, z, weight, depth)
    return (area_name, pts_w_weight_depth, np.squeeze(pts[indices, :2])) # area_name, 3D point weights depths, 2D point coordinates xy

def get_descriptors_from_feature_map(feats, coords_xy, H, W):
    C, _, _ = feats.shape
    grids = torch.from_numpy(coords_xy).float().reshape(1, -1, 1, 2)
    grids[:,:,:,0] *= 2/(W-1)
    grids[:,:,:,1] *= 2/(H-1)
    grids -= 1
    single_feats = torch.nn.functional.grid_sample(feats[None, ...], grids.to(feats.device), mode='bilinear', padding_mode='border')
    single_feats = torch.nn.functional.normalize(single_feats, p=2, dim=1)
    return single_feats.reshape(C, -1).transpose(1, 0)


def select_descriptors(descriptors):
    if descriptors.shape[0] == 1:
        return descriptors[0]
    # select via least median distance
    # dist_matrix = squareform(pdist(descriptors))
    sim_matrix = descriptors @ descriptors.transpose(1, 0)
    sim_matrix = sim_matrix[~np.eye(sim_matrix.shape[0],dtype=bool)].reshape(sim_matrix.shape[0],-1)
    # idx = np.argmin(np.median(dist_matrix, axis=1))
    idx = np.argmax(np.median(sim_matrix, axis=1))
    # print(idx, np.median(sim_matrix, axis=1))
    return descriptors[idx]


def chamfer_distance_simple(pt1, pt2):
    if pt1 is None or pt1.size == 0 or pt2 is None or pt2.size == 0:
        pt2_offset = np.zeros((pt2.shape[0], 4))
        pt2_offset[:, :3] = pt2[:,:3]
        return 0.0, None, pt2_offset
    # TODO: not optimized
    pt1 = pt1[:,:3]
    pt2 = pt2[:,:3]
    dist_sum = 0
    pt1_offset = np.empty((pt1.shape[0], 4))
    pt2_offset = np.empty((pt2.shape[0], 4))
    for i, pt in enumerate(pt1):
        dist = np.min(np.linalg.norm(pt2 - pt, axis=1, keepdims=False))
        pt1_offset[i] = np.concatenate([pt, [dist]])
        dist_sum += dist
    for i, pt in enumerate(pt2):
        dist = np.min(np.linalg.norm(pt1 - pt, axis=1, keepdims=False))
        pt2_offset[i] = np.concatenate([pt, [dist]])
        dist_sum += dist
    return dist_sum / (pt1.shape[0] + pt2.shape[0]), pt1_offset, pt2_offset

def save_keypints_to_file(keypoints, filename):
    util.ensure_dir(os.path.dirname(filename))
    save_dict = {}
    for k, v in keypoints.items():
        save_dict[k] = v.tolist()
    if '.json' in filename:
        util.write_json(save_dict, filename)
    else:
        util.write_pkl(save_dict, filename)

def load_keypints_from_file(filename):
    keypoints = util.read_json(filename)
    for k, v in keypoints.items():
        keypoints[k] = np.asarray(v)
    return keypoints

def remove_border_for_batch_heatmap(batch_heatmap_np, border_size=30):
    batch_heatmap_np[:,:,:border_size,:] = 0.0
    batch_heatmap_np[:,:,-border_size:,:] = 0.0
    batch_heatmap_np[:,:,:,:border_size] = 0.0
    batch_heatmap_np[:,:,:,-border_size:] = 0.0
    return batch_heatmap_np

def save_pointcloud(pcd_np, pcd_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    o3d.io.write_point_cloud(pcd_path, pcd)

def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    """
    return torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """

    diffs = all_diffs(a, b)
    if metric == 'sqeuclidean':
        return torch.sum(diffs ** 2, dim=-1)
    elif metric == 'euclidean':
        return torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-12)
    elif metric == 'cityblock':
        return torch.sum(torch.abs(diffs), dim=-1)
    else:
        raise NotImplementedError(
            'The following metric is not implemented by `cdist` yet: {}'.format(metric))

def transform_pcd_pose(pcd, mat):
    mat = np.linalg.inv(mat)
    R = mat[:3, :3]
    t = mat[:3, 3]
    if type(pcd) is np.ndarray:
        points = np.matmul(np.transpose(R), np.transpose(pcd - t))
        points = np.transpose(points)
        return points
    elif type(pcd) is torch.Tensor:
        R = torch.from_numpy(R).float().to(pcd.device)
        t = torch.from_numpy(t).float().to(pcd.device)
        points = torch.matmul(R.transpose(1, 0), (pcd - t).transpose(1, 0))
        points = points.transpose(1, 0).contiguous()
        return points

def rotation_matrix(augment_axis, augment_rotation):
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if augment_axis == 1:
        return random.choice([Rx, Ry, Rz]) 
    return Rx @ Ry @ Rz
    
def translation_matrix(augment_translation):
    T = np.random.rand(3) * augment_translation
    return T

def rand_transformation_matrix(augment_rot_axis=1, augment_rotation=1.0, augment_translation=0.5):
    gt_trans = np.eye(4).astype(np.float32)
    R = rotation_matrix(augment_rot_axis, augment_rotation)
    T = translation_matrix(augment_translation)
    gt_trans[0:3, 0:3] = R
    gt_trans[0:3, 3] = T
    return gt_trans
