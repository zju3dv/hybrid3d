import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

import sys
import os

sys.path.append(os.getcwd())  # noqa
import utils.vis_3d as vis_3d
from utils.sampler import FarthestSampler, nms_3D

def colorize_open3d_pcd(pcd):
    # use z axis to map color
    pts = np.asarray(pcd.points)
    pts_z_min = np.min(pts[:, 2])
    pts_z_max = np.max(pts[:, 2])
    colors = plt.cm.viridis(
        (pts[:, 2] - pts_z_min) / (pts_z_max - pts_z_min))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def fps_clustering(keypoints, cluster_radius, nms_space, max_K, iteration=10):
    if keypoints.shape[0]>500000:
        indices = np.random.choice(keypoints.shape[0], 500000, replace=False)  
        keypoints = keypoints[indices, ...]

    # voxel downsampling, uniform density
    sampler = FarthestSampler()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(keypoints)
    pcd = pcd.voxel_down_sample(voxel_size = 0.03)
    uniformed_keypoints = np.asarray(pcd.points)
    farthest_pts = sampler.sample(uniformed_keypoints, max_K)

    clustered_pts = np.empty((0, 3))
    for i in range(iteration):
        # for the first iteration, candidate nodes from fps points
        if i == 0:
            candidate_nodes = farthest_pts.copy()
        else:
            candidate_nodes = clustered_pts.copy()
        # reset clustered_pts
        clustered_pts = np.empty((max_K, 3))
        count = 0
        for idx, pt_node in enumerate(candidate_nodes):
            indices = np.argwhere(np.linalg.norm(keypoints - pt_node, axis=1) < cluster_radius)
            # indices = np.squeeze(indices)
            center = np.mean(keypoints[indices], axis=0)
            if nms_space >= 0 and count > 0:
                # if pt node near the existing point, we skip it
                if np.min(np.linalg.norm(clustered_pts[:count] - center, axis=1)) < nms_space:
                    continue
                # if not enough points, we skip?
            clustered_pts[count, :] = center
            count += 1
        clustered_pts = clustered_pts[:count, :]
    return clustered_pts

def random_clustering(keypoints, cluster_radius, nms_space, max_K, iteration=10):
    indices = np.random.choice(keypoints.shape[0], max_K, replace=False)  
    random_pts = keypoints[indices, ...]
    clustered_pts = np.empty((0, 3))
    for i in range(iteration):
        # for the first iteration, candidate nodes from fps points
        if i == 0:
            candidate_nodes = random_pts.copy()
        else:
            candidate_nodes = clustered_pts.copy()
        # reset clustered_pts
        clustered_pts = np.empty((max_K, 3))
        count = 0
        for idx, pt_node in enumerate(candidate_nodes):
            indices = np.argwhere(np.linalg.norm(keypoints - pt_node, axis=1) < cluster_radius)
            # indices = np.squeeze(indices)
            center = np.mean(keypoints[indices], axis=0)
            if nms_space >= 0 and count > 0:
                # if pt node near the existing point, we skip it
                if np.min(np.linalg.norm(clustered_pts[:count] - center, axis=1)) < nms_space:
                    continue
                # if not enough points, we skip?
            clustered_pts[count, :] = center
            count += 1
        clustered_pts = clustered_pts[:count, :]
    return clustered_pts

def mean_clustering(points_info, cluster_radius, nms_space, max_K, min_weight_thresh, initial_node_method, iteration, min_pt_each_cluster=3):
    '''
    initial_node_method: fps or nms
    '''
    keypoints, weights = points_info[:, :3], points_info[:, 3]
    if max_K <= 0:
        # print('Warning max_K == 0!')
        max_K = 1
    clustered_pts = np.empty((max_K, 3))
    for i in range(iteration):
        # for the first iteration, candidate nodes from fps points
        if i == 0:
            candidate_nodes, _, _ = nms_3D(keypoints, weights, nms_space, max_K)
        else:
            candidate_nodes = clustered_pts.copy()
        # reset clustered_pts
        clustered_pts = np.empty((max_K, 3))
        clustered_weights = np.empty((max_K))
        indices_list = []
        count = 0
        for idx, pt_node in enumerate(candidate_nodes):
            indices = np.argwhere(np.linalg.norm(keypoints - pt_node, axis=1) < cluster_radius)
            indices = indices.reshape(-1)
            if indices.size < min_pt_each_cluster:
                continue
            selected_keypoints = keypoints[indices]
            selected_weights = weights[indices]
            centroid_weight = np.mean(selected_weights)
            center = np.mean(selected_keypoints, axis=0).reshape(1, 3)
            clustered_pts[count, :] = center
            clustered_weights[count] = centroid_weight
            indices_list.append(indices)
            count += 1
        clustered_pts = clustered_pts[:count, :]
        clustered_weights = clustered_weights[:count]
        # nms
        if nms_space >= 0 and clustered_pts.shape[0] > 0:
            clustered_pts, clustered_weights, ind = nms_3D(clustered_pts, clustered_weights, nms_space, max_K)
            indices_list = [indices_list[i] for i in ind]
    return clustered_pts, clustered_weights, indices_list

def weighted_clustering(points_info, cluster_radius, nms_space, max_K, min_weight_thresh, initial_node_method, iteration, min_pt_each_cluster=3):
    '''
    initial_node_method: fps or nms
    FIXME: indices_list may not be okey!
    '''
    winner_take_all = False
    choose_closest_points = False
    only_preserve_fragment_wise_covisible_points = False

    keypoints, weights = points_info[:, :3], points_info[:, 3]

    # check if has depth    
    has_depths = False
    if points_info.shape[1] >= 5:
        depths = points_info[:, 4]
        has_depths = True
    # check if has fragment id
    has_fragment_id = False
    if points_info.shape[1] >= 6:
        fragment_ids = points_info[:, 5]
        has_fragment_id = True

    if choose_closest_points and has_depths:
        iteration = 1

    if max_K <= 0:
        # print('Warning max_K == 0!')
        max_K = 1
    clustered_pts = np.empty((max_K, 3))
    for i in range(iteration):
        # for the first iteration, candidate nodes from fps points
        if i == 0:
            if initial_node_method == 'fps':
                sampler = FarthestSampler()
                candidate_nodes = sampler.sample(keypoints, max_K)
            elif initial_node_method == 'nms':
                candidate_nodes, _, _ = nms_3D(keypoints, weights, nms_space, max_K)
        else:
            candidate_nodes = clustered_pts.copy()
        # reset clustered_pts
        clustered_pts = np.empty((max_K, 3))
        clustered_weights = np.empty((max_K))
        clustered_pts_label = np.empty((max_K))
        indices_list = []
        count = 0
        if winner_take_all:
            exist_indices = np.empty((0), np.int)
        for idx, pt_node in enumerate(candidate_nodes):
            indices = np.argwhere(np.linalg.norm(keypoints - pt_node, axis=1) < cluster_radius)
            indices = indices.reshape(-1)
            if winner_take_all:
                indices = np.setdiff1d(indices, exist_indices, assume_unique=True)
                exist_indices = np.concatenate([exist_indices, indices])
            if indices.size < min_pt_each_cluster:
                continue
            selected_keypoints = keypoints[indices]
            selected_weights = weights[indices]

            # when we enabled only perserve fragment wise covisible points
            if only_preserve_fragment_wise_covisible_points and has_fragment_id:
                selected_fragment_ids = fragment_ids[indices]
                if np.unique(selected_fragment_ids).size < 2:
                    continue
                
            if choose_closest_points and has_depths:
                selected_depths = depths[indices]
                closest_pt_idx = np.argmin(selected_depths)
                center = selected_keypoints[closest_pt_idx].reshape(1, 3)
                centroid_weight = selected_weights[closest_pt_idx]
            else:
                # we choose top k 
                top_k = max(1, int(selected_weights.shape[0] * 0.2))
                centroid_weight = np.mean(np.sort(selected_weights)[-top_k:])
                # centroid_weight = np.mean(selected_weights)
                if centroid_weight < min_weight_thresh:
                    continue
                weighted_keypoints = np.multiply(selected_keypoints, selected_weights.reshape(-1, 1))
                center = np.sum(weighted_keypoints, axis=0) / np.sum(selected_weights, axis=0)
                center = center.reshape(1, 3)

            if has_fragment_id:
                selected_fragment_ids = fragment_ids[indices]
                unique_ids = np.unique(selected_fragment_ids)
                # 0: left fragment visible, 1: right fragment visible, 2: both fragment covisible
                # print(unique_ids)
                if unique_ids.size == 2:
                    clustered_pts_label[count] = 2
                else:
                    clustered_pts_label[count] = unique_ids

            clustered_pts[count, :] = center
            clustered_weights[count] = centroid_weight
            indices_list.append(indices)
            count += 1
        clustered_pts = clustered_pts[:count, :]
        clustered_weights = clustered_weights[:count]
        clustered_pts_label = clustered_pts_label[:count]
        # nms
        if nms_space >= 0 and clustered_pts.shape[0] > 0:
            clustered_pts, clustered_weights, ind = nms_3D(clustered_pts, clustered_weights, nms_space, max_K)
            indices_list = [indices_list[i] for i in ind]
            clustered_pts_label = clustered_pts_label[ind]
    clustered_pts = np.concatenate([clustered_pts, clustered_pts_label[:, None]], axis=1)
    return clustered_pts, clustered_weights, indices_list

def weighted_voxel_clustering(keypoints, weights, cluster_radius, nms_space, max_K, min_weight_thresh, initial_node_method, iteration):
    '''
    initial_node_method: fps or nms
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(keypoints)
    
    pcd, cubic_id, indices = pcd.voxel_down_sample_and_trace(0.03, np.min(keypoints, axis=0), np.max(keypoints, axis=0))
    indices = [x[0] for x in indices]
    idx = np.asarray(indices)
    keypoints = keypoints[idx, ...]
    weights = weights[idx, ...]

    clustered_pts = np.empty((max_K, 3))
    for i in range(iteration):
        # for the first iteration, candidate nodes from fps points
        if i == 0:
            if initial_node_method == 'fps':
                sampler = FarthestSampler()
                candidate_nodes = sampler.sample(keypoints, max_K)
            elif initial_node_method == 'nms':
                candidate_nodes, _, _ = nms_3D(keypoints, weights, nms_space, max_K)
        else:
            candidate_nodes = clustered_pts.copy()
        # reset clustered_pts
        clustered_pts = np.empty((max_K, 3))
        clustered_weights = np.empty((max_K))
        count = 0
        for idx, pt_node in enumerate(candidate_nodes):
            indices = np.argwhere(np.linalg.norm(keypoints - pt_node, axis=1) < cluster_radius)
            indices = indices.reshape(-1)
            if (indices.size == 0):
                continue
            selected_keypoints = keypoints[indices]
            selected_weights = weights[indices]
            # we choose top k 
            top_k = max(1, int(selected_weights.shape[0] * 0.2))
            centroid_weight = np.mean(np.sort(selected_weights)[-top_k:])
            # centroid_weight = np.mean(selected_weights)
            if centroid_weight < min_weight_thresh:
                continue
            weighted_keypoints = np.multiply(selected_keypoints, selected_weights.reshape(-1, 1))
            center = np.sum(weighted_keypoints, axis=0) / np.sum(selected_weights, axis=0)
            center = center.reshape(1, 3)
            clustered_pts[count, :] = center
            clustered_weights[count] = centroid_weight
            count += 1
        clustered_pts = clustered_pts[:count, :]
        clustered_weights = clustered_weights[:count]
        # nms
        if nms_space >= 0 and clustered_pts.shape[0] > 0:
            clustered_pts, clustered_weights, _ = nms_3D(clustered_pts, clustered_weights, nms_space, max_K)
    return clustered_pts

def K_means_clustering(keypoints, max_K, iteration=10):
    if keypoints.shape[1] == 3:
        kmeans = MiniBatchKMeans(n_clusters=max_K, max_no_improvement=iteration, init_size=max_K * 3).fit(keypoints)
    elif keypoints.shape[1] == 4:
        kmeans = MiniBatchKMeans(n_clusters=max_K, max_no_improvement=iteration, init_size=max_K * 3).fit(keypoints[:, :3], sample_weight=keypoints[:,3])
    return kmeans.cluster_centers_

if __name__ == '__main__':
    # pcd_orig_path = 'data/redwood_lidar/lidar_resampled/aligned_low_apartment/apartment_simple.ply'
    # pcd_orig_o3d = o3d.io.read_point_cloud(pcd_orig_path, print_progress=True)
    # pcd_orig_o3d = pcd_orig_o3d.voxel_down_sample(voxel_size=0.05)
    # pcd_orig_np = np.asarray(pcd_orig_o3d.points)

    pcd_path = 'data/redwood_lidar/initial_keypoints/usip/apartment.ply'
    pcd_o3d = o3d.io.read_point_cloud(pcd_path, print_progress=True)
    pcd_np = np.asarray(pcd_o3d.points)
    clustered_pts = fps_clustering(pcd_np, 0.2, 0.2, 1000)
    print(clustered_pts.shape)

    ax = vis_3d.plot_pc(pcd_np, size=3)
    # ax = vis_3d.plot_pc(pcd_orig_np, size=3)
    ax = vis_3d.plot_pc(clustered_pts, ax=ax, size=30, color=np.asarray([1, 0, 0]).reshape((1, 3)))
    plt.show()
    # fps_clustering(clustered_pts, 0.2, 0.2, 1000)

    # ax = vis_3d.plot_pc(pcd_np, size=3)
    # ax = vis_3d.plot_pc(pcd_orig_np, size=3)
    # ax = vis_3d.plot_pc(clustered_pts, ax=ax, size=30, color=np.asarray([1, 0, 0]).reshape((1, 3)))

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(clustered_pts.astype(np.float32))
    # ply_path = os.path.join(output_path, '%s.ply' % scene_name)
    # o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True, compressed=False, print_progress=True)
