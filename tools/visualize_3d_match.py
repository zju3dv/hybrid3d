import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.getcwd())  # noqa
import utils.vis_3d as vis_3d
import utils.utils_algo as utils_algo
import utils.util as util
from utils.vis_3d_o3d import map_to_color, O3dVisualizer
from utils.sampler import nms_3D
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def compute_color_from_features(list_feat, method='PCA'):
    feats = np.vstack(list_feat)
    if method == 'PCA':
        pca = PCA(n_components=3)
        pca.fit(feats)
        min_col = pca.transform(feats).min(axis=0)
        max_col = pca.transform(feats).max(axis=0)
        list_color = []
        for feat in list_feat:
            color = pca.transform(feat)
            color = (color - min_col) / (max_col - min_col)
            # color = plt.cm.Spectral(color.squeeze())[:, :3]
            list_color.append(color)
        return list_color
    elif method == 'tsne':
        def embed_tsne(data):
            """
            N x D np.array data
            """
            tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
            tsne_results = tsne.fit_transform(data)
            tsne_results = np.squeeze(tsne_results)
            tsne_min = np.min(tsne_results)
            tsne_max = np.max(tsne_results)
            return (tsne_results - tsne_min) / (tsne_max - tsne_min)
        tsne_results = embed_tsne(feats)
        colors = plt.cm.Spectral(tsne_results.squeeze())[:, :3]
        assert(len(list_feat) == 2)
        N_1 = list_feat[0].shape[0]
        list_color = [colors[:N_1], colors[N_1:]]

if __name__ == '__main__':
    scene = sys.argv[1]
    id1 = sys.argv[2]
    id2 = sys.argv[3]
    result_dir = sys.argv[4]
    gt_base_dir = 'data/3dmatch/geometric_registration_adaptive'
    # gt_base_dir = 'data/redwood_lidar/fragments_adaptive'
    # pcd_1_path = '{}/{}/cloud_bin_{}.ply'.format(gt_base_dir, scene, id1)
    # pcd_2_path = '{}/{}/cloud_bin_{}.ply'.format(gt_base_dir, scene, id2)
    # pcd_1_np = np.asarray(o3d.io.read_point_cloud(pcd_1_path, print_progress=True).points)
    # pcd_2_np = np.asarray(o3d.io.read_point_cloud(pcd_2_path, print_progress=True).points)

    desc_prefix = 'D3Feat.' if 'D3Feat' in result_dir else ''
    # desc_prefix = ''

    # corr_path = '{}/pred_result/{}/results/cloud_bin_{}_cloud_bin_{}.corr.npy'.format(result_dir, scene, id1, id2)
    kpt_1_path = '{}/keypoints/{}/cloud_bin_{}.npy'.format(result_dir, scene, id1)
    kpt_2_path = '{}/keypoints/{}/cloud_bin_{}.npy'.format(result_dir, scene, id2)
    scores_1_path = '{}/scores/{}/cloud_bin_{}.npy'.format(result_dir, scene, id1)
    scores_2_path = '{}/scores/{}/cloud_bin_{}.npy'.format(result_dir, scene, id2)
    desc_1_path = '{}/descriptors/{}/cloud_bin_{}.{}npy'.format(result_dir, scene, id1, desc_prefix)
    desc_2_path = '{}/descriptors/{}/cloud_bin_{}.{}npy'.format(result_dir, scene, id2, desc_prefix)
    kpt1 = np.load(kpt_1_path)
    kpt2 = np.load(kpt_2_path)
    desc1 = np.load(desc_1_path)
    desc2 = np.load(desc_2_path)
    scores1 = np.squeeze(np.load(scores_1_path))
    scores2 = np.squeeze(np.load(scores_2_path))
    print(desc1.shape)
    # print(desc1)
    # print(kpt1.shape, scores1.shape)
    # kpt1, scores1, _ = nms_3D(kpt1, scores1, 0.01, method='max')
    # print(kpt1.shape, scores1.shape)
    # print('0:', desc2[0])
    # print('132:', desc2[132])
    # exit(0)

    # compute score from average negative
    # desc1_sample = desc1.copy()
    # rand_idx = np.arange(desc1_sample.shape[0])
    # np.random.shuffle(rand_idx)
    # desc1_sample = desc1_sample[rand_idx[:1024], :]
    # desc_diff = utils_algo.cdist(torch.from_numpy(desc1), torch.from_numpy(desc1_sample))
    # average_negative = (desc_diff.sum(-1)) / (desc_diff.shape[-1] - 1)
    # print(average_negative.shape,scores1.shape)
    # # exit(0)
    # scores1 = average_negative.numpy()

    # visualize neighbor
    # kpt_select = 4000
    # dist_diff = utils_algo.cdist(torch.from_numpy(kpt1), torch.from_numpy(kpt1[kpt_select]))
    # scores1 = np.zeros_like(scores1)
    # mask = dist_diff.squeeze()<0.0625
    # scores1[mask] = 1

    # exit(0)

    # kpt1 = kpt1[:1024]
    # desc1 = desc1[:1024]
    # scores1 = scores1[:1024]

    # corr = np.load(corr_path)

    # select_kpt = 60
    # kpt1 = kpt1[select_kpt:select_kpt+1, ...]
    # desc1 = desc1[select_kpt:select_kpt+1, ...]
    # print(desc2.shape, desc1.shape)
    # sim = np.einsum('ij,kj->i', desc2, desc1, optimize = True)
    # print(np.sort(sim))
    # # print('desc1',desc1)
    # idx = (-sim).argsort()[:2]
    # print(idx)
    # # print('desc2', desc2[idx])
    # print(np.min(sim), np.max(sim))
    
    
    N_corr = kpt1.shape[0]
    # N_corr = 25
    corr = np.empty((N_corr, 2), dtype=np.uint32)
    count = 0
    for i, d in enumerate(desc1):
        if i >= N_corr:
            break
        # dists = np.linalg.norm(desc2 - d, axis=1)
        # print(desc2.shape, d.shape)
        sim = np.einsum('ij,j->i', desc2, d, optimize = True)
        # print(np.min(sim), np.max(sim))
        match2_idx = np.argmax(sim)
        if sim[match2_idx] < 0.85: continue
        # print(sim[match2_idx])
        corr[count] = (i, match2_idx)
        count += 1
        if count > 100: break
    corr = corr[:count, ...]

    left_right_offset = np.array([1, 1, 0])
    # pcd_1_np += left_right_offset
    kpt1 += left_right_offset
    # pcd_2_np -= left_right_offset
    kpt2 -= left_right_offset

    o3d.visualization.RenderOption.line_width = 8.0
    visualizer = O3dVisualizer()
    # visualizer.add_o3d_geometry(pcd_orig_o3d)

    color_type = 'score'
    # color_type = 'desc'

    # visualizer.add_np_points(pcd_1_np, size=0.005, color=map_to_color(pcd_1_np[:, 0], cmap='viridis'))
    # visualizer.add_np_points(kpt1, size=0.02, color=(0, 1, 0))
    # print((scores1 > 0.5).sum() / float(scores1.shape[0]))
    print((scores1 > 0.5).sum() / float(scores1.shape[0]))
    # print((scores2 > 0.5).sum(), (scores2 < 0.5).sum())
    print((scores1>0).sum())
    if color_type == 'score':
        print(f'score min{scores1.min()}, max{scores1.max()}')
        color = map_to_color(scores1, cmap='coolwarm')
    elif color_type == 'desc':
        color_pca = compute_color_from_features([desc1, desc2], method='PCA')
        color = color_pca[0]
    visualizer.add_np_points(kpt1, size=0.02, color=color)

    if False: # draw correspondence
        line_pts = []
        line_indices = []
        for i, c in enumerate(corr):
            line_pts.append(kpt1[c[0]])
            line_pts.append(kpt2[c[1]])
            line_indices.append([i*2, i*2+1])
        visualizer.add_line_set(line_pts, line_indices)

    # visualizer.add_np_points(pcd_2_np, size=0.005, color=map_to_color(pcd_2_np[:, 0], cmap='viridis'))
    # visualizer.add_np_points(kpt2, size=0.02, color=(0, 1, 0))
    # mask = dists < 0.8
    # visualizer.add_np_points(kpt2, size=0.02, color=map_to_color(sim, cmap='coolwarm', vmin=0.5, vmax=1.0))
    if color_type == 'score':
        color = map_to_color(scores2, cmap='coolwarm')
    elif color_type == 'desc':
        color = color_pca[1]
    visualizer.add_np_points(kpt2, size=0.02, color=color)
    
    visualizer.run_visualize()
