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

import random

SEED = 10
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def compute_color_from_features(list_feat, method='PCA', cmap='Spectral'):
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
        colors = plt.cm.get_cmap(cmap)(tsne_results.squeeze())[:, :3]
        assert(len(list_feat) == 2)
        N_1 = list_feat[0].shape[0]
        list_color = [colors[:N_1], colors[N_1:]]
        return list_color

if __name__ == '__main__':
    scene = '7-scenes-redkitchen'
    id1 = 8
    result_dir = 'saved/0_paper_selected_3dmatch/2d_multi_tower/log_hf_1024_003916_H3DMultiTower_X3DMatchFragmentDataLoader_fcgf_voxel_0.02_epoch99_no_nms_73cd8b'
    result_fcgf_dir = 'saved/0_paper_selected_3dmatch/fcgf_0.02_rand_score/log_hf_1024_150751_H3DNetCoordVote_X3DMatchFragmentDataLoader_pos_neg_256_1024_symmetric_contras_epoch99_fcgf_torch_points_voxel_0.02_rand_score_1a3091'

    gt_base_dir = 'data/3dmatch/geometric_registration_adaptive'
    pcd_1_path = '{}/{}/cloud_bin_{}.ply'.format(gt_base_dir, scene, id1)
    pcd_1_np = np.asarray(o3d.io.read_point_cloud(pcd_1_path, print_progress=True).points)

    desc_prefix = 'D3Feat.' if 'D3Feat' in result_dir else ''
    # desc_prefix = ''

    # corr_path = '{}/pred_result/{}/results/cloud_bin_{}_cloud_bin_{}.corr.npy'.format(result_dir, scene, id1, id2)
    kpt_1_path = '{}/keypoints/{}/cloud_bin_{}.npy'.format(result_dir, scene, id1)
    scores_1_path = '{}/scores/{}/cloud_bin_{}.npy'.format(result_dir, scene, id1)
    desc_1_path = '{}/descriptors/{}/cloud_bin_{}.{}npy'.format(result_dir, scene, id1, desc_prefix)
    kpt1 = np.load(kpt_1_path)
    desc1 = np.load(desc_1_path)
    scores1 = np.squeeze(np.load(scores_1_path))

    kpt_2_path = '{}/keypoints/{}/cloud_bin_{}.npy'.format(result_fcgf_dir, scene, id1)
    scores_2_path = '{}/scores/{}/cloud_bin_{}.npy'.format(result_fcgf_dir, scene, id1)
    desc_2_path = '{}/descriptors/{}/cloud_bin_{}.{}npy'.format(result_fcgf_dir, scene, id1, desc_prefix)
    kpt2 = np.load(kpt_2_path)
    desc2 = np.load(desc_2_path)
    scores2 = np.squeeze(np.load(scores_2_path))


    print(desc1.shape)


    # o3d.visualization.RenderOption.line_width = 8.0
    visualizer = O3dVisualizer()
    # visualizer.add_o3d_geometry(pcd_orig_o3d)

    color_type = 'score'
    # color_type = 'desc'


    print((scores1 > 0.5).sum() / float(scores1.shape[0]))
    print((scores1>0).sum())


    radius_remove = False
    radius = 0.5
    center_point = np.array([0.62, 0.29, 1.18]) # 3DMatch redkitchen 3

    if radius_remove:
        kpt2 = kpt2[np.linalg.norm(kpt2 - center_point, axis=1) < radius]

    # draw fcgf points
    # color_pca = compute_color_from_features([desc2,desc2[:2]], method='tsne', cmap='Spectral')
    # color = color_pca[0]
    color = np.array([0.7, 0.7, 0.7])
    # visualizer.add_np_points(kpt2, size=0.004, color=color, resolution=20, with_normal=True)
    visualizer.add_np_points(kpt2, size=0.005, color=color, resolution=20, with_normal=True)
    
    print(f'score min{scores1.min()}, max{scores1.max()}')
    color = map_to_color(scores1, cmap='Reds', vmin=-0.3, vmax=1)
    rand1 = np.random.rand(scores1.shape[0])
    color = map_to_color(rand1, cmap='Paired')
    # color = np.array([0.8, 0.0, 0.0])

    if radius_remove:
        ind = np.linalg.norm(kpt1 - center_point, axis=1) < radius
        kpt1 = kpt1[ind]
        color = color[ind]
    # visualizer.add_np_points(kpt1, size=0.015, color=color, resolution=20, with_normal=True)
    # visualizer.add_np_points(kpt1, size=0.0065, color=color, resolution=20, with_normal=True)


    # visualizer.add_np_points(pcd_1_np, size=0.003, color=map_to_color(pcd_1_np[:, 0], cmap='RdYlBu'), resolution=5)

    
    visualizer.run_visualize()
