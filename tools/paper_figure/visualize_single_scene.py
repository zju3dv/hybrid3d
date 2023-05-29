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

def compute_color_from_features(list_feat, method='PCA', cmap='Spectral'):
    feats = np.vstack(list_feat)
    if method == 'PCA':
        pca = PCA(n_components=1)
        pca.fit(feats)
        min_col = pca.transform(feats).min(axis=0)
        max_col = pca.transform(feats).max(axis=0)
        list_color = []
        for feat in list_feat:
            color = pca.transform(feat)
            color = (color - min_col) / (max_col - min_col)
            color = plt.cm.get_cmap(cmap)(color.squeeze())[:, :3]
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
        colors = plt.cm.get_cmap(cmap)(1-tsne_results.squeeze())[:, :3]
        assert(len(list_feat) == 2)
        N_1 = list_feat[0].shape[0]
        list_color = [colors[:N_1], colors[N_1:]]
        return list_color

if __name__ == '__main__':
    scene = '7-scenes-redkitchen'
    id1 = 8
    result_dir = '/home/ybbbbt/Developer/hybrid_feature/saved/0_paper_selected_test/figure_only/log_hf_1101_165354_H3DNetCoordVote_X3DMatchFragmentDataLoader_pos_neg_256_256_601_subset_no_balanced_layer_epoch99_visualize_score_097374'
    # result_dir = 'saved/0_paper_selected_test/fcgf_0.02_rand_score/log_hf_1024_150751_H3DNetCoordVote_X3DMatchFragmentDataLoader_pos_neg_256_1024_symmetric_contras_epoch99_fcgf_torch_points_voxel_0.02_rand_score_1a3091'

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


    print(desc1.shape)


    # o3d.visualization.RenderOption.line_width = 8.0
    visualizer = O3dVisualizer()
    # visualizer.add_o3d_geometry(pcd_orig_o3d)

    # color_type = 'score'
    # color_type = 'desc'


    print((scores1 > 0.5).sum() / float(scores1.shape[0]))
    print((scores1>0).sum())


    # color from desc
    # color_pca = compute_color_from_features([desc1,desc1[:2]], method='tsne', cmap='Spectral')
    # color = color_pca[0]
    
    # color from score
    print(f'score min{scores1.min()}, max{scores1.max()}')
    color = map_to_color(scores1, cmap='coolwarm')

    color = np.array([0.5, 0.5, 0.5])

    # visualizer.add_np_points(kpt1, size=0.007, color=color, resolution=20, with_normal=True)
    visualizer.add_np_points(kpt1, size=0.005, color=color, resolution=20)


    # visualizer.add_np_points(pcd_1_np, size=0.003, color=map_to_color(pcd_1_np[:, 0], cmap='RdYlBu'), resolution=5)

    
    visualizer.run_visualize()
