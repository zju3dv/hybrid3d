from utils import vis, utils_algo, clustering
import os
import numpy as np
from utils.sampler import FarthestSampler, nms_3D

def write_clustering_keypoints_to_files(epoch, config, dataset, inferece_keypoints_raw):
    save_path = os.path.join(config.log_dir, 'kpts', 'kpts-offset-%03d.json' % epoch)
    utils_algo.save_keypints_to_file(dataset.keypoints_chamfer_offset, save_path)
    save_path = os.path.join(config.log_dir, 'kpts', 'kpts-%03d.json' % epoch)
    utils_algo.save_keypints_to_file(dataset.keypoints, save_path)
    save_path = os.path.join(config.log_dir, 'kpts', 'kpts-raw-%03d.json' % epoch)
    utils_algo.save_keypints_to_file(inferece_keypoints_raw, save_path)

def update_dataset_keypoints_via_clustering(epoch, config, kpt_dict, logger, writer, data_loader, write_tb=True, verbose=True, write_to_file=True):
    # clustering
    cluster_conf = config['trainer']['clustering']
    clustered_kpt_dict = {}
    clustered_kpt_chamfer_offset_dict = {}
    mean_chamfer_dist = 0
    # order by key, for better logging
    keys = list(kpt_dict.keys())
    keys.sort()
    for k in keys:
        v = kpt_dict[k]
        # clustering 3D points with weights
        clustered_kpt_dict[k], _, _ = getattr(clustering, cluster_conf['method'])(
            v, cluster_conf['cluster_radius'], cluster_conf['nms_space'], 
                data_loader.dataset.keypoints_max_K[k], cluster_conf['min_weight_thresh'], 'nms', cluster_conf['max_iteration'],
                    cluster_conf['min_pt_each_cluster']
        )
        bidirectional_chamfer_dist, _, cluster_kpt_offset = utils_algo.chamfer_distance_simple(data_loader.dataset.keypoints.get(k), clustered_kpt_dict[k])
        # curr_chamfer_dist: unidirectional chamfer distance
        # when switching from initial keypoints to clutering keypoints, bidirectional_chamfer_dist may be large(e.g. due to ceiling points)
        curr_chamfer_dist = np.mean(cluster_kpt_offset[:, 3])
        mean_chamfer_dist += curr_chamfer_dist
        # prepare to save chamfer offset dict
        clustered_kpt_chamfer_offset_dict[k] = cluster_kpt_offset
        if verbose:
            logger.info('%s: kpt_num = %d | chamfer_dist = %.5f' % (k, clustered_kpt_dict[k].shape[0], curr_chamfer_dist))
    mean_chamfer_dist /= len(keys)
    # write chamfer distance to tensorboard
    if write_tb:
        writer.set_step(epoch)
        writer.add_scalar('mean_chamfer_distance', mean_chamfer_dist)
        if verbose:
            logger.info('mean_chamfer_distance = %.6f' % mean_chamfer_dist)
    # dump offset
    if write_to_file:
        save_path = os.path.join(config.log_dir, 'kpts', 'kpts-offset-%03d.json' % epoch)
        utils_algo.save_keypints_to_file(clustered_kpt_chamfer_offset_dict, save_path)
    # update keypoints
    data_loader.dataset.update_keypoints(clustered_kpt_dict)
    data_loader.dataset.update_keypoints_chamfer_offset(clustered_kpt_chamfer_offset_dict)
    return clustered_kpt_dict, mean_chamfer_dist

def update_dataset_keypoints_via_nms(epoch, config, kpt_dict, logger, writer, data_loader):
    # clustering
    cluster_conf = config['trainer']['clustering']
    clustered_kpt_dict = {}
    clustered_kpt_chamfer_offset_dict = {}
    mean_chamfer_dist = 0
    # order by key, for better logging
    keys = list(kpt_dict.keys())
    keys.sort()
    for k in keys:
        v = kpt_dict[k]
        clustered_kpt_dict[k], _, _ = nms_3D(v[:, :3], v[:, 3], cluster_conf['nms_space'], data_loader.dataset.keypoints_max_K[k])
        clustered_kpt_chamfer_offset_dict[k] = None
    # update keypoints
    data_loader.dataset.update_keypoints(clustered_kpt_dict)
    data_loader.dataset.update_keypoints_chamfer_offset(clustered_kpt_chamfer_offset_dict)
    return clustered_kpt_dict, mean_chamfer_dist
