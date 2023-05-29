import argparse
import collections
import open3d
import torch
import numpy as np
import random
from tqdm import tqdm
import copy
import multiprocessing as mp
import signal
import os
import sys
import cv2
sys.path.append('.')  # noqa
# import matplotlib.pyplot as plt
import matplotlib
import evaluation.superpoint as superpoint
from utils import clustering_strategy, utils_algo, clustering, vis, util

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

from parse_config import ConfigParser

def save_results(path, scene_name, fragment_idx, descriptors, keypoints, scores):
    if not descriptors is None:
        descriptors_save_path = os.path.join(path, 'descriptors', scene_name)
        util.ensure_dir(descriptors_save_path)
        np.save(os.path.join(descriptors_save_path, 'cloud_bin_{}.npy'.format(fragment_idx)), descriptors)
    keypoints_save_path = os.path.join(path, 'keypoints', scene_name)
    scores_save_path = os.path.join(path, 'scores', scene_name)
    util.ensure_dir(keypoints_save_path)
    util.ensure_dir(scores_save_path)
    np.save(os.path.join(keypoints_save_path, 'cloud_bin_{}.npy'.format(fragment_idx)), keypoints)
    np.save(os.path.join(scores_save_path, 'cloud_bin_{}.npy'.format(fragment_idx)), scores)

def main(config):
    from data_loader.data_loader_factory import get_data_loader_by_name

    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader_config = config['data_loader']
    data_loader = getattr(get_data_loader_by_name(data_loader_config['type']), data_loader_config['type'])(
        config=config,
        data_dir=data_loader_config['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0,
        num_workers=4,
        training=False)

    sp_front = superpoint.SuperPointFrontend('data/superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=True)
    config['trainer']['point_lifting']['nms_dist'] = 4
    config['trainer']['point_lifting']['conf_thresh'] = 0.015
    print('Superpoint loaded.')

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frame_batch_size = 8
    frame_skip_step = 1
    max_keypoints_per_fragment = 99999

    # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
    # Catch Ctrl+C / SIGINT and exit multiprocesses gracefully in python
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    mp_pool = mp.Pool(processes = 8)
    signal.signal(signal.SIGINT, original_sigint_handler)

    clustered_kpts_weights_dict = {}
    dump_vis_dir = str(config.log_dir) + '/vis'
    util.ensure_dir(dump_vis_dir)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, dynamic_ncols=True)):
            data_idx = batch['idx']
            fragment_key = batch['fragment_key']
            all_frame_data = batch['frame_data']
            pcd_data = batch['pcd_data']

            # inferece pts and weights
            all_params = []
            output_descriptors_list = []
            
            # generate batch list
            frame_list = list(range(random.randint(0, frame_skip_step), len(all_frame_data), frame_skip_step))
            batch_list = []
            for i in range(len(frame_list) // frame_batch_size):
                batch_list.append(frame_list[i*frame_batch_size:(i+1)*frame_batch_size])
            last_N = len(frame_list) % frame_batch_size
            if last_N > 0:
                batch_list.append(frame_list[-last_N:])

            # inference heatmaps
            for i_idx, indices in enumerate(batch_list):
                input_rgb = torch.stack([all_frame_data[i]['rgb'] for i in indices]).to(device, dtype=torch.float32)
                # input_sparse_depth = torch.stack([all_frame_data[i]['sparse_depth'] for i in indices]).to(device, dtype=torch.float32)
                # pcd_crsp_idx = torch.stack([all_frame_data[i]['pcd_crsp_idx'] for i in indices]).to(device, dtype=torch.long)
                target_depth = torch.stack([all_frame_data[i]['depth'] for i in indices]).to(device, dtype=torch.float32)

                _, _, H, W = input_rgb.shape

                # model_input_data = {
                #     'rgb' : input_rgb,
                #     'depth': input_sparse_depth,
                #     'pcd' : copy.deepcopy(pcd_data),
                #     'pcd_crsp_idx': pcd_crsp_idx,
                #     'fragment_key': fragment_key
                # }
                # output = model(model_input_data)
                # output_depth, output_heatmap = output['depth'], output['heatmap']
                output_descriptor = []
                output_heatmap = []
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_rgb_np = input_rgb.cpu().detach().numpy() * std.reshape([1, 3, 1, 1]) + mean.reshape([1, 3, 1, 1])
                for i_frame in range(input_rgb_np.shape[0]):
                    gray = cv2.cvtColor((input_rgb_np[i_frame].transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                    pts, desc_per_pt, desc, heatmap = sp_front.run(gray)
                    output_descriptor.append(desc)
                    output_heatmap.append(heatmap[None, ...])
                output_descriptor = torch.cat(output_descriptor, axis=0)
                # output_descriptor = torch.from_numpy(output_descriptor).float().to(device)
                output_heatmap = np.concatenate(output_heatmap, axis=0)
                output_depth = None
                # output_descriptor = output.get('descriptor')
                output_descriptors_list.extend([output_descriptor[i] for i in range(output_descriptor.shape[0])])
                
                depth_trunc = config['trainer']['clustering']['depth_trunc']
                target_depth[target_depth > depth_trunc] = 0
                # output_heatmap_np = output_heatmap.cpu().detach().numpy()
                output_heatmap_np = output_heatmap
                output_heatmap = torch.from_numpy(output_heatmap)[:, None, :, :]
                # print(output_heatmap.shape)
                # handling checkerboard artifact
                # output_heatmap_np = utils_algo.remove_border_for_batch_heatmap(output_heatmap_np)
                target_depth_np = target_depth.data.cpu().detach().numpy()

                batch_Twc = np.stack([all_frame_data[i]['camera_pose_Twc'] for i in indices])
                batch_camera_intrinsics = np.stack([all_frame_data[i]['camera_intrinsics'] for i in indices])
                conf = config['trainer']['point_lifting']
                batch_size = input_rgb.shape[0]
                all_params.extend([(
                        conf,
                        fragment_key,
                        np.squeeze(output_heatmap_np[i,...]),
                        np.squeeze(target_depth_np[i,...]),
                        batch_Twc[i, ...],
                        batch_camera_intrinsics[i, ...]) for i in range(batch_size)])
                # dump for visualize similarity
                if False and i_idx == 0:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    input_rgb_np = input_rgb.cpu().detach().numpy() * std.reshape([1, 3, 1, 1]) + mean.reshape([1, 3, 1, 1])
                    np.save(f'debug/rgb_{fragment_key}.npy', input_rgb_np)
                    np.save(f'debug/desc_{fragment_key}.npy', output_descriptor.cpu().detach().numpy())
                    # np.save(f'debug/xyz_{fragment_key}.npy', output['xyz'].cpu().detach().numpy())
                    # exit(0)
                # dump for visualize
                if i_idx == 0 and batch_idx % 20 == 0:
                    logger.debug('Image Dump: batch_idx:{}, fragment_key:{}, img_idx:{}'.format(
                        batch_idx, fragment_key, [all_frame_data[i]['idx'] for i in indices]))
                    vis_size = 4
                    figure = vis.save_fig_simple(
                        config, input_rgb[:vis_size, ...],
                        None if output_depth is None else output_depth[:vis_size, ...],
                        output_heatmap[:vis_size, ...],
                        target_depth[:vis_size, ...])
                    matplotlib.image.imsave('{}/{:05d}.jpg'.format(dump_vis_dir, batch_idx), figure)
            # get result from heatmap nms
            results = mp_pool.map_async(utils_algo.lift_heatmap_depth_to_space, all_params).get(60)
            pts_w_weight_depth_fragment = []
            descriptors = []
            for idx, (area_names, pts_w_weight, coords_xy) in enumerate(results):
                if np.isnan(np.sum(pts_w_weight)):
                    print('NaN found in {} , camera pose may contain NaN.'.format(area_names))
                    continue
                # get weighted 3D points
                pts_w_weight_depth_fragment.append(pts_w_weight)
                # get descriptor
                if len(output_descriptors_list) > 0:
                    descriptors.append(utils_algo.get_descriptors_from_feature_map(output_descriptors_list[idx], coords_xy, H, W))
            # concat 3D points_weights and descriptors
            pts_w_weight_depth_fragment = np.concatenate(pts_w_weight_depth_fragment, axis=0)
            has_descriptor = len(descriptors) > 0
            if has_descriptor > 0:
                descriptors = torch.cat(descriptors, dim=0).detach().cpu().numpy()


            clustered_kpts_weights = pts_w_weight_depth_fragment
            clustered_descriptors = descriptors

            # sort by weights
            new_order = np.argsort(-clustered_kpts_weights[:, 3])
            clustered_kpts_weights = clustered_kpts_weights[new_order]
            clustered_kpts_weights_dict[fragment_key] = clustered_kpts_weights
            clustered_descriptors = clustered_descriptors[new_order] if has_descriptor else None
            
            # save to npy
            area, fragment_idx = fragment_key.rsplit('_', 1)
            save_results(config.log_dir, area, fragment_idx, clustered_descriptors, clustered_kpts_weights[:, :3], clustered_kpts_weights[:, 3])


    save_path = os.path.join(config.log_dir, 'clustered_kpts_weights.json')
    utils_algo.save_keypints_to_file(clustered_kpts_weights_dict, save_path)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='test args')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--info', default=None, type=str,
                      help='training info (default: NONE)')

    config = ConfigParser.from_args(args)
    main(config)
