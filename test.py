import argparse
import collections
import open3d
import torch
import numpy as np
import random
from tqdm import tqdm
import copy
from utils import clustering_strategy, utils_algo, clustering, vis, util
import multiprocessing as mp
import signal
import os
import time
# import matplotlib.pyplot as plt
import matplotlib

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

def nms_score_cpu(xyz, scores, radius, max_neighbor_sample=64):
    import torch_points_kernels as tp
    xyz = torch.from_numpy(xyz)
    scores = torch.from_numpy(scores)
    idx, dist = tp.ball_query(radius, max_neighbor_sample, xyz.unsqueeze(0).contiguous(), xyz.unsqueeze(0).contiguous(), mode="dense", sort=True)
    idx = idx.squeeze()
    dist = dist.squeeze()
    idx[dist < 0] = -1
    scores_with_shadow = torch.cat([scores, torch.zeros_like(scores[0:1])], dim=0)
    neighbor_scores = scores_with_shadow[idx]
    neighbor_max_scores, _ = torch.max(neighbor_scores, dim=1)
    scores = scores * (neighbor_max_scores == scores).float()
    return scores.numpy()

def main(config, cmd_args):
    from data_loader.data_loader_factory import get_data_loader_by_name
    # import model.loss as module_loss
    # import model.metric as module_metric
    from model.model_factory import get_model_by_name
    from trainer import Trainer
    from trainer import TrainerFragment

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

    # build model architecture
    model = config.init_obj('arch', get_model_by_name(config['arch']['type']), config)
    logger.debug(model)

    # get function handles of loss and metrics
    # loss_fn = config.init_obj('loss', module_loss, config)
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    logger.info('Checkpoint Epoch: {}'.format(checkpoint['epoch']))
    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)
    # model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # model.train()
    model.eval()
    # kpconv module has bug with eval()
    # if hasattr(model, 'kpconv'):
    #     model.kpconv.train()
    # if hasattr(model, 'output_multi_tower'):
    #     model.output_multi_tower = False

    if hasattr(model, 'get_spatial_ops'):
        data_loader.dataset.set_spatial_ops(model.get_spatial_ops())

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

    model_type = 'heat_desc' # or heat_coord_desc

    measure_speed = False

    random_rotate = True

    rand_T = None

    time_accu = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, dynamic_ncols=True)):
            if random_rotate:
                # random 3 axis rotate, no translation
                rand_T = utils_algo.rand_transformation_matrix(3, 1, 0)
            t_batch_begin = time.time()
            data_idx = batch['idx']
            fragment_key = batch['fragment_key']
            all_frame_data = batch['frame_data']
            pcd_data = batch['pcd_data']
            if random_rotate:
                pcd_data.pos = utils_algo.transform_pcd_pose(pcd_data.pos, rand_T)

            if config['data_loader']['read_frame_data']:
                # inferece pts and weights
                all_params = []
                output_descriptors_list = []
                
                # generate batch list
                frame_list = list(range(random.randint(0, frame_skip_step), len(all_frame_data), frame_skip_step))
                batch_list = [frame_list]
                # for i in range(len(frame_list) // frame_batch_size):
                #     batch_list.append(frame_list[i*frame_batch_size:(i+1)*frame_batch_size])
                # last_N = len(frame_list) % frame_batch_size
                # if last_N > 0:
                #     batch_list.append(frame_list[-last_N:])

                # inference heatmaps
                for i_idx, indices in enumerate(batch_list):
                    input_rgb = torch.stack([all_frame_data[i]['rgb'] for i in indices]).to(device, dtype=torch.float32)
                    input_sparse_depth = torch.stack([all_frame_data[i]['sparse_depth'] for i in indices]).to(device, dtype=torch.float32)
                    pcd_crsp_idx = torch.stack([all_frame_data[i]['pcd_crsp_idx'] for i in indices]).to(device, dtype=torch.long)
                    target_depth = torch.stack([all_frame_data[i]['depth'] for i in indices]).to(device, dtype=torch.float32)

                    _, _, H, W = input_rgb.shape

                    model_input_data = {
                        'rgb' : input_rgb,
                        'depth': input_sparse_depth,
                        'pcd' : copy.deepcopy(pcd_data),
                        'pcd_crsp_idx': pcd_crsp_idx,
                        'fragment_key': fragment_key
                    }
                    t1 = time.time()
                    output = model(model_input_data, cache_pcd_inference=True)
                    t2 = time.time()
                    # print('2d time', t2 - t1)
                    output_depth, output_heatmap = output['depth'], output['heatmap']
                    output_descriptor = output.get('descriptor')
                    output_descriptors_list.extend([output_descriptor[i] for i in range(output_descriptor.shape[0])])

                    if 'coord' in output:
                        model_type = 'heat_coord_desc'
                        output_coord = output.get('coord')
                        output_coord_np = output_coord.cpu().detach().numpy()
                    
                    depth_trunc = config['trainer']['clustering']['depth_trunc']
                    target_depth[target_depth > depth_trunc] = 0
                    output_heatmap_np = output_heatmap.cpu().detach().numpy()
                    if model_type == 'heat_desc':
                        # handling checkerboard artifact
                        output_heatmap_np = utils_algo.remove_border_for_batch_heatmap(output_heatmap_np)
                    target_depth_np = target_depth.data.cpu().detach().numpy()

                    batch_Twc = np.stack([all_frame_data[i]['camera_pose_Twc'] for i in indices])
                    batch_camera_intrinsics = np.stack([all_frame_data[i]['camera_intrinsics'] for i in indices])
                    conf = config['trainer']['point_lifting']
                    batch_size = input_rgb.shape[0]
                    if model_type == 'heat_desc':
                        all_params.extend([(
                                conf,
                                fragment_key,
                                np.squeeze(output_heatmap_np[i,...]),
                                np.squeeze(target_depth_np[i,...]),
                                batch_Twc[i, ...],
                                batch_camera_intrinsics[i, ...]) for i in range(batch_size)])
                    elif model_type == 'heat_coord_desc':
                        all_params.extend([(
                                conf,
                                fragment_key,
                                np.squeeze(output_heatmap_np[i,...]),
                                np.squeeze(output_coord_np[i,...]),
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
                    if not measure_speed and i_idx == 0 and batch_idx % 20 == 0:
                        logger.debug('Image Dump: batch_idx:{}, fragment_key:{}, img_idx:{}'.format(
                            batch_idx, fragment_key, [all_frame_data[i]['idx'] for i in indices]))
                        vis_size = 4
                        figure = vis.save_fig_auto_choice(config, {
                            'rgb': input_rgb[:vis_size, ...],
                            'target_depth': target_depth[:vis_size, ...],
                            'output_heatmap': output_heatmap[:vis_size, ...],
                            'output_rgb_heatmap': output['rgb_heatmap'][:vis_size, ...] if 'rgb_heatmap' in output else None,
                            'output_pcd_heatmap': output['pcd_heatmap'][:vis_size, ...] if 'pcd_heatmap' in output else None,
                            'output_coord': output_coord[:vis_size, ...] if 'coord' in model_type else None,
                        })
                        matplotlib.image.imsave('{}/{:05d}.jpg'.format(dump_vis_dir, batch_idx), figure)
                # get result from heatmap nms
                if model_type == 'heat_desc':
                    results = mp_pool.map_async(utils_algo.lift_heatmap_depth_to_space, all_params).get(60)
                elif model_type == 'heat_coord_desc':
                    results = mp_pool.map_async(utils_algo.lift_heatmap_coord_depth_to_space, all_params).get(60)
                pts_w_weight_depth_fragment = []
                descriptors = []

                t3 = time.time()
                for idx, (area_names, pts_w_weight, coords_xy) in enumerate(results):
                    if np.isnan(np.sum(pts_w_weight)):
                        print('NaN found in {} , camera pose may contain NaN.'.format(area_names))
                        continue
                    # get weighted 3D points
                    pts_w_weight_depth_fragment.append(pts_w_weight)
                    # get descriptor
                    if len(output_descriptors_list) > 0:
                        descriptors.append(utils_algo.get_descriptors_from_feature_map(output_descriptors_list[idx], coords_xy, H, W))
                t4 = time.time()
                # print('interpolate feature 2d time', t4 - t3)
                # concat 3D points_weights and descriptors
                pts_w_weight_depth_fragment = np.concatenate(pts_w_weight_depth_fragment, axis=0)
                has_descriptor = len(descriptors) > 0
                if has_descriptor > 0:
                    descriptors = torch.cat(descriptors, dim=0).detach().cpu().numpy()
            else:
                pts_w_weight_depth_fragment = batch['2d_candidate_pts'].numpy()
                descriptors = batch['2d_candidate_desc'].numpy()

            # if model has vote part
            if hasattr(model, 'forward_vote'):
                t1 = time.time()
                xyz = torch.from_numpy(pts_w_weight_depth_fragment[:, :3]).float().to(device)[None, ...].contiguous()
                if random_rotate:
                    xyz = utils_algo.transform_pcd_pose(xyz.squeeze(), rand_T)
                features = torch.from_numpy(descriptors).float().to(device)[None, ...].transpose(2, 1)
                vote_output = model.forward_vote({
                    'rgb_xyz': xyz,
                    'rgb_features': features,
                    'pcd_xyz': pcd_data,
                }, use_cache=True)
                t2 = time.time()
                # print('forward time', t2 - t1)
                if random_rotate:
                    vote_output['vote_xyz'] = utils_algo.transform_pcd_pose(vote_output['vote_xyz'], np.linalg.inv(rand_T))
                # clustered_kpts_weights = pts_w_weight_depth_fragment
                clustered_kpts_weights = np.concatenate([
                    vote_output['vote_xyz'].detach().cpu().numpy(), vote_output['vote_scores'][:, None].detach().cpu().numpy()], axis=1)
                # clustered_kpts_weights[:, 3] = pts_w_weight_depth_fragment[:, 3]
                # random scores
                if cmd_args.random_score:
                    clustered_kpts_weights[:, 3] = np.random.rand(clustered_kpts_weights.shape[0])
                clustered_descriptors = vote_output['vote_features'].detach().cpu().numpy()
            else:
                clustered_kpts_weights = pts_w_weight_depth_fragment
                clustered_descriptors = descriptors
            if measure_speed:
                t_batch_end = time.time()
                time_accu.append(t_batch_end - t_batch_begin)
                # print('total time', sum(time_accu), sum(time_accu) / len(time_accu))
                continue
            # sort by weights
            new_order = np.argsort(-clustered_kpts_weights[:, 3])
            clustered_kpts_weights = clustered_kpts_weights[new_order]
            clustered_kpts_weights_dict[fragment_key] = clustered_kpts_weights
            clustered_descriptors = clustered_descriptors[new_order]
            
            # save to npy
            area, fragment_idx = fragment_key.rsplit('_', 1)
            xyz = clustered_kpts_weights[:, :3]
            scores = clustered_kpts_weights[:, 3]

            # scores = nms_score_cpu(xyz, scores, 0.03)

            save_results(config.log_dir, area, fragment_idx, clustered_descriptors, xyz, scores)




    save_path = os.path.join(config.log_dir, 'clustered_kpts_weights.pkl')
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
    args.add_argument('--random_score', default=False, action='store_true')
    
    cmd_args = args.parse_args()

    config = ConfigParser.from_args(args)
    main(config, cmd_args)
