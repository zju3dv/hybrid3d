import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, AverageMeterAutoKey
import tqdm
from utils import vis, utils_algo, clustering, clustering_strategy
import time
import os
import multiprocessing as mp
import signal
import utils
import statistics
import random
import copy
import math
import contextlib
from trainer.trainer_helper import *
from utils.utils_algo import *
from model.losses.vote_loss import VoteLoss
from utils.sampler import nms_3D

class TrainerFusion(BaseTrainer):
    """
    TrainerFusion class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.subset_ratio = config['trainer']['subset_ratio']
        self.len_epoch = self.subset_ratio if self.subset_ratio > 1 else int(len(self.data_loader) * self.subset_ratio)
        self.valid_data_loader = valid_data_loader
        self.len_valid_epoch = int(len(self.valid_data_loader) * self.len_epoch / float(len(self.data_loader)))
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        # self.log_step = int(np.sqrt(data_loader.batch_size))
        self.log_step = config['trainer']['log_step']
        self.frame_batch_size = self.config['trainer']['frame_batch_size']
        self.training_frame_size = self.config['trainer']['training_frame_size']

        # since we do not train 2d model, we just start voting train at epoch 1
        if self.config['trainer']['skip_2d_training']:
            self.config['trainer']['voting_train_start'] = 0

        self.vote_loss = VoteLoss(self.config)

        # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        # Catch Ctrl+C / SIGINT and exit multiprocesses gracefully in python
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.mp_pool = mp.Pool(processes = 8)
        signal.signal(signal.SIGINT, original_sigint_handler)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.train_average_meter = AverageMeterAutoKey()
        self.valid_average_meter = AverageMeterAutoKey()

        # set spatial operation for dataset loader, so that we can precompute multiscale info
        if hasattr(self.model, 'get_spatial_ops'):
            self.data_loader.dataset.set_spatial_ops(self.model.get_spatial_ops())
    
    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def _choose_random_indices(self, max_idx, num_idx):
        indices = list(range(max_idx))
        random.shuffle(indices)
        return indices[:num_idx]
    
    def _concat_loss_input(self, loss_input_dict, loss_input_local):
        for name, val in loss_input_local.items():
            if val is None:
                loss_input_dict[name] = None
            elif isinstance(val, torch.Tensor):
                loss_input_dict[name] = val if not name in loss_input_dict else torch.cat([loss_input_dict[name], val])
            elif isinstance(val, list):
                loss_input_dict[name] = val if not name in loss_input_dict else loss_input_dict[name] + val
            elif isinstance(val, int):
                loss_input_dict[name] = val
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()
        self.logger.info('current lr {:.4e}'.format(self._get_lr()))
        progress_bar = tqdm.tqdm(total=self.len_epoch, dynamic_ncols=True)
        # chamfer_distance_list = []
        inference_kpts_raw = {}
        # self.data_loader.dataset.update_dynamic_subset(epoch, self.data_loader.sampler.indices.tolist(), self.len_epoch)
        self.data_loader.sampler.set_current_seed(epoch)
        run_count = 0
        for batch_idx, batch in enumerate(self.data_loader):
            if batch_idx == self.len_epoch:
                break
            # if batch is None: continue

            data_idx = batch[0]['idx']
            fragment_pair_key = '{}_{}'.format(batch[0]['fragment_key'], batch[1]['fragment_key'])
            fragment_key = tuple(b['fragment_key'] for b in batch)
            all_frame_data = tuple(b['frame_data'] for b in batch)
            pcd_data = tuple(b['pcd_data'] for b in batch)
            pcd_data_aug = tuple(b['pcd_data_aug'] for b in batch)

            # update fragment pcd max_K
            self.data_loader.dataset.update_max_keypoint_of_fragment(data_idx)

            # inferece
            # t3 = time.time()
            step = (epoch - 1) * self.len_epoch + batch_idx
            self._vote_on_fragment(epoch, fragment_key, batch, all_frame_data, pcd_data, train_vote=True, step=step, call_func_mode='train')

            # skip 2D model training, we only train voteing model
            progress_bar.update(1)
    
        progress_bar.close()

        self.writer.set_step(epoch)
        # clustering_strategy.write_clustering_keypoints_to_files(epoch, self.config, self.data_loader.dataset, inference_kpts_raw)

        self.writer.add_scalar('mean_loss', self.train_metrics.avg('loss'))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.config['trainer']['voting_use_start'] == epoch - 1 or self.config['trainer']['voting_train_start'] == epoch - 1:
            self._save_checkpoint(epoch, force_save=True)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        # progress_bar = tqdm.tqdm(total=len(self.valid_data_loader), dynamic_ncols=True)
        # chamfer_distance_list = []
        self.valid_data_loader.dataset.update_dynamic_subset(epoch, self.valid_data_loader.sampler.indices.tolist(), self.len_valid_epoch)
        with torch.no_grad():
            run_count = 0
            self.valid_data_loader.sampler.set_current_seed(epoch)
            for batch_idx, batch in enumerate(self.valid_data_loader):
                if batch_idx == self.len_valid_epoch:
                    break
                # if batch is None: continue
                data_idx = batch[0]['idx']
                fragment_pair_key = '{}_{}'.format(batch[0]['fragment_key'], batch[1]['fragment_key'])
                fragment_key = tuple(b['fragment_key'] for b in batch)
                all_frame_data = tuple(b['frame_data'] for b in batch)
                pcd_data = tuple(b['pcd_data'] for b in batch)
                pcd_data_aug = tuple(b['pcd_data_aug'] for b in batch)


                # update fragment pcd max_K
                self.data_loader.dataset.update_max_keypoint_of_fragment(data_idx)

                step = (epoch - 1) * self.len_valid_epoch + batch_idx

                # vote on fragment
                self._vote_on_fragment(epoch, fragment_key, batch, all_frame_data, pcd_data, train_vote=False, step=step, call_func_mode='valid')
        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('mean_loss', self.valid_metrics.avg('loss'))
        return self.valid_metrics.result()

    def _vote_on_fragment(self, epoch, fragment_key, batch, all_frame_data, pcd_data, frame_skip_step=1, train_vote=False, step=0, call_func_mode='train'):
        assert(self.config['trainer']['voting_use_start'] > self.config['trainer']['voting_train_start'])
        use_voted_points = epoch >= self.config['trainer']['voting_use_start']
        train_vote = train_vote and epoch >= self.config['trainer']['voting_train_start']
        # assume fragment pair
        assert(len(fragment_key) == 2)

        if self.config['data_loader']['read_frame_data']:
            frame_batch_size = self.frame_batch_size * 4 + 1
            # inferece pts and weights
            all_params = [[], []]
            output_descriptors_list = [[], []]
            for pair_idx in range(2):
                # generate batch list
                frame_list = list(range(random.randint(0, frame_skip_step-1), len(all_frame_data[pair_idx]), frame_skip_step))
                batch_list = []
                for i in range(len(frame_list) // frame_batch_size):
                    batch_list.append(frame_list[i*frame_batch_size:(i+1)*frame_batch_size])
                last_N = len(frame_list) % frame_batch_size
                if last_N > 0:
                    batch_list.append(frame_list[-last_N:])

                # inference or training
                # with torch.no_grad():
                with torch.no_grad():
                    for indices in batch_list:
                        input_rgb = torch.stack([all_frame_data[pair_idx][i]['rgb'] for i in indices]).to(self.device, dtype=torch.float32)
                        pcd_crsp_idx = torch.stack([all_frame_data[pair_idx][i]['pcd_crsp_idx'] for i in indices]).to(self.device, dtype=torch.long)
                        target_depth = torch.stack([all_frame_data[pair_idx][i]['depth'] for i in indices]).to(self.device, dtype=torch.float32)
                        _, _, H, W = input_rgb.shape
                        model_input_data = {
                            'rgb' : input_rgb,
                            'pcd' : copy.deepcopy(pcd_data[pair_idx]),
                            'pcd_crsp_idx': pcd_crsp_idx,
                            'fragment_key': fragment_key[pair_idx],
                        }
                        output = self.model(model_input_data, True)
                        output_depth, output_heatmap, output_coord = output['depth'], output['heatmap'], output['coord']
                        output_descriptor = output.get('descriptor')
                        output_descriptors_list[pair_idx].extend([output_descriptor[i] for i in range(output_descriptor.shape[0])])
                        
                        depth_trunc = self.config['trainer']['clustering']['depth_trunc']
                        target_depth[target_depth > depth_trunc] = 0
                        output_heatmap_np = output_heatmap.cpu().detach().numpy()
                        output_coord_np = output_coord.cpu().detach().numpy()
                        # handling checkerboard artifact
                        # output_heatmap_np = utils_algo.remove_border_for_batch_heatmap(output_heatmap_np)
                        target_depth_np = target_depth.data.cpu().detach().numpy()
                        batch_Twc = np.stack([all_frame_data[pair_idx][i]['camera_pose_Twc'] for i in indices])
                        batch_camera_intrinsics = np.stack([all_frame_data[pair_idx][i]['camera_intrinsics'] for i in indices])
                        conf = self.config['trainer']['point_lifting']
                        batch_size = input_rgb.shape[0]
                        all_params[pair_idx].extend([(
                                conf,
                                fragment_key[pair_idx],
                                np.squeeze(output_heatmap_np[i,...]),
                                np.squeeze(output_coord_np[i,...]),
                                np.squeeze(target_depth_np[i,...]),
                                batch_Twc[i, ...],
                                batch_camera_intrinsics[i, ...]) for i in range(batch_size)])
            results = [self.mp_pool.map_async(utils_algo.lift_heatmap_coord_depth_to_space, all_params[i]).get(60) for i in range(2)]
            # pts_w_weight_depth_fragment = np.concatenate([pts_w_weight for area_name, pts_w_weight, _ in results], axis=0)

            # unpack result to assemble vote input
            input_pts = []
            input_descs = []
            pts_w_weight_frag_full = []
            for pair_idx in range(2):
                pts_w_weight_frag = [] # 3D xyz, weight
                descriptors_frag = []
                for idx, (area_names, pts_w_weight, coords_xy)  in enumerate(results[pair_idx]):
                    pts_w_weight_frag.append(pts_w_weight[:, :4])
                    descriptors_frag.append(utils_algo.get_descriptors_from_feature_map(output_descriptors_list[pair_idx][idx], coords_xy, H, W))
                pts_w_weight_frag = np.concatenate(pts_w_weight_frag, axis=0)
                descriptors_frag = torch.cat(descriptors_frag, dim=0).detach().cpu().numpy()
                input_pts.append(pts_w_weight_frag[:, :3])
                pts_w_weight_frag_full.append(pts_w_weight_frag)
                # append 2D weight to descriptor
                # input_descs.append(np.concatenate([pts_w_weight_frag[:, 3, None], descriptors_frag], axis=1))
                input_descs.append(descriptors_frag)
        else: # without rgb feature
            # input_pts = [np.empty((0, 3)) for i in range(2)]
            # input_descs = [np.empty((0, 128)) for i in range(2)]
            pair_0_base_Twc = batch[0]['base_Twc'] if batch[0]['pair_tag'] == 0 else batch[1]['base_Twc']
            input_pts = [
                transform_pcd_pose(batch[i]['2d_candidate_pts'].numpy(), np.matmul(np.linalg.inv(pair_0_base_Twc), batch[i]['base_Twc']))
                for i in range(2)]
            input_descs = [batch[i]['2d_candidate_desc'].numpy() for i in range(2)]

        output = []
        pcd_list = [batch[i]['pcd_data_aug'].pos if call_func_mode == 'train' else batch[i]['pcd_data'].pos for i in range(2)]
        augment_vote_input_pts = self.config['trainer']['augment_vote_input'] and call_func_mode == 'train'
        if augment_vote_input_pts :
            rand_T_Mat = [rand_transformation_matrix() for i in range(2)]
        has_zero_candidates = False
        self.optimizer.zero_grad()
        for pair_idx in range(2):
            # if input_pts[pair_idx].shape[0] < 3:
            #     has_zero_candidates = True
            #     break
            rgb_xyz = torch.from_numpy(input_pts[pair_idx]).float().to(self.device)
            rgb_features = torch.from_numpy(input_descs[pair_idx]).float().to(self.device).transpose(0, 1)
            with (contextlib.nullcontext() if train_vote else torch.no_grad()):
                # TODO: add random rotation and translation?
                if augment_vote_input_pts:
                    aug_rgb_xyz = transform_pcd_pose(rgb_xyz.squeeze(), rand_T_Mat[pair_idx])
                    aug_pcd_xyz = transform_pcd_pose(pcd_list[pair_idx].squeeze(), rand_T_Mat[pair_idx])
                    output_vote = self.model.forward_vote({
                        'rgb_xyz': aug_rgb_xyz,
                        'rgb_features': rgb_features,
                        'pcd_xyz': aug_pcd_xyz,
                    })
                    output_vote['vote_xyz'] = transform_pcd_pose(output_vote['vote_xyz'], np.linalg.inv(rand_T_Mat[pair_idx]))
                    output.append(output_vote)
                else:
                    output.append(self.model.forward_vote({
                        'rgb_xyz': rgb_xyz,
                        'rgb_features': rgb_features,
                        'pcd_xyz': pcd_list[pair_idx].squeeze(),
                    }))

        self.writer.set_step(step, call_func_mode)

        if not has_zero_candidates:
            # clip src points to overlap area
            # print(0, output[0]['vote_xyz'].shape)
            # store full xyz for symmetric negative pairs
            output[0]['vote_xyz_full'] = output[0]['vote_xyz']
            output[0]['vote_features_full'] = output[0]['vote_features']

            overlap_idx = find_overlap_indices(output[0]['vote_xyz'], batch[0]['pcd_overlap_np'], 0.15).unique()
            output[0]['vote_xyz'] = output[0]['vote_xyz'][overlap_idx, ...].contiguous()
            output[0]['vote_scores'] = output[0]['vote_scores'][overlap_idx, ...]
            output[0]['vote_features'] = output[0]['vote_features'][overlap_idx, ...]

            # more anchor for rgb points
            if self.config['trainer']['more_rgb_anchor']:
                rgb_xyz = torch.from_numpy(input_pts[0]).float().contiguous()
                # print(1, output[0]['vote_xyz'].shape)
                rgb_overlap_idx = find_overlap_indices(output[0]['vote_xyz'], rgb_xyz, 0.05).unique()
                full_idx = np.arange(output[0]['vote_xyz'].shape[0])

                MAX_SCR_NUM = self.config['loss']['vote_desc_anchor_num']
                if full_idx.shape[0] > MAX_SCR_NUM:
                    # perform nms to avoid rgb too close
                    _, _, valid_rgb_idx = nms_3D(output[0]['vote_xyz'][rgb_overlap_idx].detach().cpu().numpy(), None, 0.1)
                    rgb_overlap_idx = rgb_overlap_idx[valid_rgb_idx]
                    # no more than max_rgb_anchor_ratio
                    MAX_RGB_ANCHOR = round(MAX_SCR_NUM * self.config['trainer']['max_rgb_anchor_ratio'])
                    if rgb_overlap_idx.shape[0] > MAX_RGB_ANCHOR:
                        np.random.shuffle(rgb_overlap_idx)
                        rgb_overlap_idx = rgb_overlap_idx[:MAX_RGB_ANCHOR]
                    remain_idx = np.setdiff1d(full_idx, rgb_overlap_idx)
                    np.random.shuffle(remain_idx)
                    remain_idx = remain_idx[:MAX_SCR_NUM - rgb_overlap_idx.shape[0]]
                    selected_balanced_idx = np.concatenate([rgb_overlap_idx, remain_idx])
                else:
                    selected_balanced_idx = full_idx
                
                output[0]['vote_xyz'] = output[0]['vote_xyz'][selected_balanced_idx, ...].contiguous()
                output[0]['vote_scores'] = output[0]['vote_scores'][selected_balanced_idx, ...]
                output[0]['vote_features'] = output[0]['vote_features'][selected_balanced_idx, ...]
            
            # check if we still have enough anchors
            # if not, we skip this round
            if output[0]['vote_xyz'].shape[0] < self.config['loss']['vote_desc_anchor_num'] // 2:
                return

            loss, loss_dict, extra_info_dict = self.vote_loss(epoch, output)

            avg_meter = self.train_average_meter
            if loss_dict and train_vote:
                self.train_metrics.update('loss', loss.item())
                avg_meter = self.train_average_meter
                loss.backward()
                self.optimizer.step()
            elif loss_dict:
                self.valid_metrics.update('loss', loss.item())
                avg_meter = self.valid_average_meter
            
            for k, v in loss_dict.items():
                avg_meter.update(k, v)
            for k, v in extra_info_dict.items():
                avg_meter.update(k, v)

            if step % self.log_step == 0:
                avg_meter.write_avg_to_board(self.writer)
                avg_meter.reset()
