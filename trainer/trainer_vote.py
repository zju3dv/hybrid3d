import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
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
from model.losses.vote_loss import VoteLoss

class TrainerCoordVote(BaseTrainer):
    """
    TrainerCoordVote class
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
            self.config['trainer']['voting_train_start'] = 1

        self.vote_loss = VoteLoss(self.config)
        self.vote_optimizer = torch.optim.Adam(self.model.votenet.parameters(), lr = 1.0e-3, weight_decay=1.0e-4, amsgrad=True)
        self.vote_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.vote_optimizer, 0.95)

        # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        # Catch Ctrl+C / SIGINT and exit multiprocesses gracefully in python
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.mp_pool = mp.Pool(processes = 8)
        signal.signal(signal.SIGINT, original_sigint_handler)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # set spatial operation for dataset loader, so that we can precompute multiscale info
        if hasattr(self.model, 'get_spatial_ops'):
            self.data_loader.dataset.set_spatial_ops(self.model.get_spatial_ops())

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
            chamfer_distance, kpts_raw = self._vote_on_fragment(epoch, fragment_key, batch, all_frame_data, pcd_data, train_vote=True, step=step)
            if batch[0]['inference_only']: continue

            # skip 2D model training, we only train voteing model
            if self.config['trainer']['skip_2d_training']:
                progress_bar.update(1)
                continue

            inference_kpts_raw.update(kpts_raw)
            # if chamfer_distance > 0:
            #     chamfer_distance_list.append(chamfer_distance)
            # t4 = time.time()
            # print('inference', t4 - t3)
            assert(len(all_frame_data) == 2)

            self.model.train()
            
            # choose frame to train
            training_choices = tuple(self._choose_random_indices(len(all_frame_data[i]), self.training_frame_size // 2)
                for i in range(2))

            # prepare heatmap and chamfer offset map
            # supervision training_frame_num x [heatmap, chamfer_offset_map]
            supervision = tuple(prepare_supervision(self.config, self.data_loader.dataset, fragment_pair_key, all_frame_data[i], training_choices[i], i)
                for i in range(2))

            # assemble data, perform data augmentation
            batch_frame_data = tuple(prepare_batch_frame_data(training_choices[i], all_frame_data[i], supervision[i], self.frame_batch_size // 2, True)
                for i in range(2))
            
            assert(len(batch_frame_data[0]) == len(batch_frame_data[1]))
            N_inner_batches = len(batch_frame_data[0])
            # prepare batch data
            for i in range(N_inner_batches):
                self.optimizer.zero_grad()
                loss_input = dict()
                model_input_data_list = []
                # for source and target fragments
                for pair_idx in range(2):
                    data = batch_frame_data[pair_idx][i]
                    input_rgb = data['rgb'].to(self.device, dtype=torch.float32)
                    pcd_crsp_idx = data['pcd_crsp_idx'].to(self.device, dtype=torch.long)
                    target_depth = data['depth'].to(self.device, dtype=torch.float32)
                    target_heatmap = data['heatmap'].to(self.device, dtype=torch.float32)
                    chamfer_offset_map = data['chamfer_offset_map'].to(self.device, dtype=torch.float32)
                    projection_pts_info = data['projection_pts_info']
                    # prepare model input
                    model_input_data = {
                        'rgb' : input_rgb,
                        'pcd' : copy.deepcopy(pcd_data_aug[pair_idx]),
                        'pcd_crsp_idx': pcd_crsp_idx,
                        'fragment_key': fragment_key
                    }
                    model_input_data_list.append(model_input_data)
                    output = self.model(model_input_data)

                    # concat data for loss input
                    loss_input_local = {
                        'epoch': epoch,
                        'output_depth': output['depth'],
                        'output_heatmap' : output['heatmap'],
                        'output_coord': output['coord'],
                        'output_descriptor' : output.get('descriptor'),
                        'projection_pts_info': projection_pts_info,
                        'target_depth' : target_depth,
                        'target_heatmap' : target_heatmap,
                        'chamfer_offset_map' : chamfer_offset_map
                    }
                    self._concat_loss_input(loss_input, loss_input_local)

                loss, loss_dict = self.criterion(loss_input)
                if not loss_dict:
                    continue
                loss.backward()
                if self.config['trainer']['grad_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config['trainer']['grad_clip_norm'])
                self.optimizer.step()

                current_step = (epoch - 1) * self.len_epoch * N_inner_batches + run_count * N_inner_batches + i
                self.writer.set_step(current_step)
                if not math.isnan(loss.item()):
                    self.train_metrics.update('loss', loss.item())
                for k, v in loss_dict.items():
                    self.writer.add_scalar(k, v)
                # for met in self.metric_ftns:
                #     self.train_metrics.update(met.__name__, met(output_depth, target))
                if i == 0 and (run_count % self.log_step == 0 or (epoch == self.start_epoch and run_count % (self.log_step//10) == 0)):
                    self.logger.debug('Current Step: {}'.format(current_step))
                    self.logger.debug('Image Dump: fragment_key:{}, img_idx{}'.format(fragment_key, data['idx'][:4]))
                    input_rgb = torch.cat([x['rgb'] for x in model_input_data_list])
                    fig = vis.save_fig_full(self.config, input_rgb,\
                            loss_input['output_depth'], loss_input['output_heatmap'], loss_input['target_depth'], \
                                loss_input['target_heatmap'], loss_input['chamfer_offset_map'],
                                    coord=loss_input['output_coord'])
                    self.writer.add_image('full_vis', torch.from_numpy(fig), dataformats='HWC')
            run_count += 1
            progress_bar.update(1)
            # progress_bar.set_postfix_str("Loss=%.8e" % (loss.item()))
    
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
        # kpconv module has bug with eval()
        if hasattr(self.model, 'kpconv'):
            self.model.kpconv.train()
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

                # inferece
                chamfer_distance, kpts_raw = self._vote_on_fragment(epoch, fragment_key, batch, all_frame_data, pcd_data, train_vote=False, step=step)
                if batch[0]['inference_only']: continue

                # if chamfer_distance > 0:
                #     chamfer_distance_list.append(chamfer_distance)
                assert(len(all_frame_data) == 2)

                self.model.train()

                # choose frame to train
                training_choices = tuple(self._choose_random_indices(len(all_frame_data[i]), self.training_frame_size // 2)
                    for i in range(2))

                # prepare heatmap and chamfer offset map
                # supervision training_frame_num x [heatmap, chamfer_offset_map]
                supervision = tuple(prepare_supervision(self.config, self.data_loader.dataset, fragment_pair_key, all_frame_data[i], training_choices[i], i)
                    for i in range(2))

                # assemble data, perform data augmentation
                batch_frame_data = tuple(prepare_batch_frame_data(training_choices[i], all_frame_data[i], supervision[i], self.frame_batch_size // 2, False)
                    for i in range(2))

                assert(len(batch_frame_data[0]) == len(batch_frame_data[1]))
                N_inner_batches = len(batch_frame_data[0])

                for idx in range(N_inner_batches):
                    loss_input = dict()
                    model_input_data_list = []
                    # for source and target fragments
                    for pair_idx in range(2):
                        data = batch_frame_data[pair_idx][idx]
                        input_rgb = data['rgb'].to(self.device, dtype=torch.float32)
                        pcd_crsp_idx = data['pcd_crsp_idx'].to(self.device, dtype=torch.long)
                        target_depth = data['depth'].to(self.device, dtype=torch.float32)
                        target_heatmap = data['heatmap'].to(self.device, dtype=torch.float32)
                        chamfer_offset_map = data['chamfer_offset_map'].to(self.device, dtype=torch.float32)
                        projection_pts_info = data['projection_pts_info']

                        model_input_data = {
                            'rgb' : input_rgb,
                            'pcd' : copy.deepcopy(pcd_data[pair_idx]),
                            'pcd_crsp_idx': pcd_crsp_idx,
                            'fragment_key': fragment_key
                        }
                        model_input_data_list.append(model_input_data)
                        output = self.model(model_input_data)
                        output_depth, output_heatmap = output['depth'], output['heatmap']
                        # concat data for loss input
                        loss_input_local = {
                            'epoch': epoch,
                            'output_depth': output['depth'],
                            'output_heatmap' : output['heatmap'],
                            'output_coord': output['coord'],
                            'output_descriptor' : output.get('descriptor'),
                            'projection_pts_info': projection_pts_info,
                            'target_depth' : target_depth,
                            'target_heatmap' : target_heatmap,
                            'chamfer_offset_map' : chamfer_offset_map
                        }
                        self._concat_loss_input(loss_input, loss_input_local)
                    loss, loss_dict = self.criterion(loss_input)
                    if not loss_dict:
                        continue

                    current_step = (epoch - 1) * self.len_valid_epoch * N_inner_batches + run_count * N_inner_batches + idx
                    self.writer.set_step(current_step, 'valid')
                    if not math.isnan(loss.item()):
                        self.valid_metrics.update('loss', loss.item())
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(k, v)
                    # for met in self.metric_ftns:
                    #     self.train_metrics.update(met.__name__, met(output_depth, target))
                    if run_count % (self.log_step // 5) == 0 and idx == 0:
                        self.logger.debug('Validation Current Step: {}'.format(run_count))
                        self.logger.debug('Image Dump: fragment_key:{}, img_idx{}'.format(fragment_key, data['idx'][:4]))
                        input_rgb = torch.cat([x['rgb'] for x in model_input_data_list])
                        fig = vis.save_fig_full(self.config, input_rgb,\
                             loss_input['output_depth'], loss_input['output_heatmap'], loss_input['target_depth'], \
                                loss_input['target_heatmap'], loss_input['chamfer_offset_map'],\
                                    coord=loss_input['output_coord'])
                        self.writer.add_image('full_vis', torch.from_numpy(fig), dataformats='HWC')
                run_count += 1
        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('mean_loss', self.valid_metrics.avg('loss'))
        return self.valid_metrics.result()

    def _vote_on_fragment(self, epoch, fragment_key, batch, all_frame_data, pcd_data, frame_skip_step=1, train_vote=False, step=0):
        self.model.train()
        assert(self.config['trainer']['voting_use_start'] > self.config['trainer']['voting_train_start'])
        use_voted_points = epoch >= self.config['trainer']['voting_use_start']
        train_vote = train_vote and epoch >= self.config['trainer']['voting_train_start']
        # assume fragment pair
        assert(len(fragment_key) == 2)
        frame_batch_size = self.frame_batch_size * 4
        # inferece pts and weights
        all_params = [[], []]
        output_descriptors_list = [[], []]
        for pair_idx in range(2):
            # generate batch list
            frame_list = list(range(random.randint(0, frame_skip_step), len(all_frame_data[pair_idx]), frame_skip_step))
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
                        'fragment_key': fragment_key[pair_idx]
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
            input_descs.append(np.concatenate([pts_w_weight_frag[:, 3, None], descriptors_frag], axis=1))
        
        output = []
        augment_vote_input_pts = self.config['trainer']['augment_vote_input']
        if augment_vote_input_pts:
            rand_T_Mat = [rand_transformation_matrix() for i in range(2)]
        has_zero_candidates = False
        self.vote_optimizer.zero_grad()
        self.vote_lr_scheduler.step(epoch)
        for pair_idx in range(2):
            if input_pts[pair_idx].shape[0] < 3:
                has_zero_candidates = True
                break
            xyz = torch.from_numpy(input_pts[pair_idx]).float().to(self.device)[None, ...]
            features = torch.from_numpy(input_descs[pair_idx]).float().to(self.device)[None, ...].transpose(1, 2)
            with (contextlib.nullcontext() if train_vote else torch.no_grad()):
                # TODO: add random rotation and translation?
                if augment_vote_input_pts:
                    aug_xyz = transform_pcd_pose(xyz.squeeze(), rand_T_Mat[pair_idx])[None, ...]
                    output_vote = self.model.forward_vote({
                        'xyz': aug_xyz,
                        'features': features,
                    })
                    output_vote['vote_xyz'] = transform_pcd_pose(output_vote['vote_xyz'], np.linalg.inv(rand_T_Mat[pair_idx]))
                    output.append(output_vote)
                else:
                    output.append(self.model.forward_vote({
                        'xyz': aug_xyz,
                        'features': features,
                    }))

        self.writer.set_step(step, 'train' if train_vote else 'valid')

        if train_vote and not has_zero_candidates:
            # orig_chamfer_dist, _, _ = utils_algo.chamfer_distance_simple(input_pts[0], input_pts[1])
            # voted_chamfer_dist, _, _ = utils_algo.chamfer_distance_simple(output[0]['vote_xyz'].detach().cpu().numpy(), output[1]['vote_xyz'].detach().cpu().numpy())
            # self.writer.add_scalar('orig_chamfer_dist', orig_chamfer_dist)
            # self.writer.add_scalar('voted_chamfer_dist', voted_chamfer_dist)
            loss, loss_dict, extra_info_dict = self.vote_loss(epoch, output)
            pts_w_3d_scores = None
            if not loss_dict:
                pts_w_3d_scores = np.empty((0, 4))
            else:
                loss.backward()
                self.vote_optimizer.step()

            for k, v in loss_dict.items():
                self.writer.add_scalar(k, v)
            for k, v in extra_info_dict.items():
                self.writer.add_scalar(k, v)
            
            pts_w_3d_scores = torch.cat([torch.cat([output[i]['vote_xyz'], output[i]['vote_scores'][:, None]], dim=1) for i in range(2)], dim=0).detach().cpu().numpy() if pts_w_3d_scores is None else pts_w_3d_scores
        else:
            pts_w_3d_scores = np.empty((0, 4))
        # clustering and update fragment keypoints
        fragment_pair_key = '{}_{}'.format(fragment_key[0], fragment_key[1])
        # for early state, we use classical weighted clustering, since the voting network is not stable
        # currently, voting network cannot handle with zero candidates
        #   (e.g. too few covisible area, which is rarely to be happened)
        if not use_voted_points or pts_w_3d_scores.size == 0:
            pts_w_3d_scores = np.concatenate(pts_w_weight_frag_full, axis=0)
            inferece_keypoints_dict = {fragment_pair_key: pts_w_3d_scores}
            clustered_kpts, mean_chamfer_distance = clustering_strategy.update_dataset_keypoints_via_clustering(
                    epoch, self.config, inferece_keypoints_dict, self.logger, self.writer, self.data_loader, False, False, False)
        else:
            # use orig 2d scores
            pts_w_3d_scores_orig_2d = np.concatenate(pts_w_weight_frag_full, axis=0)
            pts_w_3d_scores[:, 3] = pts_w_3d_scores_orig_2d[:, 3]
            inferece_keypoints_dict = {fragment_pair_key: pts_w_3d_scores}
            # clustered_kpts, mean_chamfer_distance = clustering_strategy.update_dataset_keypoints_via_nms(
            #         epoch, self.config, inferece_keypoints_dict, self.logger, self.writer, self.data_loader)
            clustered_kpts, mean_chamfer_distance = clustering_strategy.update_dataset_keypoints_via_clustering(
                    epoch, self.config, inferece_keypoints_dict, self.logger, self.writer, self.data_loader, False, False, False)
        
        return mean_chamfer_distance, inferece_keypoints_dict
