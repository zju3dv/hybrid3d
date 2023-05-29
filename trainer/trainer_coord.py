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
from trainer.trainer_helper import prepare_supervision, prepare_batch_frame_data, generate_target_info_by_closest_clusters


class TrainerCoord(BaseTrainer):
    """
    TrainerCoord class
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
            chamfer_distance, kpts_raw = self._inference_on_fragment(epoch, fragment_key, batch, all_frame_data, pcd_data)
            if batch[0]['inference_only']: continue

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
                        'output_rgb_heatmap' : output.get('rgb_heatmap'),
                        'output_pcd_heatmap' : output.get('pcd_heatmap'),
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
                    fig = vis.save_fig_auto_choice(self.config, {
                        'rgb': input_rgb,
                        'target_depth': loss_input['target_depth'],
                        'output_heatmap': loss_input['output_heatmap'],
                        'output_rgb_heatmap': loss_input.get('output_rgb_heatmap'),
                        'output_pcd_heatmap': loss_input.get('output_pcd_heatmap'),
                        'target_heatmap': loss_input['target_heatmap'],
                        'output_coord': loss_input['output_coord'],
                    })
                    self.writer.add_image('full_vis', torch.from_numpy(fig), dataformats='HWC')
            run_count += 1
            progress_bar.update(1)
            # progress_bar.set_postfix_str("Loss=%.8e" % (loss.item()))
    
        progress_bar.close()

        self.writer.set_step(epoch)
        clustering_strategy.write_clustering_keypoints_to_files(epoch, self.config, self.data_loader.dataset, inference_kpts_raw)

        self.writer.add_scalar('mean_loss', self.train_metrics.avg('loss'))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        # kpconv module has bug with eval()
        # if hasattr(self.model, 'kpconv'):
        #     self.model.kpconv.train()
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

                # inferece
                chamfer_distance, kpts_raw = self._inference_on_fragment(epoch, fragment_key, batch, all_frame_data, pcd_data)
                if batch[0]['inference_only']: continue

                # if chamfer_distance > 0:
                #     chamfer_distance_list.append(chamfer_distance)
                assert(len(all_frame_data) == 2)

                # self.model.eval()

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
                            'output_rgb_heatmap' : output.get('rgb_heatmap'),
                            'output_pcd_heatmap' : output.get('pcd_heatmap'),
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
                        fig = vis.save_fig_auto_choice(self.config, {
                            'rgb': input_rgb,
                            'target_depth': loss_input['target_depth'],
                            'output_heatmap': loss_input['output_heatmap'],
                            'output_rgb_heatmap': loss_input.get('output_rgb_heatmap'),
                            'output_pcd_heatmap': loss_input.get('output_pcd_heatmap'),
                            'target_heatmap': loss_input['target_heatmap'],
                            'output_coord': loss_input['output_coord'],
                        })
                        self.writer.add_image('full_vis', torch.from_numpy(fig), dataformats='HWC')
                run_count += 1
        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('mean_loss', self.valid_metrics.avg('loss'))
        return self.valid_metrics.result()

    def _inference_on_fragment(self, epoch, fragment_key, batch, all_frame_data, pcd_data, frame_skip_step=1):
        self.model.eval()
        # assume fragment pair
        assert(len(fragment_key) == 2)
        frame_batch_size = self.frame_batch_size * 4
        # inferece pts and weights
        all_params = []
        for pair_idx in range(2):
            # generate batch list
            frame_list = list(range(random.randint(0, frame_skip_step), len(all_frame_data[pair_idx]), frame_skip_step))
            batch_list = []
            for i in range(len(frame_list) // frame_batch_size):
                batch_list.append(frame_list[i*frame_batch_size:(i+1)*frame_batch_size])
            last_N = len(frame_list) % frame_batch_size
            if last_N > 0:
                batch_list.append(frame_list[-last_N:])

            # inference heatmaps
            with torch.no_grad():
                for indices in batch_list:
                    input_rgb = torch.stack([all_frame_data[pair_idx][i]['rgb'] for i in indices]).to(self.device, dtype=torch.float32)
                    pcd_crsp_idx = torch.stack([all_frame_data[pair_idx][i]['pcd_crsp_idx'] for i in indices]).to(self.device, dtype=torch.long)
                    target_depth = torch.stack([all_frame_data[pair_idx][i]['depth'] for i in indices]).to(self.device, dtype=torch.float32)

                    model_input_data = {
                        'rgb' : input_rgb,
                        'pcd' : copy.deepcopy(pcd_data[pair_idx]),
                        'pcd_crsp_idx': pcd_crsp_idx,
                        'fragment_key': fragment_key[pair_idx]
                    }
                    output = self.model(model_input_data, True)
                    output_depth, output_heatmap, output_coord = output['depth'], output['heatmap'], output['coord']
                    
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
                    all_params.extend([(
                            conf,
                            fragment_key[pair_idx],
                            np.squeeze(output_heatmap_np[i,...]),
                            np.squeeze(output_coord_np[i,...]),
                            np.squeeze(target_depth_np[i,...]),
                            batch_Twc[i, ...],
                            batch_camera_intrinsics[i, ...]) for i in range(batch_size)])
        results = self.mp_pool.map_async(utils_algo.lift_heatmap_coord_depth_to_space, all_params).get(60)
        # pts_w_weight_depth_fragment = np.concatenate([pts_w_weight for area_name, pts_w_weight, _ in results], axis=0)

        # unpack result to assemble pts_w_weight_depth_fragment
        pts_w_weight_depth_fragment = [] # 3D xyz, weight, depth, fragment_id(0 or 1)
        for area_name, pts_w_weight, _ in results:
            # add fragment_id to last axis
            if area_name == fragment_key[0]:
                pts_full_info = np.concatenate([pts_w_weight, np.zeros((pts_w_weight.shape[0], 1))], axis=1)
            elif area_name == fragment_key[1]:
                pts_full_info = np.concatenate([pts_w_weight, np.ones((pts_w_weight.shape[0], 1))], axis=1)
            else:
                raise RuntimeError('area_name not in fragment_key')
            pts_w_weight_depth_fragment.append(pts_full_info)
        pts_w_weight_depth_fragment = np.concatenate(pts_w_weight_depth_fragment, axis=0)

        # clustering and update fragment keypoints
        fragment_pair_key = '{}_{}'.format(fragment_key[0], fragment_key[1])
        inferece_keypoints_dict = {fragment_pair_key: pts_w_weight_depth_fragment}
        clustered_kpts, mean_chamfer_distance = clustering_strategy.update_dataset_keypoints_via_clustering(
                epoch, self.config, inferece_keypoints_dict, self.logger, self.writer, self.data_loader, False, False, False)
        return mean_chamfer_distance, inferece_keypoints_dict
