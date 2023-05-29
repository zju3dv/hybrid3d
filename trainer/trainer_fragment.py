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
from trainer.trainer_helper import prepare_supervision, prepare_batch_frame_data


class TrainerFragment(BaseTrainer):
    """
    TrainerFragment class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.subset_ratio = config['trainer']['subset_ratio']
        self.len_epoch = int(len(self.data_loader) * self.subset_ratio)
        self.valid_data_loader = valid_data_loader
        self.len_valid_epoch = int(len(self.valid_data_loader) * self.subset_ratio)
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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # handling fresh start and resume
        if epoch == self.start_epoch and epoch > 1:
            resume_log_dir = os.path.dirname(self.config.resume).replace('models_', 'log_')
            # resume keypoints
            kpts_path = os.path.join(resume_log_dir, 'kpts', 'kpts-%03d.json' % (epoch - 1))
            self.logger.info('Resume dataset.keypoints from: {}'.format(kpts_path))
            resume_keypoints = utils_algo.load_keypints_from_file(kpts_path)
            self.data_loader.dataset.update_keypoints(resume_keypoints)
            # resume keypoints offset
            kpts_offset_path = os.path.join(resume_log_dir, 'kpts', 'kpts-offset-%03d.json' % (epoch - 1))
            self.logger.info('Resume dataset.keypoints_chamfer_offset from: {}'.format(kpts_offset_path))
            resume_keypoints_offset = utils_algo.load_keypints_from_file(kpts_offset_path)
            self.data_loader.dataset.update_keypoints_chamfer_offset(resume_keypoints_offset)

        self.model.train()
        self.train_metrics.reset()
        progress_bar = tqdm.tqdm(total=self.len_epoch, dynamic_ncols=True)
        chamfer_distance_list = []
        inference_kpts_raw = {}
        self.data_loader.dataset.update_dynamic_subset(epoch, self.data_loader.sampler.indices.tolist(), self.len_epoch)
        
        run_count = 0
        for batch_idx, batch in enumerate(self.data_loader):
            # if batch_idx == self.len_epoch:
            #     break
            if batch is None: continue

            data_idx = batch['idx']
            fragment_key = batch['fragment_key']
            all_frame_data = batch['frame_data']
            pcd_data = batch['pcd_data']
            pcd_data_aug = batch['pcd_data_aug']

            # update fragment pcd max_K
            self.data_loader.dataset.update_max_keypoint_of_fragment(data_idx)

            # inferece
            # t3 = time.time()
            chamfer_distance, kpts_raw = self._inference_on_fragment(epoch, fragment_key, all_frame_data, pcd_data)
            if batch['inference_only']: continue

            inference_kpts_raw.update(kpts_raw)
            if chamfer_distance > 0:
                chamfer_distance_list.append(chamfer_distance)
            # t4 = time.time()
            # print('inference', t4 - t3)

            # choose frame to train
            training_choices = self._choose_random_indices(len(all_frame_data), self.training_frame_size)

            # prepare heatmap and chamfer offset map
            # supervision training_frame_num x [heatmap, chamfer_offset_map]
            supervision = prepare_supervision(self.config, self.data_loader.dataset, fragment_key, all_frame_data, training_choices)
            # assemble data, perform data augmentation
            batch_frame_data = prepare_batch_frame_data(training_choices, all_frame_data, supervision, self.frame_batch_size, True)

            # optimize
            self.model.train()
            # prepare batch data

            for i, data in enumerate(batch_frame_data):
                input_rgb = data['rgb'].to(self.device, dtype=torch.float32)
                input_sparse_depth = data['sparse_depth'].to(self.device, dtype=torch.float32)
                pcd_crsp_idx = data['pcd_crsp_idx'].to(self.device, dtype=torch.long)
                target_depth = data['depth'].to(self.device, dtype=torch.float32)
                target_heatmap = data['heatmap'].to(self.device, dtype=torch.float32)
                chamfer_offset_map = data['chamfer_offset_map'].to(self.device, dtype=torch.float32)
                projection_pts_info = data['projection_pts_info']

                self.optimizer.zero_grad()
                model_input_data = {
                    'rgb' : input_rgb,
                    'depth': input_sparse_depth,
                    'pcd' : copy.deepcopy(pcd_data_aug),
                    'pcd_crsp_idx': pcd_crsp_idx,
                    'fragment_key': fragment_key
                }
                output = self.model(model_input_data)
                output_depth, output_heatmap = output['depth'], output['heatmap']
                loss_input = {
                    'epoch': epoch,
                    'output_depth': output_depth,
                    'output_heatmap' : output_heatmap,
                    'output_descriptor' : output.get('descriptor'),
                    'projection_pts_info': projection_pts_info,
                    'target_depth' : target_depth,
                    'target_heatmap' : target_heatmap,
                    'chamfer_offset_map' : chamfer_offset_map
                }
                loss, loss_dict = self.criterion(loss_input)
                loss.backward()
                self.optimizer.step()

                current_step = (epoch - 1) * self.len_epoch * len(batch_frame_data) + run_count * len(batch_frame_data) + i
                self.writer.set_step(current_step)
                self.train_metrics.update('loss', loss.item())
                for k, v in loss_dict.items():
                    self.writer.add_scalar(k, v)
                # for met in self.metric_ftns:
                #     self.train_metrics.update(met.__name__, met(output_depth, target))
                if i == 0 and (run_count % self.log_step == 0 or (epoch == self.start_epoch and run_count % (self.log_step//10) == 0)):
                    self.logger.debug('Current Step: {}'.format(current_step))
                    self.logger.debug('Image Dump: fragment_key:{}, img_idx{}'.format(fragment_key, data['idx'][:4]))
                    fig = vis.save_fig_full(self.config, input_rgb,\
                            output_depth, output_heatmap, target_depth, target_heatmap, chamfer_offset_map)
                    self.writer.add_image('full_vis', torch.from_numpy(fig), dataformats='HWC')
            run_count += 1
            progress_bar.update(1)
            # progress_bar.set_postfix_str("Loss=%.8e" % (loss.item()))
    
        progress_bar.close()

        self.writer.set_step(epoch)
        if len(chamfer_distance_list) > 0:
            mean_chamfer_distance = statistics.mean(chamfer_distance_list)
            self.writer.add_scalar('mean_chamfer_distance', mean_chamfer_distance)
            self.logger.info('mean_chamfer_distance = {}'.format(mean_chamfer_distance))
        clustering_strategy.write_clustering_keypoints_to_files(epoch, self.config, self.data_loader.dataset, inference_kpts_raw)

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
        if hasattr(self.model, 'kpconv'):
            self.model.kpconv.train()
        self.valid_metrics.reset()
        # progress_bar = tqdm.tqdm(total=len(self.valid_data_loader), dynamic_ncols=True)
        chamfer_distance_list = []
        self.valid_data_loader.dataset.update_dynamic_subset(epoch, self.valid_data_loader.sampler.indices.tolist(), self.len_valid_epoch)
        with torch.no_grad():
            run_count = 0
            for batch_idx, batch in enumerate(self.valid_data_loader):
                # if batch_idx == self.len_valid_epoch:
                #     break
                if batch is None: continue
                data_idx = batch['idx']
                fragment_key = batch['fragment_key']
                all_frame_data = batch['frame_data']
                pcd_data = batch['pcd_data']

                # update fragment pcd max_K
                self.data_loader.dataset.update_max_keypoint_of_fragment(data_idx)

                # inferece
                # t3 = time.time()
                chamfer_distance, _ = self._inference_on_fragment(epoch, fragment_key, all_frame_data, pcd_data)
                if batch['inference_only']: continue

                if chamfer_distance > 0:
                    chamfer_distance_list.append(chamfer_distance)
                # t4 = time.time()
                # print('inference', t4 - t3)

                # choose frame to train
                training_choices = self._choose_random_indices(len(all_frame_data), self.training_frame_size)

                # prepare heatmap and chamfer offset map
                # supervision training_frame_num x [heatmap, chamfer_offset_map]
                supervision = prepare_supervision(self.config, self.data_loader.dataset, fragment_key, all_frame_data, training_choices)
                # assemble data, perform data augmentation
                batch_frame_data = prepare_batch_frame_data(training_choices, all_frame_data, supervision, self.frame_batch_size, False)
                for idx, data in enumerate(batch_frame_data):
                    input_rgb = data['rgb'].to(self.device, dtype=torch.float32)
                    input_sparse_depth = data['sparse_depth'].to(self.device, dtype=torch.float32)
                    pcd_crsp_idx = data['pcd_crsp_idx'].to(self.device, dtype=torch.long)
                    target_depth = data['depth'].to(self.device, dtype=torch.float32)
                    target_heatmap = data['heatmap'].to(self.device, dtype=torch.float32)
                    chamfer_offset_map = data['chamfer_offset_map'].to(self.device, dtype=torch.float32)
                    projection_pts_info = data['projection_pts_info']

                    model_input_data = {
                        'rgb' : input_rgb,
                        'depth': input_sparse_depth,
                        'pcd' : copy.deepcopy(pcd_data),
                        'pcd_crsp_idx': pcd_crsp_idx,
                        'fragment_key': fragment_key
                    }
                    output = self.model(model_input_data)
                    output_depth, output_heatmap = output['depth'], output['heatmap']
                    loss_input = {
                        'epoch': epoch,
                        'output_depth': output_depth,
                        'output_heatmap' : output_heatmap,
                        'output_descriptor' : output.get('descriptor'),
                        'projection_pts_info': projection_pts_info,
                        'target_depth' : target_depth,
                        'target_heatmap' : target_heatmap,
                        'chamfer_offset_map' : chamfer_offset_map
                    }
                    loss, loss_dict = self.criterion(loss_input)

                    current_step = (epoch - 1) * self.len_valid_epoch * len(batch_frame_data) + run_count * len(batch_frame_data) + idx
                    self.writer.set_step(current_step, 'valid')
                    self.valid_metrics.update('loss', loss.item())
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(k, v)
                    # for met in self.metric_ftns:
                    #     self.train_metrics.update(met.__name__, met(output_depth, target))
                    if run_count % (self.log_step // 5) == 0 and idx == 0:
                        self.logger.debug('Validation Current Step: {}'.format(current_step))
                        self.logger.debug('Image Dump: fragment_key:{}, img_idx{}'.format(fragment_key, data['idx'][:4]))
                        fig = vis.save_fig_full(self.config, input_rgb,\
                                output_depth, output_heatmap, target_depth, target_heatmap, chamfer_offset_map)
                        self.writer.add_image('full_vis', torch.from_numpy(fig), dataformats='HWC')
                run_count += 1
        self.writer.set_step(epoch, 'valid')
        if len(chamfer_distance_list) > 0:
            mean_chamfer_distance = statistics.mean(chamfer_distance_list)
            self.writer.add_scalar('mean_chamfer_distance', mean_chamfer_distance)
        return self.valid_metrics.result()

    def _inference_on_fragment(self, epoch, fragment_key, data, pcd_data, frame_skip_step=1):
        # self.model.eval()
        # kpconv module has bug with eval()
        # if hasattr(self.model, 'kpconv'):
        #     self.model.kpconv.train()
        self.model.train()
        frame_batch_size = self.frame_batch_size * 4
        # inferece pts and weights
        all_params = []
        
        # generate batch list
        frame_list = list(range(random.randint(0, frame_skip_step), len(data), frame_skip_step))
        batch_list = []
        for i in range(len(frame_list) // frame_batch_size):
            batch_list.append(frame_list[i*frame_batch_size:(i+1)*frame_batch_size])
        last_N = len(frame_list) % frame_batch_size
        if last_N > 0:
            batch_list.append(frame_list[-last_N:])

        # inference heatmaps
        with torch.no_grad():
            for indices in batch_list:
                # input_rgb = frame_data['rgb'].to(self.device, dtype=torch.float32)
                # input_sparse_depth = frame_data['sparse_depth'].to(self.device, dtype=torch.float32)
                # target_depth = frame_data['depth'].to(self.device, dtype=torch.float32)
                input_rgb = torch.stack([data[i]['rgb'] for i in indices]).to(self.device, dtype=torch.float32)
                input_sparse_depth = torch.stack([data[i]['sparse_depth'] for i in indices]).to(self.device, dtype=torch.float32)
                pcd_crsp_idx = torch.stack([data[i]['pcd_crsp_idx'] for i in indices]).to(self.device, dtype=torch.long)
                target_depth = torch.stack([data[i]['depth'] for i in indices]).to(self.device, dtype=torch.float32)


                model_input_data = {
                    'rgb' : input_rgb,
                    'depth': input_sparse_depth,
                    'pcd' : copy.deepcopy(pcd_data),
                    'pcd_crsp_idx': pcd_crsp_idx,
                    'fragment_key': fragment_key
                }
                output = self.model(model_input_data)
                output_depth, output_heatmap = output['depth'], output['heatmap']
                
                depth_trunc = self.config['trainer']['clustering']['depth_trunc']
                target_depth[target_depth > depth_trunc] = 0
                output_heatmap_np = output_heatmap.cpu().detach().numpy()
                # handling checkerboard artifact
                output_heatmap_np = utils_algo.remove_border_for_batch_heatmap(output_heatmap_np)
                target_depth_np = target_depth.data.cpu().detach().numpy()
                # batch_Twc = frame_data['camera_pose_Twc'].numpy()
                # batch_camera_intrinsics = frame_data['camera_intrinsics'].numpy()
                batch_Twc = np.stack([data[i]['camera_pose_Twc'] for i in indices])
                batch_camera_intrinsics = np.stack([data[i]['camera_intrinsics'] for i in indices])
                conf = self.config['trainer']['point_lifting']
                batch_size = input_rgb.shape[0]
                all_params.extend([(
                        conf,
                        fragment_key,
                        np.squeeze(output_heatmap_np[i,...]),
                        np.squeeze(target_depth_np[i,...]),
                        batch_Twc[i, ...],
                        batch_camera_intrinsics[i, ...]) for i in range(batch_size)])
        results = self.mp_pool.map_async(utils_algo.lift_heatmap_depth_to_space, all_params).get(60)
        pts_w_weght_fragment = np.concatenate([pts_w_weight for area_name, pts_w_weight, _ in results], axis=0)
        # clustering and update fragment keypoints
        inferece_keypoints_dict = {fragment_key: pts_w_weght_fragment}
        clustered_kpts, mean_chamfer_distance = clustering_strategy.update_dataset_keypoints_via_clustering(
                epoch, self.config, inferece_keypoints_dict, self.logger, self.writer, self.data_loader, False, False, False)
        return mean_chamfer_distance, inferece_keypoints_dict
