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


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.subset_ratio = config['trainer']['subset_ratio']
        self.clustering_subset_ratio = config['trainer']['clustering_subset_ratio']
        self.len_epoch = int(len(self.data_loader) * self.subset_ratio)
        self.len_clustering_epoch = int(len(self.data_loader) * self.clustering_subset_ratio)
        self.valid_data_loader = valid_data_loader
        self.len_valid_epoch = int(len(self.valid_data_loader) * self.subset_ratio)
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        # self.log_step = int(np.sqrt(data_loader.batch_size))
        self.log_step = config['trainer']['log_step']
        self.use_chamfer_offset_map = config['trainer']['use_chamfer_offset_map']

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # handling fresh start and resume
        if epoch == self.start_epoch:
            if epoch <= self.config['trainer']['clustering_start_epoch']:
                # use initial keypoints when epoch <= clustering_start_epoch 
                clustering_strategy.update_dataset_keypoints_via_clustering(\
                    epoch, self.config, self.data_loader.dataset.keypoints, self.logger, self.writer, self.data_loader, False)
            else:
                # use clustering keypoints from resume path when epoch > clustering_start_epoch
                resume_log_dir = os.path.dirname(self.config.resume).replace('models_', 'log_')
                kpts_path = os.path.join(resume_log_dir, 'kpts', 'kpts-%03d.json' % (epoch - 1))
                self.logger.info('Resume dataset.keypoints from: {}'.format(kpts_path))
                resume_keypoints = utils_algo.load_keypints_from_file(kpts_path)
                self.data_loader.dataset.update_keypoints(resume_keypoints)
                

        self.model.train()
        self.train_metrics.reset()
        progress_bar = tqdm.tqdm(total=self.len_epoch, dynamic_ncols=True)

        for batch_idx, batch in enumerate(self.data_loader):
            if batch_idx == self.len_epoch:
                break
            input_rgb = batch['rgb'].to(self.device, dtype=torch.float32)
            input_sparse_depth = batch['sparse_depth'].to(self.device, dtype=torch.float32)
            target_depth = batch['depth'].to(self.device, dtype=torch.float32)
            target_heatmap = batch['heatmap'].to(self.device, dtype=torch.float32)
            if self.use_chamfer_offset_map:
                chamfer_offset_map = batch['chamfer_offset_map'].to(self.device, dtype=torch.float32)
            else:
                chamfer_offset_map = None
            # target_heatmap = batch['heatmap'].to(self.device, dtype=torch.int64)

            self.optimizer.zero_grad()
            output_depth, output_heatmap = self.model(input_rgb, input_sparse_depth)
            loss, loss_dict = self.criterion(epoch, output_depth, output_heatmap, target_depth, target_heatmap, chamfer_offset_map)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for k, v in loss_dict.items():
                self.writer.add_scalar(k, v)
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output_depth, target))

            progress_bar.update(1)
            progress_bar.set_postfix_str("Loss=%.8e" % (loss.item()))

            if batch_idx % self.log_step == 0:
                # self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                #     epoch,
                #     self._progress(batch_idx),
                #     loss.item()))
                # pass
                self.logger.debug('Current Step: {}'.format((epoch - 1) * self.len_valid_epoch + batch_idx))
                self.logger.debug('Image Dump: Area:{}, img_idx{}'.format(batch['area'][:4], batch['idx'][:4]))
                fig = vis.save_fig_full(self.config, input_rgb,\
                         output_depth, output_heatmap, target_depth, target_heatmap, chamfer_offset_map)
                self.writer.add_image('full_vis', torch.from_numpy(fig), dataformats='HWC')

        progress_bar.close()
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if epoch >= self.config['trainer']['clustering_start_epoch']:
            self._inferece_clustering_points(epoch)

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
        self.valid_metrics.reset()
        # progress_bar = tqdm.tqdm(total=len(self.valid_data_loader), dynamic_ncols=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                if batch_idx == self.len_valid_epoch:
                    break
                input_rgb = batch['rgb'].to(self.device, dtype=torch.float32)
                input_sparse_depth = batch['sparse_depth'].to(self.device, dtype=torch.float32)
                target_depth = batch['depth'].to(self.device, dtype=torch.float32)
                target_heatmap = batch['heatmap'].to(self.device, dtype=torch.float32)
                if self.use_chamfer_offset_map:
                    chamfer_offset_map = batch['chamfer_offset_map'].to(self.device, dtype=torch.float32)
                else:
                    chamfer_offset_map = None
                # target_heatmap = batch['heatmap'].to(self.device, dtype=torch.int64)

                output_depth, output_heatmap = self.model(input_rgb, input_sparse_depth)
                loss, loss_dict = self.criterion(epoch, output_depth, output_heatmap, target_depth, target_heatmap, chamfer_offset_map)

                self.writer.set_step((epoch - 1) * self.len_valid_epoch + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for k, v in loss_dict.items():
                    self.writer.add_scalar(k, v)
                # for met in self.metric_ftns:
                #     self.valid_metrics.update(met.__name__, met(output_depth, target))
                # progress_bar.update(1)
                # progress_bar.set_postfix_str("valid Loss=%.8e" % (loss.item()))
                if batch_idx % self.log_step == 0:
                    fig = vis.save_fig_full(self.config, input_rgb,\
                         output_depth, output_heatmap, target_depth, target_heatmap, chamfer_offset_map)
                    self.logger.debug('Current Step: {}'.format((epoch - 1) * self.len_valid_epoch + batch_idx))
                    self.logger.debug('Image Dump: Area:{}, img_idx{}'.format(batch['area'][:4], batch['idx'][:4]))
                    self.writer.add_image('full_vis', torch.from_numpy(fig), dataformats='HWC')

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


    def _inferece_clustering_points(self, epoch):
        """
        inferece clustering points

        :param epoch: Integer, current training epoch.
        :return: point set dict
        """
        self.logger.info('Update_clustering_points...')
        inferece_keypoints_dict = {}
        self.model.eval()
        augmentation_status = self.data_loader.dataset.augmentation
        self.data_loader.dataset.augmentation = False

        # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        # Catch Ctrl+C / SIGINT and exit multiprocesses gracefully in python
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        mp_pool = mp.Pool(processes = self.data_loader.batch_size)
        signal.signal(signal.SIGINT, original_sigint_handler)

        progress_bar = tqdm.tqdm(total=self.len_clustering_epoch, dynamic_ncols=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                if batch_idx == self.len_clustering_epoch:
                    break
                input_rgb = batch['rgb'].to(self.device, dtype=torch.float32)
                input_sparse_depth = batch['sparse_depth'].to(self.device, dtype=torch.float32)
                target_depth = batch['depth'].to(self.device, dtype=torch.float32)
                target_heatmap = batch['heatmap'].to(self.device, dtype=torch.float32)

                output_depth, output_heatmap = self.model(input_rgb, input_sparse_depth)
                
                depth_trunc = self.config['trainer']['clustering']['depth_trunc']
                target_depth[target_depth > depth_trunc] = 0

                output_heatmap_np = output_heatmap.cpu().detach().numpy()
                # handling checkerboard artifact
                output_heatmap_np = utils_algo.remove_border_for_batch_heatmap(output_heatmap_np)
                target_depth_np = target_depth.data.cpu().detach().numpy()
                batch_Twc = batch['camera_pose_Twc'].numpy()
                batch_camera_intrinsics = batch['camera_intrinsics'].numpy()
                conf = self.config['trainer']['point_lifting']

                batch_size = output_heatmap_np.shape[0]
                params = [(
                        conf,
                        batch['area'][i],
                        np.squeeze(output_heatmap_np[i,...]),
                        np.squeeze(target_depth_np[i,...]),
                        batch_Twc[i, ...],
                        batch_camera_intrinsics[i, ...]) for i in range(batch_size)]
                results = mp_pool.map_async(utils_algo.lift_heatmap_depth_to_space, params).get(60)
                for area_name, pts_w_weight, _ in results:
                    if area_name in inferece_keypoints_dict:
                        inferece_keypoints_dict[area_name] = np.concatenate([inferece_keypoints_dict[area_name], pts_w_weight], axis=0)
                    else:
                        inferece_keypoints_dict[area_name] = pts_w_weight
                progress_bar.update(1)
        mp_pool.close()
        mp_pool.join()
        progress_bar.close()
        # save keypoints before clustering
        save_path = os.path.join(self.config.log_dir, 'kpts', 'kpts-raw-%03d.json' % epoch)
        utils_algo.save_keypints_to_file(inferece_keypoints_dict, save_path)
        # clustering and update keypoints in dataset
        clustered_keypoints_dict. _ = \
            clustering_strategy.update_dataset_keypoints_via_clustering(\
                epoch, self.config, inferece_keypoints_dict, self.logger, self.writer, self.data_loader)
        # save keypoints after clustering
        save_path = os.path.join(self.config.log_dir, 'kpts', 'kpts-%03d.json' % epoch)
        utils_algo.save_keypints_to_file(clustered_keypoints_dict, save_path)

        self.data_loader.dataset.augmentation = augmentation_status
