import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from trainer.trainer_helper import *
from utils.utils_algo import cdist, save_pointcloud

class PeaknessLoss(torch.nn.Module):
    def __init__(self, config):
        super(PeaknessLoss, self).__init__()
        self.config = config
        from model.pointnet2.pointnet2_utils import ball_query
        self.ball_query = ball_query
        self.radius = self.config['loss']['vote_score_peakness_loss_radius']
        self.neighbor_max_sample = self.config['loss']['vote_score_peakness_loss_neighbor_max_sample']
        self.margin = self.config['loss']['vote_score_peakness_loss_margin']

    def forward(self, xyz, scores):
        neighbor = self.ball_query(self.radius, self.neighbor_max_sample, xyz.unsqueeze(0), xyz.unsqueeze(0), False)[0].long()
        N = xyz.shape[0]
        scores = scores.squeeze().unsqueeze(1)
        shadow_scores = torch.zeros_like(scores[:1, :])
        scores = torch.cat([scores, shadow_scores], dim=0)

        neighbor_scores = scores[neighbor, :] # [n_points, n_neighbors, 1]

        neighbor_num = torch.sum((neighbor >= 0).float(), dim=1, keepdims=True) # [n_points, 1]
        # print(neighbor_num.mean(), neighbor_num.max(), neighbor_num.min(), neighbor_num.median())
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num)) # at least one
        mean_scores = torch.sum(neighbor_scores, dim=1) / neighbor_num  # [n_points, 1]
        max_scores, _ = torch.max(neighbor_scores, dim=1)  # [n_points, 1]
        # return 1 - (F.relu(max_scores - mean_scores - self.margin)).mean()
        return F.relu(mean_scores - max_scores + self.margin).mean()
        # return (mean_scores - max_scores).mean()

class ScoreConsistencyLoss(torch.nn.Module):
    def __init__(self, config):
        super(ScoreConsistencyLoss, self).__init__()
        self.config = config
        self.radius = self.config['loss']['vote_desc_positive_radius']

    def forward(self, src_xyz, src_scores, dst_xyz, dst_scores):
        src_match_idx, dst_match_idx = get_matching_indices(src_xyz, dst_xyz, self.radius)
        return (src_scores[src_match_idx] - dst_scores[dst_match_idx]).pow(2).mean()

class VoteLoss(torch.nn.Module):
    def __init__(self, config):
        super(VoteLoss, self).__init__()
        self.config = config
        self.vote_score_loss_weight = self.get_weights('vote_score_loss_weight')
        self.vote_desc_loss_weight = self.get_weights('vote_desc_loss_weight')
        self.vote_score_peakness_loss_weight = self.get_weights('vote_score_peakness_loss_weight')
        self.vote_score_consistency_loss_weight = self.get_weights('vote_score_consistency_loss_weight')
        self.peakness_loss = PeaknessLoss(config)
        self.score_consistency_loss = ScoreConsistencyLoss(config)
    
    def get_weights(self, name, default_val=0):
        return self.config['loss'][name] if name in self.config['loss'] else default_val

    def forward(self, epoch, loss_input):
        losses = []
        loss_dict = {}
        extra_info_dict = {}

        anc_match_idx = None

        # vote features should be distinctive according to distance#
        # enabled until xyz has been trained before
        if self.vote_desc_loss_weight > 0 and epoch > self.config['trainer']['voting_train_start']:
            pc_src_input = loss_input[0]['vote_xyz'] # Mx3
            pc_dst_input = loss_input[1]['vote_xyz'] # Nx3
            desc_src_input = loss_input[0]['vote_features'] # MxC
            desc_dst_input = loss_input[1]['vote_features'] # NxC

            MAX_ANCHOR_KPT_NUM = self.config['loss']['vote_desc_anchor_num']
            MAX_DST_KPT_NUM = self.config['loss']['vote_desc_neg_num']

            if pc_src_input.shape[0] > MAX_ANCHOR_KPT_NUM:
                rand_idx = torch.randperm(pc_src_input.shape[0])[:MAX_ANCHOR_KPT_NUM]
                pc_src_input = pc_src_input[rand_idx, :]
                desc_src_input = desc_src_input[rand_idx, :]

            # if pc_src_input.numel() == 0:
                # return [], {}, {}
            anc_match_idx, pos_match_idx = get_matching_indices(
                pc_src_input, pc_dst_input, self.config['loss']['vote_desc_positive_radius'])
            
            if anc_match_idx.numel() < MAX_ANCHOR_KPT_NUM // 2:
                # print(anc_match_idx.numel())
                return [], {}, {}
            
            pc_src_input = pc_src_input[anc_match_idx, :]
            desc_src_input = desc_src_input[anc_match_idx, :]
            anc_desc = desc_src_input
            pos_desc = desc_dst_input[pos_match_idx, :]
            pos_xyz = pc_dst_input[pos_match_idx, :]

            if pc_dst_input.shape[0] > MAX_DST_KPT_NUM:
                rand_idx = torch.randperm(pc_dst_input.shape[0])[:MAX_DST_KPT_NUM]
                pc_dst_input = pc_dst_input[rand_idx, :]
                desc_dst_input = desc_dst_input[rand_idx, :]

            # compute distance diff
            dist_diff = cdist(pc_src_input, pc_dst_input, metric='euclidean')

            # compute desc diff
            desc_diff = cdist(desc_src_input, desc_dst_input, metric='euclidean')

            assert(dist_diff.shape == desc_diff.shape)

            # find negative pairs via radius
            negative_radius = self.config['loss']['vote_desc_negative_safe_radius']
            desc_diff_clone = desc_diff.clone()
            desc_diff_clone[dist_diff<negative_radius] += 1e10 # mark as invalid

            negative_min, _ = torch.min(desc_diff_clone, dim=1, keepdim=False) # (M, 1)

            # remove desc diff mannually setting to large val (which means no points out of safe radius)
            valid_pair_mask = negative_min < 1e5
            assert((~valid_pair_mask).sum() == 0)

            # construct symmetric negative pairs
            negative_min_symmetric = None
            if self.config['loss']['symmetric_negative_pair']:
                pc_src_input_full = loss_input[0]['vote_xyz_full'] # Mx3
                desc_src_input_full = loss_input[0]['vote_features_full'] # MxC
                if pc_src_input_full.shape[0] > MAX_DST_KPT_NUM:
                    rand_idx = torch.randperm(pc_src_input_full.shape[0])[:MAX_DST_KPT_NUM]
                    pc_src_input_full = pc_src_input_full[rand_idx, :]
                    desc_src_input_full = desc_src_input_full[rand_idx, :]
                    # compute distance diff
                    dist_diff_symmetric = cdist(pos_xyz, pc_src_input_full, metric='euclidean')
                    # compute desc diff
                    desc_diff_symmetric = cdist(pos_desc, desc_src_input_full, metric='euclidean')
                    desc_diff_symmetric_clone = desc_diff_symmetric.clone()
                    desc_diff_symmetric_clone[desc_diff_symmetric<negative_radius] += 1e10 # mark as invalid

                    negative_min_symmetric, _ = torch.min(desc_diff_symmetric_clone, dim=1, keepdim=False) # (M, 1)
                

            if negative_min.numel() > 0:
                positive_max = torch.sqrt((anc_desc - pos_desc).pow(2).sum(1) + 1e-12)
                assert(positive_max.shape == negative_min.shape)
                p_n_diff = positive_max - negative_min
                extra_info_dict['vote_desc_accuracy'] = (p_n_diff < 0).sum() * 100.0 / p_n_diff.shape[0]
                extra_info_dict['vote_desc_pos_mean'] = positive_max.mean().item()
                extra_info_dict['vote_desc_neg_mean'] = negative_min.mean().item()
                average_negative = (desc_diff.sum(-1) - positive_max) / (desc_diff.shape[-1] - 1)
                extra_info_dict['vote_desc_average_negative'] = average_negative.mean().item()
                desc_loss_type = self.config['loss']['vote_desc_loss_type']
                if desc_loss_type == 'triplet':
                    desc_loss = F.relu(p_n_diff + self.config['loss']['vote_desc_triplet_margin']).mean()
                elif desc_loss_type == 'contrastive':
                    pos_loss = F.relu(positive_max - self.config['loss']['vote_desc_pos_margin']).pow(2)
                    neg_loss = F.relu(self.config['loss']['vote_desc_neg_margin'] - negative_min).pow(2)
                    if not negative_min_symmetric is None:
                        neg_loss = (neg_loss + F.relu(self.config['loss']['vote_desc_neg_margin'] - negative_min_symmetric).pow(2)) / 2
                    desc_loss = 0.5 * (pos_loss + neg_loss)
                    desc_loss = desc_loss.mean()
                loss_dict['vote_desc_loss'] = desc_loss.item()
                desc_loss = self.vote_desc_loss_weight * desc_loss
                losses.append(desc_loss)
            
        # good vote should have lower p_n_diff
        if self.vote_score_loss_weight > 0:
            if anc_match_idx != None and anc_match_idx.numel() > 3:
                sigma_src = loss_input[0]['vote_scores'] # M
                sigma_dst = loss_input[1]['vote_scores'] # N
                selected_src_sigma = sigma_src[anc_match_idx]
                selected_dst_sigma = sigma_dst[pos_match_idx]
                selected_p_n_diff = p_n_diff
                # p_n_diff : smaller is better
                # average_negative: larger is better
                # KP2D like score, which extremely make score to 0 or 1
                selected_sigma = (selected_src_sigma + selected_dst_sigma) / 2
                # do not let score loss gradient affect descritor learning
                # selected_sigma = selected_sigma.detach()
                # average_negative = average_negative.detach()

                vote_score_type = self.config['loss']['vote_score_loss_type']
                if vote_score_type == 'D3Feat':
                    vote_score_loss = selected_sigma.mul(selected_p_n_diff).mean()
                elif vote_score_type == 'pndiff_mean':
                    vote_score_loss = selected_sigma.mul(selected_p_n_diff - selected_p_n_diff.mean()).mean()
                elif vote_score_type == 'pndiff_rank':
                    # vote_score_loss = selected_sigma.mul(selected_p_n_diff - selected_p_n_diff.mean()).mean()
                    # pairwise distance matrix to rank
                    rank_mask = ((selected_p_n_diff.unsqueeze(1) - selected_p_n_diff.unsqueeze(0)) > 0).float() * 2 - 1
                    # pairwise distance of sigma
                    sigma_pairwise_diff = selected_sigma.unsqueeze(1) - selected_sigma.unsqueeze(0)
                    # pn_diff lower -> better pair
                    vote_score_loss = F.relu(sigma_pairwise_diff * rank_mask).mean()
                elif vote_score_type == 'pndiff_rank_margin':
                    # vote_score_loss = selected_sigma.mul(selected_p_n_diff - selected_p_n_diff.mean()).mean()
                    # pairwise distance matrix to rank
                    rank_mask = ((selected_p_n_diff.unsqueeze(1) - selected_p_n_diff.unsqueeze(0)) > 0).float() * 2 - 1
                    # pairwise distance of sigma
                    sigma_pairwise_diff = selected_sigma.unsqueeze(1) - selected_sigma.unsqueeze(0)
                    # pn_diff lower -> better pair
                    margin = 1.0 / selected_sigma.shape[0]
                    vote_score_loss = F.relu(sigma_pairwise_diff * rank_mask + margin).mean()
                elif vote_score_type == 'avg_neg_mean':
                    assert(selected_sigma.shape == average_negative.shape)
                    vote_score_loss = selected_sigma.mul(-(average_negative - average_negative.mean())).mean()
                elif vote_score_type == 'avg_neg_rank':
                    # pairwise distance matrix to rank
                    rank_mask = ((average_negative.unsqueeze(1) - average_negative.unsqueeze(0)) > 0).float() * 2 - 1
                    # pairwise distance of sigma
                    sigma_pairwise_diff = selected_sigma.unsqueeze(1) - selected_sigma.unsqueeze(0)
                    # average_negative larger -> better points
                    vote_score_loss = F.relu(-sigma_pairwise_diff * rank_mask).mean()
                loss_dict['vote_score_loss'] = vote_score_loss.item()
                losses.append(self.vote_score_loss_weight * vote_score_loss)
                # print(sigma_src_dst.detach().cpu().numpy())

        # encourgage score peakness
        if self.vote_score_peakness_loss_weight > 0:
            # src_peakness = self.peakness_loss(loss_input[0]['vote_xyz'], loss_input[0]['vote_scores'])
            # dst_peakness = self.peakness_loss(loss_input[1]['vote_xyz'], loss_input[1]['vote_scores'])
            # peakness_loss = (src_peakness + dst_peakness) / 2
            # src points has become sparse anchors
            peakness_loss = self.peakness_loss(loss_input[1]['vote_xyz'], loss_input[1]['vote_scores'])
            loss_dict['vote_score_peakness_loss'] = peakness_loss.item()
            losses.append(self.vote_score_peakness_loss_weight * peakness_loss)
        
        # score should be consistent among pairs
        if self.vote_score_consistency_loss_weight > 0:
            score_consistency_loss = self.score_consistency_loss(
                loss_input[0]['vote_xyz'], loss_input[0]['vote_scores'],
                loss_input[1]['vote_xyz'], loss_input[1]['vote_scores'])
            loss_dict['vote_score_consistency_loss'] = score_consistency_loss.item()
            losses.append(self.vote_score_consistency_loss_weight * score_consistency_loss)

        sigma_src = loss_input[0]['vote_scores'] # M
        sigma_dst = loss_input[1]['vote_scores'] # N
        extra_info_dict['mean_score'] = (sigma_src.mean() + sigma_dst.mean()).item() / 2
        extra_info_dict['std_score'] = (sigma_src.std() + sigma_dst.std()).item() / 2
        extra_info_dict['min_score'] = torch.min(sigma_src.min(), sigma_dst.min()).item()
        extra_info_dict['max_score'] = torch.max(sigma_src.max(), sigma_dst.max()).item()

        loss = sum(losses)
        return loss, loss_dict, extra_info_dict
