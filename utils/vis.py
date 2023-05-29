import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import pandas as pd
from PIL import Image
import utils.utils_algo


def reshape_to_horizontal_image(x, sphere_group_size, H, W):
    return x.reshape(sphere_group_size, H, W).transpose(0, 2, 1).reshape(sphere_group_size * W, H).transpose(1, 0)

def colored_data(depth, cmap='viridis', d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    cmap_ = plt.cm.get_cmap(cmap)
    return 255 * cmap_(depth_relative)[:,:,:3] # H, W, C

def batch_to_hstack(data, max_batch_size=4):
    data = np.transpose(data, (0, 2, 3, 1)) # B C H W -> B H W C
    clip_batch = min(4, data.shape[0])
    return np.hstack([data[i, ...] for i in range(clip_batch)])

def save_fig_simple(config, input_rgb, output_depth, output_heatmap, target_depth, coord=None):
    B, _, H, W = input_rgb.shape
    # denormalize rgb
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # rgb = (rgb - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    input_rgb_np = input_rgb.cpu().detach().numpy() * std.reshape([1, 3, 1, 1]) + mean.reshape([1, 3, 1, 1])
    # input_rgb_np = input_rgb.cpu().detach().numpy()
    input_rgb_h = 255 * batch_to_hstack(input_rgb_np)
    if not output_depth is None:
        output_depth_h = np.squeeze(batch_to_hstack(output_depth.cpu().detach().numpy()))
    if output_heatmap.shape[2] != H and output_heatmap.shape[3] != W: # resize if need
        output_heatmap_orig_np = output_heatmap.cpu().detach().numpy()
        output_heatmap = torch.nn.functional.interpolate(output_heatmap, size=(H, W), mode='nearest')
    output_heatmap_np = output_heatmap.cpu().detach().numpy()
    output_heatmap_h = np.squeeze(batch_to_hstack(output_heatmap_np))
    target_depth_h = np.squeeze(batch_to_hstack(target_depth.data.cpu().detach().numpy()))

    if not output_depth is None:
        output_depth_row = colored_data(output_depth_h, 'viridis', 0, 7)
    output_heatmap_row = colored_data(output_heatmap_h, 'Reds', 0, 1)
    target_depth_row = colored_data(target_depth_h, 'viridis', 0, 7)

    # draw nms keypoints
    kpt_vis = []
    # handling checkerboard artifact
    output_heatmap_np = utils.utils_algo.remove_border_for_batch_heatmap(output_heatmap_np)
    input_rgb_np = 255 * input_rgb_np
    heatmap_vis_conf = config['visualization']['heatmap']
    for i in range(output_heatmap.shape[0]):
        if coord is None:
            single_heatmap = np.squeeze(output_heatmap_np[i,...])
            H, W = single_heatmap.shape
            pts = utils.utils_algo.heatmap_to_pts(single_heatmap, conf_thresh=heatmap_vis_conf['conf_thresh'], nms_dist=heatmap_vis_conf['nms_dist'])
            kpt_vis.append(np.expand_dims(utils.utils_algo.draw_pts_with_confidence(pts, H, W, input_rgb_np[i]), axis=0))
        else:
            single_heatmap = np.squeeze(output_heatmap_orig_np[i,...])
            pts = utils.utils_algo.heatmap_to_pts(single_heatmap, conf_thresh=heatmap_vis_conf['conf_thresh'], nms_dist=heatmap_vis_conf['nms_dist'])
            pts = np.transpose(pts) # Nx3 (x in heatmap, y in heatmap, weight)
            # get coordinate from single_coord
            pts_xy = coord.squeeze()[i, ...].detach().cpu().numpy()[:, pts[:, 1].round().astype(int), pts[:, 0].round().astype(int)]
            pts[:, :2] = pts_xy.transpose(1, 0)
            pts = np.transpose(pts)
            kpt_vis.append(np.expand_dims(utils.utils_algo.draw_pts_with_confidence(pts, H, W, input_rgb_np[i]), axis=0))
    kpt_vis = batch_to_hstack(np.concatenate(kpt_vis, axis=0))

    # draw heatmap on input rgb
    alpha = 0.5
    heatmap_on_images = input_rgb_h * (1 - alpha) + colored_data(output_heatmap_h, 'coolwarm', 0, 1) * alpha
    if not output_depth is None:
        img_merge = np.vstack(
            [input_rgb_h, output_depth_row, target_depth_row, \
                output_heatmap_row, heatmap_on_images, kpt_vis])
    else:
        img_merge = np.vstack(
            [input_rgb_h, target_depth_row, \
                output_heatmap_row, heatmap_on_images, kpt_vis])
    return img_merge.astype('uint8')

def save_fig_full(config, input_rgb, output_depth, output_heatmap, target_depth, target_heatmap, chamfer_offset_map=None, resize_ratio=0.5, coord=None):
    B, _, H, W = input_rgb.shape
    # denormalize rgb
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # rgb = (rgb - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    input_rgb_np = input_rgb.cpu().detach().numpy() * std.reshape([1, 3, 1, 1]) + mean.reshape([1, 3, 1, 1])
    # input_rgb_np = input_rgb.cpu().detach().numpy()
    input_rgb_h = 255 * batch_to_hstack(input_rgb_np)
    if not output_depth is None:
        output_depth_h = np.squeeze(batch_to_hstack(output_depth.cpu().detach().numpy()))
    if output_heatmap.shape[2] != H and output_heatmap.shape[3] != W: # resize if need
        output_heatmap_orig_np = output_heatmap.cpu().detach().numpy()
        output_heatmap = torch.nn.functional.interpolate(output_heatmap, size=(H, W), mode='nearest')
    output_heatmap_np = output_heatmap.cpu().detach().numpy()
    output_heatmap_h = np.squeeze(batch_to_hstack(output_heatmap_np))
    target_depth_h = np.squeeze(batch_to_hstack(target_depth.data.cpu().detach().numpy()))
    target_heatmap_h = np.squeeze(batch_to_hstack(target_heatmap.data.cpu().detach().numpy()))
    # chamfer_offset_map_h = np.squeeze(batch_to_hstack(chamfer_offset_map.data.cpu().detach().numpy()))

    if not output_depth is None:
        output_depth_row = colored_data(output_depth_h, 'viridis', 0, 7)
    output_heatmap_row = colored_data(output_heatmap_h, 'Reds', 0, 1)
    target_depth_row = colored_data(target_depth_h, 'viridis', 0, 7)
    target_heatmap_row = colored_data(target_heatmap_h, 'Reds', 0, 1)
    # chamfer_offset_map_row = colored_data(chamfer_offset_map_h, 'viridis', 0, 0.2)

    # draw nms keypoints
    kpt_vis = []
    # handling checkerboard artifact
    output_heatmap_np = utils.utils_algo.remove_border_for_batch_heatmap(output_heatmap_np)
    input_rgb_np = 255 * input_rgb_np
    heatmap_vis_conf = config['visualization']['heatmap']
    for i in range(output_heatmap.shape[0]):
        if coord is None:
            single_heatmap = np.squeeze(output_heatmap_np[i,...])
            H, W = single_heatmap.shape
            pts = utils.utils_algo.heatmap_to_pts(single_heatmap, conf_thresh=heatmap_vis_conf['conf_thresh'], nms_dist=heatmap_vis_conf['nms_dist'])
            kpt_vis.append(np.expand_dims(utils.utils_algo.draw_pts_with_confidence(pts, H, W, input_rgb_np[i]), axis=0))
        else:
            single_heatmap = np.squeeze(output_heatmap_orig_np[i,...])
            pts = utils.utils_algo.heatmap_to_pts(single_heatmap, conf_thresh=heatmap_vis_conf['conf_thresh'], nms_dist=heatmap_vis_conf['nms_dist'])
            pts = np.transpose(pts) # Nx3 (x in heatmap, y in heatmap, weight)
            # get coordinate from single_coord
            pts_xy = coord.squeeze()[i, ...].detach().cpu().numpy()[:, pts[:, 1].round().astype(int), pts[:, 0].round().astype(int)]
            pts[:, :2] = pts_xy.transpose(1, 0)
            pts = np.transpose(pts)
            kpt_vis.append(np.expand_dims(utils.utils_algo.draw_pts_with_confidence(pts, H, W, input_rgb_np[i]), axis=0))
    kpt_vis = batch_to_hstack(np.concatenate(kpt_vis, axis=0))

    # draw output heatmap on input rgb
    alpha = 0.5
    output_heatmap_on_images = input_rgb_h * (1 - alpha) + colored_data(output_heatmap_h, 'coolwarm', 0, 1) * alpha

    # draw target_heatmap on input rgb
    target_heatmap_on_images = input_rgb_h * (1 - alpha) + target_heatmap_row * alpha

    if not output_depth is None:
        img_merge = np.vstack(
            [input_rgb_h, output_depth_row, target_depth_row, \
                output_heatmap_row, target_heatmap_on_images, output_heatmap_on_images, kpt_vis])
    else:
        img_merge = np.vstack(
            [input_rgb_h, target_depth_row, \
                output_heatmap_row, target_heatmap_on_images, output_heatmap_on_images, kpt_vis])
    img_merge = cv2.resize(img_merge.astype('uint8'), dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
    return img_merge



def save_fig_auto_choice(config, input_dict, resize_ratio=0.5):
    visualize_list = []
    input_rgb = input_dict['rgb']
    B, _, H, W = input_rgb.shape
    # denormalize rgb
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # rgb = (rgb - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    input_rgb_np = input_rgb.cpu().detach().numpy() * std.reshape([1, 3, 1, 1]) + mean.reshape([1, 3, 1, 1])
    # input_rgb_np = input_rgb.cpu().detach().numpy()
    input_rgb_h = 255 * batch_to_hstack(input_rgb_np)
    visualize_list.append(input_rgb_h)

    if not input_dict.get('output_depth') is None:
        output_depth_h = np.squeeze(batch_to_hstack(input_dict['output_depth'].cpu().detach().numpy()))
        visualize_list.append(colored_data(output_depth_h, 'viridis', 0, 7))

    if not input_dict.get('target_depth') is None:
        target_depth_h = np.squeeze(batch_to_hstack(input_dict['target_depth'].data.cpu().detach().numpy()))
        visualize_list.append(colored_data(target_depth_h, 'viridis', 0, 7))

    if not input_dict.get('output_heatmap') is None:
        output_heatmap = input_dict['output_heatmap']
        if output_heatmap.shape[2] != H and output_heatmap.shape[3] != W: # resize if need
            output_heatmap_orig_np = output_heatmap.cpu().detach().numpy()
            output_heatmap = torch.nn.functional.interpolate(output_heatmap, size=(H, W), mode='nearest')
        output_heatmap_np = output_heatmap.cpu().detach().numpy()
        output_heatmap_h = np.squeeze(batch_to_hstack(output_heatmap_np))
        visualize_list.append(colored_data(output_heatmap_h, 'Reds', 0, 1))
    
    # heatmap from rgb tower
    if not input_dict.get('output_rgb_heatmap') is None:
        output_rgb_heatmap = input_dict['output_rgb_heatmap']
        if output_rgb_heatmap.shape[2] != H and output_rgb_heatmap.shape[3] != W: # resize if need
            output_rgb_heatmap = torch.nn.functional.interpolate(output_rgb_heatmap, size=(H, W), mode='nearest')
        output_rgb_heatmap_np = output_rgb_heatmap.cpu().detach().numpy()
        output_rgb_heatmap_h = np.squeeze(batch_to_hstack(output_rgb_heatmap_np))
        visualize_list.append(colored_data(output_rgb_heatmap_h, 'Reds', 0, 1))

    # heatmap from pcd tower
    if not input_dict.get('output_pcd_heatmap') is None:
        output_pcd_heatmap = input_dict['output_pcd_heatmap']
        if output_pcd_heatmap.shape[2] != H and output_pcd_heatmap.shape[3] != W: # resize if need
            output_pcd_heatmap = torch.nn.functional.interpolate(output_pcd_heatmap, size=(H, W), mode='nearest')
        output_heapcd_tmap_pc = output_pcd_heatmap.cpu().detach().numpy()
        output_hepcd_atmap_p = np.squeeze(batch_to_hstack(output_heapcd_tmap_pc))
        visualize_list.append(colored_data(output_hepcd_atmap_p, 'Reds', 0, 1))
    
    alpha = 0.5
    
    # draw target_heatmap on input rgb
    if not input_dict.get('target_heatmap') is None:
        target_heatmap_h = np.squeeze(batch_to_hstack(input_dict['target_heatmap'].data.cpu().detach().numpy()))
        target_heatmap_on_images = input_rgb_h * (1 - alpha) + colored_data(target_heatmap_h, 'Reds', 0, 1) * alpha
        visualize_list.append(target_heatmap_on_images)
    
    # draw output heatmap on input rgb
    if not input_dict.get('output_heatmap') is None:
        output_heatmap_on_images = input_rgb_h * (1 - alpha) + colored_data(output_heatmap_h, 'coolwarm', 0, 1) * alpha
        visualize_list.append(output_heatmap_on_images)

    # draw nms keypoints
    if not input_dict.get('output_heatmap') is None:
        kpt_vis = []
        # handling checkerboard artifact
        output_heatmap_np = utils.utils_algo.remove_border_for_batch_heatmap(output_heatmap_np)
        input_rgb_np = 255 * input_rgb_np
        heatmap_vis_conf = config['visualization']['heatmap']
        coord = input_dict.get('output_coord')
        for i in range(output_heatmap.shape[0]):
            if coord is None:
                single_heatmap = np.squeeze(output_heatmap_np[i,...])
                H, W = single_heatmap.shape
                pts = utils.utils_algo.heatmap_to_pts(single_heatmap, conf_thresh=heatmap_vis_conf['conf_thresh'], nms_dist=heatmap_vis_conf['nms_dist'])
                kpt_vis.append(np.expand_dims(utils.utils_algo.draw_pts_with_confidence(pts, H, W, input_rgb_np[i]), axis=0))
            else:
                single_heatmap = np.squeeze(output_heatmap_orig_np[i,...])
                pts = utils.utils_algo.heatmap_to_pts(single_heatmap, conf_thresh=heatmap_vis_conf['conf_thresh'], nms_dist=heatmap_vis_conf['nms_dist'])
                pts = np.transpose(pts) # Nx3 (x in heatmap, y in heatmap, weight)
                # get coordinate from single_coord
                pts_xy = coord.squeeze()[i, ...].detach().cpu().numpy()[:, pts[:, 1].round().astype(int), pts[:, 0].round().astype(int)]
                pts[:, :2] = pts_xy.transpose(1, 0)
                pts = np.transpose(pts)
                kpt_vis.append(np.expand_dims(utils.utils_algo.draw_pts_with_confidence(pts, H, W, input_rgb_np[i]), axis=0))
        kpt_vis = batch_to_hstack(np.concatenate(kpt_vis, axis=0))
        visualize_list.append(kpt_vis)

    img_merge = np.vstack(visualize_list)
    img_merge = cv2.resize(img_merge.astype('uint8'), dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
    return img_merge