import cv2
import numpy as np
import os
import time
from shutil import copyfile
from pathlib import Path

scene_list = [
    '7-scenes-chess',
    '7-scenes-fire',
    '7-scenes-heads',
    '7-scenes-office',
    '7-scenes-pumpkin',
    '7-scenes-redkitchen',
    '7-scenes-stairs',
]


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        print(dirname)
        os.makedirs(dirname, exist_ok=True)


def pose_inverse(T):
    T_inv = T.copy()
    T_inv[:3, :3] = np.transpose(T[:3, :3])
    # print(-np.transpose(T[:3, :3]))
    # print(T[:3, 3].reshape(3, 1))
    T_inv[:3, 3] = np.dot(-np.transpose(T[:3, :3]), T[:3, 3])
    return T_inv


def warp_depth(depth, src_K, src_Twc, dst_K, dst_Twc):
    ys, xs = np.where(depth > 0)
    warped_depth = np.zeros_like(depth)
    pts = np.empty((len(xs), 3))
    pts[:, 0] = (xs - src_K[2]) / src_K[0]
    pts[:, 1] = (ys - src_K[3]) / src_K[1]
    pts[:, 2] = 1.0
    pts = pts * np.repeat(depth[ys, xs].reshape(-1, 1), 3, axis=1)
    # lift to space
    pts_w = np.transpose(np.matmul(src_Twc[:3, :3], np.transpose(pts))) + src_Twc[:3, 3]

    # project back
    pts_cam = np.matmul(np.transpose(dst_Twc[:3, :3]), np.transpose(pts_w - dst_Twc[:3, 3]))
    pts_cam = np.transpose(pts_cam)

    # remove pts out of frustum
    fx, fy, cx, cy = dst_K
    H, W = warped_depth.shape
    pts_cam = pts_cam[pts_cam[:, 2] > 0, ...]
    ds = pts_cam[:, 2]
    pts_plane = np.array([(pts_cam[:, 0] / pts_cam[:, 2]) * fx + cx, (pts_cam[:, 1] / pts_cam[:, 2]) * fy + cy])
    pts_plane = np.transpose(pts_plane).astype(int)
    valid_mask = (pts_plane[:, 0] <= W - 1) & (pts_plane[:, 0] >=
                                               0) & (pts_plane[:, 1] <= H - 1) & (pts_plane[:, 1] >= 0)
    ds = ds[valid_mask]
    pts_plane = pts_plane[valid_mask]

    warped_depth[pts_plane[:, 1], pts_plane[:, 0]] = ds
    return warped_depth


if __name__ == "__main__":
    src_dir = 'data/3dmatch/preprocess_7scenes/7scenes_not_aligned/'
    dst_dir = 'data/3dmatch/preprocess_7scenes/7scenes_align/'
    src_K = (585.0, 585.0, 320.0, 240.0)
    dst_K = (520.0, 520.0, 320.0, 240.0)
    for scene in scene_list:
        scene_path = os.path.join(src_dir, scene)
        dst_scene_path = os.path.join(dst_dir, scene)
        for seq_idx in range(1, 100):
            seq_path = '{}/seq-{:02d}'.format(scene_path, seq_idx)
            dst_seq_path = '{}/seq-{:02d}'.format(dst_scene_path, seq_idx)
            if not os.path.isdir(seq_path):
                continue
            print(dst_seq_path)
            ensure_dir(dst_seq_path)
            for i in range(999999):
                print('\r%06d' % i, end='')
                pose_file = '{}/frame-{:06d}.pose.txt'.format(seq_path, i)
                if not os.path.isfile(pose_file):
                    break
                rgb_file = '{}/frame-{:06d}.color.png'.format(seq_path, i)
                depth_file = '{}/frame-{:06d}.depth.png'.format(seq_path, i)
                # convert pose from depth to rgb camera
                src_Twc = np.loadtxt(pose_file)
                dst_Twc = src_Twc.copy()
                dst_Twc[:3, 3] -= np.matmul(src_Twc[:3, :3], np.asarray([-0.0245, 0, 0]))
                # warp depth
                depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) * 1.0e-3

                warped_depth = warp_depth(depth, src_K, src_Twc, dst_K, dst_Twc)
                warped_depth[warped_depth > 60] = 0
                warped_depth *= 1.0e3
                warped_depth = warped_depth.astype(np.uint16)

                dst_depth_file = '{}/frame-{:06d}.depth.png'.format(dst_seq_path, i)
                dst_rgb_file = '{}/frame-{:06d}.color.png'.format(dst_seq_path, i)
                dst_pose_file = '{}/frame-{:06d}.pose.txt'.format(dst_seq_path, i)

                cv2.imwrite(dst_depth_file, warped_depth)
                np.savetxt(dst_pose_file, dst_Twc, delimiter=' \t')
                copyfile(rgb_file, dst_rgb_file)

