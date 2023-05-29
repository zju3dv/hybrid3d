import sys
import os
sys.path.append(os.getcwd())  # noqa
import numpy as np
import cv2
import shutil
from utils.util import ensure_dir


class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


scene_list = [
    'apartment',
    'bedroom',
    'boardroom',
    'lobby',
    'loft'
]

if __name__ == '__main__':
    for scene in scene_list:
        pose_path = f'/home/ybbbbt/Data-2T/redwood_lidar/pose/pose_{scene}/{scene}.log'
        rgbd_src_path = f'/home/ybbbbt/Data-2T/redwood_lidar/rgbd/rgbd_{scene}/{scene}/'
        rgbd_dst_path = f'/mnt/ssd_disk/redwood_lidar_3dmatch_format/rgbd/{scene}/seq-01/'

        ensure_dir(rgbd_dst_path)

        print(scene)

        traj = read_trajectory(pose_path)
        for idx, pose in enumerate(traj):
            print('\r%06d : %06d' % (idx, len(traj)), end='')
            Twc = pose.pose
            image_src_name = '{}/image/{:06d}.jpg'.format(rgbd_src_path, idx)
            depth_src_name = '{}/depth/{:06d}.png'.format(rgbd_src_path, idx)
            image_dst_name = '{}/frame-{:06d}.color.png'.format(rgbd_dst_path, idx)
            depth_dst_name = '{}/frame-{:06d}.depth.png'.format(rgbd_dst_path, idx)
            pose_dst_name = '{}/frame-{:06d}.pose.txt'.format(rgbd_dst_path, idx)
            np.savetxt(pose_dst_name, Twc, delimiter=' \t')
            shutil.copy2(depth_src_name, depth_dst_name)
            cv2.imwrite(image_dst_name, cv2.imread(image_src_name))
