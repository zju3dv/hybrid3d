# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/trajectory_io.py

import numpy as np
import os.path
from pathlib import Path

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


def write_trajectory(traj, filename):
    with open(filename, 'w') as f:
        for x in traj:
            p = x.pose.tolist()
            f.write(' '.join(map(str, x.metadata)) + '\n')
            f.write('\n'.join(
                ' '.join(map('{0:.12f}'.format, p[i])) for i in range(4)))
            f.write('\n')

def get_3dmatch_frame_num(dir):
    cnt = 0
    while(True):
        if not os.path.isfile("{}/frame-{:06d}.color.png".format(dir, cnt)):
            return cnt
        cnt += 1

def read_3dmatch_trajectory(dir):
    traj = []
    for i in range(get_3dmatch_frame_num(dir)):
        mat = np.zeros((4, 4))
        filename = os.path.join(dir, 'frame-{:06d}.pose.txt'.format(i))
        with open(filename, 'r') as f:
            for j in range(4):
                matstr = f.readline()
                mat[j, :] = np.fromstring(matstr, dtype=float, sep=' \t')
        info = {
            'idx': i,
            'rgb_filename': os.path.join(dir, 'frame-{:06d}.color.png'.format(i)),
            'depth_filename': os.path.join(dir, 'frame-{:06d}.depth.png'.format(i))
        }
        traj.append(CameraPose(info, mat))
    return traj

def read_3dmatch_camera_intrinsics(filename):
    mat = np.zeros((3, 3))
    with open(filename, 'r') as f:
        for j in range(3):
            matstr = f.readline()
            mat[j, :] = np.fromstring(matstr, dtype=float, sep=' \t')
    return mat

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
