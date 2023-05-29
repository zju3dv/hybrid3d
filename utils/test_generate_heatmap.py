import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import argparse
import cv2
import sys, os
import json
from pathlib import Path
import time

class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat
    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)

def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline();
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape = (4, 4))
            for i in range(4):
                matstr = f.readline();
                mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj



if __name__ == "__main__":
    pose_file = sys.argv[1]
    keypoint_file = sys.argv[2]
    image_file = sys.argv[3]
    depth_file = sys.argv[4]

    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) * 1e-3

    keypoints = read_json(keypoint_file)
    keypoints = np.asarray(keypoints)
    
    img = plt.imread(image_file)
    traj = read_trajectory(pose_file)
    focal = 525
    cx = 319.5
    cy = 239.5
    width = 640
    height = 480
    # origin input : Twc
    Twc = traj[0]
    Rwc = Twc.pose[:3, :3]
    twc = Twc.pose[:3, 3]
    time_start = time.time()
    H, W, _ = img.shape
    heatmap = np.zeros((H, W, 1), np.uint8)
    # the second np.transpose is only for batch compute rotation
    pts_cam = np.matmul(np.transpose(Rwc), np.transpose(keypoints - twc))
    pts_cam = np.transpose(pts_cam)
    for i in range(pts_cam.shape[0]):
        pt = pts_cam[i]
        if pt[2] < 0: continue
        pt_plane = ((pt[0] / pt[2]) * focal + cx, (pt[1] / pt[2]) * focal + cy)
        if pt_plane[0] > width - 1 or pt_plane[0] < 0 or pt_plane[1] > height - 1 or pt_plane[1] < 0: continue
        pt_plane_int = tuple(round(x) for x in pt_plane)
        # print(pt_plane_int)
        if abs(pt[2] - depth[pt_plane_int[1], pt_plane_int[0]]) > 0.1: continue
        # cv2.drawMarker(img, pt_plane_int, (0, 255, 0))
        cv2.circle(heatmap, center=pt_plane_int, radius=8, color=(255), thickness=-1, lineType=cv2.LINE_AA)
    time_end = time.time()
    print('time', time_end - time_start)
    heatmap = np.squeeze(heatmap)
    plt.imshow(heatmap, cmap='Reds')
    plt.show()
