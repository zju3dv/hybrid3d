import cv2
import os
import numpy as np


path_list = [
    'data/3dmatch/raw_download/analysis-by-synthesis/apt1/kitchen/data',
    'data/3dmatch/raw_download/analysis-by-synthesis/apt1/living/data',
    'data/3dmatch/raw_download/analysis-by-synthesis/apt2/bed/data',
    'data/3dmatch/raw_download/analysis-by-synthesis/apt2/kitchen/data',
    'data/3dmatch/raw_download/analysis-by-synthesis/apt2/living/data',
    'data/3dmatch/raw_download/analysis-by-synthesis/apt2/luke/data',
    'data/3dmatch/raw_download/analysis-by-synthesis/office2/5a/data',
    'data/3dmatch/raw_download/analysis-by-synthesis/office2/5b/data'
]

dst_path_list = [
    'data/3dmatch/rgbd/analysis-by-synthesis-apt1-kitchen/seq-01',
    'data/3dmatch/rgbd/analysis-by-synthesis-apt1-living/seq-01',
    'data/3dmatch/rgbd/analysis-by-synthesis-apt2-bed/seq-01',
    'data/3dmatch/rgbd/analysis-by-synthesis-apt2-kitchen/seq-01',
    'data/3dmatch/rgbd/analysis-by-synthesis-apt2-living/seq-01',
    'data/3dmatch/rgbd/analysis-by-synthesis-apt2-luke/seq-01',
    'data/3dmatch/rgbd/analysis-by-synthesis-office2-5a/seq-01',
    'data/3dmatch/rgbd/analysis-by-synthesis-office2-5b/seq-01'
]

if __name__ == "__main__":
    for path, dst_path in zip(path_list, dst_path_list):
        print(path)
        j = 0
        has_invalid_pose = False
        for i in range(999999):
            filename = '{}/frame-{:06d}.color.jpg'.format(path, i)
            if not os.path.isfile(filename):
                break
            mat = np.loadtxt('{}/frame-{:06d}.pose.txt'.format(path, i))
            if mat.shape != (4, 4):
                # print(i)
                has_invalid_pose = True
                continue
            im = cv2.imread(filename)
            im2 = cv2.resize(im, (640, 480))
            cv2.imwrite('{}/frame-{:06d}.color.png'.format(dst_path, j), im2)
            j += 1
        print('has_invalid_pose:', has_invalid_pose)
