import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.getcwd())  # noqa
import utils.vis_3d as vis_3d
import utils.utils_algo as utils_algo
import utils.util as util
from utils.sampler import FarthestSampler
from utils.vis_3d_o3d import map_to_color, O3dVisualizer

if __name__ == '__main__':
    voxel_size = float(sys.argv[1])
    max_K_div = float(sys.argv[2])
    pcd_path = sys.argv[3]
    pcd = o3d.io.read_point_cloud(pcd_path, print_progress=True)
    pcd_np = np.asarray(pcd.points).copy()

    pcd_downsample = pcd.voxel_down_sample(voxel_size=voxel_size)

    max_K = np.asarray(pcd_downsample.points).shape[0] // max_K_div

    print('max_K', max_K)
    
    sampler = FarthestSampler()

    pcd_sampled_np = sampler.sample(pcd_np, int(max_K))

    o3d.visualization.RenderOption.line_width = 8.0
    visualizer = O3dVisualizer()

    visualizer.add_np_points(pcd_np, size=0.005, color=map_to_color(pcd_np[:, 0], cmap='viridis'))
    visualizer.add_np_points(pcd_sampled_np, size=0.02, color=(0, 1, 0))
    
    visualizer.run_visualize()
