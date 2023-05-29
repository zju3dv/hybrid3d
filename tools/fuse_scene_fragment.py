# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/rgbd_integration_uniform.py

import open3d as o3d
import sys
import os
import numpy as np
import glob
import subprocess
import shutil
from pathlib import Path
import json
import cv2
from tqdm import tqdm

sys.path.append(os.getcwd())  # noqa
from tools.trajectory_io import *

# scenes = [
#     '7-scenes-chess',
#     '7-scenes-redkitchen',
#     '7-scenes-fire',
#     '7-scenes-heads',
#     '7-scenes-office',
#     '7-scenes-pumpkin',
#     '7-scenes-stairs',
#     'analysis-by-synthesis-apt1-kitchen',
#     'analysis-by-synthesis-apt1-living',
#     'analysis-by-synthesis-apt2-bed',
#     'analysis-by-synthesis-apt2-kitchen',
#     'analysis-by-synthesis-apt2-living',
#     'analysis-by-synthesis-apt2-luke',
#     'analysis-by-synthesis-office2-5a',
#     'analysis-by-synthesis-office2-5b',
#     'bundlefusion-apt0',
#     'bundlefusion-apt1',
#     'bundlefusion-apt2',
#     'bundlefusion-copyroom',
#     'bundlefusion-office0',
#     'bundlefusion-office1',
#     'bundlefusion-office2',
#     'bundlefusion-office3',
#     'rgbd-scenes-v2-scene_01',
#     'rgbd-scenes-v2-scene_02',
#     'rgbd-scenes-v2-scene_03',
#     'rgbd-scenes-v2-scene_04',
#     'rgbd-scenes-v2-scene_05',
#     'rgbd-scenes-v2-scene_06',
#     'rgbd-scenes-v2-scene_07',
#     'rgbd-scenes-v2-scene_08',
#     'rgbd-scenes-v2-scene_09',
#     'rgbd-scenes-v2-scene_10',
#     'rgbd-scenes-v2-scene_11',
#     'rgbd-scenes-v2-scene_12',
#     'rgbd-scenes-v2-scene_13',
#     'rgbd-scenes-v2-scene_14',
#     'sun3d-brown_bm_1-brown_bm_1',
#     'sun3d-brown_bm_4-brown_bm_4',
#     'sun3d-brown_cogsci_1-brown_cogsci_1',
#     'sun3d-brown_cs_2-brown_cs2',
#     'sun3d-brown_cs_3-brown_cs3',
#     'sun3d-harvard_c11-hv_c11_2',
#     'sun3d-harvard_c3-hv_c3_1',
#     'sun3d-harvard_c5-hv_c5_1',
#     'sun3d-harvard_c6-hv_c6_1',
#     'sun3d-harvard_c8-hv_c8_3',
#     'sun3d-home_at-home_at_scan1_2013_jan_1',
#     'sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika',
#     'sun3d-home_md-home_md_scan9_2012_sep_30',
#     'sun3d-hotel_nips2012-nips_4',
#     'sun3d-hotel_sf-scan1',
#     'sun3d-hotel_uc-scan3',
#     'sun3d-hotel_umd-maryland_hotel1',
#     'sun3d-hotel_umd-maryland_hotel3',
#     'sun3d-mit_32_d507-d507_2',
#     'sun3d-mit_46_ted_lab1-ted_lab_2',
#     'sun3d-mit_76_417-76-417b',
#     'sun3d-mit_76_studyroom-76-1studyroom2',
#     'sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika',
#     'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
#     'sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika'
# ]

scenes = [
    '7-scenes-redkitchen',
    'sun3d-home_at-home_at_scan1_2013_jan_1',
    'sun3d-home_md-home_md_scan9_2012_sep_30',
    'sun3d-hotel_uc-scan3',
    'sun3d-hotel_umd-maryland_hotel1',
    'sun3d-hotel_umd-maryland_hotel3',
    'sun3d-mit_76_studyroom-76-1studyroom2',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
]

# scenes = [
#     'apartment',
#     'bedroom',
#     'boardroom',
#     'lobby',
#     'loft'
# ]

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def open3d_fusion(K, camera_poses, pcd_path):
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    voxel_length = 6.0 / 512
    trunc_margin = 5 * voxel_length
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        # voxel_length=6.0/512,
        # sdf_trunc=0.02,
        sdf_trunc=trunc_margin,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    for i, camera_pose in enumerate(camera_poses):
        info = camera_pose.metadata
        color = o3d.io.read_image(info['rgb_filename'])
        depth = o3d.io.read_image(info['depth_filename'])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=6, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            camera_intrinsics,
            np.linalg.inv(np.linalg.inv(camera_poses[0].pose) @ camera_poses[i].pose),
        )
    pcd = volume.extract_point_cloud()
        # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(pcd_path, pcd)

def get_volumn_size(cam_intrinsic, camera_poses, voxel_size, depth_clip=20):
    import tools.fusion_module as fusion
    vol_bnds = np.zeros((3,2))
    for i, camera_pose in enumerate(camera_poses):
        info = camera_pose.metadata
        # print("\rReading Depth frame %d/%d"%(i+1, len(camera_poses)), end='')
        # Read depth image and camera pose
        depth_im = cv2.imread(info['depth_filename'], cv2.IMREAD_ANYDEPTH).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > depth_clip] = 0
        cam_pose = camera_pose.pose

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intrinsic, np.linalg.inv(camera_poses[0].pose) @ cam_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    return np.ceil((vol_bnds[:,1]-vol_bnds[:,0])/voxel_size).copy(order='C').astype(int)

def adaptive_fusion(cam_intrinsic, camera_poses, pcd_path, depth_clip=20):
    import tools.fusion_module as fusion
    # 
    vol_bnds = np.zeros((3,2))
    for i, camera_pose in enumerate(camera_poses):
        info = camera_pose.metadata
        print("\rReading Depth frame %d/%d"%(i+1, len(camera_poses)), end='')
        # Read depth image and camera pose
        depth_im = cv2.imread(info['depth_filename'], cv2.IMREAD_ANYDEPTH).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > depth_clip] = 0
        cam_pose = camera_pose.pose

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intrinsic, np.linalg.inv(camera_poses[0].pose) @ cam_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    # Initialize voxel volume
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.006)
    for i, camera_pose in enumerate(camera_poses):
        print("\rFusing frame %d/%d"%(i+1, len(camera_poses)), end='')
        info = camera_pose.metadata
        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread(info['rgb_filename']), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(info['depth_filename'], cv2.IMREAD_ANYDEPTH).astype(float)
        # print(info['depth_filename'])
        # print(depth_im.shape)
        depth_im /= 1000.
        depth_im[depth_im > depth_clip] = 0
        cam_pose = camera_pose.pose
        if depth_im.shape != color_image.shape[:2]:
            color_image = cv2.resize(color_image, (depth_im[0], depth_im[1], 3))
        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intrinsic, np.linalg.inv(camera_poses[0].pose) @ cam_pose, obs_weight=1.)
    point_cloud = tsdf_vol.get_point_cloud()
    # fusion.pcwrite(pcd_path, point_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    o3d.io.write_point_cloud(pcd_path, pcd)

numFramesPerFrag = 50

# fuse_method = '3dmatch'
fuse_method = 'open3d'
# fuse_method = 'adaptive'

if __name__ == "__main__":
    # data_base_dir = 'data/3dmatch/rgbd/'
    data_base_dir = 'data/3dmatch/rgbd/'
    # output_base_dir = 'data/3dmatch/fragments_adaptive/'
    output_base_dir = 'data/3dmatch/fragments_color/'
    # data_base_dir = 'data/redwood_lidar/rgbd/'
    # output_base_dir = 'data/redwood_lidar/fragments_adaptive/'
    # output_base_dir = 'data/redwood_lidar/fragments_color/'
    fragment_list = []
    for scene in scenes:
        print('current scene: ', scene)
        scene_path = os.path.join(data_base_dir, scene)
        cam_K_path = os.path.join(scene_path, 'camera-intrinsics.txt')
        K = read_3dmatch_camera_intrinsics(os.path.join(scene_path, 'camera-intrinsics.txt'))
        ensure_dir('{}/{}'.format(output_base_dir, scene))
        fragment_idx = 0
        for seq_idx in range(1, 3):
            seq_name = 'seq-{:02d}'.format(seq_idx)
            seq_path = '{}/{}'.format(scene_path, seq_name)
            if not os.path.isdir(seq_path):
                continue
            print('read camera_poses from {}'.format(seq_path))
            camera_poses = read_3dmatch_trajectory(seq_path)
            end_idx = len(camera_poses) - numFramesPerFrag - 1
            for frame_idx in tqdm(range(0, end_idx, numFramesPerFrag)):
                # print('\r{} / {}'.format(frame_idx, end_idx), end='')
                fragment_pcd_file = '{}/{}/cloud_bin_{}.ply'.format(output_base_dir, scene, fragment_idx)

                # 3DMatch ToolBox .cu fusion
                if fuse_method == '3dmatch':
                    voxelSize = 0.006
                    truncMargin = voxelSize * 5
                    voxelGridOrigin = [-1.5,-1.5,0.5]
                    vol_size = get_volumn_size(K, camera_poses[fragment_idx: fragment_idx+numFramesPerFrag], voxelSize, 6)
                    if vol_size[0] * vol_size[1] * vol_size[2] > 900**3 or vol_size[0] * vol_size[1] * vol_size[2] <= 0:
                        print('Large scene out of bound: ', scene, frame_idx, vol_size)
                        vol_size[0] = vol_size[1] = vol_size[2] = 900
                    cmd_output = subprocess.check_output(
                        ['tools/fragment/demo', cam_K_path, seq_path, str(frame_idx), str(frame_idx), str(numFramesPerFrag),\
                            str(voxelGridOrigin[0]), str(voxelGridOrigin[1]), str(voxelGridOrigin[2]), str(voxelSize), str(truncMargin),\
                                str(vol_size[0]), str(vol_size[1]), str(vol_size[2])]).decode(sys.stdout.encoding)
                    shutil.move('tsdf.ply', fragment_pcd_file)
                # Open3D Fusion
                elif fuse_method == 'open3d':
                    open3d_fusion(K, camera_poses[fragment_idx*numFramesPerFrag: fragment_idx*numFramesPerFrag+numFramesPerFrag], fragment_pcd_file)
                elif fuse_method == 'adaptive':
                    adaptive_fusion(K, camera_poses[fragment_idx*numFramesPerFrag: fragment_idx*numFramesPerFrag+numFramesPerFrag], fragment_pcd_file)
                
                fragment_info_file = '{}/{}/cloud_bin_{}.info.txt'.format(output_base_dir, scene, fragment_idx)
                with open(fragment_info_file, 'w') as f:
                    f.write('%s\t %s\t %d\t %d\t\n'%(scene, seq_name, frame_idx, frame_idx + numFramesPerFrag - 1))
                    mat = camera_poses[frame_idx].pose
                    for i in range(4):
                        f.write('%15.8e\t %15.8e\t %15.8e\t %15.8e\t\n'%(mat[i, 0], mat[i, 1], mat[i, 2], mat[i, 3]))
                fragment_list.append([scene, fragment_idx, seq_name, frame_idx, frame_idx + numFramesPerFrag - 1])
                fragment_idx += 1
            # print('')
    write_json(fragment_list, '{}/fragment_list_v2.json'.format(output_base_dir))
