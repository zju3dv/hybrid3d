# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/rgbd_integration_uniform.py

import open3d as o3d
import sys
import os

sys.path.append(os.getcwd())  # noqa
from tools.trajectory_io import *
import numpy as np

scenes = [
    '7-scenes-chess',
    '7-scenes-redkitchen',
    '7-scenes-fire',
    '7-scenes-heads',
    '7-scenes-office',
    '7-scenes-pumpkin',
    '7-scenes-stairs',
    # 'analysis-by-synthesis-apt1-kitchen',
    # 'analysis-by-synthesis-apt1-living',
    # 'analysis-by-synthesis-apt2-bed',
    # 'analysis-by-synthesis-apt2-kitchen',
    # 'analysis-by-synthesis-apt2-living',
    # 'analysis-by-synthesis-apt2-luke',
    # 'analysis-by-synthesis-office2-5a',
    # 'analysis-by-synthesis-office2-5b'
    # 'bundlefusion-apt0',
    # 'bundlefusion-apt1',
    # 'bundlefusion-apt2',
    # 'bundlefusion-copyroom',
    # 'bundlefusion-office0',
    # 'bundlefusion-office1',
    # 'bundlefusion-office2',
    # 'bundlefusion-office3',
    # 'rgbd-scenes-v2-scene_01',
    # 'rgbd-scenes-v2-scene_02',
    # 'rgbd-scenes-v2-scene_03',
    # 'rgbd-scenes-v2-scene_04',
    # 'rgbd-scenes-v2-scene_05',
    # 'rgbd-scenes-v2-scene_06',
    # 'rgbd-scenes-v2-scene_07',
    # 'rgbd-scenes-v2-scene_08',
    # 'rgbd-scenes-v2-scene_09',
    # 'rgbd-scenes-v2-scene_10',
    # 'rgbd-scenes-v2-scene_11',
    # 'rgbd-scenes-v2-scene_12',
    # 'rgbd-scenes-v2-scene_13',
    # 'rgbd-scenes-v2-scene_14',
    # 'sun3d-brown_bm_1-brown_bm_1',
    # 'sun3d-brown_bm_4-brown_bm_4',
    # 'sun3d-brown_cogsci_1-brown_cogsci_1',
    # 'sun3d-brown_cs_2-brown_cs2',
    # 'sun3d-brown_cs_3-brown_cs3',
    # 'sun3d-harvard_c11-hv_c11_2',
    # 'sun3d-harvard_c3-hv_c3_1',
    # 'sun3d-harvard_c5-hv_c5_1',
    # 'sun3d-harvard_c6-hv_c6_1',
    # 'sun3d-harvard_c8-hv_c8_3',
    # 'sun3d-home_at-home_at_scan1_2013_jan_1',
    # 'sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika',
    # 'sun3d-home_md-home_md_scan9_2012_sep_30',
    # 'sun3d-hotel_nips2012-nips_4',
    # 'sun3d-hotel_sf-scan1',
    # 'sun3d-hotel_uc-scan3',
    # 'sun3d-hotel_umd-maryland_hotel1',
    # 'sun3d-hotel_umd-maryland_hotel3',
    # 'sun3d-mit_32_d507-d507_2',
    # 'sun3d-mit_46_ted_lab1-ted_lab_2',
    # 'sun3d-mit_76_417-76-417b',
    # 'sun3d-mit_76_studyroom-76-1studyroom2',
    # 'sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika',
    # 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
    # 'sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika'
]

if __name__ == "__main__":
    data_base_dir = 'data/3dmatch/rgbd/'
    output_base_dir = 'data/3dmatch/reconstruction/'
    for scene in scenes:
        print('current scene: ', scene)
        scene_path = os.path.join(data_base_dir, scene)
        K = read_3dmatch_camera_intrinsics(os.path.join(scene_path, 'camera-intrinsics.txt'))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            640, 480, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        # read camera poses
        camera_poses = []
        for i in range(1, 1000):
            seq_path = '{}/seq-{:02d}'.format(scene_path, i)
            if not os.path.isdir(seq_path):
                continue
            print('read camera_poses from {}'.format(seq_path))
            camera_poses += read_3dmatch_trajectory(seq_path)
        
        
        # volume = o3d.integration.UniformTSDFVolume(
        #     length=4.0,
        #     resolution=512,
        #     sdf_trunc=0.04,
        #     color_type=o3d.integration.TSDFVolumeColorType.RGB8,
        # )
        voxel_length = 6.0 / 512
        trunc_margin = 5 * voxel_length
        volume = o3d.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            # voxel_length=6.0/512,
            # sdf_trunc=0.02,
            sdf_trunc=trunc_margin,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        # camera_poses = camera_poses[:500]
        depth_trunc = 60.0
        if 'sun3d' in scene:
            depth_trunc = 4.0
        for i, camera_pose in enumerate(camera_poses):
            print("\rIntegrate {:d} : {:d}".format(i, len(camera_poses)), end='')
            info = camera_pose.metadata
            color = o3d.io.read_image(info['rgb_filename'])
            depth = o3d.io.read_image(info['depth_filename'])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                camera_intrinsics,
                np.linalg.inv(camera_poses[i].pose),
            )

        output_scene_dir = os.path.join(output_base_dir, scene)
        ensure_dir(output_scene_dir)

        print("Extract triangle mesh")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh])
        o3d.io.write_triangle_mesh("{}/mesh.ply".format(output_scene_dir), mesh)

        print("Extract voxel-aligned debugging point cloud")
        voxel_pcd = volume.extract_voxel_point_cloud()
        # o3d.visualization.draw_geometries([voxel_pcd])
        o3d.io.write_point_cloud("{}/voxel_pcd.ply".format(output_scene_dir), voxel_pcd)

        # print("Extract voxel-aligned debugging voxel grid")
        # voxel_grid = volume.extract_voxel_grid()
        # o3d.visualization.draw_geometries([voxel_grid])
        # o3d.io.write_voxel_grid("{}/voxel_grid.ply".format(output_scene_dir), voxel_grid)

        print("Extract point cloud")
        pcd = volume.extract_point_cloud()
        # o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud("{}/pcd.ply".format(output_scene_dir), pcd)
