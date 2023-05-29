import open3d as o3d
import numpy as np
import copy
import os
import sys

# adopt code from http://www.open3d.org/docs/release/tutorial/Basic/mesh.html?highlight=mesh#Connected-components
def remove_small_clusters(input_mesh_path, small_cluster_th=500, visualize=False):
    mesh = o3d.io.read_triangle_mesh(input_mesh_path, print_progress=True)
    vert = np.asarray(mesh.vertices)
    min_vert, max_vert = vert.min(axis=0), vert.max(axis=0)
    for _ in range(30):
        cube = o3d.geometry.TriangleMesh.create_box()
        cube.scale(0.005, center=cube.get_center())
        cube.translate(
            (
                np.random.uniform(min_vert[0], max_vert[0]),
                np.random.uniform(min_vert[1], max_vert[1]),
                np.random.uniform(min_vert[2], max_vert[2]),
            ),
            relative=False,
        )
        mesh += cube
    mesh.compute_vertex_normals()
    if visualize:
        print("Show input mesh")
        o3d.visualization.draw_geometries([mesh])


    # print("Cluster connected triangles")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)


    mesh_0 = copy.deepcopy(mesh)
    cluster_tri_count = cluster_n_triangles[triangle_clusters]
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < small_cluster_th
    mesh_0.remove_triangles_by_mask(triangles_to_remove)

    if visualize:
        print("Show mesh with small clusters removed")
        o3d.visualization.draw_geometries([mesh_0])
    return mesh_0

scenes = [
    '7-scenes-chess',
    '7-scenes-redkitchen',
    '7-scenes-fire',
    '7-scenes-heads',
    '7-scenes-office',
    '7-scenes-pumpkin',
    '7-scenes-stairs',
    # analysis-by-synthesis already has good mesh
    'analysis-by-synthesis-apt1-kitchen',
    'analysis-by-synthesis-apt1-living',
    'analysis-by-synthesis-apt2-bed',
    'analysis-by-synthesis-apt2-kitchen',
    'analysis-by-synthesis-apt2-living',
    'analysis-by-synthesis-apt2-luke',
    'analysis-by-synthesis-office2-5a',
    'analysis-by-synthesis-office2-5b',
    # bundlefusion already has good mesh
    'bundlefusion-apt0',
    'bundlefusion-apt1',
    'bundlefusion-apt2',
    'bundlefusion-copyroom',
    'bundlefusion-office0',
    'bundlefusion-office1',
    'bundlefusion-office2',
    'bundlefusion-office3',
    'rgbd-scenes-v2-scene_01',
    'rgbd-scenes-v2-scene_02',
    'rgbd-scenes-v2-scene_03',
    'rgbd-scenes-v2-scene_04',
    'rgbd-scenes-v2-scene_05',
    'rgbd-scenes-v2-scene_06',
    'rgbd-scenes-v2-scene_07',
    'rgbd-scenes-v2-scene_08',
    'rgbd-scenes-v2-scene_09',
    'rgbd-scenes-v2-scene_10',
    'rgbd-scenes-v2-scene_11',
    'rgbd-scenes-v2-scene_12',
    'rgbd-scenes-v2-scene_13',
    'rgbd-scenes-v2-scene_14',
    'sun3d-brown_bm_1-brown_bm_1',
    'sun3d-brown_bm_4-brown_bm_4',
    'sun3d-brown_cogsci_1-brown_cogsci_1',
    'sun3d-brown_cs_2-brown_cs2',
    'sun3d-brown_cs_3-brown_cs3',
    'sun3d-harvard_c11-hv_c11_2',
    'sun3d-harvard_c3-hv_c3_1',
    'sun3d-harvard_c5-hv_c5_1',
    'sun3d-harvard_c6-hv_c6_1',
    'sun3d-harvard_c8-hv_c8_3',
    'sun3d-home_at-home_at_scan1_2013_jan_1',
    'sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika',
    'sun3d-home_md-home_md_scan9_2012_sep_30',
    'sun3d-hotel_nips2012-nips_4',
    'sun3d-hotel_sf-scan1',
    'sun3d-hotel_uc-scan3',
    'sun3d-hotel_umd-maryland_hotel1',
    'sun3d-hotel_umd-maryland_hotel3',
    'sun3d-mit_32_d507-d507_2',
    'sun3d-mit_46_ted_lab1-ted_lab_2',
    'sun3d-mit_76_417-76-417b',
    'sun3d-mit_76_studyroom-76-1studyroom2',
    'sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
    'sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika'
]

if __name__ == "__main__":
    output_base_dir = 'data/3dmatch/reconstruction/'
    for scene in scenes:
        print('current scene: ', scene)
        output_scene_dir = os.path.join(output_base_dir, scene)
        raw_mesh_path = "{}/mesh.ply".format(output_scene_dir)
        clean_mesh_path = "{}/mesh_clean.ply".format(output_scene_dir)
        clean_pcd_path =  "{}/pcd_clean.ply".format(output_scene_dir)
        # mesh denoise
        if not os.path.isfile(clean_mesh_path):
            clean_mesh = remove_small_clusters(raw_mesh_path)
            o3d.io.write_triangle_mesh(clean_mesh_path, clean_mesh)
        else:
            clean_mesh = o3d.io.read_triangle_mesh(clean_mesh_path)

        # sample
        if not os.path.isfile(clean_pcd_path):
            pcd = clean_mesh.sample_points_uniformly(number_of_points=2000000)
            pcd = pcd.voxel_down_sample(voxel_size=0.001)
            o3d.io.write_point_cloud(clean_pcd_path, pcd)
