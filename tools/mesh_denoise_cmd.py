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

if __name__ == "__main__":
    raw_mesh_path = os.sys.argv[1]
    clean_mesh_path = os.sys.argv[2]
    clean_mesh = remove_small_clusters(raw_mesh_path)
    o3d.io.write_triangle_mesh(clean_mesh_path, clean_mesh)

