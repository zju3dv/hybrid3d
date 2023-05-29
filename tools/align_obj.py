import numpy as np
import open3d as o3d


def decompose_to_sRT(Trans):
    t = Trans[:3, 3]
    R = Trans[:3, :3]
    # assume x y z have the same scale
    scale = np.linalg.norm(R[:3, 0])
    R = R / scale
    print(scale, R, t)
    return scale, R, t


# yml_dict = read_yaml('/home/ybbbbt/Data/helmet_demo_reconstruction/zju_east_floor_3/sfm-export/1F.yaml')

# print(yml_dict)

# trans_orig = [ 6.1108805910063098e-03, 3.1335935884086979e-01,
#        7.8483500386322209e+00, 0., -7.8531099276272727e+00,
#        1.5314903568565685e-01, 0., 0., -1.5302336204882380e-01,
#        -7.8468576641948289e+00, 3.1342219565928631e-01,
#        1.6556637646499086e+00 ]

trans_orig = [-1.1384392865462958e-03, 6.0515316451277272e-02,
       -1.6322197686606004e+00, 3.9138491959260784e+02,
       1.6330527730166098e+00, 3.0716627292381116e-02, 0.,
       1.4037359815318518e+02, 3.0695393848012245e-02,
       -1.6319306671488087e+00, -6.0526749844398965e-02,
       2.6667082782862850e+00 ]

trans_orig = np.array(trans_orig).reshape(3, 4)

trans = np.eye(4)

trans[:3, :4] = trans_orig


scale, R, t = decompose_to_sRT(trans)

# mesh = o3d.io.read_triangle_mesh('/home/ybbbbt/Data/helmet_demo_reconstruction/zju_mengminwei_1027/dense-export/model_fill.obj')
mesh = o3d.io.read_triangle_mesh('/home/ybbbbt/Data/helmet_demo_reconstruction/guobo_f2_1120/guobo_f2_1120_dense-export/dense-export/model_src.obj')

print(scale, R, t)

mesh.transform(trans)

o3d.io.write_triangle_mesh('model_trans_fill.obj', mesh)

# o3d.visualization.draw_geometries([mesh])



