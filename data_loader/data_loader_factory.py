import data_loader.x3dmatch as x3dmatch
import data_loader.redwood_lidar as redwood_lidar
import data_loader.redwood_lidar_fragment as redwood_lidar_fragment
import data_loader.x3dmatch_fragment as x3dmatch_fragment

data_loader_dict = {
    'RedwoodLidarDataLoader' : redwood_lidar,
    'X3DMatchDataLoader' : x3dmatch,
    'X3DMatchFragmentDataLoader' : x3dmatch_fragment,
    'RedwoodLidarFragDataLoader' : redwood_lidar_fragment
}

def get_data_loader_by_name(name):
    return data_loader_dict[name]
