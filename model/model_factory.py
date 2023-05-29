import model.model_3d_coord_skip as module_arch_3d_coord_skip
import model.model_multi_tower as module_arch_multi_tower
import model.model_coord_vote as module_coord_vote
import model.model_rgb_only as module_rgb_only

module_arch = {
    'H3DNetCoordSkip': module_arch_3d_coord_skip,
    'H3DMultiTower': module_arch_multi_tower,
    'H3DNetCoordVote': module_coord_vote,
    'H3DCoordRGB': module_rgb_only,
}

def get_model_by_name(name):
    return module_arch[name]
