import model.model_vote_fusion_v3 as module_vote_fusion_v3
import model.model_vote_fusion_fcgf_only as model_vote_fusion_fcgf_only
import model.model_vote_fusion_direct_assign as model_vote_fusion_direct_assign

module_arch = {
    'VoteFusionModuleV3': module_vote_fusion_v3,
    'VoteFusionModuleFCGFOnly': model_vote_fusion_fcgf_only,
    'VoteFusionModuleDirectAssign' : model_vote_fusion_direct_assign,
}

def get_vote_model_by_name(name):
    return module_arch[name]
