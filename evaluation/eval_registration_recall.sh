
input_path=$1
# e.g. ../../saved/log_hf_0929_114904_H3DNet_v8_X3DMatchFragmentDataLoader_no_div_heatmap_cluster_min_pt_5_no_dropout_for_desc_epoch87_c40c4b/log_result_250

ABSOLUTE_PATH=$(cd $input_path; pwd)

matlab -nosplash -nodisplay -nodesktop -r "cd evaluation/3dmatch/; evaluate_cmd $ABSOLUTE_PATH ../../data/3dmatch/geometric_registration_adaptive; exit"


