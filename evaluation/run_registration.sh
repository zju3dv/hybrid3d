eval "$(conda shell.bash hook)"

conda activate hf
folder_name=saved/log_hf_1005_144842_H3DNetCoordSkip_X3DMatchFragmentDataLoader_desc_w_5.0_nms_2_no_dropout_epoch99_8c18e3


kpts_nums=(
50
100
250
500
1000
2500
5000
)

for ((i = 0; i < ${#kpts_nums[@]}; ++i)); do
    kpts_num=${kpts_nums[$i]}
    python evaluation/registration_3dmatch.py $kpts_num /home/ybbbbt/Data-2T/3dmatch/geometric_registration_adaptive $folder_name
done
date