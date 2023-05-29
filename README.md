# Hybrid3D: Learning 3D Hybrid Features with Point Clouds and Multi-View Images for Point Cloud Registration


# Installation

```bash
conda env create -f environment.yml
conda activate hf
```

# Data Preparation

We provide scene fragment fusion in `fuse_scene_fragment.py`, fragment indices generation in `generate_fragment_indices.py` and the pre-computation of overlapping area in `generate_overlapping_areas.py`.

The preprocessed data and the pre-trained model would be avaiable in the future.

# Training

```bash
# stage 1: training 2d coordinates
python train.py -c config/server_3dmatch_coord.yaml -i rgbd_stage
# stage 2: training 3d fusion
python train.py -c config/server_3dmatch_fusion.yaml --start_epoch 1 -r saved/model_xxx/checkpoint.pth  -i fusion_stage
```

# Testing

See `run_registration.sh` and `eval_registration_recall.sh` in evaluation folder.
