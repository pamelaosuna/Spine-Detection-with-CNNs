#!/bin/bash

#SBATCH --job-name=spine
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate spinedet
which python3
echo "loaded conda environment"
echo "start inference..."
date

# lr_0.001_warmup_None_momentum_0.6_L2_3e-06_aug_A2_F101 
PARAM_CONFIG=lr_0.001_warmup_None_momentum_0.6_L2_3e-06_aug_S1A2 #  # lr_0.0001_warmup_None_momentum_None_L2_0.05_aug_A2_F50-Tilo # 
MODEL_NAME=DefDETR # FasterRCNN #
MODEL_TYPE=Def_DETR # Faster-RCNN-101 # 
EPOCH_NUMBER=epoch_55 # epoch_7 # epoch_45
PART=valid

for DATASET in S1A1 S1 A1 A2 S1A2
do
    python -u src/spine_detection/predict_mmdet.py \
        --input "data/$DATASET/$PART/*.png" \
        --model $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --param_config $PARAM_CONFIG \
        --model_epoch $EPOCH_NUMBER \
        --theta 0.5 \
        --delta 0.5 \
        --output output/prediction/$MODEL_NAME/$PARAM_CONFIG/${PART}_${DATASET} \
        # --log_level debug
        # --save_images
done

# ## 2nd
# PARAM_CONFIG=lr_0.001_warmup_None_momentum_0.6_L2_3e-06_aug_A1 #  # lr_0.0001_warmup_None_momentum_None_L2_0.05_aug_A2_F50-Tilo # 
# MODEL_NAME=DefDETR # FasterRCNN # 
# MODEL_TYPE=Def_DETR # Faster-RCNN-50 # 
# EPOCH_NUMBER=epoch_57 # epoch_7 # epoch_45
# PART=valid

# for DATASET in S1A1 S1 A1 A2 S1A2
# do
#     python -u src/spine_detection/predict_mmdet.py \
#         --input "data/$DATASET/$PART/*.png" \
#         --model $MODEL_NAME \
#         --model_type $MODEL_TYPE \
#         --param_config $PARAM_CONFIG \
#         --model_epoch $EPOCH_NUMBER \
#         --theta 0.5 \
#         --delta 0.5 \
#         --output output/prediction/$MODEL_NAME/$PARAM_CONFIG/${PART}_${DATASET} \
#         # --log_level debug
#         # --save_images
# done

# ## 3rd
# PARAM_CONFIG=lr_0.001_warmup_None_momentum_0.6_L2_3e-06_aug_A2 #  # lr_0.0001_warmup_None_momentum_None_L2_0.05_aug_A2_F50-Tilo # 
# MODEL_NAME=DefDETR # FasterRCNN # 
# MODEL_TYPE=Def_DETR # Faster-RCNN-50 # 
# EPOCH_NUMBER=epoch_46 # epoch_7 # epoch_45
# PART=valid

# for DATASET in S1A1 S1 A1 A2 S1A2
# do
#     python -u src/spine_detection/predict_mmdet.py \
#         --input "data/$DATASET/$PART/*.png" \
#         --model $MODEL_NAME \
#         --model_type $MODEL_TYPE \
#         --param_config $PARAM_CONFIG \
#         --model_epoch $EPOCH_NUMBER \
#         --theta 0.5 \
#         --delta 0.5 \
#         --output output/prediction/$MODEL_NAME/$PARAM_CONFIG/${PART}_${DATASET} \
#         # --log_level debug
#         # --save_images
# done

# ## 4th

# PARAM_CONFIG=lr_0.001_warmup_None_momentum_0.6_L2_3e-06_aug_S1A1 #  # lr_0.0001_warmup_None_momentum_None_L2_0.05_aug_A2_F50-Tilo # 
# MODEL_NAME=DefDETR # FasterRCNN # 
# MODEL_TYPE=Def_DETR # Faster-RCNN-50 # 
# EPOCH_NUMBER=epoch_45 # epoch_7 # epoch_45
# PART=valid

# for DATASET in S1A1 S1 A1 A2 S1A2
# do
#     python -u src/spine_detection/predict_mmdet.py \
#         --input "data/$DATASET/$PART/*.png" \
#         --model $MODEL_NAME \
#         --model_type $MODEL_TYPE \
#         --param_config $PARAM_CONFIG \
#         --model_epoch $EPOCH_NUMBER \
#         --theta 0.5 \
#         --delta 0.5 \
#         --output output/prediction/$MODEL_NAME/$PARAM_CONFIG/${PART}_${DATASET} \
#         # --log_level debug
#         # --save_images
# done

# ## 5th

# # PARAM_CONFIG=lr_0.001_warmup_None_momentum_0.6_L2_3e-06_aug_S1A1 #  # lr_0.0001_warmup_None_momentum_None_L2_0.05_aug_A2_F50-Tilo # 
# # MODEL_NAME=DefDETR # FasterRCNN # 
# # MODEL_TYPE=Def_DETR # Faster-RCNN-50 # 
# # EPOCH_NUMBER=epoch_6 # epoch_7 # epoch_45
# # PART=valid

# # for DATASET in S1A1 S1 A1 A2 S1A2
# # do
# #     python -u src/spine_detection/predict_mmdet.py \
# #         --input "data/$DATASET/$PART/*.png" \
# #         --model $MODEL_NAME \
# #         --model_type $MODEL_TYPE \
# #         --param_config $PARAM_CONFIG \
# #         --model_epoch $EPOCH_NUMBER \
# #         --theta 0.5 \
# #         --delta 0.5 \
# #         --output output/prediction/$MODEL_NAME/$PARAM_CONFIG/${PART}_${DATASET} \
# #         # --log_level debug
# #         # --save_images
# # done

echo job finished
date