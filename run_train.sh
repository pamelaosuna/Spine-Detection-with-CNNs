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
echo "start training..."
date

DATASET=A2

python3 -u src/spine_detection/train_mmdet.py \
 --model FasterRCNN --model_type Faster-RCNN-101 \
 --device cuda:0 --learning_rate 0.001 \
 --max_epochs 50 --weight_decay 3e-06 --image_dir data/images \
 -sp _aug_${DATASET}_F101 --annotation_dir data/${DATASET}/default_annotations/ \
 --use_aug \
 --momentum 0.6 \

# for DATASET in S1A1 S1 A1 A2 S1A2
# do
#     python3 -u src/spine_detection/train_mmdet.py \
#     --model FasterRCNN --model_type Faster-RCNN-50 \
#     --device cuda:0 --learning_rate 0.0001 \
#     --max_epochs 50 --weight_decay 0.05 --image_dir data/images \
#     -sp _aug_${DATASET}_F50-Tilo --annotation_dir data/${DATASET}/default_annotations/ \
#     --use_aug
# done
# echo $CUDA_HOME

# for DATASET in S1A1 S1 A1 A2 S1A2
# do
#     python3 -u src/spine_detection/train_mmdet.py \
#     --model DefDETR --model_type Def_DETR \
#     --device cuda:0 --learning_rate 0.001 --momentum 0.6 \
#     --max_epochs 80 --weight_decay 3e-06 --image_dir data/images \
#     -sp _aug_${DATASET} --annotation_dir data/${DATASET}/default_annotations/ \
#     --use_aug
# done

echo "job finished"
date