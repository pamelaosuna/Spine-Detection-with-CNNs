#!/bin/bash

#SBATCH --job-name=eval-spine
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=3-00:00:00

echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate spinedet
which python3
echo "loaded conda environment"
echo "start evaluation..."
date

python3 -u src/spine_detection/eval_mmdet.py \
 --config tutorial_exps/FasterRCNN_RCNN_model/Simons_1/lr_0.001_warmup_None_momentum_0.6_L2_3e-06_aug_Simon_1/config.py \
 --checkpoint tutorial_exps/FasterRCNN_RCNN_model/Simons_1/lr_0.001_warmup_None_momentum_0.6_L2_3e-06_aug_Simon_1/epoch_13.pth \
 --new_val_ann_file data/Altug_1/default_annotations/valid.csv \
 --new_val_img_prefix data/Altug_1/raw/


echo "job finished"
date