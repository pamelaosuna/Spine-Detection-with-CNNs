import argparse
import os
import sys

import mmcv
from mmcv import Config
from mmdet.apis import init_detector, single_gpu_test
from mmdet.datasets import build_dataset, build_dataloader
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, build_runner
from mmdet.models import build_detector
from mmdet.core import EvalHook
from mmdet.utils import get_root_logger

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)) # remove for pull request
sys.path.append(parent_dir) # remove when pull request
from spine_detection.utils.data_utils import DATASETS, SpineDataset
from spine_detection.utils.logger_utils import setup_custom_logger
from spine_detection.utils.model_utils import (
    get_checkpoint_path,
    get_config_path,
    get_pretrained_checkpoint_path,
    load_config,
    parse_args,
)

def evaluate_model(config_file, checkpoint_file, new_val_ann_file, new_val_img_prefix):
    # Load the configuration file
    cfg = Config.fromfile(config_file)
    
    # Modify the validation dataset
    cfg.data.val.ann_file = new_val_ann_file
    cfg.data.val.img_prefix = new_val_img_prefix
    
    # Build the validation dataset
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    
    # Build the model and load the checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.CLASSES = ['spine'] # checkpoint['meta']['CLASSES']
    model = model.cuda()
    model.eval()
    
    # Build the dataloader
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )
    
    # Set up the logger
    log_file = os.path.join(os.path.dirname(checkpoint_file), 'eval.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)


    # # Run only the evaluation
    # runner.call_hook('before_run')
    # runner.call_hook('before_epoch')
    # runner.call_hook('before_val_epoch')
    # runner.call_hook('after_val_epoch')
    # runner.call_hook('after_epoch')
    # runner.call_hook('after_run')
    
    # Run the evaluation
     # Run the evaluation
    outputs = single_gpu_test(model, val_dataloader, show=False)
    # outputs = []
    # for i, data in enumerate(val_dataloader):
    #     with torch.no_grad():
    #         data = scatter(collate([data], samples_per_gpu=1), [torch.cuda.current_device()])[0]
    #         result = model(return_loss=False, rescale=True, **data)
    #         outputs.append(result)
    
    # Evaluate the results
    eval_results = val_dataset.evaluate(outputs, metric=['bbox'])
    # print(eval_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on a new validation partition')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint file path')
    parser.add_argument('--new_val_ann_file', help='new validation annotation file path')
    parser.add_argument('--new_val_img_prefix', help='new validation image prefix')
    args = parser.parse_args()
    
    evaluate_model(args.config, args.checkpoint, args.new_val_ann_file, args.new_val_img_prefix)