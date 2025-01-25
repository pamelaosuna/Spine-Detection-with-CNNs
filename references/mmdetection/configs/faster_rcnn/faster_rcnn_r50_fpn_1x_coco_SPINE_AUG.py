# File to build and use the model trained by Tilo

# model settings
# auto_scale_lr = dict(base_batch_size=16, enable=False)
# default_hooks = dict(
#     param_scheduler=dict(type='ParamSchedulerHook'))
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     timer=dict(type='IterTimerHook'),
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        ),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    test_cfg=dict(
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000),
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05))
    )

# dataset settings
dataset_type = 'SpineDataset'
classes = ['spine']
data_root = 'data/raw'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', keep_ratio=True, img_scale=(512, 512)),
    dict(type='RandomFlip', direction='horizontal', flip_ratio=0.0),
    dict(type='RandomFlip', direction='vertical', flip_ratio=0.0),
    dict(type='RandomFlip', direction='diagonal', flip_ratio=0.0),
    dict(type='RandomAffine',
        max_rotate_degree=10.0,
        max_shear_degree=2.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(
            0.5,
            1.5,
        )),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/default_annotations/train.csv',
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/default_annotations/valid.csv',
        img_prefix='',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/default_annotations/test.csv',
        img_prefix='',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
# optimizer_config = dict(grad_clip=dict(max_norm=2, norm_type=2))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=10)

optimizer_config = dict(grad_clip=None)
# test_cfg = dict(type='TestLoop')
# test_dataloader = dict(
#     batch_size=16,
#     dataset=dict(
#         ann_file='annotations/test_cocoformat_all.json',
#         backend_args=None,
#         data_prefix=dict(img='test/'),
#         data_root='data/MOT17_all/',
#         pipeline=[
#             dict(backend_args=None, type='LoadImageFromFile'),
#             dict(keep_ratio=True, scale=(
#                 1333,
#                 800,
#             ), type='Resize'),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 meta_keys=(
#                     'img_id',
#                     'img_path',
#                     'ori_shape',
#                     'img_shape',
#                     'scale_factor',
#                 ),
#                 type='PackDetInputs'),
#         ],
#         test_mode=True,
#         type='CocoDataset'),
#     drop_last=False,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(shuffle=False, type='DefaultSampler'))
# test_evaluator = dict(
#     ann_file='data/MOT17_all/annotations/test_cocoformat_all.json',
#     backend_args=None,
#     format_only=False,
#     metric='bbox',
#     type='CocoMetric')
# train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
# train_dataloader = dict(
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     batch_size=16,
#     dataset=dict(
#         ann_file='annotations/train_cocoformat_all.json',
#         backend_args=None,
#         data_prefix=dict(img='train/'),
#         data_root='data/MOT17_all/',
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=[
#             dict(backend_args=None, type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(keep_ratio=True, scale=(
#                 1333,
#                 800,
#             ), type='Resize'),
#             dict(direction='horizontal', type='RandomFlip', flip_ratio=0.0),
#             dict(direction='vertical', type='RandomFlip', flip_ratio=0.0),
#             dict(direction='diagonal', type='RandomFlip', flip_ratio=0.0),
#             dict(
#                 max_rotate_degree=10.0,
#                 max_shear_degree=2.0,
#                 max_translate_ratio=0.1,
#                 scaling_ratio_range=(
#                     0.5,
#                     1.5,
#                 ),
#                 type='RandomAffine'),
#             dict(type='PackDetInputs'),
#         ],
#         type='CocoDataset'),
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(shuffle=True, type='DefaultSampler'))


# val_cfg = dict(type='ValLoop')
# val_dataloader = dict(
#     batch_size=16,
#     dataset=dict(
#         ann_file='annotations/val_cocoformat_all.json',
#         backend_args=None,
#         data_prefix=dict(img='val/'),
#         data_root='data/MOT17_all/',
#         pipeline=[
#             dict(backend_args=None, type='LoadImageFromFile'),
#             dict(keep_ratio=True, scale=(
#                 1333,
#                 800,
#             ), type='Resize'),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 meta_keys=(
#                     'img_id',
#                     'img_path',
#                     'ori_shape',
#                     'img_shape',
#                     'scale_factor',
#                 ),
#                 type='PackDetInputs'),
#         ],
#         test_mode=True,
#         type='CocoDataset'),
#     drop_last=False,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(shuffle=False, type='DefaultSampler'))
# val_evaluator = dict(
#     ann_file='data/MOT17_all/annotations/val_cocoformat_all.json',
#     backend_args=None,
#     format_only=False,
#     metric='bbox',
#     type='CocoMetric')
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'tutorial_exps'