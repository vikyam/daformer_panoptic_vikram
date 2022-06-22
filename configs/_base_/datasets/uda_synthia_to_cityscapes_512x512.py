# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) # original
crop_size = (512, 512)
synthia_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPanopticAnnotations'),
    dict(type='LoadDepthAnnotations'),
    dict(type='Resize', img_scale=(1280, 760)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='GenPanopLabels', sigma=8, mode='train'), # TODO: make sure that mode is set with the correct value "train" or "val",
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img']),
    # dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_center',
                               'center_weights', 'gt_offset', 'offset_weights',
                               'gt_instance_seg', 'gt_depth_map']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadPanopticAnnotations'), # in UDA we dont need annotation on target - so we dont load the panotpic PNG
    # dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),  # TODO:for mapillary you need to add padding - for that follow your previous cvpr 2022 mapillry dataloader code and add a new transform
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='GenPanopLabels', sigma=8),  # in UDA we dont need annotation on target - so we dont generate the panoptic gt from the PANOPITC PNG and segmentinfo
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
    # dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
# use this below config to visualize cityscapes panoptic labels on trainset
# cityscapes_train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadPanopticAnnotations'), # in UDA we dont need annotation on target - so we dont load the panotpic PNG
#     dict(type='Resize', img_scale=(1024, 512)),  # TODO:for mapillary you need to add padding - for that follow your previous cvpr 2022 mapillry dataloader code and add a new transform
#     dict(type='RandomCrop', crop_size=crop_size),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='GenPanopLabels', sigma=8, mode='val'),  # in UDA we dont need annotation on target - so we dont generate the panoptic gt from the PANOPITC PNG and segmentinfo
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_center', 'center_weights', 'gt_offset', 'offset_weights', 'gt_instance_seg', 'gt_foreground_seg']),
#     ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='SynthiaDataset',
            data_root='data/synthia/',
            img_dir='RGB',
            depth_dir='Depth',
            ann_dir='panoptic-labels-crowdth-0-for-daformer/synthia_panoptic', # 'GT/LABELS',
            pipeline=synthia_train_pipeline),
        target=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            depth_dir='Depth', # not in use
            ann_dir='gtFine_panoptic/cityscapes_panoptic_train_trainId', # 'gtFine/train',
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        depth_dir='Depth', # not in use
        ann_dir='gtFine_panoptic/cityscapes_panoptic_val',        # ACTUAL MODE
        # ann_dir='gtFine_panoptic_debug/cityscapes_panoptic_val',    # DEBUG MODE you dont need to activate this here, it is hardcoded in mmseg/apis/train.py, whenever debug is true,  ann_dir will be set to 'gtFine_panoptic_debug/cityscapes_panoptic_val'
        pipeline=test_pipeline),)

    # test=dict(
    #     type='CityscapesDataset',
    #     data_root='data/cityscapes/',
    #     img_dir='leftImg8bit/val',
    #     ann_dir='gtFine/val',
    #     pipeline=test_pipeline))
