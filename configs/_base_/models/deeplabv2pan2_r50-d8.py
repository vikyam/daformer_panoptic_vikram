# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoderPanoptic',
    pretrained='open-mmlab://resnet50_v1c',
    activate_panoptic=False,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),

    decode_head=dict(
        debug='',
        activate_panoptic=False,
        type='DLV2HeadPanoptic',
        in_channels=2048,
        in_index=3,
        dilations=(6, 12, 18, 24),
        num_classes=19,
        align_corners=False,
        init_cfg=dict(type='Normal', std=0.01,
                      override=[
                                # dict(name='mtl_block'),
                                dict(name='semanitc_head'),
                                dict(name='instance_head'),
                                dict(name='center_sub_head'),
                                dict(name='offset_sub_head'),
                                dict(name='depth_head'),
                                ]
                      ),
        loss_decode=[
                        # dict(type='CrossEntropyLossDada', loss_name='loss_semantic', loss_weight=1.0), # DADA semantic loss
                        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                        # dict(type='MSELoss', loss_weight=0.001, reduction='mean'), # when MTLBlock is deactivated
                        dict(type='MSELoss', loss_weight=0.0005, reduction='mean'), # when MTLBlock is activated
                        dict(type='L1Loss', loss_weight=0.01, reduction='mean'),
                        # dict(type='BerHuLoss', loss_weight=0.001),
                    ]
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
