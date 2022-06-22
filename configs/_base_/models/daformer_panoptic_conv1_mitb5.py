# This is the same as SegFormer but with 256 embed_dims
# SegF. with C_e=256 in Tab. 7

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoderPanoptic',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        debug='',
        activate_panoptic=False,
        type='DAFormerHeadPanoptic',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='conv',
                kernel_size=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg),
        ),
        loss_decode=[
                        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                        # dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
                        dict(type='MSELoss', loss_weight=10.0, reduction='mean'),
                        dict(type='L1Loss', loss_weight=0.1, reduction='mean'),
                        # dict(type='BerHuLoss', loss_weight=0.001),
                    ]

    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

