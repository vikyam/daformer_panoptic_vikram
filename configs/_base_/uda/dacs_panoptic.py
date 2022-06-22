# Baseline UDA
uda = dict(
    type='DACSPanoptic',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=4000,
    print_grad_magnitude=False,
    panoptic_eval_interval=5000,
)
use_ddp_wrapper = True
