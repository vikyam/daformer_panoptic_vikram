# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Add ddp_wrapper from mmgen

import random
import warnings

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

from mmseg.core import DistEvalHook, EvalHook
from mmseg.core.ddp_wrapper import DistributedDataParallelWrapper
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
import os
from tools.panoptic_deeplab.utils import create_panop_eval_folders


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if cfg.debug:
        shuffle = False
    else:
        shuffle = True
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True,
            shuffle=shuffle, # TODO
                        ) for ds in dataset
                    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        use_ddp_wrapper = cfg.get('use_ddp_wrapper', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if use_ddp_wrapper:
            mmcv.print_log('Use DDP Wrapper.', 'mmseg')
            model = DistributedDataParallelWrapper(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer) # original: uncomment this during actual training
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.9) # dummy comment this during actual training

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # --- get the cfg['evaluation'] dict defined in experiments.py ---
        eval_cfg = cfg.get('evaluation', {})

        # --- cfg.runner['type'] = 'IterBasedRunner', so eval_cfg['by_epoch'] is False ---
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'

        # --- distributed is False so eval_hook = EvalHook ---
        eval_hook = DistEvalHook if distributed else EvalHook

        # --- this is for eval_type='daformer' ---
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

        # --- this is for eval_type='panop_deeplab' ---
        eval_cfg['eval_type'] = 'panop_deeplab'
        if cfg.debug:
            eval_cfg['interval'] = 3 # TODO
        else:
            eval_cfg['interval'] = runner.max_iters # the panoptic eval is done only once at the end of the training
        eval_cfg['panop_eval_folder'] = os.path.join(cfg['work_dir'], 'panoptic_eval')
        # create the panoptic evaluation root and sub folders reauired by the panoptic deeplab evaluation script
        # eval_cfg['panop_eval_temp_folder'] is created to store the intermediate semantic, instance prdictions as PNGs
        # and once evaluation is done they are removed. The final evaluation results are saved in the training log
        eval_cfg['panop_eval_temp_folder'] = create_panop_eval_folders(eval_cfg['panop_eval_folder'])
        if eval_cfg['dataset_name'] == 'Cityscapes':
            eval_cfg['gt_dir'] = os.path.join(cfg.data.val.data_root, 'gtFine', 'val')
            if cfg.debug:
                cfg.data.val.ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
            eval_cfg['gt_dir_panop'] = os.path.join(cfg.data.val.data_root, cfg.data.val.ann_dir.split('/')[0])
        elif eval_cfg['dataset_name'] == 'Mapillary':
            raise NotImplementedError("No implementation found for Mapillary!")
        # eval_cfg['debug'] = cfg['debug']
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
