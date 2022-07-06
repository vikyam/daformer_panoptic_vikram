# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Additional dataset location logging

import os
import os.path as osp
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose
from .pipelines import GenPanopLabels

import json
from PIL import Image
import time
import torch.nn.functional as F

from tools.panoptic_deeplab.eval import SemanticEvaluator, CityscapesInstanceEvaluator, CityscapesPanopticEvaluator
from tools.panoptic_deeplab.utils import rgb2id
from tools.panoptic_deeplab.post_processing import get_semantic_segmentation, get_panoptic_segmentation, get_cityscapes_instance_format

# for visualization of val predictions
from matplotlib import pyplot as plt
from mmseg.models.utils.visualization import subplotimg, subplotimgV2
from mmseg.models.utils.dacs_transforms import denorm
from mmseg.models.utils.visualization import prep_sem_for_vis, prep_cnt_for_vis, prep_ofs_for_vis, prep_ins_for_vis, prep_pan_for_vis
from mmseg.models.utils.visualization import save_validation_visuals

@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 depth_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)
            if not osp.isabs(self.depth_dir):
                self.depth_dir = osp.join(self.data_root, self.depth_dir)

        # load annotations
        self.img_infos = self.load_annotations_panoptic(self.ann_dir)
        # self.img_infos = self.load_annotations(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split) # original
        self.gen_panop_labels = GenPanopLabels(8, 'val')
        #
        self.best_miou = -1.0

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations_panoptic(self, ann_dir):
        img_infos = []
        json_filename = ann_dir + '.json'
        dataset = json.load(open(json_filename))
        self.files = {}
        for ano in dataset['annotations']:
            img_info = {}
            if 'synthia' in self.data_root:
                ano_fname = ano['file_name']
                seg_fname = ano['image_id'] + self.seg_map_suffix
            elif 'cityscapes' in self.data_root:
                ano_fname = ano['image_id']
                str1 = ano_fname.split('_')[0] + '/' + ano_fname
                ano_fname = str1 + '_leftImg8bit.png'
                seg_fname = str1 + self.seg_map_suffix
            img_info['filename'] = ano_fname
            img_info['ann'] = {}
            img_info['ann']['seg_map'] = seg_fname
            img_info['ann']['segments_info'] = ano['segments_info']
            img_infos.append(img_info)
        print_log(
            f'Loaded {len(img_infos)} images from {self.img_dir}',
            logger=get_root_logger())
        return img_infos


    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['depth_prefix'] = self.depth_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def get_gt_semantic_labels(self):
        """Get ground truth panoptic labels for evaluation."""
        gt_semantic_labels = []
        for img_info in self.img_infos:
            filename = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            panop_lbl_dict = {}
            gt_panoptic_seg = Image.open(filename)
            gt_panoptic_seg = np.asarray(gt_panoptic_seg, dtype=np.float32)  # the id values are > 255, we need np.float32 # (760,1280,3)
            panop_lbl_dict['gt_panoptic_seg'] = gt_panoptic_seg
            panop_lbl_dict['ann_info'] = {}
            panop_lbl_dict['ann_info']['segments_info'] = img_info['ann']['segments_info']
            panop_lbl_dict['seg_fields'] = []
            panop_lbl_dict['seg_fields'].append('gt_panoptic_seg')
            panoptic_labels = self.gen_panop_labels(panop_lbl_dict)
            gt_semantic_labels.append(panoptic_labels['gt_semantic_seg'])
        return gt_semantic_labels

    def get_gt_panoptic_labels(self, device, logger, debug):
        """Get ground truth panoptic labels for evaluation."""
        if debug:
            img_list = ['frankfurt/frankfurt_000000_000294_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_000576_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_001016_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_001236_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_001751_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_002196_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_002963_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_003025_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_003357_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_003920_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_004617_gtFine_panoptic.png',
                        'frankfurt/frankfurt_000000_005543_gtFine_panoptic.png']
        gt_panoptic_labels = []
        count_img = 0
        logger.info('')
        if debug:
            log_interval = 1
        else:
            log_interval = 50

        for img_info in self.img_infos:

            if debug and img_info['ann']['seg_map'] not in img_list:
                continue
            filename = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            panop_lbl_dict = {}
            gt_panoptic_seg = Image.open(filename)
            if debug:
                gt_panoptic_seg = gt_panoptic_seg.resize((1024,512), Image.NEAREST)
            gt_panoptic_seg = np.asarray(gt_panoptic_seg, dtype=np.float32)  # the id values are > 255, we need np.float32 # (760,1280,3)
            panop_lbl_dict['gt_panoptic_seg'] = gt_panoptic_seg
            panop_lbl_dict['ann_info'] = {}
            panop_lbl_dict['ann_info']['segments_info'] = img_info['ann']['segments_info']
            panop_lbl_dict['seg_fields'] = []
            panop_lbl_dict['seg_fields'].append('gt_panoptic_seg')
            data = self.gen_panop_labels(panop_lbl_dict)
            gt_panoptic_labels.append([data['gt_semantic_seg'], data['gt_center'], data['gt_offset'], data['gt_instance_seg']])
            if count_img % log_interval == 0:
                logger.info(f'generating panoptic labels for imgid:  {count_img+1}')
            count_img+=1
        return gt_panoptic_labels

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate_panoptic(self, results, device=None, panop_eval_temp_folder=None,
                          dataset_name=None, gt_dir=None, debug=False, num_samples_debug=None,
                          gt_dir_panop=None, logger=None):

        # get all the GT panoptic labels for all images in the val set
        gt_panoptic_labels = self.get_gt_panoptic_labels(device, logger, debug)

        # nbytes_sem = results[0]['semantic'].nbytes
        # nbytes_cnt = results[0]['center'].nbytes
        # nbytes_ofs = results[0]['offset'].nbytes
        # total_mem_results = ((nbytes_sem + nbytes_cnt + nbytes_ofs) * 500) / 1e+9
        # nbytes_sem_lbl = gt_panoptic_labels[0][0].nbytes
        # tot_mem_lbl = (nbytes_sem_lbl * 500) / 1e+9
        # logger.info('')
        # logger.info(f'total memory requirements for results {total_mem_results} GB')
        # logger.info(f'total memory requirements for gt_panoptic_labels {tot_mem_lbl} GB')
        # logger.info('')

        if debug:
            log_interval=1
            num_samples = num_samples_debug
        else:
            log_interval = 50
            num_samples = len(gt_panoptic_labels)
        eval_folder = {}
        eval_folder['semantic'] = os.path.join(panop_eval_temp_folder, 'semantic')
        eval_folder['instance'] = os.path.join(panop_eval_temp_folder, 'instance')
        eval_folder['panoptic'] = os.path.join(panop_eval_temp_folder, 'panoptic')
        eval_folder['visuals'] = os.path.join(panop_eval_temp_folder, 'visuals')

        image_filename_list = []
        if dataset_name == 'Cityscapes':
            panoptic_josn_file = 'cityscapes_panoptic_val.json'
            panoptic_json_folder = 'cityscapes_panoptic_val'
            for i in range(len(self.img_infos)):
                image_filename_list.append(self.img_infos[i]['ann']['seg_map'].split('.')[0])
            stuff_area = 2048
            input_image_size = (2048, 1024)
        elif dataset_name == 'Mapillary':
            stuff_area = 4096
            input_image_size = (1024, 768)
        else:
            raise NotImplementedError(f'Implementation not found for dataset: {dataset_name}')
        # inputs
        num_classes = 19
        ignore_label = 255
        train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0]

        mapillary_dataloading_style = 'DADA'
        label_divisor = 1000

        cityscapes_thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        CENTER_THRESHOLD = 0.5
        NMS_KERNEL = 7
        TOP_K_INSTANCE = 200
        from tools.panoptic_deeplab.utils import AverageMeter
        post_time = AverageMeter()
        timing_warmup_iter = 10
        INSTANCE_SCORE_TYPE = 'semantic'
        # DUMP_PANOPTIC_VISUAL_IMGS = True

        semantic_metric = SemanticEvaluator(
            num_classes=num_classes,
            ignore_label=ignore_label,
            output_dir=eval_folder['semantic'],
            train_id_to_eval_id=train_id_to_eval_id,
            logger=logger,
            dataset_name=dataset_name,
        )
        instance_metric = CityscapesInstanceEvaluator(
            output_dir=eval_folder['instance'],
            train_id_to_eval_id=train_id_to_eval_id,
            gt_dir=gt_dir,
            num_classes=num_classes,
            DEBUG=debug,
            num_samples=num_samples_debug,
            dataset_name=dataset_name,
            rgb2id=rgb2id,
            input_image_size=input_image_size,
            mapillary_dataloading_style=mapillary_dataloading_style,
            logger=logger,
        )
        panoptic_metric = CityscapesPanopticEvaluator(
            output_dir=eval_folder['panoptic'],
            train_id_to_eval_id=train_id_to_eval_id,
            label_divisor=label_divisor,
            void_label=label_divisor * ignore_label,
            gt_dir=gt_dir_panop,
            split='val',
            num_classes=num_classes,
            panoptic_josn_file=panoptic_josn_file,
            panoptic_json_folder=panoptic_json_folder,
            debug=debug,
            target_dataset_name=dataset_name,
            input_image_size=input_image_size,
            mapillary_dataloading_style=mapillary_dataloading_style,
            logger=logger,
        )
        image_filename_list_debug = []
        try:
            for i in range(num_samples):
                image_filename = image_filename_list[i].split('/')[1]
                if i == timing_warmup_iter:
                    post_time.reset()
                start_time = time.time()
                out_dict = {}
                out_dict['semantic'] = torch.from_numpy(results[i]['semantic']).to(device)
                out_dict['center'] = torch.from_numpy(results[i]['center']).to(device)
                out_dict['offset'] = torch.from_numpy(results[i]['offset']).to(device)
                gt_labels = {}
                gt_labels['semantic'] = torch.from_numpy(gt_panoptic_labels[i][0]).to(device)
                # the following three gt labels are not used by the evaluation script, they are just used for visualization purpose
                gt_labels['center'] = gt_panoptic_labels[i][1]
                gt_labels['offset'] = gt_panoptic_labels[i][2]
                gt_labels['gt_instance_seg'] = gt_panoptic_labels[i][3]

                semantic_pred = get_semantic_segmentation(out_dict['semantic'])
                panoptic_pred, center_pred = get_panoptic_segmentation(
                    semantic_pred,
                    out_dict['center'],
                    out_dict['offset'],
                    thing_list=cityscapes_thing_list,
                    label_divisor=label_divisor,
                    stuff_area=stuff_area,
                    void_label=(label_divisor * ignore_label),
                    threshold=CENTER_THRESHOLD,
                    nms_kernel=NMS_KERNEL,
                    top_k=TOP_K_INSTANCE,
                    foreground_mask=None)
                torch.cuda.synchronize(device)
                post_time.update(time.time() - start_time)

                if i % log_interval == 0:
                    logger.info('[{}/{}]\tPost-processing Time: {post_time.val:.3f}s ({post_time.avg:.3f}s)\t'.format(i, num_samples, post_time=post_time))

                semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
                panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()
                # Evaluates semantic segmentation.
                semantic_metric.update(semantic_pred, gt_labels['semantic'].squeeze(0).cpu().numpy(), image_filename, debug=debug, logger=logger)
                # Evaluates instance segmentation.
                raw_semantic = F.softmax(out_dict['semantic'], dim=1)
                # raw_semantic = F.interpolate(raw_semantic, size=upsample_dim, mode='bilinear', align_corners=False)  # Consistent with OpenCV.
                center_hmp = out_dict['center']
                # center_hmp = F.interpolate(center_hmp, size=upsample_dim, mode='bilinear', align_corners=False)  # Consistent with OpenCV.
                raw_semantic = raw_semantic.squeeze(0).cpu().numpy()
                center_hmp = center_hmp.squeeze(1).squeeze(0).cpu().numpy()
                instances = get_cityscapes_instance_format(panoptic_pred, raw_semantic, center_hmp, label_divisor=label_divisor, score_type=INSTANCE_SCORE_TYPE)
                instance_metric.update(instances, image_filename, debug=debug, logger=logger)

                # Evaluates panoptic segmentation.
                if 'Cityscapes' in dataset_name:
                    image_id = '_'.join(image_filename.split('_')[:3])
                elif 'Mapillary' in dataset_name:
                    image_id = image_filename

                panoptic_metric.update(panoptic_pred, image_filename=image_filename, image_id=image_id)
                image_filename_list_debug.append(image_filename)

                # save visualization PNGs
                save_validation_visuals(gt_labels['semantic'],
                                        gt_labels['center'],
                                        gt_labels['offset'],
                                        gt_labels['gt_instance_seg'],
                                        eval_folder['visuals'],
                                        image_filename_list[i],
                                        semantic_pred,
                                        center_hmp,
                                        out_dict['offset'],
                                        panoptic_pred,
                                        debug,
                                        dataset_name
                                        )

        except Exception:
            logger.exception("Exception during testing:")
            raise
        finally:
            logger.info("Inference finished.")
            semantic_results = semantic_metric.evaluate()
            logger.info(semantic_results)
            if instance_metric is not None:
                instance_results = instance_metric.evaluate(img_list_debug=image_filename_list_debug)
                logger.info(instance_results)
            if panoptic_metric is not None:
                panoptic_results = panoptic_metric.evaluate(logger)
                logger.info(panoptic_results)
            mIoU = semantic_results['sem_seg']['mIoU']
            mPQ = panoptic_results['All']['pq']
            eval_results = {}
            eval_results['mIoU'] = mIoU
            eval_results['mPQ'] = mPQ

            if self.best_miou < mIoU:
                self.best_miou = mIoU
                logger.info('*** BEST mIoU: {} ***'.format(self.best_miou))
                logger.info('*** Corresponding PQ: {} ***'.format(mPQ))

            # removing the intermediate results and keeping the final evaluation results (json files)
            strCmd1 = 'rm -r' + ' ' + eval_folder['instance']
            strCmd2 = 'rm -r' + ' ' + eval_folder['semantic']
            strCmd3 = 'rm -r' + ' ' + os.path.join(eval_folder['panoptic'], 'predictions')
            logger.info('Removing the intermediate results and keeping the final eval json files ...')
            os.system(strCmd1)
            os.system(strCmd2)
            os.system(strCmd3)
            logger.info('END: panoptic evaluation !')
            logger.info('')

            return eval_results



    def evaluate(self, results, metric='mIoU', logger=None, efficient_test=False,  **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}

        gt_seg_maps = self.get_gt_semantic_labels()
        # gt_seg_maps = self.get_gt_seg_maps(efficient_test) # original code for semantic daformer

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
