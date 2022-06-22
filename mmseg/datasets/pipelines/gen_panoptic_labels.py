import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES

from cityscapesscripts.helpers.labels import id2label, labels

from tools.panoptic_deeplab.utils import rgb2id

@PIPELINES.register_module()
class GenPanopLabels(object):
    """
    Generates panoptic training target for Panoptic-DeepLab.
    Annotation is assumed to have Cityscapes format.
    Arguments:
        ignore_label: Integer, the ignore label for semantic segmentation.
        rgb2id: Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
        thing_list: List, a list of thing classes
        sigma: the sigma for Gaussian kernel.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
        ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in semantic segmentation branch,
            crowd region is ignored in the original TensorFlow implementation.
    """

    def __init__(self, sigma, mode):
        self.ignore_label = 255
        self.thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        self.ignore_stuff_in_offset = True
        self.small_instance_area = 4096 # not using currently
        self.small_instance_weight = 3
        self.ignore_crowd_in_semantic = True
        self.sigma = sigma
        self.mode = mode
        #
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, results):
        panoptic = results['gt_panoptic_seg']
        segments = results['ann_info']['segments_info']
        #
        panoptic = rgb2id(panoptic)
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        instance = np.zeros_like(panoptic, dtype=np.uint8)
        foreground = np.zeros_like(panoptic, dtype=np.uint8)
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(panoptic, dtype=np.float32)
        x_coord = np.ones_like(panoptic, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        # Generate pixel-wise loss weights
        # semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)

        instance_id = 1
        for seg in segments:
            cat_id = seg["category_id"]

            # TODO:
            # as in the validation json file,
            # the cat_id are the id and not the trainIds,
            # so we need to map the id to trainId
            # because our self.thing_list uses trainIds
            # where as for train json file,
            # the cat_id are the continous trainIds (0,..,15)
            if self.mode == 'val':
                labelInfo = id2label[cat_id]
                cat_id = labelInfo.trainId

            if self.ignore_crowd_in_semantic:
                if not seg['iscrowd']:
                    semantic[panoptic == seg["id"]] = cat_id
            else:
                semantic[panoptic == seg["id"]] = cat_id

            # the foreground is used by the panoptic evaluation script for evlauting foreground semantic segmentation
            # refer to /home/suman/apps/code/CVPR2022/panoptic-deeplab-ori/tools/test_net_single_core.py
            # we dont do that
            # if cat_id in self.thing_list:
            #     foreground[panoptic == seg["id"]] = 1

            # this block is required only for synthia or source domain # TODO
            # this is required for generating the pseudo labels for center and offset by the DACS self-training module
            if cat_id in self.thing_list:
                instance[panoptic == seg["id"]] = instance_id
                instance_id += 1

            if not seg['iscrowd']:
                # Ignored regions are not in `segments`.
                # Handle crowd region.
                center_weights[panoptic == seg["id"]] = 1
                if self.ignore_stuff_in_offset:  # ignore_stuff_in_offset = True
                    # Handle stuff region.
                    if cat_id in self.thing_list:
                        offset_weights[panoptic == seg["id"]] = 1
                else:
                    offset_weights[panoptic == seg["id"]] = 1

            '''
            The below if block generates the center and offsets for thing classes.
            There are some segments belonging to thing classes which has iscrowd falg True or 1.
            The below block generates center and offsets for those segments as well,
            but the center_weights and offset_weights generated in the above if block take care of these segments
            by multipying 0 with the center and offset losses,i.e., for these crowd region segments (belong to thing class)
            we dont compute the center and offset losses.            
            '''
            # if cat_id in self.thing_list and self.dataset_name == 'cityscapes' or cat_id in self.thing_list and self.dataset_name == 'synthia' and not seg['iscrowd']:
            # ideally we should also put the condition -->  and not seg['iscrowd']: -- but does not matter,those crowd region has offset and center wieghts 0 --
            # see the above code segment - loss willbe 0 and grad won't backprop
            if cat_id in self.thing_list:  # and not seg['iscrowd']:
                # find instance center
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue


                # TODO: this block is only required for cityscapes by original panoptic deeplab semantic loss,
                #  they use this weight semantic_weights, I am not using it
                # Find instance area
                # if self.dataset_name == 'cityscapes':
                #     ins_area = len(mask_index[0])
                #     if ins_area < self.small_instance_area:
                #         semantic_weights[panoptic == seg["id"]] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])

                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(
                    center[0, aa:bb, cc:dd], self.g[a:b, c:d])

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]

        # if self.mode == 'train' or self.mode == 'all':  # train for cityscapes and mapillary and all for synthia
        results['gt_semantic_seg'] = semantic.astype('long')
        results['seg_fields'].append('gt_semantic_seg')
        results['gt_center'] = center.astype(np.float32)
        results['seg_fields'].append('gt_center')
        results['gt_offset'] = offset.astype(np.float32)
        results['seg_fields'].append('gt_offset')
        results['center_weights'] = center_weights.astype(np.float32)
        results['seg_fields'].append('center_weights')
        results['offset_weights'] = offset_weights.astype(np.float32)
        results['seg_fields'].append('offset_weights')
        results['gt_instance_seg'] = instance.astype('long')
        results['seg_fields'].append('gt_instance_seg')

        # if self.mode == 'val':
            #  results['gt_foreground_seg'] = foreground.astype('long')
            #  results['seg_fields'].append('gt_foreground_seg')
            # results['semantic_weights'] = semantic_weights.astype(np.float32)
            # results['seg_fields'].append('semantic_weights')
            # results['gt_center_points'] = center_pts
            # results['seg_fields'].append('gt_center_points')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(ignore_label={self.ignore_label}, ' \
                    f'(thing_list={self.thing_list}, ' \
                    f'(ignore_stuff_in_offset={self.ignore_stuff_in_offset}, ' \
                    f'(small_instance_area={self.small_instance_area}, ' \
                    f'(small_instance_weight={self.small_instance_weight}, ' \
                    f'(sigma={self.sigma}, ' \
                    f'(g={self.g}, '
        return repr_str

