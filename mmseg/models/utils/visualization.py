# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from tools.panoptic_deeplab.save_annotations import flow_compute_color
#
from tools.panoptic_deeplab.save_annotations import label_to_color_image, random_color
from tools.panoptic_deeplab.utils import create_label_colormap
import os


Cityscapes_palette = [
    128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153,
    153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130,
    180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100,
    0, 0, 230, 119, 11, 32, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128,
    128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128,
    192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128,
    64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64,
    0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64,
    128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64,
    0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64,
    64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192,
    192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128,
    160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0,
    224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64,
    0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192,
    128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64,
    128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32,
    128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128,
    192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0,
    192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64,
    160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96,
    64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192,
    96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0,
    0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32,
    0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192,
    160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128,
    96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0,
    192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32,
    64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0,
    160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160,
    64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128,
    96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192,
    128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96,
    192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32,
    160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160,
    128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32,
    128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160,
    224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0,
    224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224,
    128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32,
    32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32,
    64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192,
    224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96,
    192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64,
    96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 0, 0, 0
]


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def _colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image


def subplotimg(ax, img, title, range_in_title=False, palette=Cityscapes_palette, **kwargs):
    if img is None:
        return
    with torch.no_grad():
        if torch.is_tensor(img):
            img = img.cpu()
        if len(img.shape) == 2:
            if torch.is_tensor(img):
                img = img.numpy()
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            if not torch.is_tensor(img):
                img = img.numpy()
        if kwargs.get('cmap', '') == 'cityscapes':
            kwargs.pop('cmap')
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize_mask(img, palette)

    if range_in_title:
        vmin = np.min(img)
        vmax = np.max(img)
        title += f' {vmin:.3f}-{vmax:.3f}'

    ax.imshow(img, **kwargs)
    ax.set_title(title)


def subplotimgV2(ax, img, title, **kwargs):
    ax.imshow(img, **kwargs)
    ax.set_title(title)


def get_np_array(pt_tensor, img=None, ratio=None, type=None, palette=Cityscapes_palette):
    with torch.no_grad():
        if ratio:
            img = img * 255
            img = img.permute(1, 2, 0).cpu().numpy()
        if type == 'gt_center' or type == 'pred_center':
            np_array = pt_tensor.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
        if type == 'gt_center_w' or type == 'gt_offset_w':
            np_array = pt_tensor.squeeze(dim=0).cpu().numpy()
        if type == 'gt_offset' or type == 'pred_offset':
            np_array = pt_tensor.squeeze(dim=0).cpu().numpy().transpose(1, 2, 0)
            np_array = flow_compute_color(np_array[:, :, 1], np_array[:, :, 0])
        if type == 'pred_semantic':
            pt_tensor = torch.softmax(pt_tensor.detach(), dim=0)
            _, pt_tensor = torch.max(pt_tensor, dim=0)
            np_array = pt_tensor.cpu().numpy()
            pil_array = colorize_mask(np_array, palette)
            np_array = pil_array
            # np_array = np.array(pil_array)
            # np_array = np.expand_dims(np_array, axis=2)
        if type != 'gt_offset' and type != 'pred_offset' and type != 'gt_semantic' and type != 'pred_semantic':
            np_array = np_array[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
            np_array = np_array.clip(0, 255)

        if ratio:
            np_array = ratio * np_array + (1 - ratio) * img

        if type != 'pred_semantic':
            np_array = np_array.astype(dtype=np.uint8)
        return np_array
# ema_softmax = torch.softmax(ema_semantic_logits.detach(), dim=1)
# pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)


def prep_sem_for_vis(sem_map, mode='GT', dataset_name='Cityscapes', debug=False, blend_ratio=0.7, img=None):
    if mode == 'GT':
        sem_map = sem_map.cpu().numpy()
    if dataset_name=='Cityscapes':
        if debug:
            IGNORE_TOP_VAL = 15
            IGNORE_BOTTOM_VAL = 90
        else:
            IGNORE_TOP_VAL = 15 * 2
            IGNORE_BOTTOM_VAL = 90 * 2
        sem_map[:IGNORE_TOP_VAL, :] = 255
        sem_map[-IGNORE_BOTTOM_VAL:, :] = 255
        colored_label = label_to_color_image(sem_map, colormap=create_label_colormap())
        colored_label = blend_ratio * colored_label + (1 - blend_ratio) * img
    elif dataset_name == 'Mapillary':
        raise NotImplementedError('Implementation not found for Mapillary dataset --> mmseg/datasets/custom.py !!')
    return colored_label.astype(int)

def prep_cnt_for_vis(cnt_map, mode='GT', dataset_name='Cityscapes', debug=False, blend_ratio=0.7, img=None, sem_id=None):
    if mode == 'GT':
        cnt_map = cnt_map[0]
    else:
        # set the stuff regions to 0
        cnt_map[np.logical_and.reduce((sem_id != 11, sem_id != 12, sem_id != 13, sem_id != 14, sem_id != 15, sem_id != 16, sem_id != 17, sem_id != 18))] = 0

    if dataset_name=='Cityscapes':
        if debug:
            IGNORE_TOP_VAL = 15
            IGNORE_BOTTOM_VAL = 90
        else:
            IGNORE_TOP_VAL = 15 * 2
            IGNORE_BOTTOM_VAL = 90 * 2
        cnt_map[:IGNORE_TOP_VAL, :] = 255
        cnt_map[-IGNORE_BOTTOM_VAL:, :] = 255
        cnt_map = cnt_map[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
        cnt_map = cnt_map.clip(0, 255)
        img = blend_ratio * cnt_map + (1 - blend_ratio) * img
    elif dataset_name == 'Mapillary':
        raise NotImplementedError('Implementation not found for Mapillary dataset --> mmseg/datasets/custom.py !!')
    return img.astype(int)

def prep_ofs_for_vis(ofs_map, mode='GT', dataset_name='Cityscapes', debug=False, blend_ratio=0.7, img=None, sem_id=None):
    if mode == 'PD':
        ofs_map = ofs_map.cpu().numpy()
        ofs_map = ofs_map[0]
        ofs_map = ofs_map.transpose([1, 2, 0])
        ofs_map[np.logical_and.reduce((sem_id != 11, sem_id != 12, sem_id != 13, sem_id != 14, sem_id != 15, sem_id != 16, sem_id != 17, sem_id != 18)), :] = 0
    else:
        ofs_map = ofs_map.transpose([1, 2, 0])

    if dataset_name=='Cityscapes':
        if debug:
            IGNORE_TOP_VAL = 15
            IGNORE_BOTTOM_VAL = 90
        else:
            IGNORE_TOP_VAL = 15 * 2
            IGNORE_BOTTOM_VAL = 90 * 2
        ofs_map[:IGNORE_TOP_VAL, :] = 255
        ofs_map[-IGNORE_BOTTOM_VAL:, :] = 255
        offset_image = flow_compute_color(ofs_map[:, :, 1], ofs_map[:, :, 0])
        img = blend_ratio * offset_image + (1 - blend_ratio) * img
    elif dataset_name == 'Mapillary':
        raise NotImplementedError('Implementation not found for Mapillary dataset --> mmseg/datasets/custom.py !!')
    return img.astype(int)


def prep_ins_for_vis(inst_map, mode='GT', dataset_name='Cityscapes', debug=False, blend_ratio=0.7, img=None):
    stuff_id = 0
    if dataset_name=='Cityscapes':
        if debug:
            IGNORE_TOP_VAL = 15
            IGNORE_BOTTOM_VAL = 90
        else:
            IGNORE_TOP_VAL = 15 * 2
            IGNORE_BOTTOM_VAL = 90 * 2
        inst_map[:IGNORE_TOP_VAL, :] = 255
        inst_map[-IGNORE_BOTTOM_VAL:, :] = 255
        #----------------------------------------
        ids = np.unique(inst_map)
        num_colors = len(ids)
        colormap = np.zeros((num_colors, 3), dtype=np.uint8)
        # Maps inst_map to continuous value.
        for i in range(num_colors):
            inst_map[inst_map == ids[i]] = i
            colormap[i, :] = random_color(rgb=True, maximum=255)
            if ids[i] == stuff_id:
                colormap[i, :] = np.array([0, 0, 0])  # np.array([255, 0, 0])
        colored_label = colormap[inst_map]
        colored_label = blend_ratio * colored_label + (1 - blend_ratio) * img
    elif dataset_name == 'Mapillary':
        raise NotImplementedError('Implementation not found for Mapillary dataset --> mmseg/datasets/custom.py !!')
    return colored_label.astype(int)


def _random_color(base, max_dist=30):
    new_color = base + np.random.randint(low=-max_dist, high=max_dist + 1, size=3)
    return tuple(np.maximum(0, np.minimum(255, new_color)))


def prep_pan_for_vis(pan_map, mode='GT', dataset_name='Cityscapes', debug=False, blend_ratio=0.7, img=None):
    label_divisor = 1000
    if dataset_name=='Cityscapes':
        if debug:
            IGNORE_TOP_VAL = 15
            IGNORE_BOTTOM_VAL = 90
        else:
            IGNORE_TOP_VAL = 15 * 2
            IGNORE_BOTTOM_VAL = 90 * 2
        pan_map[:IGNORE_TOP_VAL, :] = 255
        pan_map[-IGNORE_BOTTOM_VAL:, :] = 255
        #
        colormap = create_label_colormap()
        colored_label = np.zeros((pan_map.shape[0], pan_map.shape[1], 3), dtype=np.uint8)
        taken_colors = set([0, 0, 0])
        for lab in np.unique(pan_map):
            mask = pan_map == lab
            base_color = colormap[lab // label_divisor]
            if tuple(base_color) not in taken_colors:
                taken_colors.add(tuple(base_color))
                color = base_color
            else:
                while True:
                    color = _random_color(base_color)
                    if color not in taken_colors:
                        taken_colors.add(color)
                        break
            colored_label[mask] = color
            colored_label = blend_ratio * colored_label + (1 - blend_ratio) * img
    elif dataset_name == 'Mapillary':
        raise NotImplementedError('Implementation not found for Mapillary dataset --> mmseg/datasets/custom.py !!')
    return colored_label.astype(int)


def gen_inst_seg_for_vis(panoptic_pred):
    # cityscapes_thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
    label_divisor = 1000
    TrgInstPd = None
    ins_id = panoptic_pred % label_divisor
    sem_id = np.around(panoptic_pred / 1000).astype(int)
    no_instance_mask = np.logical_and.reduce((sem_id != 11, sem_id != 12, sem_id != 13, sem_id != 14, sem_id != 15, sem_id != 16, sem_id != 17, sem_id != 18))
    pan_to_ins = panoptic_pred.copy()
    pan_to_ins[ins_id == 0] = 0
    pan_to_ins[no_instance_mask] = 0
    return pan_to_ins, sem_id


def save_validation_visuals(TrgSemGT,
                            TrgCntGT,
                            TrgOfsGT,
                            TrgInstGT,
                            out_dir,
                            image_filename,
                            TrgSemPd,
                            TrgCntPd,
                            TrgOfsPd,
                            TrgPanPd,
                            debug,
                            dataset_name
                            ):  # TrgInstPd

    # generate instance segmentation map for visualization
    TrgInstPd, sem_id = gen_inst_seg_for_vis(TrgPanPd)

    if debug and dataset_name == 'Cityscapes':
        vis_W = 1024
        vis_H = 512
    elif not debug and dataset_name == 'Cityscapes':
        vis_W = 2048
        vis_H = 1024
    elif dataset_name == 'Mapillary':
        raise NotImplementedError('Implementation not found for Mapillary dataset --> mmseg/datasets/custom.py !!')

    input_img_path = image_filename.replace('_gtFine_panoptic', '_leftImg8bit')
    input_img_path = input_img_path + '.png'
    input_img_path = 'data/cityscapes/leftImg8bit/val/' + input_img_path
    #
    output_img_path = image_filename.split('/')[0]
    output_img_path = os.path.join(out_dir, output_img_path)
    os.makedirs(output_img_path, exist_ok=True)
    output_filename = image_filename.split('/')[1].replace('_gtFine_panoptic', '_predictions') + '.png'
    output_img_path = os.path.join(output_img_path, output_filename)

    # load input_filename
    vis_trg_img = Image.open(input_img_path)
    # vis_trg_img.show()
    vis_trg_img = vis_trg_img.convert('RGB')
    vis_trg_img = vis_trg_img.resize((vis_W, vis_H), Image.BICUBIC)
    vis_trg_img = np.asarray(vis_trg_img, np.int)
    rows, cols = 4, 4
    # gridspec_kw = {'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
    # fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), gridspec_kw=gridspec_kw)
    fig, axs = plt.subplots(rows, cols, figsize=(12, 6), constrained_layout=True)
    # plotting the input image
    subplotimgV2(axs[0][0], vis_trg_img, 'TrgImg')
    # plot the GT labels
    subplotimgV2(axs[1][0], prep_sem_for_vis(TrgSemGT, mode='GT', dataset_name=dataset_name, debug=debug, blend_ratio=0.7, img=vis_trg_img), 'TrgSemGT')
    subplotimgV2(axs[1][1], prep_cnt_for_vis(TrgCntGT, mode='GT', dataset_name=dataset_name, debug=debug, blend_ratio=0.7, img=vis_trg_img), 'TrgCntGT')
    subplotimgV2(axs[1][2], prep_ofs_for_vis(TrgOfsGT, mode='GT', dataset_name=dataset_name, debug=debug, blend_ratio=0.7, img=vis_trg_img), 'TrgOfsGT')
    subplotimgV2(axs[1][3], prep_ins_for_vis(TrgInstGT, mode='GT', dataset_name=dataset_name, debug=debug, blend_ratio=0.7, img=vis_trg_img), 'TrgInstGT')
    # plot the predictions
    subplotimgV2(axs[2][0], prep_sem_for_vis(TrgSemPd, mode='PD', dataset_name=dataset_name, debug=debug, blend_ratio=0.7, img=vis_trg_img), 'TrgSemPd')
    subplotimgV2(axs[2][1], prep_cnt_for_vis(TrgCntPd, mode='PD', dataset_name=dataset_name, debug=debug, blend_ratio=0.7, img=vis_trg_img, sem_id=sem_id), 'TrgCntPd')
    subplotimgV2(axs[2][2], prep_ofs_for_vis(TrgOfsPd, mode='PD', dataset_name=dataset_name, debug=debug, blend_ratio=0.7, img=vis_trg_img, sem_id=sem_id), 'TrgOfsPd')
    subplotimgV2(axs[2][3], prep_ins_for_vis(TrgInstPd, mode='PD', dataset_name=dataset_name, debug=debug, blend_ratio=0.7, img=vis_trg_img), 'TrgInstPd')
    #
    subplotimgV2(axs[3][3], prep_pan_for_vis(TrgPanPd, mode='PD', dataset_name=dataset_name, debug=debug, blend_ratio=0.7, img=vis_trg_img), 'TrgPanPd')
    #
    for ax in axs.flat:
        ax.axis('off')
    plt.savefig(output_img_path)
    plt.close()