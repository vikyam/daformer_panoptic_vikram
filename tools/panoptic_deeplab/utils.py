import numpy as np
import os
from datetime import datetime
import torch


def rgb2id(color):
    """Converts the color to panoptic label.
    Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
    Args:
        color: Ndarray or a tuple, color encoded image.
    Returns:
        Panoptic label.
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def create_label_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def create_panop_eval_folders(panoptic_eval_root_folder, panop_eval_temp_folder=None):
    # generate folder for panoptic evaluation with a unique name
    # so that everytime when we evalute and then remove the folder and create new root folder and subfolders
    # they don't interfer one another, i mean if you have the sameeval_root folder then once you delte that
    # and create once again the same root folder there should be a time gap as the eval root folder is large
    # in size and it may take few seconds to reomve all the png and text files generated by the evaluation
    # script, so to be in a safer side I generate the eval root folder witha unique name every time
    # so that even if the deletion of the previous eval root folder takes sometime, i can instanly create
    # anothe unique named eval root foler for the next evaluation

    os.makedirs(panoptic_eval_root_folder, exist_ok=True)

    if not panop_eval_temp_folder:
        # str1 = datetime.now().strftime("%m-%Y")
        str2 = datetime.now().strftime("%d-%m-%Y")
        str3 = datetime.now().strftime("%H-%M-%S-%f")
        panop_eval_temp_folder = 'panop_eval_{}_{}'.format(str2, str3)

    panop_eval_temp_folder_abs_path = os.path.join(panoptic_eval_root_folder, panop_eval_temp_folder)
    panop_eval_folder_dict = {}
    panop_eval_folder_dict['semantic'] = os.path.join(panop_eval_temp_folder_abs_path, 'semantic')
    panop_eval_folder_dict['instance'] = os.path.join(panop_eval_temp_folder_abs_path, 'instance')
    panop_eval_folder_dict['panoptic'] = os.path.join(panop_eval_temp_folder_abs_path, 'panoptic')
    panop_eval_folder_dict['visuals'] = os.path.join(panop_eval_temp_folder_abs_path, 'visuals')
    # panop_eval_folder_dict['sem_vis'] = os.path.join(panop_eval_temp_folder_abs_path, 'sem_vis')
    # panop_eval_folder_dict['pan_vis'] = os.path.join(panop_eval_temp_folder_abs_path, 'pan_vis')
    # panop_eval_folder_dict['ins_vis'] = os.path.join(panop_eval_temp_folder_abs_path, 'ins_vis')
    # panop_eval_folder_dict['debug_test'] = os.path.join(panop_eval_temp_folder_abs_path, 'debug_test')
    # panop_eval_folder_dict['logger_eval'] = os.path.join(panop_eval_temp_folder_abs_path, 'logger_eval')
    # panop_eval_folder_dict['tensorboard'] = os.path.join(panop_eval_temp_folder_abs_path, 'tensorboard')

    os.makedirs(panop_eval_folder_dict['semantic'],  exist_ok=True)
    os.makedirs(panop_eval_folder_dict['instance'],  exist_ok=True)
    os.makedirs(panop_eval_folder_dict['panoptic'],  exist_ok=True)
    os.makedirs(panop_eval_folder_dict['visuals'], exist_ok=True)
    # os.makedirs(panop_eval_folder_dict['debug_test'],  exist_ok=True)
    # os.makedirs(panop_eval_folder_dict['logger_eval'],  exist_ok=True)
    # os.makedirs(panop_eval_folder_dict['sem_vis'],  exist_ok=True)
    # os.makedirs(panop_eval_folder_dict['pan_vis'],  exist_ok=True)
    # os.makedirs(panop_eval_folder_dict['ins_vis'],  exist_ok=True)
    return panop_eval_temp_folder_abs_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_loss_info_str(loss_meter_dict):
    msg = ''
    for key in loss_meter_dict.keys():
        msg += '{name}: {meter.val:.6f} ({meter.avg:.6f})\t'.format(name=key, meter=loss_meter_dict[key])

    return msg


def to_cuda(batch, device):
    if type(batch) == torch.Tensor:
        batch = batch.to(device)
    elif type(batch) == dict:
        for key in batch.keys():
            batch[key] = to_cuda(batch[key], device)
    elif type(batch) == list:
        for i in range(len(batch)):
            batch[i] = to_cuda(batch[i], device)
    return batch


def get_module(model, distributed):
    if distributed:
        return model.module
    else:
        return model