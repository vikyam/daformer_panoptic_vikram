# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.uda.uda_decorator_panoptic import UDADecoratorPanoptic
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg, subplotimgV2
from mmseg.utils.utils import downscale_label_ratio
from mmseg.models.utils.visualization import get_np_array
from mmseg.ops import resize


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACSPanoptic(UDADecoratorPanoptic):

    def __init__(self, **cfg):
        super(DACSPanoptic, self).__init__(**cfg)
        self.act_panop = cfg['activate_panoptic']
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      gt_center,
                      center_weights,
                      gt_offset,
                      offset_weights,
                      gt_instance_seg,
                      gt_depth_map,
                      target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)

        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
                                                    img,
                                                    img_metas,
                                                    gt_semantic_seg,
                                                    gt_center,
                                                    center_weights,
                                                    gt_offset,
                                                    offset_weights,
                                                    gt_instance_seg,
                                                    gt_depth_map,
                                                    return_feat=True)

        if clean_losses['decode.loss_center'].item() > 14.0:
            clean_losses['decode.loss_center'] = torch.zeros(1)

        # getting the source predictions for debug visualization
        if self.local_iter !=0 and self.local_iter % self.debug_img_interval == 0:
            debug_output = self.get_model().decode_head.debug_output
            semantic_pred_src = debug_output['semantic']
            semantic_pred_src = resize(input=semantic_pred_src, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.get_model().align_corners)
            if self.act_panop:
                center_pred_src = debug_output['center']
                offset_pred_src = debug_output['offset']
                depth_pred_src = debug_output['depth']
                center_pred_src = resize(input=center_pred_src, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.get_model().align_corners)
                offset_pred_src = resize(input=offset_pred_src, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.get_model().align_corners)
                # depth_pred_src = resize(input=depth_pred_src, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.get_model().align_corners)

        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist) # todo ADD UP THE LOSSES AND DO ONE BACKPROP
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        if not self.act_panop:
            ema_semantic_logits = self.get_ema_model().encode_decode(target_img, target_img_metas)
        else:
            ema_semantic_logits, ema_center_logits, ema_offset_logits = self.get_ema_model().encode_decode(target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_semantic_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mixed_img_metas = []
        mix_losses = self.get_model().forward_train(
                                                    mixed_img, mixed_img_metas, mixed_lbl,
                                                    gt_center,
                                                    center_weights,
                                                    gt_offset,
                                                    offset_weights,
                                                    gt_instance_seg,
                                                    gt_depth_map,
                                                    seg_weight=pseudo_weight, return_feat=True)

        # getting the source predictions for debug visualization
        if self.local_iter !=0 and self.local_iter % self.debug_img_interval == 0:
            debug_output = self.get_model().decode_head.debug_output
            semantic_pred_mix = debug_output['semantic']
            semantic_pred_mix = resize(input=semantic_pred_mix, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.get_model().align_corners)
            if self.act_panop:
                center_pred_mix = debug_output['center']
                offset_pred_mix = debug_output['offset']
                depth_pred_mix = debug_output['depth']
                center_pred_mix = resize(input=center_pred_mix, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.get_model().align_corners)
                offset_pred_mix = resize(input=offset_pred_mix, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.get_model().align_corners)
                # depth_pred_mix = resize(input=depth_pred_mix, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.get_model().align_corners)

        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        # visualization
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html#list-colormaps
        if self.local_iter !=0 and self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1) # TODO
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 6, 5
                gridspec_kw = {'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
                fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), gridspec_kw=gridspec_kw)

                # Source image and labels
                subplotimg(  axs[0][0], vis_img[j], 'SrcImg')
                subplotimg(  axs[0][1], gt_semantic_seg[j], 'SrcSemGT', cmap='cityscapes')
                subplotimgV2(axs[0][2], get_np_array(gt_center[j], img=vis_img[j], ratio=0.7, type='gt_center'), 'SrcCntGT')
                subplotimgV2(axs[0][3], get_np_array(gt_offset[j], img=vis_img[j], ratio=0.7, type='gt_offset'), 'SrcOfsGT')
                # subplotimgV2(axs[0][4], get_np_array(gt_offset[j], type='gt_depth'), 'SrcDepGT')

                # Source center and offset loss weights
                subplotimgV2(axs[1][2], get_np_array(center_weights[j], type='gt_center_w'), 'SrcCntWt')
                subplotimgV2(axs[1][3], get_np_array(offset_weights[j], type='gt_offset_w'), 'SrcOfsWt')

                # Source predictions
                subplotimgV2(axs[2][1], get_np_array(semantic_pred_src[j], type='pred_semantic'), 'SrcSemPd')
                if self.act_panop:
                    subplotimgV2(axs[2][2], get_np_array(center_pred_src[j], img=vis_img[j], ratio=0.7, type='pred_center'), 'SrcCntPd', cmap='viridis')
                    subplotimgV2(axs[2][3], get_np_array(offset_pred_src[j], img=vis_img[j], ratio=0.7, type='pred_offset'), 'SrcOfsPd')
                    # subplotimgV2(axs[2][4], get_np_array(depth_pred_src[j], img=vis_img[j], ratio=0.7, type='pred_depth'), 'SrcDepPd')

                # Target image and predictions
                subplotimg(axs[3][0], vis_trg_img[j], 'TrgImg') # TODO
                subplotimg(axs[3][1], pseudo_label[j], 'TrgSemPd', cmap='cityscapes') # TODO
                if self.act_panop:
                    subplotimgV2(axs[3][2], get_np_array(ema_center_logits[j], img=vis_trg_img[j], ratio=0.7, type='pred_center'), 'TrgCntPd', cmap='viridis') # TODO
                    subplotimgV2(axs[3][3], get_np_array(ema_offset_logits[j], img=vis_trg_img[j], ratio=0.7, type='pred_offset'), 'TrgOfsPd') # TODO

                # Mixed image and pseudo labels
                subplotimg(axs[4][0], vis_mixed_img[j], 'MixImg')
                subplotimg(axs[4][1], mixed_lbl[j], 'MixSemPL', cmap='cityscapes')

                # Mixed image predictions
                subplotimg(axs[5][0], mix_masks[j][0], 'MixMask', cmap='gray')
                subplotimgV2(axs[5][1], get_np_array(semantic_pred_mix[j], type='pred_semantic'), 'MixSemPd')
                if self.act_panop:
                    subplotimgV2(axs[5][2], get_np_array(center_pred_mix[j], img=vis_mixed_img[j], ratio=0.7, type='pred_center'), 'MixCntPd', cmap='viridis')
                    subplotimgV2(axs[5][3], get_np_array(offset_pred_mix[j], img=vis_mixed_img[j], ratio=0.7, type='pred_offset'), 'MixOfsPd')
                    # subplotimgV2(axs[5][4], get_np_array(depth_pred_mix[j], img=vis_mixed_img[j], ratio=0.7, type='pred_depth'), 'MixDepPd')

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

        self.local_iter += 1
        return log_vars


