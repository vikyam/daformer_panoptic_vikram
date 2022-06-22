# @force_fp32(apply_to=('semantic_logits', 'center_logits', 'offset_logits', 'depth_logits', ))
def resize():
    pass
def losses_source_domain(self,
           semantic_logits,
           center_logits,
           offset_logits,
           depth_logits,
           gt_semantic_seg,
           gt_center,
           center_weights,
           gt_offset,
           offset_weights,
           gt_instance_seg,
           gt_depth_map,
           train_cfg,
           semantic_weight=None,):
    """Compute segmentation loss."""
    losses = dict()
    # upsample the predictions
    semantic_logits = resize(input=semantic_logits, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.align_corners)
    center_logits = resize(input=center_logits, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.align_corners)
    offset_logits = resize(input=offset_logits, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.align_corners)
    depth_logits = resize(input=depth_logits, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.align_corners)

    # daformer  ce_loss
    # if self.sampler is not None:
    #     semantic_weight = self.sampler.sample(semantic_logits, gt_semantic_seg)
    # gt_semantic_seg = gt_semantic_seg.squeeze(1)
    # losses['loss_semantic'] = self.loss_decode[0](semantic_logits, gt_semantic_seg, weight=semantic_weight, ignore_index=self.ignore_index)

    # # loss semantic
    # gt_semantic_seg = gt_semantic_seg.squeeze(1)
    # losses['loss_semantic'] = self.loss_decode[0](semantic_logits, gt_semantic_seg) * self.loss_semanitc_lambda
    # losses['acc_semantic'] = accuracy(semantic_logits, gt_semantic_seg)


    # # loss center
    # center_weights = center_weights.squeeze(dim=1)
    # # Pixel-wise loss weight
    # center_loss_weights = center_weights[:, None, :, :].expand_as(center_logits)
    # loss_center = self.loss_decode[1](center_logits, gt_center.squeeze(dim=1)) * center_loss_weights
    # # safe division
    # if center_loss_weights.sum() > 0:
    #     loss_center = loss_center.sum() / center_loss_weights.sum() * self.loss_center_lambda + 1e-10
    # else:
    #     loss_center = loss_center.sum() * 0
    # losses['loss_center'] = loss_center



    # loss offset
    offset_weights = offset_weights.squeeze(dim=1)
    # Pixel-wise loss weight
    offset_loss_weights = offset_weights[:, None, :, :].expand_as(offset_logits)
    loss_offset = self.loss_decode[2](offset_logits, gt_offset.squeeze(dim=1)) * offset_loss_weights
    # safe division
    if offset_loss_weights.sum() > 0:
        loss_offset = loss_offset.sum() / offset_loss_weights.sum() * self.loss_offset_lambda + 1e-10
    else:
        loss_offset = loss_offset.sum() * 0
    losses['loss_offset'] = loss_offset
    # loss depth
    losses['loss_depth'] = self.loss_decode[3](depth_logits, gt_depth_map.squeeze(dim=1)) * self.loss_depth_lambda
    return losses

# @force_fp32(apply_to=('semantic_logits',))
# def losses_mixed_domain(self, semantic_logits, gt_semantic_seg, seg_weight=None):
#     """Compute segmentation loss."""
#     # upsample the predictions
#     semantic_logits = resize(input=semantic_logits, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.align_corners)
#     losses = dict()
#     # loss semantic
#     gt_semantic_seg = gt_semantic_seg.squeeze(1)
#     losses['loss_semantic'] = self.loss_decode[0](semantic_logits, gt_semantic_seg) * self.loss_semanitc_lambda
#     losses['acc_semantic'] = accuracy(semantic_logits, gt_semantic_seg)
#     return losses