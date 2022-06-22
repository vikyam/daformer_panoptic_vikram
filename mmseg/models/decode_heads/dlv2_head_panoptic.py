from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .decode_head_panoptic import BaseDecodeHeadPanoptic
from mmcv.cnn import build_conv_layer
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch

def sum_aspp_outs(aspp_outs):
    out = aspp_outs[0]
    for i in range(len(aspp_outs) - 1):
        out += aspp_outs[i + 1]
    return out


class MTLBlock(nn.Module):
    '''
    Multi-task learning block in the ECCV 2022 submission.
    The network arch, is adopted from DADA ICCV 2019.
    The DADA adapted it from Residual Auximilary MTL block (ROCK) NIPS paper.
    The differences between our MTLB and DADA auxiliary block are disucssed in our ECCV 2022 submission.
    '''
    def __init__(self):
        super(MTLBlock, self).__init__()
        # DADA auxiliary Encoder Part
        self.enc1 = ConvModule(2048, 512, 1, stride=1, padding=0) # conv + relu
        self.enc2 = ConvModule(512, 512, 3, stride=1, padding=1) # conv + relu
        self.enc3 = ConvModule(512, 128, 1, stride=1, padding=0, act_cfg=None) # only conv
        # DADA auxiliary Decoder Part
        self.dec = ConvModule(128, 2048, 1, stride=1, padding=0)  # conv + relu

    def forward(self, x):
        """Forward function."""
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x) # DADA takes this conv feature, apply maxpool and consider that as depth prediction
        x = self.dec(x)
        return x

@HEADS.register_module()
class DLV2HeadPanoptic(BaseDecodeHeadPanoptic):
    '''
    this is the implementation of the DAPNet model submitted in the ECCV 2022
    it has a mtl_block, semantic, instance and depth decoder
    and depth fusion
    '''
    #  debug=False, activate_panoptic=False,
    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        assert 'channels' not in kwargs
        assert 'dropout_ratio' not in kwargs
        assert 'norm_cfg' not in kwargs
        kwargs['channels'] = 1
        kwargs['dropout_ratio'] = 0
        kwargs['norm_cfg'] = None

        super(DLV2HeadPanoptic, self).__init__(**kwargs)

        del self.conv_seg
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        #
        self.out_channels_instance = 128
        self.out_channels_depth = 1
        self.semanitc_head =    ASPPModule(dilations, self.in_channels, self.num_classes, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
        self.debug = kwargs['debug']
        self.debug_output = {}
        self.act_panop = kwargs['activate_panoptic']
        if self.act_panop:
            self.depth_head =       ASPPModule(dilations, self.in_channels, self.out_channels_depth, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
            self.instance_head =    ASPPModule(dilations, self.in_channels, self.out_channels_instance, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
            self.center_sub_head =  ConvModule(128, 1, 1, stride=1, padding=0, act_cfg=None)  # only conv
            self.offset_sub_head =  ConvModule(128, 2, 1, stride=1, padding=0, act_cfg=None)  # only conv
            self.mtl_block = MTLBlock() # TODO:comment


    def forward(self, inputs):
        """Forward function."""
        if not self.act_panop: # original daformer forward pass
            x = self._transform_inputs(inputs)
            aspp_outs = self.semanitc_head(x)
            out = aspp_outs[0]
            for i in range(len(aspp_outs) - 1):
                out += aspp_outs[i + 1]
            self.debug_output.update({'semantic': out.detach()})
            return out
        else:
            x = self._transform_inputs(inputs)
            # TODO:comment
            mtlb_out = self.mtl_block(x)
            x = mtlb_out * x
            semantic_pred = sum_aspp_outs(self.semanitc_head(x))
            depth_pred = sum_aspp_outs(self.depth_head(x))
            instance_out = sum_aspp_outs(self.instance_head(x))
            center_pred = self.center_sub_head(instance_out)
            offset_pred = self.offset_sub_head(instance_out)

            self.debug_output.update({'semantic': semantic_pred.detach()})
            self.debug_output.update({'center': center_pred.detach()})
            self.debug_output.update({'offset': offset_pred.detach()})
            self.debug_output.update({'depth': depth_pred.detach()})

            return semantic_pred, center_pred, offset_pred, depth_pred
