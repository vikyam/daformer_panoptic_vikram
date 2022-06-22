from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead
from .daformer_head_panoptic import DAFormerHeadPanoptic
from .dlv2_head import DLV2Head
from .dlv2_head_panoptic import DLV2HeadPanoptic
from .fcn_head import FCNHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'DAHead',
    'DLV2Head',
    'DLV2HeadPanoptic',
    'SegFormerHead',
    'DAFormerHead',
    'DAFormerHeadPanoptic',
    'ISAHead',
]
