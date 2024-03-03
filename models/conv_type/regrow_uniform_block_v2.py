import torch

from models.conv_type.masker import HardParameterMasker as Hard, SoftParameterMasker as Soft
from models.conv_type.regrow_uniform_block import RegrowUniformBlockConv2d


class BaseUBPConv2d(RegrowUniformBlockConv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(BaseUBPConv2d, self).__init__(mask_type, in_channels, out_channels, kernel_size, stride, **kwargs)

    @torch.no_grad()
    def get_mask(self, weight):
        weight = weight * self.mask
        return super(BaseUBPConv2d, self).get_mask(weight)


class SUBPConv2dV2(BaseUBPConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(SUBPConv2dV2, self).__init__(Soft, in_channels, out_channels, kernel_size, stride, **kwargs)


class HUBPConv2dV2(BaseUBPConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(HUBPConv2dV2, self).__init__(Hard, in_channels, out_channels, kernel_size, stride, **kwargs)
