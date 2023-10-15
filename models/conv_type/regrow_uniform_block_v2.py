import torch

from models.conv_type.masker import HardParameterMasker, SoftParameterMasker
from models.conv_type.regrow_uniform_block import RegrowUniformBlockConv2d


# implemention for Uniform Block Conv2d
class RegrowUniformBlockConv2dV2(RegrowUniformBlockConv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(RegrowUniformBlockConv2dV2, self).__init__(mask_type, in_channels, out_channels, kernel_size,
                                                         stride=stride, **kwargs)

    @torch.no_grad()
    def get_mask(self, weight):
        weight = weight * self.mask
        return super(RegrowUniformBlockConv2dV2, self).get_mask(weight)


class SoftRegrowUniformBlockConv2dV2(RegrowUniformBlockConv2dV2):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(SoftRegrowUniformBlockConv2dV2, self).__init__(SoftParameterMasker, in_channels, out_channels,
                                                             kernel_size, stride=stride, **kwargs)


class HardRegrowUniformBlockConv2dV2(RegrowUniformBlockConv2dV2):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(HardRegrowUniformBlockConv2dV2, self).__init__(HardParameterMasker, in_channels, out_channels,
                                                             kernel_size, stride=stride, **kwargs)
