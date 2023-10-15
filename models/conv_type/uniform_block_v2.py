import torch

from models.conv_type.masker import HardParameterMasker, SoftParameterMasker
from models.conv_type.uniform_block import UniformBlockConv2d


# implemention for Uniform Block Conv2d
class UniformBlockConv2dV2(UniformBlockConv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(UniformBlockConv2dV2, self).__init__(mask_type, in_channels, out_channels, kernel_size, stride=stride,
                                                   **kwargs)

    @torch.no_grad()
    def get_mask(self, weight):
        weight = weight * self.mask
        return super(UniformBlockConv2dV2, self).get_mask(weight)


class SoftUniformBlockConv2dV2(UniformBlockConv2dV2):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(SoftUniformBlockConv2dV2, self).__init__(SoftParameterMasker, in_channels, out_channels, kernel_size,
                                                       stride=stride, **kwargs)


class HardUniformBlockConv2dV2(UniformBlockConv2dV2):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(HardUniformBlockConv2dV2, self).__init__(HardParameterMasker, in_channels, out_channels, kernel_size,
                                                       stride=stride, **kwargs)
