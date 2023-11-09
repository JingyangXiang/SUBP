import torch

from models.conv_type.base_conv import BaseNMConv
from models.conv_type.masker import HardParameterMasker, SoftParameterMasker


# implemention for Uniform Block Conv2d
class UniformBlockConv2dFine(BaseNMConv):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(UniformBlockConv2dFine, self).__init__(in_channels, out_channels, kernel_size, stride=stride, **kwargs)
        assert kernel_size in [1, 3]
        self.mask_type = mask_type
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x):
        weight = self.mask_type.apply(self.weight, self.mask)
        output = self._conv_forward(x, weight, self.bias)
        return output

    @torch.no_grad()
    def get_mask(self, weight):
        # from 0 to prune_rate, when prune_rate_cur_epoch is 0,               keep dense training;
        #                       when prune_rate_cur_epoch rate is prune_rate, prune all need pruned blocks.
        prune_rate_cur_epoch = self.get_prune_rate_cur_epoch(self.cur_epoch, self.decay_start, self.decay_end)
        assert prune_rate_cur_epoch <= self.prune_rate, f"Epoch: {self.cur_epoch}, prune_rate_cur_epoch > prune_rate"

        # (out_channels, in_channels, kernel_size, kernel_size)
        # (out_channels // N, N, in_channels, kernel_size, kernel_size)
        # (out_channels // N, in_channels, N x kernel_size x kernel_size)
        weight = weight.reshape(self.out_channels // self.N, self.N, self.in_channels, *self.kernel_size)
        weight = weight.flatten(2).permute(0, 2, 1)

        # get and sort importance
        score = self.get_weight_importance(weight)
        sorted_local, index_local = torch.sort(score, dim=1)

        # generate mask
        mask = torch.ones(score.shape, device=score.device)
        index_prune_end_cur_epoch = int(
            prune_rate_cur_epoch * self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        mask = mask.scatter_(dim=1, index=index_local[:, :index_prune_end_cur_epoch], value=0.)
        mask = mask[:, :, None].repeat(1, 1, self.N).permute(0, 2, 1)
        mask = mask.reshape(self.out_channels, self.in_channels, *self.kernel_size)
        return mask


class SoftUniformBlockConv2dFine(UniformBlockConv2dFine):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(SoftUniformBlockConv2dFine, self).__init__(SoftParameterMasker, in_channels, out_channels, kernel_size,
                                                         stride=stride, **kwargs)


class HardUniformBlockConv2dFine(UniformBlockConv2dFine):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(HardUniformBlockConv2dFine, self).__init__(HardParameterMasker, in_channels, out_channels, kernel_size,
                                                         stride=stride, **kwargs)
