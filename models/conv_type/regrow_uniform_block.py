import torch
import torch.nn.functional as F

from models.conv_type.base_conv import BaseNMConv
from .masker import HardParameterMasker as Hard, SoftParameterMasker as Soft


# implemention for Uniform Block Conv2d
class RegrowUniformBlockConv2d(BaseNMConv):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(RegrowUniformBlockConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, **kwargs)
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

        regrow_rate_cur_epoch = self.prune_rate - prune_rate_cur_epoch
        regrow_num_cur_epoch = int(regrow_rate_cur_epoch * self.in_channels)

        # (out_channels, in_channels, kernel_size, kernel_size) -> (out_channels // N, N, in_channels, kernel_size, kernel_size)
        weight = weight.reshape(self.out_channels // self.N, self.N, self.in_channels, *self.kernel_size)
        # (out_channels // N, in_channels, N x kernel_size x kernel_size)
        weight = weight.transpose(1, 2).flatten(2)

        # get and sort importance
        score = self.get_weight_importance(weight)
        sorted_local, index_local = torch.sort(score, dim=1)

        # generate prune mask
        mask = torch.ones(score.shape, device=score.device)
        index_prune_end = int(self.prune_rate * self.in_channels)
        mask = mask.scatter_(dim=1, index=index_local[:, :index_prune_end], value=0.)

        # 1. get probability
        # 2. sample index
        # 3. gather index from index_local
        # 4. regrow mask
        if regrow_num_cur_epoch > 0:
            prob = torch.softmax(F.normalize(sorted_local[:, :index_prune_end], dim=-1) / self.tau, dim=-1)
            sample_index = torch.multinomial(prob, num_samples=regrow_num_cur_epoch)
            index_regrow_cur_epoch = torch.gather(index_local, dim=1, index=sample_index)
            mask = mask.scatter_(dim=1, index=index_regrow_cur_epoch, value=1.)

        mask = mask[:, None, :, None, None].repeat(1, self.N, 1, *self.kernel_size)
        mask = mask.reshape(self.out_channels, self.in_channels, *self.kernel_size)
        return mask


class SUBPConv2dV1(RegrowUniformBlockConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(SUBPConv2dV1, self).__init__(Soft, in_channels, out_channels, kernel_size, stride=stride, **kwargs)


class HUBPConv2dV1(RegrowUniformBlockConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(HUBPConv2dV1, self).__init__(Hard, in_channels, out_channels, kernel_size, stride=stride, **kwargs)
