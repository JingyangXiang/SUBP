import numpy as np
import torch
import torch.nn as nn

from utils.prune_criterion import BlockAngularRedundancy


class BaseNMConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BaseNMConv, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.alpha = None
        self.tau = None
        self.prune_criterion = None
        self.prune_rate = None
        self.N = None
        self.cur_epoch = None
        self.total_epoch = None
        self.decay_end = None
        self.decay_start = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super(BaseNMConv, self).reset_parameters()
        # init epoch params
        self.decay_start = self.decay_end = self.total_epoch = self.cur_epoch = 0
        # 1xN block
        self.N = 1
        # prune rate
        self.prune_rate = 0.
        # prune schedule
        self.prune_criterion = "L1"
        # temperature
        self.tau = 1.
        # cos similarity importance
        self.alpha = 1.

    @torch.no_grad()
    def init_N(self, N: int):
        # 初始化1xN的N
        self.N = N
        assert N > 0 and self.out_channels % N == 0
        print(f"==> (1) init 1xN: 1x{self.N}")

    @torch.no_grad()
    def init_prune_schedule(self, schedule: str):
        self.schedule = schedule
        print(f"==> (2) init prune schedule: {schedule}")

    @torch.no_grad()
    def init_prune_rate(self, prune_rate: float):
        self.prune_rate = prune_rate
        print(f"==> (3) init prune rate: {self.prune_rate}")

    @torch.no_grad()
    def init_prune_criterion(self, prune_criterion: str):
        self.prune_criterion = prune_criterion
        print(f"==> (4) init prune criterion: {self.prune_criterion}")

    @torch.no_grad()
    def get_weight_importance(self, weight):
        assert len(weight.shape) == 3
        if self.prune_criterion == "L1":
            return weight.abs().sum(dim=-1)
        elif self.prune_criterion == "L2":
            return torch.norm(weight, dim=-1, keepdim=False)
        elif self.prune_criterion == "BPAR":
            # (out_channels, in_channels, kernel_size, kernel_size)
            # (out_channels // N, N, in_channels, kernel_size, kernel_size)
            # (out_channels // N, in_channels, N x kernel_size x kernel_size)
            mask = self.mask.reshape(self.out_channels // self.N, self.N, self.in_channels, *self.kernel_size)
            mask = mask.transpose(1, 2).flatten(2)
            return BlockAngularRedundancy(weight, mask=mask, alpha=self.alpha)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def update_mask(self):
        self.mask.data = self.get_mask(self.weight)

    @torch.no_grad()
    def show_zero_num(self):
        mask = self.mask
        print(f"weight num: {mask.numel()}, zero num: {torch.sum(torch.eq(mask, 0))}")

    @torch.no_grad()
    def do_grad_decay(self, decay):
        self.weight.grad.add_(decay * (1 - torch.clamp(self.mask, min=0, max=1)) * self.weight)

    @torch.no_grad()
    def get_mask(self, weight):
        raise NotImplementedError

    @torch.no_grad()
    def update_epoch(self, cur_epoch, total_epoch, decay_start, decay_end):
        assert cur_epoch < total_epoch and total_epoch > 0
        assert 0 <= decay_start <= decay_end < total_epoch
        self.cur_epoch = cur_epoch
        self.total_epoch = total_epoch
        self.decay_start = decay_start
        self.decay_end = decay_end
        print(f"=> (cur_epoch, total_epoch, decay_start, decay_end) "
              f"= ({self.cur_epoch}, {self.total_epoch}, {self.decay_start}, {self.decay_end})")

    def get_prune_rate_cur_epoch(self, cur_epoch, decay_start, decay_end) -> float:
        # prune all weight according to prune rate
        if decay_start == decay_end and decay_start == 0:
            return self.prune_rate
        # from o to 1
        if self.schedule == 'linear':
            prune_rate_ratio = min(max((cur_epoch - decay_start) / (decay_end - decay_start), 0), 1)
        elif self.schedule == 'cos':
            if cur_epoch < decay_start:
                prune_rate_ratio = 0.
            elif decay_start <= cur_epoch < decay_end:
                prune_rate_ratio = 1 - np.cos((cur_epoch - decay_start) / (decay_end - decay_start) * np.pi * 0.5)
            else:
                prune_rate_ratio = 1.
        elif self.schedule == 'exp':
            if cur_epoch < decay_start:
                prune_rate_ratio = 0.
            elif decay_start <= cur_epoch < decay_end:
                b = 0.05
                sch = 1 - np.exp(-b * np.arange(decay_end - decay_start))
                prune_rate_ratio = sch[cur_epoch - decay_start] / sch[-1]
            else:
                prune_rate_ratio = 1.
        elif self.schedule == 'cubic':
            prune_rate_ratio = min(max((((cur_epoch - decay_start) / (decay_end - decay_start)) ** 3, 0)), 1)
        else:
            raise ValueError(f"{self.schedule} has not been support")
        assert 0 <= prune_rate_ratio <= 1
        return prune_rate_ratio * self.prune_rate
