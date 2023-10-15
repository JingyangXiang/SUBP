import torch
from torch import autograd as autograd


class SoftParameterMasker(autograd.Function):
    """Dynamic STE (straight-through estimator) parameter masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor):
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class SoftParameterMaskerV2(autograd.Function):
    """Dynamic STE (straight-through estimator) parameter masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor):
        ctx.save_for_backward(mask)
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        return grad_output * torch.clamp(mask, min=1.), None


class HardParameterMasker(autograd.Function):
    """Hard parameter masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mask)
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        return grad_output * mask, None
