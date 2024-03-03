import torch.nn as nn

from models.conv_type.regrow_uniform_block import SUBPConv2dV1
from models.conv_type.regrow_uniform_block_v2 import SUBPConv2dV2

DenseConv2d = nn.Conv2d

__all__ = ['SUBPConv2dV1', "SUBPConv2dV2", "DenseConv2d"]
