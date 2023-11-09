import torch.nn as nn

from models.conv_type.regrow_uniform_block import HardRegrowUniformBlockConv2d, SoftRegrowUniformBlockConv2d
from models.conv_type.regrow_uniform_block_v2 import HardRegrowUniformBlockConv2dV2, SoftRegrowUniformBlockConv2dV2
from models.conv_type.uniform_block import HardUniformBlockConv2d, SoftUniformBlockConv2d
from models.conv_type.uniform_block_fine import HardUniformBlockConv2dFine, SoftUniformBlockConv2dFine
from models.conv_type.uniform_block_v2 import HardUniformBlockConv2dV2, SoftUniformBlockConv2dV2
from models.conv_type.uniform_block_v2_fine import HardUniformBlockConv2dV2Fine, SoftUniformBlockConv2dV2Fine

DenseConv2d = nn.Conv2d

__all__ = ['SoftUniformBlockConv2d', 'HardUniformBlockConv2d',
           'SoftUniformBlockConv2dV2', 'HardUniformBlockConv2dV2',
           'SoftRegrowUniformBlockConv2d', 'HardRegrowUniformBlockConv2d',
           'SoftRegrowUniformBlockConv2dV2', 'HardRegrowUniformBlockConv2dV2',
           "SoftUniformBlockConv2dFine", "HardUniformBlockConv2dFine",
           "SoftUniformBlockConv2dV2Fine", "HardUniformBlockConv2dV2Fine",
           'DenseConv2d']
