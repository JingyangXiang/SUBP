import argparse
import ast

import torch

import models.bn_type
import models.conv_type
import utils.engine
from utils.net_utils import time_file_str


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # misc
    parser.add_argument('--save_dir', type=str, default='./', help='folder to save checkpoints and log')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    # for model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=models.__all__)
    parser.add_argument('--nonlinearity', type=str, default='relu', help='activation for model (default: relu)')
    parser.add_argument('--conv-type', type=str, default='SUBPConv2dV2',
                        choices=models.conv_type.__all__, help='convbn type for network (default: SUBPConv2dV2)')
    parser.add_argument('--bn-type', type=str, default='LearnedBatchNorm',
                        choices=models.bn_type.__all__, help='decay type for bn type')

    # for datatset
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--set', default='ImageNet', type=str, choices=["ImageNet", "ImageNetDali"],
                        help='dataset (default: ImageNet)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='Number of data loading workers (default: 12)')

    # for epoch train
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='Manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='Number of total epochs to run')

    # for learning rate
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--lr-schedule', default='cos', choices=['step', 'cos'], type=str, help='lr scheduler')
    parser.add_argument('--lr-adjust', type=int, default=30, help='number of epochs that change learning rate')
    parser.add_argument('--warmup-length', type=int, default=0, help='number of epochs that warms up learning rate')

    # for optimizer
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', dest='nesterov', type=ast.literal_eval,
                        help='nesterov for SGD', default=False)
    parser.add_argument('--no-bn-decay', dest='no_bn_decay', type=ast.literal_eval, default=False,
                        help='not apply weight decay for bn layer')

    # for pretrain, resume or evaluate
    parser.add_argument('--use-pretrain', dest='use_pretrain', action='store_true',
                        help='use pre-trained model or not in torchvision')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # SUBP
    parser.add_argument('--N', type=int, default=16, help='N for 1xN sparsity')
    parser.add_argument('--decay', type=float, default=0.0002, help='decay for SR-STE method')
    parser.add_argument('--decay-type', type=str, default='v1', choices=['v1', ], help='decay type for conv type')

    # progressive pruning
    parser.add_argument('--decay-start', default=10, type=int)
    parser.add_argument('--decay-end', default=180, type=int)
    parser.add_argument('--prune-schedule', default='cubic', choices=['linear', 'exp', 'cos', 'cubic'],
                        type=str, help='prune scheduler')
    parser.add_argument('--prune-rate', default=0.0, type=float, help='prune rate')
    parser.add_argument('--prune-criterion', default='L1', choices=['L1', 'L2', 'BPAR'],
                        type=str, help='prune criterion')
    parser.add_argument('--engine', default='epoch', choices=utils.engine.__all__,
                        type=str, help='prune engine')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    assert args.use_cuda, "torch.cuds is not available!"
    args.prefix = time_file_str()

    # check params
    if args.set.lower() == 'imagenet':
        assert args.nesterov is False
        assert args.no_bn_decay is True

    return args
