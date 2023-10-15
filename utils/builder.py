import torch.nn as nn

import models.bn_type
import models.conv_type


class Builder(object):
    def __init__(self, conv_layer, bn_layer, nonlinearity):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.nonlinearity = nonlinearity

    def batchnorm(self, planes):
        return self.bn_layer(planes)

    def conv(self, kernel_size, in_planes, out_planes, stride=1, bias=False):
        conv_bn_layer = self.conv_layer

        if kernel_size == 3:
            conv = conv_bn_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
            )
        elif kernel_size == 1:
            conv = conv_bn_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
            )
        elif kernel_size == 5:
            conv = conv_bn_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=bias,
            )
        elif kernel_size == 7:
            conv = conv_bn_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=bias,
            )
        else:
            return None

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, bias=False):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, bias=bias)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def activation(self, **kwargs):
        if self.nonlinearity == "relu":
            return (lambda: nn.ReLU(inplace=True))()
        else:
            raise ValueError(f"{self.nonlinearity} is not an initialization option!")


def get_builder(conv_type='DenseConv2d', bn_type='LearnedBatchNorm', nonlinearity='relu'):
    print('==' * 50)
    print("==> Conv Type: {}".format(conv_type))
    print("==> BN   Type: {}".format(bn_type))
    print("==> Act  Type: {}".format(nonlinearity))

    conv_layer = getattr(models.conv_type, conv_type)
    bn_layer = getattr(models.bn_type, bn_type)
    nonlinearity = nonlinearity

    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer, nonlinearity=nonlinearity)

    return builder
