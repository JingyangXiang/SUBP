import math

import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv_bn(inp, oup, stride, builder):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        builder.batchnorm(oup),
        builder.activation(inplace=True)
    )


def conv_dw(inp, oup, stride, builder):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        builder.batchnorm(inp),
        builder.activation(inplace=True),
        builder.conv1x1(inp, oup, bias=False),
        builder.batchnorm(oup),
        builder.activation(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, num_classes, builder):
        super(MobileNet, self).__init__()

        if builder is None:
            from utils.builder import get_builder
            builder = get_builder()

        in_planes = 32
        cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

        self.conv1 = conv_bn(3, in_planes, stride=2, builder=builder)

        self.features = self._make_layers(in_planes, cfg, conv_dw, builder)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(cfg[-1], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)

        x = self.fc(x)

        return x

    def _make_layers(self, in_planes, cfg, layer, builder):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride, builder))
            in_planes = out_planes
        return nn.Sequential(*layers)


def mobilenet_v1(pretrained=False, builder=None, num_classes=1000):
    if pretrained:
        raise NotImplementedError
    return MobileNet(num_classes=num_classes, builder=builder)


if __name__ == "__main__":
    model = mobilenet_v1(num_classes=1000)
    data = torch.randn(1, 3, 224, 224)
    print(model(data).shape)
