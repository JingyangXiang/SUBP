# resnet18_cifar, resnet34_cifar, resnet50_cifar
import os

Ns = [16, 32]
data_path = './dataset/cifar100'
set = "CIFAR100"
archs = ['resnet18_cifar', 'resnet34_cifar', 'resnet50_cifar']
conv_type = 'SoftUniformBlockConv2d'
weight_decay = 0.0005
nesterov = False
no_bn_decay = False
workers = 16

os.system(f'rm {set.lower()}.sh')
for arch in archs:
    for N in Ns:
        os.system(f"echo python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
                  f"--save_dir ./{set.lower()}/{arch}-{N}-{conv_type} --N {N} --conv-type {conv_type} "
                  f"--weight-decay {weight_decay} --nesterov {nesterov} --workers {workers} >> {set.lower()}.sh")
