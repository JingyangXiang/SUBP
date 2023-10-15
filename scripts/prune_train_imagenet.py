# ResNet18, ResNet34, ResNet50, mobilenet_v1
import os

Ns = [16, 32]
data_path = './dataset/imagenet'
set = "ImageNet"
archs = ['mobilenet_v1', 'resnet18', 'resnet34', 'resnet50']
conv_type = 'SoftUniformBlockConv2dV2'
weight_decay = 3.0517578125e-05
nesterov = False
no_bn_decay = True
engine = 'epoch'
no_dali = True
workers = 16
batch_size = 256
lr = batch_size * 0.001

arch_prune_rates = {
    'resnet18': [60 / 128, ],
    'resnet34': [59 / 128, ],
    'resnet50': [66 / 128, 98.5 / 128],
    'mobilenet_v1': [69 / 128, ]
}

os.system(f'rm {set.lower()}.sh')
for arch in archs:
    for N in Ns:
        arch_prune_rate = arch_prune_rates[arch]
        for prune_rate in arch_prune_rate:
            save_dir = f"./{set.lower()}/{arch}-{N}-{conv_type}-{engine}-{str(prune_rate).replace('.', '-')} "
            os.system(f"echo python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
                      f"--no-dali {no_dali} --save_dir {save_dir} --N {N} --conv-type {conv_type} "
                      f"--weight-decay {weight_decay} --nesterov {nesterov} --workers {workers} --engine {engine} "
                      f"--prune-rate {prune_rate} --batch-size {batch_size} --lr {lr} >> {set.lower()}.sh")
