# ResNet18, ResNet34, ResNet50, mobilenet_v1

Ns = [16, 32]
data_path = './dataset/imagenet'
set = "ImageNet"
archs = ['mobilenet_v1', 'resnet18', 'resnet34', 'resnet50']
conv_type = 'SUBPConv2dV2'
weight_decay = 0.0001
no_bn_decay = True
workers = 16
batch_size = 512
lr = 0.1

arch_prune_rates = {
    'resnet18': [60 / 128, ],
    'resnet34': [59 / 128, ],
    'resnet50': [66 / 128, 98.5 / 128],
    'mobilenet_v1': [69 / 128, ]
}

for arch in archs:
    for N in Ns:
        arch_prune_rate = arch_prune_rates[arch]
        for prune_rate in arch_prune_rate:
            if arch == 'resnet18':
                set = 'ImageNetDali'
                batch_size = 200
            save_dir = f"./{set.lower()}/{arch}-{N}-{conv_type}-{str(prune_rate).replace('.', '-')} "
            print(f"python pruning_train.py {data_path} --set {set} -a {arch} --no-bn-decay {no_bn_decay} "
                  f"--save_dir {save_dir} --N {N} --conv-type {conv_type} "
                  f"--weight-decay {weight_decay} --workers {workers} "
                  f"--prune-rate {prune_rate} --batch-size {batch_size} --lr {lr} --evaluate --pretrained imagenet/{arch}-{N}-")
