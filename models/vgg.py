import torch
import torch.nn as nn

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_vgg_layers(cfg: list, batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
    layers = []
    for ind,v in enumerate(cfg):
        if v == "M":
            continue
        else:
            # v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # if ind == len(cfg) - 1:
                #     layers += [nn.Sequential(*[conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])]
                if cfg[ind + 1] == "M":
                    layers += [nn.Sequential(*[conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])]
                else:
                    layers += [nn.Sequential(*[conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])]
            else:
                # if ind == len(cfg) - 1:
                #     layers += [nn.Sequential(*[conv2d, nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])]
                if cfg[ind + 1] == "M":
                    layers += [nn.Sequential(*[conv2d, nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])]
                else:
                    layers += [nn.Sequential(*[conv2d, nn.ReLU(inplace=True)])]
            in_channels = v
    return layers

def vgg_set(type: str = "vgg11", num_classes: int = 10, batch_norm: bool = True, num_channels: int = 3) -> list:
    if type not in cfgs:
        raise ValueError(f"Unsupported VGG type '{type}'")

    features = make_vgg_layers(cfgs[type], batch_norm=batch_norm, in_channels=num_channels)
    # print(features)
    classifier = [
        nn.Sequential(*[nn.Flatten(), nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout()]),
        nn.Sequential(*[nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout()]),
        nn.Linear(4096, num_classes)
    ]

    return features + classifier