import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )


        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Bottleneck(nn.Module):
    """Bottleneck block for resnet 50

    """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != Bottleneck.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


def has_identity_shortcut(block) -> bool:
    """True for residual blocks that can be replaced by an identity (same input/output shape)."""
    return isinstance(block, (BasicBlock, Bottleneck)) and len(block.shortcut) == 0


def make_resnet_layer(block, in_channels, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(block(in_channels, out_channels, stride))
        in_channels = out_channels * block.expansion

    return layers


def resnet_set(type:str = "resnet10", num_classes: int = 10, num_channels: int = 3, stem: str = "cifar") -> list:
    """
    stem: 'cifar'    -> 3x3 conv, stride 1 (32x32 inputs)
          'cub'      -> 7x7 conv, stride 2, no max-pooling (setting used for CUB-200-2011 in the paper)
          'imagenet' -> 7x7 conv, stride 2, followed by 3x3 max-pooling (standard ImageNet ResNet)
    """
    if type == "resnet10":
        block, num_block = BasicBlock, [1, 1, 1, 1]
    elif type == "resnet18":
        block, num_block = BasicBlock, [2, 2, 2, 2]
    elif type == "resnet34":
        block, num_block = BasicBlock, [3, 4, 6, 3]
    elif type == "resnet50":
        block, num_block = Bottleneck, [3, 4, 6, 3]
    else:
        raise ValueError("Unsupported ResNet type")
    res = []
    if stem == "cifar":
        res.append(nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ))
    elif stem == "cub":
        res.append(nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ))
    elif stem == "imagenet":
        res.append(nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))
    else:
        raise ValueError(f"Unsupported ResNet stem '{stem}'")

    res += make_resnet_layer(block, 64, 64, num_block[0], stride=1)
    res += make_resnet_layer(block, 64 * block.expansion, 128, num_block[1], stride=2)
    res += make_resnet_layer(block, 128 * block.expansion, 256, num_block[2], stride=2)
    res += make_resnet_layer(block, 256 * block.expansion, 512, num_block[3], stride=2)

    res.append(nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    ))
    res.append(nn.Linear(512 * block.expansion, num_classes))

    return res
