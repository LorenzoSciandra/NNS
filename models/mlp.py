import torch
import torch.nn as nn


def first_layer(features: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(features, output_dim),
        nn.ReLU(),
    )

def middle_layer(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
    )

def last_layer(input_dim: int, num_classes: int) -> nn.Sequential:
    return nn.Linear(input_dim, num_classes, bias=False)
    

def mlp_set(num_classes, num_features, num_layers=3) -> list:
    layers = []
    layers.append(first_layer(num_features, 1024))
    
    for _ in range(num_layers - 2):
        layers.append(middle_layer(1024, 1024))

    layers.append(last_layer(1024, num_classes))

    return layers