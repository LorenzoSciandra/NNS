import torch
import torch.nn as nn
from dataset import DATASET_CLASSES, DATASET_NUM_CHANNELS, DATASET_IMG_SIZE
from model import Extractor, Tunnel, Wrapped_NN, return_layer_list, DEFAULT_TUNNEL_ARGS


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def input_shape(dataset: str) -> tuple:
    return (DATASET_NUM_CHANNELS[dataset], DATASET_IMG_SIZE[dataset], DATASET_IMG_SIZE[dataset])


def forward_flops(model: nn.Module, in_shape: tuple, device='cpu') -> int:
    """
    FLOPs (2 x multiply-accumulates) of one forward pass for a single example, counting convolutions,
    linear layers and batch normalizations. Training FLOPs are usually estimated as 3x the forward ones.
    """
    flops = []

    def conv_hook(m, inp, out):
        k = m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        flops.append(2 * out[0].numel() * k + (out[0].numel() if m.bias is not None else 0))

    def linear_hook(m, inp, out):
        flops.append(2 * out[0].numel() * m.in_features + (out[0].numel() if m.bias is not None else 0))

    def bn_hook(m, inp, out):
        flops.append(2 * out[0].numel())

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            hooks.append(m.register_forward_hook(bn_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.zeros((2,) + tuple(in_shape), device=device)
        if isinstance(model, Wrapped_NN):
            inference = model.tunnel.inference
            model.tunnel.inference = True
            model(x, torch.zeros(2, dtype=torch.long, device=device))
            model.tunnel.inference = inference
        else:
            model(x)
    model.train(was_training)
    for h in hooks:
        h.remove()
    # the hooks see a batch of 2 but only count the first example
    return int(sum(flops))


def build_net(model_type: str, dataset: str, num_classes: int = None, batch_norm: bool = True,
              split_layer: int = None, tunnel_args: dict = None, device='cpu') -> Wrapped_NN:
    """Full network or, if split_layer is given, the network simplified by NNS at that layer."""
    num_classes = DATASET_CLASSES[dataset] if num_classes is None else num_classes
    tunnel_args = {**DEFAULT_TUNNEL_ARGS, **(tunnel_args or {})}
    tunnel_args.pop("num_classes", None)
    extractor = Extractor(return_layer_list(model_type, dataset, batch_norm, num_classes=num_classes), num_classes)
    tunnel = Tunnel(active=False, num_classes=num_classes, device=device, **tunnel_args)
    net = Wrapped_NN(extractor, tunnel, enable_tunnel=split_layer is not None).to(device)
    if split_layer is not None:
        with torch.no_grad():
            net.eval()
            net(torch.zeros((2,) + input_shape(dataset), device=device), torch.zeros(2, dtype=torch.long, device=device))
        net.activate_tunnel(split_layer, num_classes, verbose=False)
        net.to(device)
    return net


def split_cost(model_type: str, dataset: str, split_layer: int, num_classes: int = None, batch_norm: bool = True,
               tunnel_args: dict = None) -> dict:
    """Parameters and forward FLOPs of the full and of the simplified model (linear-probing head by default)."""
    full = build_net(model_type, dataset, num_classes, batch_norm)
    simplified = build_net(model_type, dataset, num_classes, batch_norm, split_layer, tunnel_args)
    shape = input_shape(dataset)
    res = {
        "params_full": count_params(full),
        "params_split": count_params(simplified, trainable_only=True),
        "flops_full": forward_flops(full, shape),
        "flops_split": forward_flops(simplified, shape),
    }
    res["param_reduction"] = 1 - res["params_split"] / res["params_full"]
    res["flops_reduction"] = 1 - res["flops_split"] / res["flops_full"]
    return res


def nns_training_flops(flops_full: int, flops_split: int, split_epoch: int, epochs: int, n_train: int,
                       n_tunnel: int = 0, resumed_twice: bool = False) -> dict:
    """
    Training FLOPs (3x forward) of full training and of NNS: full model up to the split epoch (included),
    simplified model afterwards, plus one forward pass on the tunnel set per monitored epoch.
    resumed_twice=True accounts for the two-stage pipeline, where the split epoch is trained again
    after resuming from its checkpoint.
    """
    full_epochs = split_epoch + 1
    split_epochs = epochs - split_epoch - (0 if resumed_twice else 1)
    full = 3 * flops_full * n_train * epochs
    nns = 3 * n_train * (flops_full * full_epochs + flops_split * split_epochs) + flops_full * n_tunnel * full_epochs
    return {"train_flops_full": full, "train_flops_nns": nns, "train_flops_reduction": 1 - nns / full}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parameters and FLOPs of full and NNS-simplified models")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", type=int, nargs="+", required=True)
    parser.add_argument("--num_classes", type=int, default=None)
    args = parser.parse_args()
    for split in args.splits:
        c = split_cost(args.model, args.dataset, split, args.num_classes)
        print(f"{args.model} {args.dataset} split={split}: params {c['params_split']:,}/{c['params_full']:,} "
              f"(-{100 * c['param_reduction']:.2f}%), fwd GFLOPs {c['flops_split'] / 1e9:.4f}/{c['flops_full'] / 1e9:.4f} "
              f"(-{100 * c['flops_reduction']:.2f}%)")
