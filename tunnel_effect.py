"""
Post-hoc Tunnel Effect split (Masarczyk et al., 2023) of a trained model: a linear probe is trained on the
(frozen) output of every layer, and the TE split is the first layer whose probe reaches at least
`threshold` (95%) of the test accuracy of the full model.

Layer indices follow the NNS convention: the probe on layer i uses the output of layers [0, i], so the
corresponding truncated model is obtained with split_layer = i + 1 (NNS keeps the layers [0, split_layer)).

Example:
    python tunnel_effect.py -trainer configs/trainer/resnet18.yaml -set dataset.name=cifar10 settings.seed=2025 \
        --checkpoint SGD_checkpoints/main/cifar10/resnet18/2025/final.pth --out results/te/cifar10_resnet18_2025.json
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from posthoc import base_parser, setup, load_checkpoint, loaders, LayerOutputs, evaluate, save_json
from flops import split_cost


def probe_input(x, max_dim):
    """Flatten the layer output; feature maps larger than max_dim are average-pooled first."""
    if x.dim() == 4 and max_dim is not None and x[0].numel() > max_dim:
        side = max(1, int(math.sqrt(max_dim / x.shape[1])))
        x = F.adaptive_avg_pool2d(x, side)
    return x.reshape(x.size(0), -1).float()


def main():
    parser = base_parser("Tunnel Effect split with linear probes")
    parser.add_argument('--checkpoint', required=True, help='checkpoint of the fully trained model (final.pth)')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_dim', type=int, default=65536, help='feature maps larger than this are average-pooled (32x32 datasets are never pooled)')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--nns_split', type=int, default=None, help='split found by NNS for this run (only stored in the output)')
    args = parser.parse_args()

    config, dataset, net = setup(args)
    device = config["settings"]["device"]
    load_checkpoint(net, args.checkpoint, device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    train_loader, _, test_loader = loaders(config, dataset)
    num_classes = dataset.num_classes
    n_layers = len(net.extractor.net)

    capture = LayerOutputs(net)
    X, y = next(iter(test_loader))
    with torch.no_grad():
        net(X.to(device), y.to(device))
    dims = [probe_input(capture.outputs[i], args.max_dim).shape[1] for i in range(n_layers)]
    # standardization (BatchNorm without affine parameters) + linear layer: still a linear probe
    probes = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(d, affine=False), nn.Linear(d, num_classes)) for d in dims]).to(device)
    optimizer = torch.optim.Adam(probes.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        probes.train()
        losses = torch.zeros(n_layers)
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                net(X, y)
            loss = 0
            for i in range(n_layers):
                l = F.cross_entropy(probes[i](probe_input(capture.outputs[i], args.max_dim)), y)
                losses[i] += l.item()
                loss = loss + l
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"epoch {epoch}: probe losses {[round(v, 3) for v in (losses / len(train_loader)).tolist()]}")

    probes.eval()
    correct = torch.zeros(n_layers)
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                net(X, y)
            for i in range(n_layers):
                correct[i] += (probes[i](probe_input(capture.outputs[i], args.max_dim)).argmax(1) == y).sum().item()
            total += y.numel()
    capture.remove()
    probe_acc = (correct / total).tolist()
    full_acc = evaluate(net, test_loader, device, args.amp)

    te_layer = next((i for i, acc in enumerate(probe_acc) if acc >= args.threshold * full_acc), n_layers - 1)
    te_split = te_layer + 1
    result = {
        "model": config["model"]["type"],
        "dataset": config["dataset"]["name"],
        "num_classes": num_classes,
        "seed": config["settings"]["seed"],
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "full_accuracy": full_acc,
        "probe_accuracy": probe_acc,
        "probe_dims": dims,
        "te_layer": te_layer,
        "te_split": te_split,
        "nns_split": args.nns_split,
    }
    if te_split < n_layers:
        cost = split_cost(config["model"]["type"], config["dataset"]["name"], te_split, num_classes, config["model"]["batch_norm"])
        result["te_param_reduction"] = cost["param_reduction"]
    if args.nns_split is not None:
        cost = split_cost(config["model"]["type"], config["dataset"]["name"], args.nns_split, num_classes, config["model"]["batch_norm"])
        result["nns_param_reduction"] = cost["param_reduction"]
        # accuracy of the probe on the last layer kept by NNS (layer nns_split - 1)
        result["nns_probe_accuracy"] = probe_acc[args.nns_split - 1]
    print(f"full accuracy {full_acc:.4f}; probe accuracies {[round(a, 4) for a in probe_acc]}")
    print(f"TE split: {te_split} (layer {te_layer}); NNS split: {args.nns_split}")
    save_json(result, args.out or f"results/te/{config['dataset']['name']}_{config['model']['type']}_{config['settings']['seed']}.json")


if __name__ == "__main__":
    main()
