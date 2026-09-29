"""
Depth-reduction baselines for ResNets, removing residual blocks with an identity shortcut
(blocks with a projection shortcut change the tensor shape and cannot be removed).

cka     Layer pruning through representation similarity (Pons et al., ICPR 2024): starting from a fully
        trained model, iteratively remove the block whose removal keeps the output representation
        (penultimate layer, on the tunnel set) most similar to the original one according to linear CKA,
        and fine-tune the pruned model after each removal.
lacoot  LaCoOT (Quetu et al., ICCV 2025): train from scratch with a regularizer that minimizes the
        Max-Sliced Wasserstein distance between the input and the output distributions of each removable
        block; after training, remove the blocks with the smallest distance (optionally fine-tuning).

Both report test accuracy, parameters, forward FLOPs and training wall-clock time after each removal.

Examples:
    python depth_baselines.py cka -trainer configs/trainer/resnet18.yaml -set dataset.name=cifar10 settings.seed=2025 \
        --checkpoint SGD_checkpoints/main/cifar10/resnet18/2025/final.pth --ft_epochs 10
    python depth_baselines.py lacoot -trainer configs/trainer/resnet18.yaml -set dataset.name=cifar10 settings.seed=2025 --lam 1.0
"""
import copy
import time
import torch
import torch.nn.functional as F
from posthoc import (base_parser, setup, load_checkpoint, loaders, LayerOutputs, evaluate, removable_blocks,
                     remove_blocks, train_epochs, save_json)
from flops import count_params, forward_flops, input_shape


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    return ((Y.T @ X).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


@torch.no_grad()
def penultimate(net, loader, device):
    """Output of the layer before the final classifier (pooled features)."""
    net.eval()
    layer = len(net.extractor.net) - 2
    capture = LayerOutputs(net, [layer])
    feats = []
    for X, y in loader:
        net(X.to(device), y.to(device))
        feats.append(capture.outputs[layer].reshape(X.size(0), -1).double())
    capture.remove()
    return torch.cat(feats)


def cost(net, dataset_name):
    return {"params": count_params(net), "flops": forward_flops(net, input_shape(dataset_name), next(net.parameters()).device)}


def run_cka(args, config, dataset, net):
    device = config["settings"]["device"]
    load_checkpoint(net, args.checkpoint, device)
    train_loader, tunnel_loader, test_loader = loaders(config, dataset, eval_transform_train=False)
    ft_opt = {"name": "SGD", "lr": args.ft_lr, "momentum": 0.9, "weight_decay": config["optimizer"]["weight_decay"]}

    reference = penultimate(net, tunnel_loader, device)
    candidates = removable_blocks(net)
    steps = [{"removed": [], "test_accuracy": evaluate(net, test_loader, device), **cost(net, config["dataset"]["name"]), "time_finetune": 0.0}]
    removed, total_ft_time = [], 0.0
    n_remove = len(candidates) if args.num_remove is None else min(args.num_remove, len(candidates))
    for step in range(n_remove):
        t0 = time.time()
        scores = {}
        for b in candidates:
            if b in removed:
                continue
            pruned = remove_blocks(copy.deepcopy(net), [b])
            scores[b] = linear_cka(reference, penultimate(pruned, tunnel_loader, device))
        best = max(scores, key=scores.get)
        selection_time = time.time() - t0
        remove_blocks(net, [best])
        removed.append(best)
        acc_before_ft = evaluate(net, test_loader, device)
        ft_time = train_epochs(net, train_loader, args.ft_epochs, device, ft_opt, amp=args.amp) if args.ft_epochs > 0 else 0.0
        total_ft_time += ft_time + selection_time
        steps.append({"removed": list(removed), "cka_scores": scores, "test_accuracy_before_ft": acc_before_ft,
                      "test_accuracy": evaluate(net, test_loader, device), **cost(net, config["dataset"]["name"]),
                      "time_selection": selection_time, "time_finetune": ft_time, "time_cumulative": total_ft_time})
        print(f"removed {removed}: accuracy {steps[-1]['test_accuracy']:.4f} (before fine-tuning {acc_before_ft:.4f})")
    return {"method": "cka_pruning", "checkpoint": args.checkpoint, "ft_epochs": args.ft_epochs, "ft_lr": args.ft_lr, "steps": steps}


def max_sliced_wasserstein(a, b, n_iter=10, relative=True):
    """
    Max-Sliced 2-Wasserstein distance between two empirical distributions of the same size (rows of a and b),
    with the slicing direction found by normalized gradient ascent on the unit sphere.
    relative=True divides the distance by the standard deviation of the projected input (detached): the raw
    distance scales with the feature norms and can be reduced by shrinking the features instead of making
    the block closer to an identity.
    """
    a, b = a.reshape(a.size(0), -1).float(), b.reshape(b.size(0), -1).float()
    theta = F.normalize(torch.randn(a.size(1), device=a.device), dim=0)
    a_d, b_d = a.detach(), b.detach()
    for _ in range(n_iter):
        theta.requires_grad_(True)
        with torch.enable_grad():
            w = ((torch.sort(a_d @ theta)[0] - torch.sort(b_d @ theta)[0]) ** 2).mean()
            grad, = torch.autograd.grad(w, theta)
        theta = F.normalize(theta.detach() + grad / (grad.norm() + 1e-12), dim=0)
    proj_a = a @ theta
    distance = torch.sqrt(((torch.sort(proj_a)[0] - torch.sort(b @ theta)[0]) ** 2).mean() + 1e-12)
    if relative:
        distance = distance / (proj_a.detach().std() + 1e-6)
    return distance


class BlockIO:
    """Captures input and output of the given blocks."""
    def __init__(self, net, blocks):
        self.io = {}
        self.handles = [net.extractor.net[i].register_forward_hook(self._hook(i)) for i in blocks]

    def _hook(self, i):
        def hook(module, inp, out):
            self.io[i] = (inp[0], out)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def run_lacoot(args, config, dataset, net):
    device = config["settings"]["device"]
    train_loader, tunnel_loader, test_loader = loaders(config, dataset, eval_transform_train=False)
    blocks = removable_blocks(net)
    io = BlockIO(net, blocks)

    def regularizer():
        return args.lam * torch.stack([max_sliced_wasserstein(i, o, args.msw_iter, not args.msw_raw) for i, o in io.io.values()]).mean()

    epochs = config["experiment"]["epochs"] if args.epochs is None else args.epochs
    train_time = train_epochs(net, train_loader, epochs, device, config["optimizer"], config["scheduler"], extra_loss=regularizer, amp=args.amp)

    # distance of every block on the tunnel set
    net.eval()
    distances = {b: 0.0 for b in blocks}
    with torch.no_grad():
        for X, y in tunnel_loader:
            net(X.to(device), y.to(device))
            for b, (i, o) in io.io.items():
                distances[b] += max_sliced_wasserstein(i, o, args.msw_iter, not args.msw_raw).item() / len(tunnel_loader)
    io.remove()
    order = sorted(blocks, key=distances.get)
    ft_opt = {"name": "SGD", "lr": args.ft_lr, "momentum": 0.9, "weight_decay": config["optimizer"]["weight_decay"]}

    trained = copy.deepcopy(net)
    steps = []
    for k in range(len(blocks) + 1):
        pruned = remove_blocks(copy.deepcopy(trained), order[:k])
        step = {"removed": order[:k], "test_accuracy": evaluate(pruned, test_loader, device), **cost(pruned, config["dataset"]["name"])}
        if args.ft_epochs > 0 and k > 0:
            step["test_accuracy_before_ft"] = step["test_accuracy"]
            step["time_finetune"] = train_epochs(pruned, train_loader, args.ft_epochs, device, ft_opt, amp=args.amp)
            step["test_accuracy"] = evaluate(pruned, test_loader, device)
        steps.append(step)
        print(f"removed {order[:k]}: accuracy {step['test_accuracy']:.4f}")
    return {"method": "lacoot", "lam": args.lam, "msw_iter": args.msw_iter, "msw_relative": not args.msw_raw, "epochs": epochs, "distances": distances,
            "time_train": train_time, "ft_epochs": args.ft_epochs, "steps": steps}


def main():
    parser = base_parser("Depth-reduction baselines")
    parser.add_argument('method', choices=['cka', 'lacoot'])
    parser.add_argument('--checkpoint', default=None, help='(cka) fully trained model')
    parser.add_argument('--num_remove', type=int, default=None, help='(cka) number of blocks to remove (default: all removable)')
    parser.add_argument('--ft_epochs', type=int, default=None, help='fine-tuning epochs after each removal (default: 10 for cka, 0 for lacoot)')
    parser.add_argument('--ft_lr', type=float, default=0.01)
    parser.add_argument('--lam', type=float, default=1.0, help='(lacoot) weight of the Max-Sliced Wasserstein regularizer')
    parser.add_argument('--msw_iter', type=int, default=10, help='(lacoot) ascent steps to find the slicing direction')
    parser.add_argument('--msw_raw', action='store_true', help='(lacoot) use the raw (scale-dependent) distance instead of the relative one')
    parser.add_argument('--epochs', type=int, default=None, help='(lacoot) training epochs (default: from the trainer config)')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    if args.ft_epochs is None:
        args.ft_epochs = 10 if args.method == 'cka' else 0

    config, dataset, net = setup(args)
    assert "resnet" in config["model"]["type"], "depth baselines are implemented for ResNets (residual blocks)"
    t0 = time.time()
    result = run_cka(args, config, dataset, net) if args.method == 'cka' else run_lacoot(args, config, dataset, net)
    result.update({"model": config["model"]["type"], "dataset": config["dataset"]["name"], "seed": config["settings"]["seed"],
                   "num_classes": dataset.num_classes, "time_total": time.time() - t0, "n_train": len(dataset.train_set)})
    name = f"{args.method}" + (f"_lam{args.lam}" if args.method == 'lacoot' else "")
    save_json(result, args.out or f"results/baselines/{config['dataset']['name']}_{config['model']['type']}_{config['settings']['seed']}_{name}.json")


if __name__ == "__main__":
    main()
