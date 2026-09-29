"""
Numerical Rank check (R1, Appendix B.5): why NR of MLP12 on CIFAR-10 reaches ~1200 in our figures, while it is
below 800 in Masarczyk et al. (2023). For every layer it reports
  - legacy:        NR as computed for the paper figures (uncentered N x N Gram matrix of the tunnel set, threshold
                   1e-3 times the *smallest* eigenvalue, because eigh returns ascending eigenvalues)
  - legacy_fixed:  same Gram matrix, threshold relative to the largest eigenvalue
  - cov_tunnel:    NR of the sample covariance, threshold 1e-3 x largest eigenvalue (Masarczyk et al.), tunnel set
  - cov_<n>:       same, on n training examples (e.g. the 10000 samples used in the TE paper)
together with the layer width, which bounds the true rank.

Example:
    python nr_check.py -trainer configs/trainer/mlp_12.yaml -set dataset.name=cifar10 settings.seed=2025 \
        --checkpoint SGD_checkpoints/main/cifar10/mlp_12/2025/final.pth --n_samples 10000
"""
import torch
from torch.utils.data import DataLoader, Subset
from posthoc import base_parser, setup, load_checkpoint, loaders, LayerOutputs, save_json
from metrics import numerical_rank, numerical_rank_legacy


@torch.no_grad()
def layer_features(net, loader, device, max_samples=None):
    capture = LayerOutputs(net)
    feats, n = {}, 0
    net.eval()
    for X, y in loader:
        net(X.to(device), y.to(device))
        for i, out in capture.outputs.items():
            feats.setdefault(i, []).append(out.reshape(out.size(0), -1).float())
        n += X.size(0)
        if max_samples and n >= max_samples:
            break
    capture.remove()
    return {i: torch.cat(f)[:max_samples] for i, f in feats.items()}


def legacy_fixed(embs, eps=1e-3):
    L = torch.linalg.eigvalsh(embs.double() @ embs.double().T)
    return int((L > eps * L.max()).sum())


def main():
    parser = base_parser("Numerical rank: legacy vs corrected computation")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n_samples', type=int, nargs='*', default=[10000])
    args = parser.parse_args()

    config, dataset, net = setup(args)
    device = config["settings"]["device"]
    load_checkpoint(net, args.checkpoint, device)
    train_loader, tunnel_loader, _ = loaders(config, dataset)

    tunnel = layer_features(net, tunnel_loader, device)
    rows = []
    for i, f in tunnel.items():
        rows.append({"layer": i, "width": f.shape[1], "legacy": int(numerical_rank_legacy(f)), "legacy_fixed": legacy_fixed(f),
                     "cov_tunnel": int(numerical_rank(f))})
    del tunnel
    for n in args.n_samples:
        feats = layer_features(net, train_loader, device, n)
        for row in rows:
            row[f"cov_{n}"] = int(numerical_rank(feats[row["layer"]]))
        del feats
    n_tunnel = len(dataset.tunnel_set)
    print(f"tunnel set size: {n_tunnel}")
    print("\t".join(rows[0].keys()))
    for row in rows:
        print("\t".join(str(v) for v in row.values()))
    save_json({"model": config["model"]["type"], "dataset": config["dataset"]["name"], "seed": config["settings"]["seed"],
               "checkpoint": args.checkpoint, "n_tunnel": n_tunnel, "layers": rows},
              args.out or f"results/nr/{config['dataset']['name']}_{config['model']['type']}_{config['settings']['seed']}.json")


if __name__ == "__main__":
    main()
