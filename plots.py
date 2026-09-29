"""
Figures with uncertainty (mean +- std over seeds) and the split decision made visible (R2).

  metric-epochs   metric of every layer over epochs (Fig. 2 style), split epoch and split layer marked
  metric-layers   metric vs layer for several epochs (Fig. 3 style), split layer marked
  labels          extractor/contractor label of every layer over epochs until the stopping epoch
                  (fraction of seeds labeling the layer as contractor)
  compare-layers  relative IFC change per layer for several groups of runs (e.g. number of classes)
  oracle          test accuracy of simplified models vs split epoch, one curve per split layer (Fig. 7 style)
  oracle-params   test accuracy vs fraction of parameters of the simplified model (Fig. 6 style)

The split is replayed from the recorded IFC with the same SplitDetector used during training.

Examples:
    python plots.py metric-epochs --runs "log_metrics/main/cifar10/resnet18/*/metrics.pth" --metric proxy_nc --out figs/resnet18_cifar10_ifc.pdf
    python plots.py labels --runs "log_metrics/main/cifar10/resnet18/*/metrics.pth" --out figs/resnet18_cifar10_labels.pdf
    python plots.py compare-layers --groups "log_metrics/num_classes/cifar100_c2/vgg11/*/metrics.pth" ... --names 2 10 25 50 100 --out figs/classes.pdf
    python plots.py oracle --results "results/oracle/cifar10/resnet18/*/*.json" --out figs/oracle.pdf
"""
import argparse
import glob
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from utils import replay_split

# validated categorical palette (fixed order); layers beyond 8 reuse it with a different line style
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LINESTYLES = ["-", "--", ":", "-."]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQUENTIAL = ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 1.5,
})


def layer_style(i):
    return {"color": PALETTE[i % len(PALETTE)], "linestyle": LINESTYLES[(i // len(PALETTE)) % len(LINESTYLES)]}


def load_runs(pattern, metric):
    """{run_path: (epochs x layers array of the metric, ifc trajectory, labeling config)}"""
    runs = {}
    for path in sorted(glob.glob(pattern, recursive=True)):
        m = torch.load(path, weights_only=False, map_location="cpu")
        epochs = sorted(e for e in m if any(re.match(r"extr_\d+/proxy_nc$", k) for k in m[e]))
        if not epochs:
            continue
        layers = sorted(int(re.match(r"extr_(\d+)/", k).group(1)) for k in m[epochs[0]] if re.match(rf"extr_\d+/{metric}$", k))
        values = np.array([[float(m[e][f"extr_{l}/{metric}"]) if m[e].get(f"extr_{l}/{metric}") is not None else np.nan
                            for l in layers] for e in epochs])
        ifc = [{int(re.match(r"extr_(\d+)/", k).group(1)): float(v) for k, v in m[e].items() if re.match(r"extr_\d+/proxy_nc$", k)} for e in epochs]
        config = next((m[e]["split/config"] for e in epochs if "split/config" in m[e]), None)
        runs[path] = (values, ifc, config)
    assert runs, f"no metrics.pth found for {pattern}"
    return runs


def stack(arrays):
    """Stack arrays of different lengths (runs stopped at different epochs) padding with NaN."""
    n = max(a.shape[0] for a in arrays)
    out = np.full((len(arrays), n) + arrays[0].shape[1:], np.nan)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return out


def splits(runs, labeling):
    res = []
    for values, ifc, config in runs.values():
        cfg = labeling or config or {"patience": 15}
        detector, labels = replay_split(ifc, **cfg)
        res.append((detector.split_layer, detector.split_epoch, labels))
    return res


def split_text(split_info):
    layers = [s[0] for s in split_info if s[0] is not None]
    epochs = [s[1] for s in split_info if s[1] is not None]
    if not layers:
        return "no split detected", None, None
    txt = f"split layer {np.mean(layers):.1f} ± {np.std(layers):.1f}, epoch {np.mean(epochs):.1f} ± {np.std(epochs):.1f} (n={len(layers)})"
    mode_layer = max(set(layers), key=layers.count)
    return txt, mode_layer, epochs


def caption_note(n_runs):
    return f"mean ± std over {n_runs} runs" if n_runs > 1 else "single run"


def finish(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


def labeling_args(args):
    if args.patience is None and args.rule is None:
        return None
    return {"patience": args.patience or 15, "rule": args.rule or "sign", "threshold": args.threshold,
            "reference": args.reference, "ref_epochs": args.ref_epochs}


def cmd_metric_epochs(args):
    runs = load_runs(args.runs, args.metric)
    data = stack([v for v, _, _ in runs.values()])  # runs x epochs x layers
    mean, std = np.nanmean(data, 0), np.nanstd(data, 0)
    split_info = splits(runs, labeling_args(args))
    txt, split_layer, split_epochs = split_text(split_info)
    epochs = np.arange(mean.shape[0])
    layers = range(mean.shape[1]) if args.layers is None else args.layers
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for l in layers:
        is_split = l == split_layer
        style = layer_style(l)
        ax.plot(epochs, mean[:, l], lw=3.0 if is_split else 1.5, zorder=3 if is_split else 2,
                label=f"Layer {l}" + (" (split)" if is_split else ""), **style)
        if len(runs) > 1:
            ax.fill_between(epochs, mean[:, l] - std[:, l], mean[:, l] + std[:, l], color=style["color"], alpha=0.15, lw=0)
    if split_epochs:
        ax.axvspan(min(split_epochs), max(split_epochs), color=TEXT_2, alpha=0.08, lw=0)
        ax.axvline(np.mean(split_epochs), color=TEXT, ls="--", lw=1.0)
        ax.text(np.mean(split_epochs), 0.98, " split epoch", transform=ax.get_xaxis_transform(), va="top", ha="left", color=TEXT, fontsize=8)
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.ylabel or args.metric)
    ax.set_title(f"{args.title or ''}\n{txt}; {caption_note(len(runs))}".strip(), fontsize=8, color=TEXT_2, loc="left")
    ax.legend(ncol=2 if mean.shape[1] > 8 else 1, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    finish(fig, args.out)


def cmd_metric_layers(args):
    runs = load_runs(args.runs, args.metric)
    data = stack([v for v, _, _ in runs.values()])
    mean, std = np.nanmean(data, 0), np.nanstd(data, 0)
    split_info = splits(runs, labeling_args(args))
    txt, split_layer, _ = split_text(split_info)
    layers_split = [s[0] for s in split_info if s[0] is not None]
    n_epochs, n_layers = mean.shape
    epochs = args.epochs or sorted(set(np.linspace(0, n_epochs - 1, 6).astype(int).tolist()))
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL[1:])
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    x = np.arange(n_layers)
    for k, e in enumerate(epochs):
        color = cmap(0.15 + 0.85 * k / max(1, len(epochs) - 1))
        ax.plot(x, mean[e], color=color, marker="o", ms=3, label=f"epoch {e}")
        if len(runs) > 1:
            ax.fill_between(x, mean[e] - std[e], mean[e] + std[e], color=color, alpha=0.2, lw=0)
    if layers_split:
        ax.axvspan(min(layers_split) - 0.5, max(layers_split) - 0.5, color=PALETTE[1], alpha=0.10, lw=0)
        ax.axvline(np.mean(layers_split) - 0.5, color=PALETTE[1], ls="--", lw=1.2)
        ax.text(np.mean(layers_split) - 0.5, 0.98, " split", transform=ax.get_xaxis_transform(), va="top", color=TEXT, fontsize=8)
    if args.logy:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xlabel("Layer")
    ax.set_ylabel(args.ylabel or args.metric)
    ax.set_title(f"{args.title or ''}\n{txt}; {caption_note(len(runs))}".strip(), fontsize=8, color=TEXT_2, loc="left")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    finish(fig, args.out)


def cmd_labels(args):
    runs = load_runs(args.runs, "proxy_nc")
    split_info = splits(runs, labeling_args(args))
    txt, _, split_epochs = split_text(split_info)
    stop = args.until or (max(split_epochs) + 1 if split_epochs else max(len(s[2]) for s in split_info))
    n_layers = max(len(s[2][0]) for s in split_info)
    frac = np.full((n_layers, stop), np.nan)
    counts = np.zeros(stop)
    acc = np.zeros((n_layers, stop))
    for _, _, labels in split_info:
        for e, lab in enumerate(labels[:stop]):
            for l, v in lab.items():
                acc[l, e] += v
            counts[e] += 1
    valid = counts > 0
    frac[:, valid] = acc[:, valid] / counts[valid]
    cmap = LinearSegmentedColormap.from_list("seq", [SEQUENTIAL[0], SEQUENTIAL[2], SEQUENTIAL[4]])
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    ax.grid(False)
    im = ax.imshow(frac, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    for layer, epoch, _ in split_info:
        if layer is not None and epoch < stop:
            ax.plot([epoch + 0.5], [layer], marker="D", ms=5, color=PALETTE[1], mec="white", mew=1, ls="none")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(n_layers))
    ax.set_title(f"{args.title or ''}\n{txt}; ◆ = split (layer, epoch) of each run".strip(), fontsize=8, color=TEXT_2, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("fraction of runs labeling\nthe layer as contractor" if len(runs) > 1 else "contractor (1) / extractor (0)", fontsize=7)
    finish(fig, args.out)


def cmd_compare_layers(args):
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for g, (pattern, name) in enumerate(zip(args.groups, args.names)):
        runs = load_runs(pattern, "proxy_nc")
        split_info = splits(runs, labeling_args(args))
        rel = []
        for (values, _, _), (layer, epoch, _) in zip(runs.values(), split_info):
            e = epoch if (args.at == "split" and epoch is not None) else values.shape[0] - 1
            rel.append((values[e] - values[0]) / np.abs(values[0]))
        rel = stack([r[None] for r in rel])[:, 0]
        mean, std = np.nanmean(rel, 0), np.nanstd(rel, 0)
        x = np.arange(len(mean))
        color = PALETTE[g % len(PALETTE)]
        layers_split = [s[0] for s in split_info if s[0] is not None]
        label = f"{name}" + (f" (split {np.mean(layers_split):.1f})" if layers_split else "")
        ax.plot(x, mean, color=color, marker="o", ms=3, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        if layers_split:
            ax.axvline(np.mean(layers_split) - 0.5, color=color, ls="--", lw=1.0)
    ax.axhline(0, color=TEXT_2, lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"relative IFC change w.r.t. epoch 1 ({'split' if args.at == 'split' else 'final'} epoch)")
    ax.set_title(f"{args.title or ''}\nmean ± std over runs; dashed lines: split layer".strip(), fontsize=8, color=TEXT_2, loc="left")
    ax.legend(title=args.legend_title, fontsize=7, title_fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    finish(fig, args.out)


def load_results(pattern):
    rows = [json.load(open(p)) for p in sorted(glob.glob(pattern, recursive=True))]
    assert rows, f"no results for {pattern}"
    return rows


def cmd_oracle(args):
    rows = [r for r in load_results(args.results) if r.get("split_layer") is not None]
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["split_layer"], {}).setdefault(r["split_epoch"], []).append(100 * r["test_accuracy"])
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for l in sorted(by_layer):
        e = sorted(by_layer[l])
        mean = np.array([np.mean(by_layer[l][k]) for k in e])
        std = np.array([np.std(by_layer[l][k]) for k in e])
        n = max(len(by_layer[l][k]) for k in e)
        style = layer_style(l)
        ax.plot(e, mean, marker="o", ms=3, label=f"Layer {l}" + (" (NNS)" if l == args.nns_layer else ""),
                lw=3.0 if l == args.nns_layer else 1.5, **style)
        if n > 1:
            ax.fill_between(e, mean - std, mean + std, color=style["color"], alpha=0.15, lw=0)
    ax.set_xlabel("Split epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{args.title or ''}\nmean ± std over seeds".strip(), fontsize=8, color=TEXT_2, loc="left")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    finish(fig, args.out)


def cmd_oracle_params(args):
    rows = [r for r in load_results(args.results) if r.get("split_layer") is not None]
    if args.epoch is not None:
        rows = [r for r in rows if r["split_epoch"] == args.epoch]
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["dataset"]), {}).setdefault(r["split_layer"], []).append(r)
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for g, ((model, dataset), layers) in enumerate(sorted(groups.items())):
        ls = sorted(layers)
        frac = np.array([np.mean([r["params"] / r["params_full"] for r in layers[l]]) for l in ls])
        acc = np.array([np.mean([100 * r["test_accuracy"] for r in layers[l]]) for l in ls])
        std = np.array([np.std([100 * r["test_accuracy"] for r in layers[l]]) for l in ls])
        color = PALETTE[g % len(PALETTE)]
        ax.errorbar(frac, acc, yerr=std, color=color, marker="o", ms=4, capsize=2, lw=1.5, label=f"{model} {dataset}")
    ax.set_xlabel("Fraction of parameters of the full model")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{args.title or ''}\nmean ± std over seeds".strip(), fontsize=8, color=TEXT_2, loc="left")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    finish(fig, args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    commands = {"metric-epochs": cmd_metric_epochs, "metric-layers": cmd_metric_layers, "labels": cmd_labels,
                "compare-layers": cmd_compare_layers, "oracle": cmd_oracle, "oracle-params": cmd_oracle_params}
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--title", default=None)
        p.add_argument("--width", type=float, default=4.5)
        p.add_argument("--height", type=float, default=3.0)
        if name in ("oracle", "oracle-params"):
            p.add_argument("--results", required=True, help="glob of results json files of simplified models")
            p.add_argument("--nns_layer", type=int, default=None)
            p.add_argument("--epoch", type=int, default=None)
            continue
        if name == "compare-layers":
            p.add_argument("--groups", nargs="+", required=True, help="one glob of metrics.pth per group")
            p.add_argument("--names", nargs="+", required=True)
            p.add_argument("--legend_title", default=None)
            p.add_argument("--at", choices=["split", "final"], default="split")
        else:
            p.add_argument("--runs", required=True, help="glob of metrics.pth files (one per seed)")
        p.add_argument("--metric", default="proxy_nc", help="proxy_nc (IFC), cka_1, numerical_rank, ncc, intrinsic_dim, ...")
        p.add_argument("--ylabel", default=None)
        p.add_argument("--logy", action="store_true")
        p.add_argument("--layers", type=int, nargs="*", default=None)
        p.add_argument("--epochs", type=int, nargs="*", default=None)
        p.add_argument("--until", type=int, default=None, help="(labels) last epoch shown (default: last split epoch)")
        # labeling used to replay the split (default: the one recorded during training, or the paper default)
        p.add_argument("--patience", type=int, default=None)
        p.add_argument("--rule", default=None)
        p.add_argument("--threshold", type=float, default=0.0)
        p.add_argument("--reference", default="first")
        p.add_argument("--ref_epochs", type=int, default=1)
    args = parser.parse_args()
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
