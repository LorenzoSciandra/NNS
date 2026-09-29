"""
Offline ablation of the NNS labeling rule (R1). The per-epoch IFC of every layer recorded during full
training (log_metrics/**/metrics.pth) is replayed through SplitDetector with different
  - labeling rules: sign of the variation (default), absolute / relative thresholds on the variation,
    absolute threshold on the IFC value,
  - reference values: first epoch (default), mean of the first k epochs, value after a k-epoch warm-up,
  - patience: 5 / 10 / 15 / 20 / 30 epochs,
and the resulting split layer, split epoch, parameter reduction and fraction of training on the full model
are reported. The replay of the default variant is checked against the split logged during training when
available. The accuracy of the simplified model for a variant requires training it
(python experiments.py labeling), since checkpoints exist only at the original split epochs.

Example:
    python split_ablation.py --metrics "log_metrics/main/**/metrics.pth" --out tables_generated/labeling_ablation
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import torch
from utils import replay_split
from flops import split_cost

DEFAULT = {"rule": "sign", "threshold": 0.0, "reference": "first", "ref_epochs": 1, "patience": 15}


def variants(patiences, ks, rel_thresholds, abs_thresholds, value_thresholds):
    out = {"default": dict(DEFAULT)}
    for p in patiences:
        out[f"patience_{p}"] = {**DEFAULT, "patience": p}
    for k in ks:
        out[f"ref_mean_{k}"] = {**DEFAULT, "reference": "mean", "ref_epochs": k}
        out[f"ref_warmup_{k}"] = {**DEFAULT, "reference": "warmup", "ref_epochs": k}
    for t in rel_thresholds:
        out[f"rel_delta_{t}"] = {**DEFAULT, "rule": "rel_delta", "threshold": t}
    for t in abs_thresholds:
        out[f"abs_delta_{t}"] = {**DEFAULT, "rule": "abs_delta", "threshold": t}
    for t in value_thresholds:
        out[f"abs_value_{t}"] = {**DEFAULT, "rule": "abs_value", "threshold": t}
    return out


def load_trajectory(path):
    """IFC per epoch ({layer: value}), and the split logged during training with its labeling config, if recorded."""
    metrics = torch.load(path, weights_only=False, map_location="cpu")
    trajectory, logged, run_config = [], None, None
    for epoch in sorted(metrics):
        m = metrics[epoch]
        ifc = {int(re.match(r"extr_(\d+)/proxy_nc", k).group(1)): float(v) for k, v in m.items() if re.match(r"extr_\d+/proxy_nc$", k)}
        if not ifc:
            break
        trajectory.append(ifc)
        run_config = m.get("split/config", run_config)
        if logged is None and run_config is not None and m["split/counter"] == run_config["patience"]:
            logged = (epoch, m["split/candidate"])
    return trajectory, logged, run_config


def parse_path(path):
    """log_metrics/[run_tag/]<dataset>/<model>/<seed>/metrics.pth"""
    parts = os.path.normpath(path).split(os.sep)
    return {"dataset": parts[-4], "model": parts[-3], "seed": parts[-2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", nargs="+", default=["log_metrics/**/metrics.pth"], help="glob(s) of metrics.pth files of full-training runs")
    parser.add_argument("--epochs", type=int, default=None, help="total training epochs (for %%training; default: from the trainer configs)")
    parser.add_argument("--patience", type=int, nargs="+", default=[5, 10, 15, 20, 30])
    parser.add_argument("--ref_epochs", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--rel_thresholds", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--abs_thresholds", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--value_thresholds", type=float, nargs="+", default=[], help="thresholds on the raw IFC value (scale dependent)")
    parser.add_argument("--out", default="tables_generated/labeling_ablation")
    args = parser.parse_args()

    import yaml
    epochs_of = lambda model: args.epochs or yaml.safe_load(open(f"configs/trainer/{model}.yaml"))["experiment"]["epochs"]
    paths = sorted({p for pattern in args.metrics for p in glob.glob(pattern, recursive=True)})
    assert paths, f"no metrics.pth found for {args.metrics}"
    all_variants = variants(args.patience, args.ref_epochs, args.rel_thresholds, args.abs_thresholds, args.value_thresholds)

    rows, reductions, checks = [], {}, []
    for path in paths:
        info = parse_path(path)
        trajectory, logged, run_config = load_trajectory(path)
        num_classes = None
        m = re.match(r"(.+?)(?:_c(\d+))?(?:_t(\d+))?$", info["dataset"])
        dataset = m.group(1)
        num_classes = int(m.group(2)) if m.group(2) else None
        for name, v in all_variants.items():
            detector, _ = replay_split(trajectory, patience=v["patience"], rule=v["rule"], threshold=v["threshold"],
                                       reference=v["reference"], ref_epochs=v["ref_epochs"])
            row = {**info, "path": path, "variant": name, **v, "epochs_recorded": len(trajectory),
                   "split_layer": detector.split_layer, "split_epoch": detector.split_epoch}
            if detector.split_layer is not None:
                key = (info["model"], dataset, num_classes, detector.split_layer)
                if key not in reductions:
                    reductions[key] = split_cost(info["model"], dataset, detector.split_layer, num_classes)["param_reduction"]
                row["param_reduction"] = 100 * reductions[key]
                row["training_fraction"] = 100 * detector.split_epoch / epochs_of(info["model"])
            rows.append(row)
        if run_config is not None:
            # the replay with the configuration used during training must give the logged split
            detector, _ = replay_split(trajectory, **run_config)
            replayed = (detector.split_epoch, detector.split_layer) if detector.split_layer is not None else None
            checks.append((path, logged, replayed))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out + "_runs.csv", index=False)
    summary = df.groupby(["model", "dataset", "variant"]).agg(
        runs=("seed", "count"), fired=("split_layer", lambda s: s.notna().sum()),
        split_mean=("split_layer", "mean"), split_std=("split_layer", "std"),
        epoch_mean=("split_epoch", "mean"), epoch_std=("split_epoch", "std"),
        reduction=("param_reduction", "mean"), training=("training_fraction", "mean")).reset_index()
    summary.to_csv(args.out + "_summary.csv", index=False)

    # compact LaTeX table: one row per variant, split and epoch (mean over seeds) for every configuration
    fmt = lambda m, s: "--" if pd.isna(m) else f"{m:.1f}\\stdf{{{0 if pd.isna(s) else s:.1f}}}"
    table = summary.assign(cell=[f"{fmt(r.split_mean, r.split_std)} / {fmt(r.epoch_mean, r.epoch_std)}" for r in summary.itertuples()])
    pivot = table.pivot_table(index="variant", columns=["model", "dataset"], values="cell", aggfunc="first")
    pivot = pivot.reindex([v for v in all_variants if v in pivot.index])
    with open(args.out + ".tex", "w") as f:
        f.write(pivot.to_latex(escape=False, na_rep="--", caption="Split layer / split epoch (mean and standard deviation over seeds) for different labeling rules, references and patience values.", label="tab:labeling_ablation"))

    if checks:
        mismatches = [c for c in checks if c[1] != c[2]]
        print(f"replay with the training labeling config vs split logged during training: {len(checks) - len(mismatches)}/{len(checks)} identical")
        for path, logged, replayed in mismatches:
            print(f"  MISMATCH {path}: logged {logged}, replayed {replayed}")
    never = df[df.split_layer.isna()]
    if len(never):
        print(f"{len(never)} (run, variant) pairs never reached the patience (e.g. trajectories truncated by -stop_at_split)")
    print(summary.to_string(index=False))
    print(f"Saved {args.out}_runs.csv, {args.out}_summary.csv, {args.out}.tex")


if __name__ == "__main__":
    main()
