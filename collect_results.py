"""
Aggregate the json files written by the experiments into tables (csv + LaTeX, in --out_dir).

  split       Table 1 (split layer / epoch, %reduction, %training) from full-training runs; the same numbers
              are also written as the %Reduction / %Training rows of the appendix accuracy tables, so that
              both come from the same set of runs
  accuracy    full training vs simplified models (mean +- std), with train accuracy and train/test gap
              (regularization hypothesis)
  te          NNS split vs post-hoc Tunnel Effect split, with the accuracy of the model truncated at each split
  cost        wall-clock time of full training and NNS, per-epoch overhead of the IFC, training FLOPs
  seeds       tunnel-set seed vs optimization seed study
  baselines   depth-reduction baselines (CKA pruning, LaCoOT) and autoencoder baseline

Example:
    python collect_results.py split accuracy te cost --tag main
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
from flops import split_cost, nns_training_flops

NAMES = {"mlp_10": "MLP10", "mlp_12": "MLP12", "resnet10": "ResNet10", "resnet18": "ResNet18", "resnet34": "ResNet34",
         "resnet50": "ResNet50", "vgg11": "VGG11", "vgg16": "VGG16", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
         "fashion_mnist": "Fashion-MNIST", "cub": "CUB", "imagenet100": "ImageNet-100"}


def pm(values, scale=1.0, digits=1):
    values = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float) * scale
    if len(values) == 0:
        return "\\multicolumn{2}{c}{--}"
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.{digits}f} & \\stdf{{{std:.{digits}f}}}"


def load(pattern):
    rows = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            r = json.load(f)
        r["path"] = path
        rows.append(r)
    return rows


def dataset_key(r):
    return r["dataset"] + (f"_c{r['num_classes']}" if r.get("config", {}).get("dataset", {}).get("num_classes") else "")


def trainer_runs(args):
    runs = [r for r in load("results/**/*.json") if "mode" in r]
    if args.tag:
        runs = [r for r in runs if (r.get("run_tag") or "") == args.tag]
    return runs


_costs = {}


POOL_DATASETS = ("cub", "imagenet100")  # datasets whose simplified model uses the average-pooling head (NNS_AVG)


def reduction(r, split):
    pool = r["dataset"] in POOL_DATASETS
    key = (r["model"], r["dataset"], r["num_classes"], split, pool)
    if key not in _costs:
        _costs[key] = split_cost(r["model"], r["dataset"], split, r["num_classes"], r["config"]["model"]["batch_norm"], {"pool": pool})
    return _costs[key]


def write(df, tex, name, args, caption=""):
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, name + ".csv"), index=False)
    with open(os.path.join(args.out_dir, name + ".tex"), "w") as f:
        f.write(tex)
    print(f"\n== {name} ({caption})\n{df.to_string(index=False)}")


def split_table(args):
    runs = [r for r in trainer_runs(args) if r["mode"] == "full" and r.get("split_layer") is not None]
    rows, lines = [], []
    for (model, dataset), g in pd.DataFrame(runs).groupby(["model", "dataset"], sort=False):
        g = g.to_dict("records")
        red = [100 * reduction(r, r["split_layer"])["param_reduction"] for r in g]
        tr = [100 * r["split_epoch"] / r["epochs"] for r in g]
        rows.append({"model": model, "dataset": dataset, "runs": len(g),
                     "epoch": np.mean([r["split_epoch"] for r in g]), "epoch_std": np.std([r["split_epoch"] for r in g], ddof=1) if len(g) > 1 else 0,
                     "layer": np.mean([r["split_layer"] for r in g]), "layer_std": np.std([r["split_layer"] for r in g], ddof=1) if len(g) > 1 else 0,
                     "reduction": np.mean(red), "training": np.mean(tr), "splits": sorted(r["split_layer"] for r in g)})
        lines.append(f"    {NAMES.get(model, model)} & {NAMES.get(dataset, dataset)} & {pm([r['split_epoch'] for r in g])} & "
                     f"{pm([r['split_layer'] for r in g])} & {np.mean(red):.1f}\\% & {np.mean(tr):.1f}\\% \\\\")
    tex = ("\\begin{tabular}{ll r@{}l r@{}l c c}\n\\toprule\nModel & Dataset & \\multicolumn{2}{c}{Epoch} & \\multicolumn{2}{c}{Layer} & "
           "\\%Reduction & \\%Training \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    tex += "% %Reduction: mean over runs of the parameter reduction of each run (linear-probing head); use the same values in the appendix tables\n"
    write(pd.DataFrame(rows), tex, "split", args, "Table 1 / appendix %Reduction and %Training rows")


def accuracy_table(args):
    runs = [r for r in trainer_runs(args) if not r.get("stopped_at_split")]
    df = pd.DataFrame([{"model": r["model"], "dataset": dataset_key(r), "method": "full" if r["mode"] == "full" else f"nns_{r['head']}" + ("_online" if r["mode"] == "nns_online" else ""),
                        "seed": r["seed"], "test": 100 * r["test_accuracy"], "train": 100 * r.get("train_accuracy", np.nan),
                        "params": r["params"], "params_full": r["params_full"]} for r in runs])
    if df.empty:
        print("no runs"); return
    df["gap"] = df.train - df.test
    agg = df.groupby(["model", "dataset", "method"]).agg(runs=("seed", "count"), test=("test", "mean"), test_std=("test", "std"),
                                                          train=("train", "mean"), gap=("gap", "mean"), gap_std=("gap", "std"),
                                                          params=("params", "mean")).reset_index()
    lines = [f"    {NAMES.get(m, m)} & {NAMES.get(d, d)} & {meth.replace('_', ' ')} & {pm(g.test, digits=2)} & {pm(g.train, digits=2)} & {pm(g.gap, digits=2)} \\\\"
             for (m, d, meth), g in df.groupby(["model", "dataset", "method"])]
    tex = ("\\begin{tabular}{lll r@{}l r@{}l r@{}l}\n\\toprule\nModel & Dataset & Method & \\multicolumn{2}{c}{Test acc.} & "
           "\\multicolumn{2}{c}{Train acc.} & \\multicolumn{2}{c}{Train-test gap} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write(agg, tex, "accuracy", args, "test/train accuracy and generalization gap")


def te_table(args):
    te = load("results/te/**/*.json")
    if not te:
        print("no Tunnel Effect results in results/te"); return
    runs = trainer_runs(args)
    simplified = {}
    for r in runs:
        if r["mode"] == "nns_two_stage":
            simplified.setdefault((r["model"], r["dataset"], r["seed"], r["split_layer"]), []).append(100 * r["test_accuracy"])
    rows = []
    for t in te:
        key = (t["model"], t["dataset"], t["seed"])
        rows.append({"model": t["model"], "dataset": t["dataset"], "seed": t["seed"], "full": 100 * t["full_accuracy"],
                     "nns_split": t.get("nns_split"), "te_split": t["te_split"],
                     "nns_acc": np.mean(simplified.get(key + (t.get("nns_split"),), [np.nan])),
                     "te_acc": np.mean(simplified.get(key + (t["te_split"],), [np.nan])),
                     "nns_reduction": 100 * t.get("nns_param_reduction", np.nan), "te_reduction": 100 * t.get("te_param_reduction", np.nan)})
    df = pd.DataFrame(rows)
    lines = [f"    {NAMES.get(m, m)} & {NAMES.get(d, d)} & {pm(g.full, digits=2)} & {pm(g.nns_split)} & {pm(g.nns_acc, digits=2)} & "
             f"{pm(g.te_split)} & {pm(g.te_acc, digits=2)} \\\\" for (m, d), g in df.groupby(["model", "dataset"])]
    tex = ("\\begin{tabular}{ll r@{}l r@{}l r@{}l r@{}l r@{}l}\n\\toprule\n & & \\multicolumn{2}{c}{Full} & \\multicolumn{4}{c}{NNS} & "
           "\\multicolumn{4}{c}{Tunnel Effect (post hoc)} \\\\\nModel & Dataset & \\multicolumn{2}{c}{Acc.} & \\multicolumn{2}{c}{Split} & "
           "\\multicolumn{2}{c}{Acc.} & \\multicolumn{2}{c}{Split} & \\multicolumn{2}{c}{Acc.} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write(df.groupby(["model", "dataset"]).mean(numeric_only=True).reset_index(), tex, "te", args, "NNS vs Tunnel Effect split")


def cost_table(args):
    runs = trainer_runs(args)
    rows = []
    for r in runs:
        if r.get("stopped_at_split"):
            continue
        train_time = sum(r["time_epochs_full"]) + sum(r["time_epochs_simplified"])
        row = {"model": r["model"], "dataset": dataset_key(r), "mode": r["mode"], "metric_mode": r["metric_mode"], "seed": r["seed"],
               "time_total_min": r["time_total"] / 60, "time_train_min": train_time / 60,
               "time_epoch_full_s": np.mean(r["time_epochs_full"]) if r["time_epochs_full"] else np.nan,
               "time_epoch_simplified_s": np.mean(r["time_epochs_simplified"]) if r["time_epochs_simplified"] else np.nan,
               "metric_per_epoch_s": np.mean(r["time_metrics"]) if r["time_metrics"] else np.nan}
        if r["mode"] == "full" and r.get("split_layer") is not None:
            c = reduction(r, r["split_layer"])
            f = nns_training_flops(c["flops_full"], c["flops_split"], r["split_epoch"], r["epochs"], r["n_train"], r["n_tunnel"])
            row.update({"train_pflops_full": f["train_flops_full"] / 1e15, "train_pflops_nns": f["train_flops_nns"] / 1e15,
                        "train_flops_reduction": 100 * f["train_flops_reduction"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no runs"); return
    agg = df.drop(columns="seed").groupby(["model", "dataset", "mode", "metric_mode"]).mean(numeric_only=True).reset_index()
    for b in load("results/baselines/**/*.json"):
        agg = pd.concat([agg, pd.DataFrame([{"model": b["model"], "dataset": b["dataset"], "mode": b["method"],
                                             "time_total_min": b["time_total"] / 60,
                                             "time_train_min": b.get("time_train", np.nan) / 60 if b.get("time_train") else np.nan}])])
    write(agg, agg.to_latex(index=False, float_format="%.2f"), "cost", args,
          "wall-clock (full-model epochs exclude the metric time, reported separately per epoch); training FLOPs estimated as 3x forward")


def seeds_table(args):
    runs = [r for r in trainer_runs(args) if r.get("tunnel_seed") is not None and r.get("split_layer") is not None and r["mode"] != "nns_two_stage"]
    if not runs:
        print("no tunnel-seed runs"); return
    df = pd.DataFrame([{"model": r["model"], "dataset": r["dataset"], "tunnel_seed": r["tunnel_seed"], "seed": r["seed"],
                        "layer": r["split_layer"], "epoch": r["split_epoch"]} for r in runs])
    rows = []
    for (m, d), g in df.groupby(["model", "dataset"]):
        # average within-group variance: fixing the tunnel set (variance due to the optimization seed) and vice versa
        var_opt = g.groupby("tunnel_seed").layer.var(ddof=0).mean()
        var_tun = g.groupby("seed").layer.var(ddof=0).mean()
        var_opt_e = g.groupby("tunnel_seed").epoch.var(ddof=0).mean()
        var_tun_e = g.groupby("seed").epoch.var(ddof=0).mean()
        rows.append({"model": m, "dataset": d, "runs": len(g), "layer_mean": g.layer.mean(), "layer_std": g.layer.std(),
                     "layer_var_given_tunnel_seed": var_opt, "layer_var_given_opt_seed": var_tun,
                     "epoch_mean": g.epoch.mean(), "epoch_std": g.epoch.std(),
                     "epoch_var_given_tunnel_seed": var_opt_e, "epoch_var_given_opt_seed": var_tun_e})
        pivot = g.pivot_table(index="tunnel_seed", columns="seed", values="layer")
        print(f"\nsplit layer, {m} {d} (rows: tunnel seed, columns: optimization seed)\n{pivot}")
    out = pd.DataFrame(rows)
    write(out, out.to_latex(index=False, float_format="%.2f"), "seeds", args,
          "var_given_tunnel_seed: variance due to the optimization seed; var_given_opt_seed: variance due to the tunnel set")
    df.to_csv(os.path.join(args.out_dir, "seeds_runs.csv"), index=False)


def baselines_table(args):
    rows = []
    for b in load("results/baselines/**/*.json"):
        for s in b["steps"]:
            rows.append({"method": b["method"] + (f" (lam={b['lam']})" if "lam" in b else ""), "model": b["model"], "dataset": b["dataset"],
                         "seed": b["seed"], "blocks_removed": len(s["removed"]), "test": 100 * s["test_accuracy"],
                         "params": s["params"], "gflops": s["flops"] / 1e9})
    for a in load("results/autoencoder/**/*.json"):
        rows.append({"method": "autoencoder + linear", "model": a["model"], "dataset": a["dataset"], "seed": a["seed"],
                     "blocks_removed": np.nan, "split": a["split"], "test": 100 * a["test_accuracy"], "params": a["params"]})
    if not rows:
        print("no baseline results"); return
    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "dataset", "method", "blocks_removed"], dropna=False).agg(
        runs=("seed", "count"), test=("test", "mean"), test_std=("test", "std"), params=("params", "mean"), gflops=("gflops", "mean")).reset_index()
    write(agg, agg.to_latex(index=False, float_format="%.2f"), "baselines", args, "depth-reduction and autoencoder baselines")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tables", nargs="+", choices=["split", "accuracy", "te", "cost", "seeds", "baselines"])
    parser.add_argument("--tag", default=None, help="only runs with this settings.run_tag")
    parser.add_argument("--out_dir", default="tables_generated")
    parser.add_argument("--pool_datasets", nargs="*", default=["cub", "imagenet100"], help="datasets whose %%reduction uses the average-pooling head")
    args = parser.parse_args()
    global POOL_DATASETS
    POOL_DATASETS = tuple(args.pool_datasets)
    fns = {"split": split_table, "accuracy": accuracy_table, "te": te_table, "cost": cost_table, "seeds": seeds_table, "baselines": baselines_table}
    for t in args.tables:
        fns[t](args)


if __name__ == "__main__":
    main()
