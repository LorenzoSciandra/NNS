"""
Driver for the experiments requested by the TMLR revision (see TODO.md). Every family prints the commands
(default) or runs them (--run, optionally several at a time with --jobs). Finished commands are marked in
runs_logs/<family>/ and skipped when re-running, so a family can be resumed after an interruption.

Families (in dependency order):
  grid           full training of the 21 model-dataset configurations x 5 seeds: IFC trajectories, NNS split,
                 split checkpoint, final checkpoint and full-training accuracy                (tag: main)
  nns            NNS second stage (simplified model resumed from the split checkpoint)       (needs grid)
  te             post-hoc Tunnel Effect split with linear probes on the final checkpoints    (needs grid)
  te-truncated   simplified model at the TE split, resumed from the NNS split checkpoint     (needs te)
  oracle         simplified models for several split layers and split epochs (Figs. 6-7)    (needs grid --save_every)
  seeds          tunnel-set seed x optimization seed (5 x 5), stopping at the split
  labeling       online NNS with alternative labeling rules / references / patience values
                 (the split/epoch for every variant and all runs: python split_ablation.py)
  num-classes    CIFAR-100 subsets with 2, 10, 25, 50, 100 classes
  cost           full training and online NNS on the same machine (wall-clock, IFC overhead)
  imagenet       ResNet18 / ResNet50 on ImageNet-100, full training + NNS
  baselines      CKA layer pruning (needs grid) and LaCoOT on ResNets
  autoencoder    autoencoder + linear classifier at the NNS split                             (needs grid)
  nr             numerical rank check on MLP12 / CIFAR-10                                    (needs grid)

Examples:
    python experiments.py grid --models resnet18 --datasets cifar10            # print the commands
    python experiments.py grid --run --jobs 2                                  # run them, two at a time
    python experiments.py nns --run
    python experiments.py imagenet --run --seeds 2025 123 42
"""
import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import yaml

PY = sys.executable
SEEDS = [2025, 123, 42, 777, 1024]
TUNNEL_SEEDS = [2025, 123, 42, 777, 1024]
MODELS = ["mlp_10", "mlp_12", "resnet10", "resnet18", "vgg11", "vgg16"]
DATASETS = ["fashion_mnist", "cifar10", "cifar100"]
GRID = [(m, d) for m in MODELS for d in DATASETS] + [(m, "cub") for m in ["resnet10", "resnet18", "resnet34"]]
GUIDING = [("mlp_10", "fashion_mnist"), ("resnet18", "cifar10"), ("vgg11", "cifar100")]
SEED_STUDY = [("mlp_10", "fashion_mnist"), ("vgg16", "cifar10"), ("resnet18", "cifar10"), ("vgg11", "cifar100")]
HEADS = {"lp": "configs/tunnel/linear_probing.yaml", "fix": "configs/tunnel/fixed_etf.yaml",
         "dcl": "configs/tunnel/declarative_etf.yaml", "avg": "configs/tunnel/avg_pooling.yaml"}
POOL_DATASETS = ("cub", "imagenet100")
EXTRA = []  # additional -set overrides for every main.py command (--set)
LABELING = {
    "default": {},
    "patience_5": {"experiment.patience": 5}, "patience_10": {"experiment.patience": 10},
    "patience_20": {"experiment.patience": 20}, "patience_30": {"experiment.patience": 30},
    "ref_mean_3": {"experiment.labeling": {"reference": "mean", "ref_epochs": 3}},
    "ref_warmup_5": {"experiment.labeling": {"reference": "warmup", "ref_epochs": 5}},
    "rel_delta_0.05": {"experiment.labeling": {"rule": "rel_delta", "threshold": 0.05}},
}


def head_for(dataset, head=None):
    return head or ("avg" if dataset in POOL_DATASETS else "lp")


def sets(**kv):
    """-set arguments from a dict (dotted keys)."""
    return [f"{k}={json.dumps(v) if isinstance(v, (dict, list, bool)) or v is None else v}" for k, v in kv.items()]


def main_cmd(model, dataset, seed, tag, name, exp="configs/experiments/exp.yaml", trainer=None, tunnel=None, flags=(), overrides=()):
    cmd = [PY, "main.py", "-exp", exp, "-trainer", trainer or f"configs/trainer/{model}.yaml"]
    if tunnel:
        cmd += ["-tunnel", tunnel]
    cmd += list(flags)
    cmd += ["-set", f"dataset.name={dataset}", f"settings.seed={seed}", f"settings.run_tag={tag}", f"logger.wandb_args.name=_{name}", *overrides, *EXTRA]
    return cmd


def results(tag, mode=None):
    out = []
    for path in sorted(glob.glob(f"results/{tag}/**/*.json", recursive=True)):
        with open(path) as f:
            r = json.load(f)
        if "mode" not in r or r.get("stopped_at_split", False):
            continue
        if mode is None or r["mode"] == mode:
            out.append(r)
    return out


def checkpoint_dir(r):
    c = r["config"]
    dataset_dir = c["dataset"]["name"] + (f"_c{c['dataset']['num_classes']}" if c["dataset"].get("num_classes") else "")
    dataset_dir += f"_t{c['dataset']['tunnel_seed']}" if c["dataset"].get("tunnel_seed") is not None else ""
    return f"{c['optimizer']['name']}_checkpoints/{r['run_tag']}/{dataset_dir}/{r['model']}/{r['seed']}/"


def config_file(r):
    """The configuration of a finished run, written to a yaml file usable as -exp and -trainer."""
    c = json.loads(json.dumps(r["config"]))
    for key in ("tunnel_args", "lth"):
        c.pop(key, None)
    c["settings"].pop("resume_from", None)
    c["experiment"].pop("nns_online", None)
    c["experiment"].pop("stop_at_split", None)
    c["optimizer"].pop("declarative_ETF", None)
    text = yaml.safe_dump(c, sort_keys=True)
    path = f"runs_cfg/{hashlib.md5(text.encode()).hexdigest()}.yaml"
    os.makedirs("runs_cfg", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def from_split(r, tag, name, split_layer, head, resume_epoch=None, extra=()):
    """Second stage of NNS: simplified model at split_layer, resumed from the checkpoint of the given run."""
    resume_epoch = r["split_epoch"] if resume_epoch is None else resume_epoch
    cfg = config_file(r)
    ckpt = checkpoint_dir(r) + f"{resume_epoch}.pth"
    return main_cmd(r["model"], r["dataset"], r["seed"], tag, name, exp=cfg, trainer=cfg, tunnel=HEADS[head],
                    flags=["-checkpoint", "-resume", str(resume_epoch), "-resume_from", ckpt],
                    overrides=[f"tunnel_args.split_layer={split_layer}", *extra])


def selected(pairs, args):
    return [(m, d) for m, d in pairs if (not args.models or m in args.models) and (not args.datasets or d in args.datasets)]


def wanted(r, args):
    return (not args.models or r["model"] in args.models) and (not args.datasets or r["dataset"] in args.datasets) and r["seed"] in args.seeds


# ---------------------------------------------------------------------------------------------- families

def fam_grid(args):
    extra = sets(**{"experiment.metrics": args.metrics, "settings.save_optimizer": False})
    if args.save_every:
        extra += [f"experiment.save_every={args.save_every}"]
    else:
        extra += ["settings.save_initial=false"]
    return [(f"{d}_{m}_{s}", main_cmd(m, d, s, args.tag or "main", f"full_{m}_{d}_{s}", overrides=extra))
            for m, d in selected(GRID, args) for s in args.seeds]


def fam_nns(args):
    source = args.source_tag or "main"
    cmds = []
    for r in results(source, "full"):
        if not wanted(r, args) or r.get("split_layer") is None:
            continue
        head = head_for(r["dataset"], args.head)
        cmds.append((f"{r['dataset']}_{r['model']}_{r['seed']}_{head}",
                     from_split(r, args.tag or f"{source}_nns_{head}", f"nns_{head}_{r['model']}_{r['dataset']}_{r['seed']}", r["split_layer"], head)))
    return cmds


def fam_te(args):
    source = args.source_tag or "main"
    cmds = []
    for r in results(source, "full"):
        if not wanted(r, args):
            continue
        cfg = config_file(r)
        cmd = [PY, "tunnel_effect.py", "-exp", cfg, "-trainer", cfg, "--checkpoint", checkpoint_dir(r) + "final.pth",
               "--out", f"results/te/{r['dataset']}_{r['model']}_{r['seed']}.json"]
        if r.get("split_layer") is not None:
            cmd += ["--nns_split", str(r["split_layer"])]
        cmds.append((f"{r['dataset']}_{r['model']}_{r['seed']}", cmd))
    return cmds


def fam_te_truncated(args):
    source = args.source_tag or "main"
    full = {(r["model"], r["dataset"], r["seed"]): r for r in results(source, "full")}
    cmds = []
    for path in sorted(glob.glob("results/te/*.json")):
        t = json.load(open(path))
        r = full.get((t["model"], t["dataset"], t["seed"]))
        if r is None or not wanted(r, args) or r.get("split_layer") is None:
            continue
        if t["te_split"] == r["split_layer"] and not args.all_splits:
            continue  # same model as the NNS second stage
        head = head_for(r["dataset"], args.head)
        cmds.append((f"{r['dataset']}_{r['model']}_{r['seed']}_te{t['te_split']}",
                     from_split(r, args.tag or f"{source}_te_{head}", f"te_{head}_{r['model']}_{r['dataset']}_{r['seed']}", t["te_split"], head)))
    return cmds


def fam_oracle(args):
    """Simplified models at several layers and split epochs, resumed from the periodic checkpoints of the grid runs."""
    source = args.source_tag or "main"
    runs = [r for r in results(source, "full") if r.get("split_layer") is not None]
    # the same layers for all the seeds of a configuration (most common split +- 1), so that seeds can be averaged
    splits = {}
    for r in runs:
        splits.setdefault((r["model"], r["dataset"]), []).append(r["split_layer"])
    cmds = []
    for r in runs:
        if not wanted(r, args):
            continue
        if (r["model"], r["dataset"]) not in selected(GUIDING, args) and not args.models:
            continue
        epochs = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(checkpoint_dir(r) + "[0-9]*.pth"))
        epochs = [e for e in epochs if args.oracle_epochs is None or e in args.oracle_epochs]
        config_splits = splits[(r["model"], r["dataset"])]
        mode = max(set(config_splits), key=config_splits.count)
        layers = args.oracle_layers or sorted({max(1, mode + k) for k in (-1, 0, 1)})
        head = head_for(r["dataset"], args.head)
        for e in epochs:
            for l in layers:
                cmds.append((f"{r['dataset']}_{r['model']}_{r['seed']}_l{l}_e{e}",
                             from_split(r, args.tag or "oracle", f"oracle_{r['model']}_{r['dataset']}_{r['seed']}_l{l}_e{e}", l, head, resume_epoch=e,
                                        extra=["settings.save_final=false"])))
    return cmds


def fam_seeds(args):
    cmds = []
    for m, d in selected(SEED_STUDY, args):
        for ts in args.tunnel_seeds:
            for s in args.seeds:
                cmds.append((f"{d}_{m}_t{ts}_s{s}", main_cmd(m, d, s, args.tag or "seeds", f"seeds_{m}_{d}_t{ts}_s{s}",
                                                            flags=["-stop_at_split"],
                                                            overrides=[f"dataset.tunnel_seed={ts}", f"experiment.metrics={args.metrics_light}",
                                                                       "settings.save_final=false", "settings.eval_train=false"])))
    return cmds


def fam_labeling(args):
    cmds = []
    variants = args.variants or list(LABELING)
    for variant in variants:
        for m, d in selected(GUIDING, args):
            for s in args.seeds:
                cmds.append((f"{variant}_{d}_{m}_{s}", main_cmd(m, d, s, args.tag or f"labeling_{variant}", f"labeling_{variant}_{m}_{d}_{s}",
                                                               tunnel=HEADS[head_for(d, args.head)], flags=["-nns"],
                                                               overrides=[f"experiment.metrics={args.metrics_light}", *sets(**LABELING[variant])])))
    return cmds


def fam_num_classes(args):
    models = args.models or ["vgg11", "resnet18"]
    cmds = []
    for m in models:
        for k in args.classes:
            for s in args.seeds:
                overrides = [f"dataset.num_classes={k}", f"experiment.metrics={args.metrics_light}"]
                cmds.append((f"c{k}_{m}_{s}_full", main_cmd(m, "cifar100", s, args.tag or "num_classes", f"classes{k}_{m}_{s}", overrides=overrides)))
                if args.online:
                    cmds.append((f"c{k}_{m}_{s}_nns", main_cmd(m, "cifar100", s, (args.tag or "num_classes") + "_nns", f"classes{k}_nns_{m}_{s}",
                                                              tunnel=HEADS["lp"], flags=["-nns"], overrides=overrides)))
    return cmds


def fam_cost(args):
    cmds = []
    for m, d in selected(GUIDING + [("resnet18", "cifar100"), ("vgg16", "cifar10")], args):
        for s in args.seeds:
            common = [f"experiment.metrics={args.metrics_light}", "settings.save_final=false"]
            cmds.append((f"{d}_{m}_{s}_full", main_cmd(m, d, s, "cost_full", f"cost_full_{m}_{d}_{s}", overrides=common)))
            cmds.append((f"{d}_{m}_{s}_nns", main_cmd(m, d, s, "cost_online", f"cost_nns_{m}_{d}_{s}", tunnel=HEADS[head_for(d, args.head)],
                                                     flags=["-nns"], overrides=common)))
    return cmds


def fam_imagenet(args):
    models = args.models or ["resnet18", "resnet50"]
    cmds = []
    for m in models:
        for s in args.seeds:
            cmds.append((f"{m}_{s}_full", main_cmd(m, "imagenet100", s, args.tag or "imagenet", f"full_{m}_imagenet100_{s}",
                                                   exp="configs/experiments/exp_imagenet100.yaml", trainer=f"configs/trainer/{m}_imagenet.yaml")))
            if args.online:
                cmds.append((f"{m}_{s}_nns", main_cmd(m, "imagenet100", s, (args.tag or "imagenet") + "_online", f"nns_{m}_imagenet100_{s}",
                                                      exp="configs/experiments/exp_imagenet100.yaml", trainer=f"configs/trainer/{m}_imagenet.yaml",
                                                      tunnel=HEADS["avg"], flags=["-nns"])))
    return cmds


def fam_baselines(args):
    source = args.source_tag or "main"
    pairs = selected([("resnet18", "cifar10"), ("resnet18", "cifar100")], args)
    cmds = []
    for r in results(source, "full"):
        if (r["model"], r["dataset"]) in pairs and wanted(r, args):
            cfg = config_file(r)
            cmds.append((f"cka_{r['dataset']}_{r['model']}_{r['seed']}",
                         [PY, "depth_baselines.py", "cka", "-exp", cfg, "-trainer", cfg, "--checkpoint", checkpoint_dir(r) + "final.pth"]))
    for m, d in pairs:
        for s in args.seeds:
            for lam in args.lams:
                cmds.append((f"lacoot_{d}_{m}_{s}_lam{lam}", [PY, "depth_baselines.py", "lacoot", "-trainer", f"configs/trainer/{m}.yaml",
                                                             "--lam", str(lam), "-set", f"dataset.name={d}", f"settings.seed={s}"]))
    return cmds


def fam_autoencoder(args):
    source = args.source_tag or "main"
    pairs = selected([("resnet18", "cifar10"), ("vgg11", "cifar10"), ("mlp_10", "fashion_mnist")], args)
    cmds = []
    for r in results(source, "full"):
        if (r["model"], r["dataset"]) in pairs and wanted(r, args) and r.get("split_layer") is not None:
            cfg = config_file(r)
            cmds.append((f"{r['dataset']}_{r['model']}_{r['seed']}", [PY, "autoencoder_baseline.py", "-exp", cfg, "-trainer", cfg,
                                                                     "-tunnel", HEADS["lp"], "--split", str(r["split_layer"])]))
    return cmds


def fam_nr(args):
    source = args.source_tag or "main"
    cmds = []
    for r in results(source, "full"):
        if r["model"] == "mlp_12" and r["dataset"] == "cifar10" and r["seed"] in args.seeds:
            cfg = config_file(r)
            cmds.append((f"{r['seed']}", [PY, "nr_check.py", "-exp", cfg, "-trainer", cfg, "--checkpoint", checkpoint_dir(r) + "final.pth"]))
    return cmds


FAMILIES = {"grid": fam_grid, "nns": fam_nns, "te": fam_te, "te-truncated": fam_te_truncated, "oracle": fam_oracle, "seeds": fam_seeds,
            "labeling": fam_labeling, "num-classes": fam_num_classes, "cost": fam_cost, "imagenet": fam_imagenet,
            "baselines": fam_baselines, "autoencoder": fam_autoencoder, "nr": fam_nr}


def run_all(family, cmds, jobs):
    log_dir = os.path.join("runs_logs", family)
    os.makedirs(log_dir, exist_ok=True)

    def run(item):
        name, cmd = item
        done = os.path.join(log_dir, name + ".done")
        if os.path.exists(done):
            return name, "skipped (done)"
        with open(os.path.join(log_dir, name + ".log"), "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if code == 0:
            open(done, "w").close()
        return name, "ok" if code == 0 else f"FAILED (exit {code}, see {log_dir}/{name}.log)"

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for name, status in pool.map(run, cmds):
            print(f"[{family}] {name}: {status}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=list(FAMILIES))
    parser.add_argument("--run", action="store_true", help="run the commands (default: print them)")
    parser.add_argument("--jobs", type=int, default=1, help="commands run in parallel")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--tunnel_seeds", type=int, nargs="*", default=TUNNEL_SEEDS)
    parser.add_argument("--tag", default=None, help="settings.run_tag of the new runs (default depends on the family)")
    parser.add_argument("--source_tag", default=None, help="run_tag of the full-training runs to start from (default: main)")
    parser.add_argument("--head", choices=list(HEADS), default=None, help="head of the simplified model (default: avg on CUB/ImageNet, lp otherwise)")
    parser.add_argument("--metrics", default="all", help="(grid) 'all' also records CKA, NR, ID, NCC for the appendix figures; 'ifc' is much cheaper")
    parser.add_argument("--metrics_light", default="ifc", help="metric mode of the other families")
    parser.add_argument("--save_every", type=int, default=None, help="(grid) periodic checkpoints, needed by the oracle family")
    parser.add_argument("--oracle_layers", type=int, nargs="*", default=None)
    parser.add_argument("--oracle_epochs", type=int, nargs="*", default=None)
    parser.add_argument("--all_splits", action="store_true", help="(te-truncated) also run when the TE split equals the NNS split")
    parser.add_argument("--variants", nargs="*", default=None, choices=list(LABELING))
    parser.add_argument("--classes", type=int, nargs="*", default=[2, 10, 25, 50, 100])
    parser.add_argument("--online", action="store_true", help="(num-classes, imagenet) also run online NNS")
    parser.add_argument("--lams", type=float, nargs="*", default=[0.1, 1.0, 5.0], help="(baselines) LaCoOT regularization weights")
    parser.add_argument("--args", dest="script_args", default="", help="extra arguments for the post-hoc scripts, e.g. '--epochs 10'")
    parser.add_argument("--set", dest="extra", nargs="*", default=[], help="extra config overrides for every main.py command, e.g. experiment.epochs=10")
    args = parser.parse_args()
    EXTRA.extend(args.extra)

    cmds = FAMILIES[args.family](args)
    if args.script_args:
        cmds = [(name, cmd + shlex.split(args.script_args) if cmd[1] != "main.py" else cmd) for name, cmd in cmds]
    if not cmds:
        print(f"no commands for {args.family}: check that the runs it depends on exist (results/<tag>/...)")
        return
    if args.run:
        run_all(args.family, cmds, args.jobs)
    else:
        for _, cmd in cmds:
            print(shlex.join(cmd))
        print(f"# {len(cmds)} commands; add --run to execute them", file=sys.stderr)


if __name__ == "__main__":
    main()
