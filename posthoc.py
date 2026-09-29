"""Helpers shared by the post-hoc experiment scripts (Tunnel Effect probes, baselines, numerical rank)."""
import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from dataset import ImageDataset
from main import load_config, set_seed, build_model
from models.resnet import has_identity_shortcut
from nc.model_structure import make_optimiser, make_scheduler


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-exp', default='configs/experiments/exp.yaml', type=str)
    parser.add_argument('-trainer', default='configs/trainer/resnet18.yaml', type=str)
    parser.add_argument('-tunnel', default=None, type=str)
    parser.add_argument('-set', dest='overrides', nargs='*', default=[], help='config overrides, e.g. dataset.name=cifar100 settings.seed=42')
    parser.add_argument('--out', default=None, type=str, help='output json file')
    return parser


def setup(args):
    """Config, dataset and (untrained) model for the given arguments."""
    config = load_config(args.exp, args.trainer, args.tunnel, args.overrides)
    set_seed(config["settings"]["seed"])
    dataset = ImageDataset(data_args=config["dataset"])
    net = build_model(config, dataset.num_classes).to(config["settings"]["device"])
    return config, dataset, net


def load_checkpoint(net, path, device):
    state = torch.load(path, weights_only=False, map_location=device)
    net.load_state_dict(state["model_state_dict"], strict=False)
    return net


def loaders(config, dataset, shuffle_train=True, eval_transform_train=True):
    """(train, tunnel, test) loaders. The training loader uses the test transform if eval_transform_train."""
    bs, nw = config["dataloader"]["batch_size"], config["dataloader"]["num_workers"]
    train_source = dataset.train_eval_set if eval_transform_train else dataset.train_set.dataset
    train = DataLoader(Subset(train_source, dataset.train_indices), batch_size=bs, num_workers=nw, shuffle=shuffle_train)
    tunnel = DataLoader(dataset.tunnel_set, batch_size=bs, num_workers=nw, shuffle=False)
    test = DataLoader(dataset.test_set, batch_size=bs, num_workers=nw, shuffle=False)
    return train, tunnel, test


class LayerOutputs:
    """Captures the (raw) output of every layer of the extractor during the forward pass."""
    def __init__(self, net, layers=None):
        self.outputs = {}
        modules = net.extractor.net
        layers = range(len(modules)) if layers is None else layers
        self.handles = [modules[i].register_forward_hook(self._hook(i)) for i in layers]

    def _hook(self, i):
        def hook(module, inp, out):
            self.outputs[i] = out
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def evaluate(net, loader, device, amp=False):
    net.eval()
    net.tunnel.inference = True
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            _, out = net(X, y)
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    net.tunnel.inference = False
    return correct / total


def removable_blocks(net):
    """Indices (in the extractor) of the residual blocks with an identity shortcut."""
    return [i for i, m in enumerate(net.extractor.net) if has_identity_shortcut(m)]


def remove_blocks(net, blocks):
    for i in blocks:
        net.extractor.net[i] = nn.Identity()
    return net


def train_epochs(net, loader, epochs, device, optimizer_args, scheduler_args=None, extra_loss=None, amp=False, log=print):
    """Plain supervised training loop (used for fine-tuning and for the LaCoOT baseline). Returns the training time."""
    optimizer = make_optimiser({"declarative_ETF": False, "sep_decay": False, **optimizer_args}, net, False)
    if scheduler_args is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    else:
        scheduler = make_scheduler(scheduler_args, optimizer, loader)
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        total, n = 0.0, 0
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                _, out = net(X, y)
            loss = F.cross_entropy(out.float(), y)
            if extra_loss is not None:
                loss = loss + extra_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        scheduler.step()
        log(f"epoch {epoch}: loss {total / max(1, n):.4f}")
    if "cuda" in str(device):
        torch.cuda.synchronize()
    return time.time() - t0


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"Saved {path}")
