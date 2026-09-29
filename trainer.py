import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import os
import json
import copy
import contextlib
from dataset import ImageDataset
import time
import torch.nn as nn
from model import *
from utils import *
from metrics import *
from nc.model_structure import make_optimiser, make_scheduler


class Trainer:
    def __init__(self,
                 dataset: ImageDataset,
                 logger,
                 model = None,
                 config = None,
                 checkpoint: bool = False,
                 resume_epoch : int = 0,
                 mixed: bool = False
                 ):

        os.environ["PYTORCH_CUDA_ALLOC_CONF"]= 'expandable_segments:True'

        if config is None or dataset is None or logger is None:
            raise ValueError("Dataset, logger, and config must be provided to initialize the Trainer.")

        if not isinstance(model, nn.Module):
            raise ValueError("Model must be an instance of torch.nn.Module.")

        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.model = model
        self.checkpoint = checkpoint
        self.resume_epoch = resume_epoch
        self.mixed = mixed

        self.tunnel_args = config["tunnel_args"]
        self.epochs = self.config["experiment"]["epochs"]
        self.config["optimizer"]["declarative_ETF"] = self.tunnel_args["declarative_ETF"]
        self.save_metrics = config["settings"].get("save_metrics", False)
        self.device = config["settings"]["device"]

        # options added for the TMLR experiments (defaults reproduce the original behavior)
        # metrics: 'all' computes every representation metric on the stored tunnel-set embeddings,
        #          'ifc' only computes the IFC in a single streaming pass (no embeddings are stored)
        self.metric_mode = self.config["experiment"].get("metrics", "all")
        # nns_online: simplify the model in the same run as soon as the split is detected
        self.online = self.config["experiment"].get("nns_online", False)
        # stop_at_split: stop training once the split is detected (split/epoch studies only)
        self.stop_at_split = self.config["experiment"].get("stop_at_split", False)
        self.amp = self.config["settings"].get("amp", False) and "cuda" in str(self.device)
        self.save_final = self.config["settings"].get("save_final", True)
        self.eval_train = self.config["settings"].get("eval_train", True)
        self.resume_from = self.config["settings"].get("resume_from", None)
        # checkpoints: the optimizer state is only needed to resume full training (the second stage of NNS
        # and the oracle runs only load the model and the scheduler); the initial model only for the oracle
        self.save_optimizer = self.config["settings"].get("save_optimizer", True)
        self.save_initial = self.config["settings"].get("save_initial", True)

        self.train_loader, self.valid_loader, self.tunnel_loader, self.test_loader = self.dataset.return_loaders(
            batch_size=self.config["dataloader"]["batch_size"],
            num_workers=self.config["dataloader"]["num_workers"],
            stratified_sampler=self.config["dataloader"]["stratified_batch"]
        )

        self.split_layer = self.tunnel_args["split_layer"]
        self.split_epoch = None

        self.model_path = self.logger.results_directory
        self.loss = CrossEntropyWithEntropyReg(alpha=0)

        self.representative_layers = set()
        self.classification_layers = set()
        self.tunnel_stopped = False
        self.patience = self.config["experiment"]["patience"]
        labeling = self.config["experiment"].get("labeling", {}) or {}
        self.detector = SplitDetector(patience=self.patience, **labeling)
        self.model.to(self.device)

        self.epoch_times = {"full": [], "simplified": []}
        self.metric_times = []


        if self.checkpoint:
            checkpoint_path = self.resume_from or self.model_path + str(self.resume_epoch) + '.pth'
            self.loaded_checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
            assert self.resume_epoch == self.loaded_checkpoint["epoch"], "resume epoch and last epoch mismatched"
            self.model.load_state_dict(self.loaded_checkpoint["model_state_dict"], strict=False)


    @property
    def counter(self):
        return self.detector.counter


    def autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()


    def sync_time(self):
        if "cuda" in str(self.device):
            torch.cuda.synchronize()
        return time.time()


    def set_optimizer(self) -> None:
        self.optimizer = make_optimiser(self.config["optimizer"], self.model, self.tunnel_args["enable_tunnel"])
        self.scheduler = make_scheduler(self.config["scheduler"], self.optimizer, self.train_loader)

        self.logger.log(f"Setting optimizer and scheduler: {self.config['optimizer']['name']}, {self.config['scheduler']['decay_type']}")

        if self.checkpoint:
            if not self.tunnel_args["enable_tunnel"]:
                self.scheduler = make_scheduler(self.config["scheduler"], self.optimizer, self.train_loader, last_epoch=self.resume_epoch)
                self.optimizer.load_state_dict(self.loaded_checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(self.loaded_checkpoint["scheduler_state_dict"])
            else:
                if self.mixed:
                    if self.config["optimizer"]["name"] == "SGD":
                        self.optimizer = make_optimiser({"declarative_ETF": self.tunnel_args["declarative_ETF"], "lr": 0.001, "name": "AdamW", "sep_decay": False, "weight_decay": 1e-5, "momentum": 0.0}, self.model, self.tunnel_args["enable_tunnel"])
                        self.scheduler = None
                    else:
                        self.optimizer = make_optimiser({"declarative_ETF": self.tunnel_args["declarative_ETF"], "lr": 0.1, "momentum": 0.9, "name": "SGD", "sep_decay": False, "weight_decay": 5e-4}, self.model, self.tunnel_args["enable_tunnel"])
                        self.scheduler = make_scheduler(self.config["scheduler"], self.optimizer, self.train_loader)
                else:
                    # self.scheduler = make_scheduler(self.config["scheduler"], self.optimizer, self.train_loader)
                    self.scheduler.load_state_dict(self.loaded_checkpoint["scheduler_state_dict"])
        elif self.save_initial:
            self.save_model(0)


    def streaming_ifc(self):
        """IFC of every layer on the tunnel set, computed in one pass without storing the embeddings."""
        ifc = StreamingIFC(self.dataset.num_classes, self.device)
        self.model.eval()
        self.model.extractor.feature_hook = ifc
        with torch.no_grad():
            for _, (X, y) in enumerate(self.tunnel_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                ifc.set_targets(y)
                self.model(X, y)
        self.model.extractor.feature_hook = None
        return ifc.compute()


    def calculate_epoch_metrics(self, epoch_metrics, epoch):

        t0 = self.sync_time()

        if self.metric_mode == "ifc":
            nc_metrics = self.streaming_ifc()
        else:
            self.eval(self.model)

            self.model.gather_embs() # gather embeddings for the tunnel

            if epoch%self.config["experiment"]["eval_every"] == 0:
                nc_metrics, ncs = calculate_all_nc_metrics(self.model, self.dataset.tunnel_targets, self.device, self.tunnel_args["enable_tunnel"], calculate_nc=True)
                for i in range(len(ncs)):
                    epoch_metrics[f"nc{i+1}"] = ncs[i]
            else:
                nc_metrics, ncs = calculate_all_nc_metrics(self.model, self.dataset.tunnel_targets, self.device, self.tunnel_args["enable_tunnel"], calculate_nc=False)

            self.model.reset_embs()  # reset embeddings for the next epoch

        metric_time = self.sync_time() - t0
        self.metric_times.append(metric_time)
        epoch_metrics["metric_time"] = round(metric_time, 3)

        ifc = {}
        for key, value in nc_metrics.items():
            for nc_metric, metric_value in value.items():
                epoch_metrics[f"{key}/{nc_metric}"] = metric_value
            ifc[int(key.split("_")[-1])] = float(value["proxy_nc"])

        fired = self.detector.update(ifc, epoch)
        if all(layer.ready and not layer.changed for layer in self.detector.layers.values()):
            print(f"No change in representative/classification layers - patience: {self.counter}")

        self.representative_layers = self.detector.extractors
        self.classification_layers = self.detector.contractors
        print(f"Representative layers: {self.representative_layers}")
        print(f"Classification layers: {self.classification_layers}")

        # label of every layer (1: contractor), used to plot the split decision
        epoch_metrics["split/labels"] = self.detector.labels()
        epoch_metrics["split/counter"] = self.counter
        epoch_metrics["split/candidate"] = self.detector.candidate
        epoch_metrics["split/config"] = {"patience": self.patience, **self.detector.label_args}

        #if not self.tunnel_args["enable_tunnel"]:
        #    abs_split_layer, soft_split_layer, curr_ranks = find_split_layer(nc_metrics)

        #    epoch_metrics["abs_split_layer"] = abs_split_layer
        #    epoch_metrics["soft_split_layer"] = soft_split_layer

        if self.save_metrics:
            save_metrics(self.logger.log_metrics, epoch_metrics, epoch)

        return epoch_metrics, fired


    def init_flatten_dim(self):
        #run the forward model for just one batch
        with torch.no_grad():
            self.model.eval()
            for _, (X, y) in enumerate(self.tunnel_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                self.model(X, y)
                break


    def simplify(self, epoch):
        """Online NNS: trim the model at the detected split and continue training with the same schedule."""
        self.logger.log(f"Simplifying the model at layer {self.split_layer} after epoch {epoch}")
        self.loss.alpha = self.tunnel_args["alpha"]
        self.model.activate_tunnel(self.split_layer, self.dataset.num_classes)
        self.model.to(self.device)

        # as when resuming from the split checkpoint: new optimizer state, same learning-rate schedule
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        self.optimizer = make_optimiser(self.config["optimizer"], self.model, True)
        if scheduler_state is not None:
            self.scheduler = make_scheduler(self.config["scheduler"], self.optimizer, self.train_loader)
            self.scheduler.load_state_dict(scheduler_state)
        for group, lr in zip(self.optimizer.param_groups, current_lrs):
            group["lr"] = lr


    def train(self):
        metric = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(self.device)
        start_time = self.sync_time()

        if self.tunnel_args["enable_tunnel"]:
            if self.tunnel_args["split_layer"] > 0:
                self.split_layer = self.tunnel_args["split_layer"]

            self.logger.log(f"Training in Tunnel mode - Split layer is {self.split_layer}")
            self.loss.alpha = self.tunnel_args["alpha"] # activate the regularization
            self.init_flatten_dim()
            self.model.activate_tunnel(self.split_layer, self.dataset.num_classes)
            self.model.to(self.device)

        self.set_optimizer()

        print("Starting training from epoch:", self.resume_epoch, " to ", self.epochs)

        epochs_run = 0
        for epoch in range(self.resume_epoch, self.epochs):

            self.model.train()
            epoch_metrics = {}
            avg_loss = 0
            t0 = self.sync_time()

            for batch, (X,y) in enumerate(self.train_loader):

                #print("Processing batch:", batch, " of ", len(self.train_loader), " in epoch:", epoch)

                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with self.autocast():
                    h, out = self.model(X, y)
                    loss_val = self.loss(out.float(), h, y)
                y_pred = F.softmax(out.float(), dim=1)

                accuracy = metric(y_pred, y)

                self.optimizer.zero_grad()
                avg_loss += loss_val.item()
                loss_val.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            t1 = self.sync_time()
            epochs_run += 1

            avg_loss = avg_loss/(batch+1)

            if torch.isnan(torch.tensor(avg_loss)):
                self.logger.log(f"Loss is NaN, for epoch: {epoch}, model: {self.config['model']['type']}, dataset: {self.config['dataset']['name']}. Stopping training.")
                raise ValueError("Loss is NaN, stopping training")

            accuracy = metric.compute().item()
            training_time = round(t1-t0, 3)
            self.epoch_times["simplified" if self.model.active_tunnel else "full"].append(t1 - t0)
            epoch_metrics["train_loss"] = avg_loss
            epoch_metrics["training_time"] = training_time
            epoch_metrics["train_accuracy"] = accuracy
            epoch_metrics["step"] = epoch

            fired = False
            if not self.model.active_tunnel:
                epoch_metrics, fired = self.calculate_epoch_metrics(epoch_metrics, epoch)

            self.logger(epoch_metrics)
            metric.reset()

            stop = False
            periodic = epoch%self.config["experiment"]["save_every"] == 0 and (epoch > 0 or self.save_initial)
            if (fired or periodic) and not self.model.active_tunnel:
                self.split_layer = self.detector.candidate
                self.logger({"split_layer": self.split_layer})
                if fired:
                    self.logger.log(f"Patience finished - Split layer detected: {self.split_layer} at epoch {epoch}")
                self.save_model(epoch)
                if fired and self.split_epoch is None:
                    self.split_epoch = epoch
                    if self.online:
                        self.simplify(epoch)
                    elif self.stop_at_split:
                        self.logger.log(f"Stopping at the split epoch {epoch}")
                        stop = True
            if stop:
                break

        self.finalize(start_time, epochs_run)


    def finalize(self, start_time, epochs_run):
        """Final evaluation, final checkpoint and a json file with the results of the run."""
        from flops import count_params, forward_flops, build_net, input_shape

        test_loss, test_accuracy = self.final_eval(self.model, self.test_loader)
        self.logger({"Test accuracy": round(test_accuracy, 4)})
        if self.tunnel_args["declarative_ETF"]:
            head = "dcl"
        elif self.tunnel_args["ETF_fc"]:
            head = "fix"
        else:
            head = "lp"
        results = {
            "name": self.logger.name,
            "model": self.config["model"]["type"],
            "dataset": self.config["dataset"]["name"],
            "num_classes": self.dataset.num_classes,
            "seed": self.config["settings"]["seed"],
            "tunnel_seed": self.config["dataset"].get("tunnel_seed"),
            "run_tag": self.config["settings"].get("run_tag"),
            "head": head if self.tunnel_args["enable_tunnel"] or self.online else None,
            "mode": "nns_two_stage" if self.tunnel_args["enable_tunnel"] else ("nns_online" if self.online else "full"),
            "labeling": {"patience": self.patience, **self.detector.label_args},
            "epochs": self.epochs,
            "epochs_run": epochs_run,
            "resume_epoch": self.resume_epoch if self.checkpoint else None,
            "stopped_at_split": self.stop_at_split and self.split_epoch is not None,
            # full runs: split at the first time the patience is reached (self.split_layer is also updated at every save_every)
            "split_layer": self.split_layer if (self.tunnel_args["enable_tunnel"] or self.online and self.split_epoch is not None) else self.detector.split_layer,
            "split_epoch": self.split_epoch if not self.tunnel_args["enable_tunnel"] else self.resume_epoch,
            "split_events": self.detector.events,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "n_train": len(self.dataset.train_set),
            "n_tunnel": len(self.dataset.tunnel_set),
            "time_total": self.sync_time() - start_time,
            "time_epochs_full": self.epoch_times["full"],
            "time_epochs_simplified": self.epoch_times["simplified"],
            "time_metrics": self.metric_times,
            "metric_mode": self.metric_mode,
            "amp": self.amp,
            "config": self.config,
        }
        if self.eval_train:
            train_loader = self.dataset.return_train_eval_loader(self.config["dataloader"]["batch_size"], self.config["dataloader"]["num_workers"])
            results["train_loss"], results["train_accuracy"] = self.final_eval(self.model, train_loader)
            self.logger({"Train accuracy (eval)": round(results["train_accuracy"], 4)})

        shape = input_shape(self.config["dataset"]["name"])
        full = build_net(self.config["model"]["type"], self.config["dataset"]["name"], self.dataset.num_classes, self.config["model"]["batch_norm"])
        results["params_full"] = count_params(full)
        results["flops_full"] = forward_flops(full, shape)
        results["params"] = count_params(self.model, trainable_only=True)
        results["flops"] = forward_flops(self.model, shape, self.device)

        if self.save_final and not self.mixed:
            torch.save({'epoch': self.epochs - 1,
                        'model_state_dict': self.model.state_dict(),
                        'split_layer': results["split_layer"] if self.model.active_tunnel else None},
                       self.model_path + ('/final_simplified.pth' if self.model.active_tunnel else '/final.pth'))

        os.makedirs(os.path.dirname(self.logger.results_json), exist_ok=True)
        with open(self.logger.results_json, "w") as f:
            json.dump(results, f, indent=1, default=str)
        self.logger.log(f"Results saved in {self.logger.results_json}")


    def eval(self, model):
        model.eval()
        model.tunnel.inference = True
        model.extractor.save = True
        model.tunnel.save = True
        with torch.no_grad():
            for _, (X, y) in enumerate(self.tunnel_loader):

                X = X.to(self.device)
                y = y.to(self.device)
                model(X, y)

        model.extractor.save = False
        model.tunnel.save = False
        model.tunnel.inference = False
        return


    def final_eval(self, model, loader):
        eval_metric = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(self.device)
        model.eval()
        model.tunnel.inference = True

        avg_loss = 0.0

        with torch.no_grad():
            for _, (X, y) in enumerate(loader):

                X = X.to(self.device)
                y = y.to(self.device)
                with self.autocast():
                    h, out = model(X, y)
                out = out.float()
                y_pred = F.softmax(out, dim=1)
                loss = self.loss(out, h, y)
                avg_loss += loss.item()
                accuracy = eval_metric(y_pred, y)

            avg_loss = avg_loss / len(loader)
            accuracy = eval_metric.compute().item()

        model.tunnel.inference = False

        eval_metric.reset()
        return avg_loss, accuracy


    def save_model(self, epoch):

        if not self.mixed:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict() if self.save_optimizer else None,
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }, self.model_path + '/' +  str(epoch) + '.pth')


    def to_string(self):
        return f"Trainer: {self.config.to_string()}"
