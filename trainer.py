import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import os
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
        self.counter = 0
        
        self.tunnel_args = config["tunnel_args"]
        self.epochs = self.config["experiment"]["epochs"]
        self.config["optimizer"]["declarative_ETF"] = self.tunnel_args["declarative_ETF"]
        self.save_metrics = config["settings"].get("save_metrics", False)
        self.device = config["settings"]["device"]
    
        self.train_loader, self.valid_loader, self.tunnel_loader, self.test_loader = self.dataset.return_loaders(
            batch_size=self.config["dataloader"]["batch_size"],
            num_workers=self.config["dataloader"]["num_workers"],
            stratified_sampler=self.config["dataloader"]["stratified_batch"]
        )
        
        self.split_layer = self.tunnel_args["split_layer"]
        
        self.model_path = self.logger.results_directory
        self.loss = CrossEntropyWithEntropyReg(alpha=0)
        
        self.representative_layers = set()
        self.classification_layers = set()
        self.moving_layers = set()
        self.tunnel_stopped = False
        self.layers = {}
        self.patience = self.config["experiment"]["patience"]
        self.model.to(self.device)
        
        
        if self.checkpoint:
            self.loaded_checkpoint = torch.load(self.model_path + str(self.resume_epoch) + '.pth', weights_only=False, map_location=self.device)
            assert self.resume_epoch == self.loaded_checkpoint["epoch"], "resume epoch and last epoch mismatched"
            self.model.load_state_dict(self.loaded_checkpoint["model_state_dict"], strict=False)
        


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
        else:
            self.save_model(0)


    def calculate_epoch_metrics(self, epoch_metrics, epoch):
        
        self.eval(self.model)

        self.model.gather_embs() # gather embeddings for the tunnel
            
        if epoch%self.config["experiment"]["eval_every"] == 0:
            nc_metrics, ncs = calculate_all_nc_metrics(self.model, self.dataset.tunnel_targets, self.device, self.tunnel_args["enable_tunnel"], calculate_nc=True)
            for i in range(len(ncs)):
                epoch_metrics[f"nc{i+1}"] = ncs[i]
        else:
            nc_metrics, ncs = calculate_all_nc_metrics(self.model, self.dataset.tunnel_targets, self.device, self.tunnel_args["enable_tunnel"], calculate_nc=False)
            
        for key, value in nc_metrics.items():
            for nc_metric, metric_value in value.items():
                epoch_metrics[f"{key}/{nc_metric}"] = metric_value
                if int(key.split("_")[-1]) == 0 or int(key.split("_")[-1]) == len(nc_metrics) - 1:
                    continue  # skip input and output layers
                if nc_metric == "proxy_nc":
                    layer_idx=int(key.split("_")[-1])
                    if layer_idx not in self.layers:
                        self.layers[layer_idx] = Layer(layer_idx=layer_idx, patience=self.config["experiment"]["patience"])
                    self.layers[layer_idx].set_val(metric_value)

        if all([not layer.changed for layer in self.layers.values()]):
            self.counter += 1
            print(f"No change in representative/classification layers - patience: {self.counter}")
        else:
            self.counter = 0

        self.representative_layers = set() # need to reinitialize for consistency
        self.classification_layers = set()
        self.moving_layers = set()
        self.representative_layers.add(0)
        self.classification_layers.add(len(nc_metrics)-1)

        for layer_name, layer in self.layers.items():
            if layer.representative:
                self.representative_layers.add(layer_name)
            else:
                self.classification_layers.add(layer_name)
            if layer.moving:
                self.moving_layers.add(layer_name)
        print(f"Representative layers: {self.representative_layers}")
        print(f"Classification layers: {self.classification_layers}")
        print(f"Moving layers: {self.moving_layers}")
        
        #if not self.tunnel_args["enable_tunnel"]:
        #    abs_split_layer, soft_split_layer, curr_ranks = find_split_layer(nc_metrics)

        #    epoch_metrics["abs_split_layer"] = abs_split_layer
        #    epoch_metrics["soft_split_layer"] = soft_split_layer
        
        if self.save_metrics:
            save_metrics(self.logger.log_metrics, epoch_metrics, epoch)
        
        self.model.reset_embs()  # reset embeddings for the next epoch

        return epoch_metrics

    
    def init_flatten_dim(self):
        #run the forward model for just one batch
        with torch.no_grad():
            self.model.eval()
            for _, (X, y) in enumerate(self.tunnel_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                self.model(X, y)
                break
    

    def train(self):
        metric = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(self.device)
        
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
        
        for epoch in range(self.resume_epoch, self.epochs):
            
            self.model.train() 
            epoch_metrics = {}
            avg_loss = 0
            t0 = time.time()
            
            for batch, (X,y) in enumerate(self.train_loader):

                #print("Processing batch:", batch, " of ", len(self.train_loader), " in epoch:", epoch)

                X = X.to(self.device)
                y = y.to(self.device)
                h, out = self.model(X, y)
                y_pred = F.softmax(out, dim=1)

                accuracy = metric(y_pred, y)

                self.optimizer.zero_grad()
                loss_val = self.loss(out, h, y)
                avg_loss += loss_val.item()
                loss_val.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            t1 = time.time()
            
            avg_loss = avg_loss/(batch+1)
            
            if torch.isnan(torch.tensor(avg_loss)):
                self.logger.log(f"Loss is NaN, for epoch: {epoch}, model: {self.config['model']['type']}, dataset: {self.config['dataset']['name']}. Stopping training.")
                raise ValueError("Loss is NaN, stopping training")
            
            accuracy = metric.compute().item()
            training_time = round(t1-t0, 3)
            epoch_metrics["train_loss"] = avg_loss
            epoch_metrics["training_time"] = training_time
            epoch_metrics["train_accuracy"] = accuracy
            epoch_metrics["step"] = epoch

            if not self.model.active_tunnel:
                epoch_metrics = self.calculate_epoch_metrics(epoch_metrics, epoch)
            
            self.logger(epoch_metrics)
            metric.reset()
            
            if (self.counter == self.patience or epoch%self.config["experiment"]["save_every"] == 0) and not self.model.active_tunnel:
                extr_layers = [int(key) for key in list(self.classification_layers)]
                
                min_extr = min(extr_layers) if len(extr_layers) > 0 else -1
                self.split_layer = min_extr
                self.logger({"split_layer": self.split_layer})
                if self.counter == self.patience:
                    self.logger.log(f"Patience finished - Split layer detected: {self.split_layer} at epoch {epoch}")
                self.save_model(epoch)

        _, test_accuracy = self.final_eval(self.model, self.test_loader)
        self.logger({"Test accuracy": round(test_accuracy, 4)})


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
                h, out = model(X, y)
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
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }, self.model_path + '/' +  str(epoch) + '.pth')
    

    def to_string(self):
        return f"Trainer: {self.config.to_string()}"