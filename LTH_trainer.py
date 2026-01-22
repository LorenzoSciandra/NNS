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
import copy

import torch.nn.utils.prune as prune
from collections import OrderedDict # Useful for tracking layer names

def count_bn_channels(model):
    """Count total BatchNorm2d weight channels."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.numel()
    return total


def verify_earlybird_pruning(model, mask):
    """
    Verify pruning percentage after applying an EarlyBird mask.
    
    model: PyTorch model already masked (or not)
    mask: the EarlyBird global flat mask (0/1)
    """
    # Total number of BN channels expected
    total_bn = count_bn_channels(model)
    
    if mask.numel() != total_bn:
        raise ValueError(
            f"Mask length ({mask.numel()}) does not match total BN channels ({total_bn})."
        )
    
    # Number of pruned entries
    pruned = (mask == 0).sum().item()
    
    # Percentage
    percentage = pruned / total_bn
    
    print(f"Total BN channels      : {total_bn}")
    print(f"Pruned channels        : {pruned}")
    print(f"Pruning fraction       : {percentage:.4f} ({percentage*100:.2f}%)")
    
    return percentage

def apply_eb_mask(model, mask, device):
    """
    Apply a flattened EarlyBird mask to the BatchNorm2d weights of a model.

    model: pytorch model
    mask: 1D tensor containing 0/1 masking values
    """
    index = 0
    mask = mask.to(device)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.numel()

            # Slice part of the global mask
            m_mask = mask[index:index+size].view_as(m.weight.data)

            # Apply mask to BN gamma
            m.weight.data.mul_(m_mask)

            # (Optional) Apply to BN beta as well:
            # m.bias.data.mul_(m_mask)

            index += size

    print("Early-Bird mask applied.")


class EarlyBird():
    '''
    Taken from https://github.com/GATECH-EIC/Early-Bird-Tickets/blob/master/main.py
    '''
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * self.percent)
        thre = y[thre_index]

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                index += size

        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def early_bird_emerge(self, model):
        mask = self.pruning(model)
        self.put(mask)
        flag = self.cal_dist()
        if flag == True:
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False


class LTH_Trainer:
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
        
        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.model = model
        self.checkpoint = checkpoint
        self.resume_epoch = resume_epoch
        self.counter = 0

        self.tunnel_args = config["tunnel_args"]
        self.epochs = self.config["experiment"]["epochs"]
        self.config["optimizer"]["declarative_ETF"] = self.tunnel_args["declarative_ETF"]
        
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
            if self.tunnel_args['reinit']:
                self.logger.log("Reinitializing weights after loading checkpoint...")
                self.model.tunnel.init_weights()
                self.model.extractor.init_weights() 


    def set_optimizer(self) -> None:
        self.optimizer = make_optimiser(self.config["optimizer"], self.model, None)
        self.scheduler = make_scheduler(self.config["scheduler"], self.optimizer, self.train_loader)

        self.logger.log(f"Setting optimizer and scheduler: {self.config['optimizer']['name']}, {self.config['scheduler']['decay_type']}")
        
        if self.checkpoint:
            if not self.tunnel_args["enable_tunnel"]:
                self.scheduler = make_scheduler(self.config["scheduler"], self.optimizer, self.train_loader, last_epoch=self.resume_epoch)
                self.optimizer.load_state_dict(self.loaded_checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(self.loaded_checkpoint["scheduler_state_dict"])
            else:
                self.scheduler = make_scheduler(self.config["scheduler"], self.optimizer, self.train_loader)
                self.scheduler.load_state_dict(self.loaded_checkpoint["scheduler_state_dict"])


    def train(self):
        metric = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(self.device)

        if self.tunnel_args["enable_tunnel"]:
            mask = torch.load(self.model_path + str(self.resume_epoch) + "_eb_mask_" + str(self.tunnel_args["prune_ratio"]) + '.pth', weights_only=False, map_location=self.device)
            apply_eb_mask(self.model, mask, self.device)
            
            verify_earlybird_pruning(self.model, mask)
        
        self.set_optimizer()
        
        print("Starting training from epoch:", self.resume_epoch, " to ", self.epochs)

        prev_mask_30 = None
        prev_mask_50 = None
        prev_mask_70 = None

        early_bird_30 = EarlyBird(0.3)
        early_bird_50 = EarlyBird(0.5)
        early_bird_70 = EarlyBird(0.7)

        flag_30 = True
        flag_50 = True
        flag_70 = True

        for epoch in range(self.resume_epoch, self.epochs):
            
            self.model.train() 
            epoch_metrics = {}
            avg_loss = 0
            t0 = time.time()
            
            for batch, (X,y) in enumerate(self.train_loader):

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
            
            self.scheduler.step()

            avg_loss = avg_loss/(batch+1)
            
            if torch.isnan(torch.tensor(avg_loss)):
                raise ValueError("Loss is NaN, stopping training")
            
            t1 = time.time()
            training_time = round(t1-t0, 3)
            accuracy = metric.compute().item()

            epoch_metrics["train_loss"] = avg_loss
            epoch_metrics["training_time"] = training_time
            epoch_metrics["train_accuracy"] = accuracy
            epoch_metrics["step"] = epoch

            self.logger(epoch_metrics)
            metric.reset()

            if not self.checkpoint:

                # curr_mask_30 = early_bird_30.pruning(copy.deepcopy(self.model))
                # curr_mask_50 = early_bird_50.compute_pruning(copy.deepcopy(self.model))
                # curr_mask_70 = early_bird_70.compute_pruning(copy.deepcopy(self.model))

                if flag_30:

                    if early_bird_30.early_bird_emerge(copy.deepcopy(self.model)):
                        self.logger.log(f"✔ Early-Bird ticket for sparsity 30% found at epoch {epoch}")
                        curr_mask_30 = early_bird_30.pruning(copy.deepcopy(self.model))
                        torch.save(curr_mask_30, self.model_path + '/' +  str(epoch) + "_eb_mask_30.pth")
                        self.save_model(epoch)
                        flag_30 = False
                    else:
                        self.logger.log(f"⚠ No Early-Bird ticket found. distances are {early_bird_30.dists}")

                if flag_50:

                    if early_bird_50.early_bird_emerge(copy.deepcopy(self.model)):
                        self.logger.log(f"✔ Early-Bird ticket for sparsity 50% found at epoch {epoch}")
                        curr_mask_50 = early_bird_50.pruning(copy.deepcopy(self.model))
                        torch.save(curr_mask_50, self.model_path + '/' +  str(epoch) + "_eb_mask_50.pth")
                        self.save_model(epoch)
                        flag_50 = False
                    else:
                        self.logger.log(f"⚠ No Early-Bird ticket found. distances are {early_bird_50.dists}")

                if flag_70:

                    if early_bird_70.early_bird_emerge(copy.deepcopy(self.model)):
                        self.logger.log(f"✔ Early-Bird ticket for sparsity 70% found at epoch {epoch}")
                        curr_mask_70 = early_bird_70.pruning(copy.deepcopy(self.model))
                        torch.save(curr_mask_70, self.model_path + '/' +  str(epoch) + "_eb_mask_70.pth")
                        self.save_model(epoch)
                        flag_70 = False
                    else:
                        self.logger.log(f"⚠ No Early-Bird ticket found. distances are {early_bird_70.dists}")

                if flag_30 == False and flag_50 == False and flag_70 == False:
                    self.logger.log("Early-Bird tickets for all sparsity levels have been found. Ending training.")
                    break

                
                # prev_mask_30 = curr_mask_30
                # prev_mask_50 = curr_mask_50
                # prev_mask_70 = curr_mask_70

        _, test_accuracy = self.final_eval(self.model, self.test_loader)
        self.logger({"Test accuracy": round(test_accuracy, 4)})
        
        if self.tunnel_args["enable_tunnel"]:    
            verify_earlybird_pruning(self.model, mask)


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
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }, self.model_path + '/' +  str(epoch) + '.pth')
    

    def to_string(self):
        return f"Trainer: {self.config.to_string()}"