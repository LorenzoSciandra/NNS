import torch
import numpy as np
from log import initialize_logger_from_config
from trainer import *
from utils import *
import argparse
from dataset import *
from model import Extractor,Tunnel, Wrapped_NN, return_layer_list
from trainer import Trainer
import random
import yaml


def init_exp(parser):
    parser.add_argument('-exp', dest='exp', default='configs/experiments/exp.yaml', type = str, help='Experiment configuration file')
    parser.add_argument('-trainer', dest='trainer', default='configs/trainer/resnet10.yaml', type = str, help='Experiment configuration file')
    parser.add_argument('-tunnel', dest='tunnel', default=None, type = str, help='Model configuration file')
    parser.add_argument('-checkpoint', dest='checkpoint', action='store_true', help='put to restart from checkpoint')
    parser.add_argument('-resume', dest='resume_epoch', default=0, type = int, help='The epoch from which to restart for the Tunnel Training')
    parser.add_argument('-mixed', dest='mixed', action='store_true', help='put to use mixed precision training')
    parser.add_argument('-lth', dest='lth', action='store_true', help='put to use LTH training')
    args = parser.parse_args()
    
    with open(args.exp) as config_file:
        config = yaml.safe_load(config_file)
    with open(args.trainer) as config_file:
        trainer_config = yaml.safe_load(config_file)
    
    config = config|trainer_config
    
    if args.tunnel:
        with open(args.tunnel) as config_file:
            tunnel = yaml.safe_load(config_file)
        config = config|tunnel
    else:
        config["tunnel_args"] = {
                        "name": "training",
                        "enable_tunnel": False,
                        "tunnel_epochs" : 0,
                        "alpha": 0,
                        "split_layer": 0,
                        "ETF_fc": False,
                        "declarative_ETF": False,
                        "inference": False,
                        "temperature": 5,
                        "fixed_etf_layers": 0,
                        "normalize": False,
                        "bias": False}
    
    config['lth'] = args.lth

    torch.manual_seed(config["settings"]["seed"])
    torch.cuda.manual_seed(config["settings"]["seed"])
    torch.cuda.manual_seed_all(config["settings"]["seed"])
    np.random.seed(config["settings"]["seed"])
    random.seed(config["settings"]["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(config["settings"]["seed"])
    return config, args


def run(config, args):    
    logger = initialize_logger_from_config(config, args.checkpoint)
    logger.log(config, header="Configuration")

    dataset = ImageDataset(data_args=config["dataset"])
    
    num_classes = DATASET_CLASSES[config["dataset"]["name"]]
    layers = return_layer_list(model_type = config["model"]["type"], 
                               dataset = config["dataset"]["name"], 
                               batch_norm= config["model"]["batch_norm"])
                        
    extractor = Extractor(layers, num_classes)
    tunnel = Tunnel(active=False,
                    num_classes=num_classes,
                    device=config["settings"]["device"],
                    **config["tunnel_args"])
    net = Wrapped_NN(extractor, tunnel, enable_tunnel=config["tunnel_args"]["enable_tunnel"])
    if args.lth:
        logger.log("Using LTH training")
        from LTH_trainer import LTH_Trainer
        trainer = LTH_Trainer(dataset=dataset,
                        logger=logger, 
                        model = net,
                        config = config,
                        checkpoint = args.checkpoint,
                        resume_epoch = args.resume_epoch,
                        mixed = args.mixed)
    else:
        trainer = Trainer(dataset=dataset,
                        logger=logger, 
                        model = net,
                        config = config,
                        checkpoint = args.checkpoint,
                        resume_epoch = args.resume_epoch,
                        mixed = args.mixed)

    trainer.train()
    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="classification")
    config, args = init_exp(parser)
    print("Starting experiment from checkpoint:", args.checkpoint)
    run(config, args)
    