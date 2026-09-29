import torch
import numpy as np
from log import initialize_logger_from_config
from trainer import *
from utils import *
import argparse
from dataset import *
from model import Extractor,Tunnel, Wrapped_NN, return_layer_list, DEFAULT_TUNNEL_ARGS
from trainer import Trainer
import random
import yaml


def apply_overrides(config, overrides):
    """Apply overrides of the form 'a.b.c=value' (value parsed as yaml, e.g. 42, 0.1, true, null, [1, 2])."""
    for override in overrides or []:
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")
        current = config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = yaml.safe_load(value)
    return config


def load_config(exp, trainer, tunnel=None, overrides=None):
    with open(exp) as config_file:
        config = yaml.safe_load(config_file)
    with open(trainer) as config_file:
        trainer_config = yaml.safe_load(config_file)

    config = config|trainer_config

    if tunnel:
        with open(tunnel) as config_file:
            tunnel = yaml.safe_load(config_file)
        config = config|tunnel
        config["tunnel_args"] = {**DEFAULT_TUNNEL_ARGS, **config["tunnel_args"]}
    else:
        config["tunnel_args"] = dict(DEFAULT_TUNNEL_ARGS)

    return apply_overrides(config, overrides)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_exp(parser):
    parser.add_argument('-exp', dest='exp', default='configs/experiments/exp.yaml', type = str, help='Experiment configuration file')
    parser.add_argument('-trainer', dest='trainer', default='configs/trainer/resnet10.yaml', type = str, help='Experiment configuration file')
    parser.add_argument('-tunnel', dest='tunnel', default=None, type = str, help='Model configuration file')
    parser.add_argument('-checkpoint', dest='checkpoint', action='store_true', help='put to restart from checkpoint')
    parser.add_argument('-resume', dest='resume_epoch', default=0, type = int, help='The epoch from which to restart for the Tunnel Training')
    parser.add_argument('-resume_from', dest='resume_from', default=None, type = str, help='Checkpoint file to restart from (default: <checkpoint dir>/<resume>.pth)')
    parser.add_argument('-mixed', dest='mixed', action='store_true', help='put to use mixed precision training')
    parser.add_argument('-lth', dest='lth', action='store_true', help='put to use LTH training')
    parser.add_argument('-nns', dest='nns', action='store_true', help='online NNS: detect the split and simplify the model in the same run (requires -tunnel for the head)')
    parser.add_argument('-stop_at_split', dest='stop_at_split', action='store_true', help='stop training when the split is detected')
    parser.add_argument('-set', dest='overrides', nargs='*', default=[], help='config overrides, e.g. -set dataset.name=cifar100 settings.seed=42')
    args = parser.parse_args()

    config = load_config(args.exp, args.trainer, args.tunnel, args.overrides)

    if args.nns:
        assert args.tunnel, "-nns requires a -tunnel configuration for the new classification head"
        config["tunnel_args"]["enable_tunnel"] = False
        config["experiment"]["nns_online"] = True
    if args.stop_at_split:
        config["experiment"]["stop_at_split"] = True
    if args.resume_from:
        config["settings"]["resume_from"] = args.resume_from

    config['lth'] = args.lth

    set_seed(config["settings"]["seed"])
    return config, args


def build_model(config, num_classes):
    layers = return_layer_list(model_type = config["model"]["type"],
                               dataset = config["dataset"]["name"],
                               batch_norm= config["model"]["batch_norm"],
                               num_classes = num_classes)

    extractor = Extractor(layers, num_classes)
    tunnel = Tunnel(active=False,
                    num_classes=num_classes,
                    device=config["settings"]["device"],
                    **config["tunnel_args"])
    return Wrapped_NN(extractor, tunnel, enable_tunnel=config["tunnel_args"]["enable_tunnel"])


def run(config, args):
    logger = initialize_logger_from_config(config, args.checkpoint)
    logger.log(config, header="Configuration")

    dataset = ImageDataset(data_args=config["dataset"])

    num_classes = dataset.num_classes
    net = build_model(config, num_classes)
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
