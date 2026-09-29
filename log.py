import logging
from typing import Dict
import wandb
import os
import json
import hashlib
from typing import Dict, Union


def generate_config_hash(config: dict) -> str:
    """Generates a unique hash for a given configuration dictionary."""
    config_str = json.dumps(config, sort_keys=True)  # Convert config to JSON string for hashing
    return hashlib.md5(config_str.encode()).hexdigest()  # Generate MD5 hash

class Logger:
    def __init__(self, name: str, logs_directory: str = './logs/', results_directory: 
                    str = './results/', log_metrics_directory: str = './log_metrics'):
        self.logger = logging.getLogger(name)
        self.run_path = None
        self.name = name
        log_file_path = os.path.join(logs_directory, f"{name}.log")

        if os.path.exists(log_file_path):
            print(f"Logger with configuration {self.name} already exists. Reusing it.")
        self.logs_directory = logs_directory
        self.results_directory = results_directory
        self.log_metrics = log_metrics_directory
        self._configure_local_logger(name)
        
    def _configure_local_logger(self, name: str):
        """Configures the local logger with console and file handlers."""
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # File handler
        file_handler = logging.FileHandler(f"{self.logs_directory}{name}.log", mode='w', encoding="utf-8")
        
        # Formatter
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M"
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log(self, message, header=None, color="white"):
        COLORS = {
            "white": "\033[97m",    # Info
            "green": "\033[92m",   # Success
            "yellow": "\033[93m",  # Warning
            "red": "\033[91m"      # Error
        }
        RESET = "\033[0m"
        color_code = COLORS.get(color, COLORS["white"])

        if header:
            line = "=" * (len(message) + 4) + header + "=" * (len(message) + 4)
            formatted_message = f"\n{line}\n  {message}\n{line}"
        else:
            formatted_message = message

        self.logger.info(f"{color_code}{formatted_message}{RESET}")
        

    def __call__(self, stats: Dict[str, float]):
        """Logs the statistics both to the console and file."""
        message = " | ".join(f"{key}: {round(value,4)}" for key, value in stats.items() if '/' not in key and value is not None)
        self.logger.info(message)
    
    def finish(self):
        self.logger.info('finish')


class WandbLogger(Logger):
    def __init__(self, name: str, logs_directory: str, results_directory: str, log_metrics_directory: str,  **kwargs):
        super().__init__(name, logs_directory, results_directory, log_metrics_directory)
        # Initialize the Wandb run
        wandb.init(name=name, **kwargs)
        
    def __call__(self, stats: Dict[str, float]):
        """Logs statistics to both the local logger and Wandb."""
        super().__call__(stats)  # Log to local logger
        if 'step' not in stats.keys():
            wandb.log(stats)
        else:    
            wandb.log(stats, step = stats['step'])         # Log to Wandb

    def finish(self):
        """Ends the Wandb run."""
        super().finish()
        wandb.finish()



def initialize_logger_from_config(config: dict, checkpoint: bool) -> Union[Logger, WandbLogger]:
    
    logger_config = config.get("logger", {})
    logger_type = logger_config.get("type", "local").lower()
    if checkpoint:
        if config['lth']:
            logs_directory = f"./LTH_{config['optimizer']['name']}_logs_checkpoint/"
        else:
            logs_directory = f"./{config['optimizer']['name']}_logs_checkpoint/"
    else:
        logs_directory = "./logs/"
    if config['lth']:
        results_directory = f"./LTH_{config['optimizer']['name']}_checkpoints/"
    else:
        results_directory = f"./{config['optimizer']['name']}_checkpoints/"

    # optional settings.run_tag keeps the outputs of different experiment families in separate folders
    dataset_dir = config["dataset"]["name"]
    if config["dataset"].get("num_classes") is not None:
        dataset_dir += f"_c{config['dataset']['num_classes']}"
    if config["dataset"].get("tunnel_seed") is not None:
        dataset_dir += f"_t{config['dataset']['tunnel_seed']}"
    run_tag = config["settings"].get("run_tag")
    run_path = (run_tag + "/" if run_tag else "") + dataset_dir + "/" + str(config["model"]["type"]) + "/" + str(config["settings"]["seed"]) + "/"
    results_directory = results_directory + run_path
    logs_directory = logs_directory + run_path
    log_metrics_directory = './log_metrics/' + run_path
    results_json_directory = './results/' + run_path
    

    os.makedirs(logs_directory, exist_ok=True)
    os.makedirs(log_metrics_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)

    config_hash = generate_config_hash(config)
    name = f"{config_hash}" + config["logger"]["wandb_args"]["name"]

    del config["logger"]["wandb_args"]["name"]

    if logger_type == "wandb":
        
        kwargs = {
            "config": config,
            **(logger_config.get("wandb_args", {}))
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        logger = WandbLogger(name=name, logs_directory=logs_directory, results_directory=results_directory, log_metrics_directory=log_metrics_directory, **kwargs)
    else:
        logger = Logger(name=name, logs_directory=logs_directory, results_directory=results_directory, log_metrics_directory=log_metrics_directory)
    logger.run_path = run_path
    logger.results_json = os.path.join(results_json_directory, f"{name}.json")
    return logger