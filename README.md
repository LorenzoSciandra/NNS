# Neural Network Simplification
This is the code for the paper "Simplifying Neural Networks During Training".

## Requirements

Install the required packages using pip:

``` pip install -r requirements.txt ```

## Usage


To run a model on a dataset, use the following command:

``` python main.py  -exp configs/experiments/exp.yaml -trainer configs/trainer/MODEL_NAME.yaml ```

This creates the following folders:
- `data/` : for storing datasets, with subfolders for each dataset.
- `logs_metrics/`: for storing the metrics logged during training.
- `logs/` : for storing the training logs.
- `OPT_checkpoints/` : for storing the model checkpoints at every #save_every epoch in the config file. OPT stands for the optimization method used (SGD, Adam, etc)

To activate the tunnel:

``` python main.py -exp configs/experiments/exp.yaml -trainer configs/trainer/MODEL_NAME.yaml -tunnel configs/tunnel/linear_probing.yaml -checkpoint -resume $last_epoch```
