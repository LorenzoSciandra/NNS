
<div align="center">

# Simplifying Neural Networks During Training
Lorenzo Sciandra, Samuele Fonio and Roberto Esposito

</div>

This repository contains the code to run and reproduce the experiments of the preprint [Simplifying Neural Networks During Training](https://arxiv.org/abs/2607.27854) paper, now submitted to a journal.


Please cite as:

```bibtex
@misc{sciandra2026nns,
      title={Simplifying Neural Networks During Training}, 
      author={Lorenzo Sciandra and Samuele Fonio and Roberto Esposito},
      year={2026},
      eprint={2607.27854},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.27854}, 
}
```

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
