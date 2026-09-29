
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

## Revision experiments

`experiments.py` prints (or runs, with `--run [--jobs N]`) the commands of each experiment family; finished commands are marked in `runs_logs/` and skipped on re-runs. Run the families in this order:

| Family | What it runs | Outputs |
|---|---|---|
| `grid` | full training of the 21 model–dataset pairs × 5 seeds (add `--save_every k` for the oracle study) | IFC trajectories, NNS split, split/final checkpoints, full accuracy |
| `nns` | second stage of NNS from the split checkpoints (`--head lp/fix/dcl/avg`) | accuracy of the simplified models |
| `te`, `te-truncated` | post-hoc Tunnel Effect split (linear probes, 95% rule) and the model truncated at that split | `results/te/` |
| `oracle` | simplified models at several layers / split epochs | Figs. 6–7 with std bands |
| `seeds` | tunnel-set seed × optimization seed (5 × 5), stopping at the split | seed study table |
| `labeling` | online NNS with alternative labeling rules, references and patience values | accuracy of each variant |
| `num-classes` | CIFAR-100 subsets with 2/10/25/50/100 classes | IFC curves and splits |
| `cost` | full training vs online NNS on the same machine | wall-clock, IFC overhead |
| `imagenet` | ResNet18/50 on ImageNet-100 (bf16, streaming IFC, pooling head) | large-scale result |
| `baselines` | CKA layer pruning (Pons et al.) and LaCoOT (Quétu et al.) on ResNet18 | accuracy/params/FLOPs/time |
| `autoencoder` | autoencoder encoder (same layers as NNS) + linear head | unsupervised baseline |
| `nr` | legacy vs corrected numerical rank, MLP12/CIFAR-10 | `results/nr/` |

Then:

- `python split_ablation.py --metrics "log_metrics/main/**/metrics.pth"`: split/epoch of every labeling variant, replayed offline on the recorded IFC.
- `python collect_results.py split accuracy te cost seeds baselines --tag main`: LaTeX/CSV tables in `tables_generated/`. Table 1 and the appendix `%Reduction`/`%Training` rows come from the same runs.
- `python plots.py {metric-epochs,metric-layers,labels,compare-layers,oracle,oracle-params} ...`: figures with mean ± std over seeds, with the split marked.

New configuration options (the defaults reproduce the original behavior):

- `-set key=value ...`: override any config entry.
- `-nns`: online NNS (the model is simplified in the same run).
- `-stop_at_split`: stop training at the split.
- `-resume_from`: explicit checkpoint file to resume from.
- `experiment.metrics: ifc`: streaming IFC, about 27× cheaper than `all` for ResNet18/CIFAR-10.
- `experiment.labeling`: `{rule, threshold, reference, ref_epochs}`.
- `dataset.tunnel_seed`, `dataset.num_classes`, `dataset.tunnel_eval_transform`.
- `settings.amp`, `settings.run_tag`.
- `tunnel_args.pool`: average-pooling head.
