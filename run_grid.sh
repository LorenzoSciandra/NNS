#!/bin/bash
# Full-training grid for the TMLR revision (tag: main). Guiding configurations first, with periodic
# checkpoints for the oracle study; finished runs are skipped on re-launch.
cd "$(dirname "$0")"
PY=~/sciandra_dl_env/bin/python
JOBS=${JOBS:-2}
$PY experiments.py grid --models mlp_10 --datasets fashion_mnist --save_every 30 --run --jobs $JOBS
$PY experiments.py grid --models resnet18 --datasets cifar10 --save_every 24 --run --jobs $JOBS
$PY experiments.py grid --models vgg11 --datasets cifar100 --save_every 20 --run --jobs $JOBS
$PY experiments.py grid --run --jobs $JOBS
echo "GRID DONE"
