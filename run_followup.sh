#!/bin/bash
# Experiments that depend on the full-training grid (run_grid.sh), then the independent ones.
# Finished commands are skipped on re-launch (markers in runs_logs/).
cd "$(dirname "$0")"
PY=~/sciandra_dl_env/bin/python
JOBS=${JOBS:-2}
until grep -q "GRID DONE" grid_launch.log 2>/dev/null; do sleep 300; done

# NNS second stage (linear probing; average-pooling head on CUB)
$PY experiments.py nns --run --jobs $JOBS
# Tunnel Effect split and model truncated at the TE split
$PY experiments.py te --run --jobs $JOBS
$PY experiments.py te-truncated --run --jobs $JOBS
# large scale: ImageNet-100 (ResNet18 / ResNet50, 3 seeds), full training then NNS second stage (pooling head)
$PY experiments.py imagenet --seeds 2025 123 42 --run --jobs 1
$PY experiments.py nns --source_tag imagenet --run --jobs 1
# numerical rank check, autoencoder and depth-reduction baselines
$PY experiments.py nr --run
$PY experiments.py autoencoder --run --jobs $JOBS
$PY experiments.py baselines --run --jobs $JOBS
# oracle study (3 seeds, subset of the periodic checkpoints)
$PY experiments.py oracle --models mlp_10 --datasets fashion_mnist --seeds 2025 123 42 --oracle_epochs 0 60 120 180 240 --run --jobs $JOBS
$PY experiments.py oracle --models resnet18 --datasets cifar10 --seeds 2025 123 42 --oracle_epochs 0 48 96 144 --run --jobs $JOBS
$PY experiments.py oracle --models vgg11 --datasets cifar100 --seeds 2025 123 42 --oracle_epochs 0 40 80 120 --run --jobs $JOBS
# independent studies
$PY experiments.py seeds --run --jobs $JOBS
$PY experiments.py labeling --run --jobs $JOBS
$PY experiments.py num-classes --online --run --jobs $JOBS
# wall-clock comparison, one run at a time (full and NNS runs of each configuration are interleaved)
$PY experiments.py cost --seeds 2025 123 42 --run --jobs 1
echo "FOLLOWUP DONE"
