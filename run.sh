#!/bin/bash

datasets="cifar100 fashion_mnist cifar10"
model_types="mlp_10 mlp_12 resnet10 resnet18 vgg11 vgg16"
seeds="2025 123 42 777 1024"

for seed in $seeds
do
    for model in $model_types
    do
        for data in $datasets
        do
         
          echo "Running model: $model on dataset: $data with seed: $seed"

          python update_yaml.py configs/experiments/exp.yaml dataset.name str "$data"
          python update_yaml.py configs/experiments/exp.yaml settings.seed int "$seed"
          python update_yaml.py configs/experiments/exp.yaml logger.wandb_args.name str "_baseline_${model}_${data}_seed${seed}"

          python main.py -exp configs/experiments/exp.yaml \
                        -trainer configs/trainer/"$model".yaml \
      
        done
    done
done