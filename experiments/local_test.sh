#!/bin/bash

dataset="WaterDropSample"
dataset_path="datasets/$dataset"

output_dir="output/$dataset"
mkdir -p $output_dir

python train.py \
    --data_path $dataset_path \
    --epoch 10 \
    --batch_size 2 \
    --lr 1e-4 \
    --num_workers 2 \
    --noise 3e-4 \
    --eval_interval 995 \
    --vis_interval 995 \
    --save_interval 995 \
    --rollout_interval 995 \
    --output_path $output_dir \
    --wandb_group "local_test" \
    --wandb_project "GNS" \
    --env "local" \
