#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: No dataset specified."
  echo "Usage: sbatch $0 <dataset_name>"
  exit 1
fi

dataset=$1
dataset_path="${SCRATCH}/224w-gns/datasets/$dataset"

output_dir="${SCRATCH}/224w-gns/datasets/$dataset/output"
mkdir -p $output_dir

cd ../

module purge
module load python
source ../224w-gns/venv/bin/activate

python train.py \
    --data_path $dataset_path \
    --epoch 10 \
    --batch_size 2 \
    --lr 1e-4 \
    --num_workers 2 \
    --noise 3e-4 \
    --eval_interval 1000 \
    --vis_interval 1000 \
    --save_interval 1000 \
    --rollout_interval 1000 \
    --output_path $output_dir \
    --wandb_group "cedar_test" \
    --wandb_project "GNS" \
    --env "cedar" \
