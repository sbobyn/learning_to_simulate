#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: No dataset specified."
  echo "Usage: sbatch $0 <dataset_name>"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Error: No data fraction size specified."
  echo "Usage: sbatch $0 <dataset_name> <data_fraction>"
  exit 1
fi

dataset=$1
dataset_path="${SCRATCH}/gns/datasets/$dataset"

output_dir="${SCRATCH}/gns/datasets/$dataset/output"
mkdir -p $output_dir

data_fraction=$2

cd ../

module purge
module load python
source venv/bin/activate

python train.py \
    --data_path $dataset_path \
    --epoch 10 \
    --batch_size 2 \
    --lr 1e-4 \
    --num_workers 2 \
    --noise 3e-4 \
    --eval_interval 10000 \
    --vis_interval 10000 \
    --save_interval 10000 \
    --rollout_interval 10000 \
    --output_path $output_dir \
    --wandb_group "dataset_size" \
    --wandb_project "GNS" \
    --env "cedar" \
    --data_fraction $data_fraction
