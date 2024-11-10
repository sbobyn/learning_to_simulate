#!/bin/bash

#SBATCH -J sample-waterdrop  # Job name
#SBATCH -o sample-waterdrop.o%j     # Name of stdout output file
#SBATCH -e sample-waterdrop.e%j     # Name of stderr error file
#SBATCH --gres=gpu:1
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2 
#SBATCH -t 00:15:00          # Run time (hh:mm:ss)
#SBATCH --mail-user=<stevenbobyn@gmail.com>
#SBATCH --mail-type=ALL

# fail on error
set -e

# start in slurm_scripts
cd ../

dataset="WaterDrop"
dataset_path="${SCRATCH}/224w-gns/datasets/$dataset"

output_dir="${SCRATCH}/224w-gns/datasets/$dataset/output"
mkdir -p $output_dir

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
