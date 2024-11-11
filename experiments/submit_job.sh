#!/bin/bash

# Check if the dataset argument is provided
if [ -z "$1" ]; then
  echo "Error: No dataset specified."
  echo "Usage: ./submit_job.sh <dataset_name>"
  exit 1
fi

dataset=$1

# Define SLURM job parameters
job_name="${dataset}_train"
output_file="${dataset}_output.o%j"
error_file="${dataset}_error.e%j"
memory="8G"
cpus_per_task=2
gpu_count=1
time_limit="00:15:00"

# Submit the job to SLURM
sbatch -J "$job_name" \
       -o "$output_file" \
       -e "$error_file" \
       --mem=$memory \
       --cpus-per-task=$cpus_per_task \
       --gres=gpu:$gpu_count \
       --time=$time_limit \
       --mail-user="stevenbobyn@gmail.com" \
       --mail-type=ALL \
       ./cedar_test.sh $dataset
