#!/bin/bash

# check if script name provided
if [ -z "$1" ]; then
  echo "Error: Args not provided."
  echo "Usage: ./submit_job.sh <job_script>.sh <dataset_name>"
  exit 1
fi

# Check if the dataset argument is provided
if [ -z "$2" ]; then
  echo "Error: Args not provided."
  echo "Usage: ./submit_job.sh <job_script>.sh <dataset_name>"
  exit 1
fi

jobscript=$1
dataset=$2

# Define SLURM job parameters
job_name="${dataset}_train"
output_file="${dataset}_output.o%j"
error_file="${dataset}_error.e%j"
memory="16G"
cpus_per_task=2
gpu_count=1
time_limit="06:00:00"

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
       ./"$jobscript" "$dataset"
