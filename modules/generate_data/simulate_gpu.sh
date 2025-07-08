#!/bin/bash

source ~/.bashrc
conda activate binns

# as=("0.2" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1" "2" "3" "5" "10")
as=("1" "3" "5")

# Create directories for every combination of parameters
for a in "${as[@]}"; do
    sbatch -p volta-gpu -N 1 -n 1 --mem=32g --qos gpu_access --gres=gpu:1 -t 6:00:00 --output="./out/slurm-%j.out" --wrap="python simulate_gpu.py $a"
done
