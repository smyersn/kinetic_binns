#!/bin/bash

source ~/.bashrc
conda activate binns

# Create configuration files
dir_name="."

# Submit the second job with dependency on the first job
sbatch -p general -N 1 -n 64 --mem=16g -t 6:00:00 --output="./$dir_name/slurm-%j.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/analysis/analysis.py $dir_name"