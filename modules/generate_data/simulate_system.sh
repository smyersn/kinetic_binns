#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10g
#SBATCH -t 2:00:00
#SBATCH --constraint=rhel8
#SBATCH --output=simulate_system.out

source ~/.bashrc
conda activate binns
python simulate_system.py