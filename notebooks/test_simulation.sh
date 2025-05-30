#!/bin/bash

source ~/.bashrc
conda activate binns

# sbatch -p volta-gpu -N 1 -n 1 --mem=32g --qos gpu_access --gres=gpu:1 -t 8:00:00 --output="slurm-%j.out" --wrap="python test_simulation.py"

python - <<’EOF’
import sys, pkgutil
print([m.name for m in pkgutil.iter_modules() if m.name=="torch"])