#!/bin/bash

source ~/.bashrc
conda activate binns

# LSTM
dus=("0.01" "0.05" "0.1" )
dvs=("0.5")
as=("5")
bs=("5")

# Create directories for every combination of parameters
for du in "${dus[@]}"; do
    for dv in "${dvs[@]}"; do
        for a in "${as[@]}"; do
            for b in "${bs[@]}"; do
                sbatch -p general -N 1 -n 1 --mem=32g -t 4:00:00 --output="./out/du_${du}_dv_${dv}_a_${a}_b_${b}.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/generate_data/parameter_sweep.py $du $dv $a $b"
            done
        done
    done
done