#!/bin/bash

source ~/.bashrc
conda activate binns

save_path="/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/wave_pinning/diff_coeffs"
dus=("0.01")
dvs=("0.1" "0.5" "1" "5")
as=("1")
bs=("1")
ks=("0.01")

# save_path="/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/wave_pinning/positive_feedback"
# dus=("0.01")
# dvs=("1")
# as=("1" "3" "5")
# bs=("1")
# ks=("0.01")

mkdir -p $save_path

# Create directories for every combination of parameters
for du in "${dus[@]}"; do
    for dv in "${dvs[@]}"; do
        for a in "${as[@]}"; do
            for b in "${bs[@]}"; do
                for k  in "${ks[@]}"; do
                    sbatch -p volta-gpu -N 1 -n 1 --mem=32g --qos gpu_access --gres=gpu:1 -t 1:00:00 --output="./out/du_${du}_dv_${dv}_a_${a}_b_${b}_k_${k}.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/simulation/simulation.py $save_path $du $dv $a $b $k"
                    # sbatch -p general -N 1 -n 1 --cpus-per-task=4 --mem=16g -t 1:00:00 --output="./out/du_${du}_dv_${dv}_a_${a}_b_${b}_k_${k}.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/simulation/simulation.py $save_path $du $dv $a $b $k"
                done
            done
        done
    done
done