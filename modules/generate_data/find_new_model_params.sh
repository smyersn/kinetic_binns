#!/bin/bash

source ~/.bashrc
conda activate binns

# Create output directories
mkdir -p ./out
mkdir -p ./gifs

# # Parameter Ranges
# # We fix a and b, then sweep Du and k to find the spot-forming regime
# as=("1.0" "2.0" "4.0" "8.0")
# bs=("1.0" "2.0" "4.0" "8.0")
# ks=("0.01" "0.05" "0.1" "0.5")

# for a in "${as[@]}"; do
#     for b in "${bs[@]}"; do
#         for k in "${ks[@]}"; do
#             JOB_NAME="a_${a}_b_${b}_k_${k}"
            
#             sbatch -p general \
#                     -N 1 \
#                     -n 1 \
#                     --cpus-per-task=4 \
#                     --mem=16g \
#                     -t 4:00:00 \
#                     --job-name=$JOB_NAME \
#                     --output="./out/${JOB_NAME}.out" \
#                     --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/generate_data/find_new_model_params.py $a $b $k"
#         done
#     done
# done

# Define specific triplets: "a b k"
triplets=(
    "1.0 1.0 0.1"
    "4.0 4.0 0.05"
    "8.0 4.0 0.05"
    "8.0 1.0 0.5"
)

for triplet in "${triplets[@]}"; do
    # This splits the string into three separate variables
    read -r a b k <<< "$triplet"

    JOB_NAME="a_${a}_b_${b}_k_${k}"
    
    sbatch -p general \
            -N 1 \
            -n 1 \
            --cpus-per-task=4 \
            --mem=16g \
            -t 18:00:00 \
            --job-name=$JOB_NAME \
            --output="./out/${JOB_NAME}.out" \
            --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/generate_data/find_new_model_params.py $a $b $k"

done