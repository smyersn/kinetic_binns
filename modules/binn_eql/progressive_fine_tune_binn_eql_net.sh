#!/bin/bash

source ~/.bashrc
source /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/utils/diff_lists.sh
conda activate binns

mkdir -p runs

# Define number of training repeats
repeats=10
job_ids=""

# Define training data params
training_data_paths=("/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/pos_feedback/a_1.0_b_1_k_0.01.pt"
"/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/pos_feedback/a_3.0_b_1_k_0.01.pt"
"/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/pos_feedback/a_5.0_b_1_k_0.01.pt")

# training_data_paths=("/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/pos_feedback/a_1.0_b_1_k_0.01.pt"
# "/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/pos_feedback/a_5.0_b_1_k_0.01.pt")

# training_data_paths=("/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/pos_feedback/a_5.0_b_1_k_0.01.pt")

# training_data_paths=("/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/pos_feedback/a_1.0_b_1_k_0.01.pt")

# training_data_paths=("/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/diff_coeffs/wave_pinning/select_torch/du_0.01_dv_0.1.pt"
# "/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/diff_coeffs/wave_pinning/select_torch/du_0.01_dv_0.5.pt"
# "/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/diff_coeffs/wave_pinning/select_torch/du_0.01_dv_1.0.pt"
# "/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/diff_coeffs/wave_pinning/select_torch/du_0.01_dv_5.0.pt")

species=("2")
dimensions=("2")
epsilons=("0")
points=("0")
# epsilons=("0.05" "0.1" "0.25")
# epsilons=("0.1" "0.25" "0.5")
# points=("25" "50" "100") # value of 0 will use all points with no interpolation

# Define BINN params
diff_coeffs=("0.01" "1") # len(diff_coeffs)=species, empty list means params learned
# diff_coeffs=() # len(diff_coeffs)=species, empty list means params learned
# duplicates=("20")
duplicates=("5")
degree=("2")
pde_weights=("1")
l0_weights=("1") 
warm_ups=("20000")
lux_taxes=('1')
param_bounds=("10")

# Create directories for every combination of parameters
for repeat in $(seq 1 $repeats); do
    for training_data_path in "${training_data_paths[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            for point in "${points[@]}"; do
                for duplicate in "${duplicates[@]}"; do
                    for pde_weight in "${pde_weights[@]}"; do
                        for l0_weight in "${l0_weights[@]}"; do
                            for param_bound in "${param_bounds[@]}"; do
                                for warm_up in "${warm_ups[@]}"; do
                                    for lux_tax in "${lux_taxes[@]}"; do
                                        # Name model
                                        base=$(basename "$training_data_path" .pt)
                                        model_name="pos_feedback/78_final_pos_test/$base/binn_eql_"
                                        
                                        if diff_lists "${pde_weights[*]}" "1"; then
                                            model_name+="pde_${pde_weight[@]}_"
                                        fi

                                        if diff_lists "${l0_weights[*]}" "0.001"; then
                                            model_name+="l0_${l0_weight[@]}_"
                                        fi

                                        if diff_lists "${epsilons[*]}" "0"; then
                                            model_name+="epsilon_${epsilon[@]}_"
                                        fi

                                        if diff_lists "${points[*]}" "0"; then
                                            model_name+="points_${point[@]}_"
                                        fi

                                        if diff_lists "${duplicates[*]}" "1"; then
                                            model_name+="duplicates_${duplicate[@]}_"
                                        fi

                                        if diff_lists "${warm_ups[*]}" "0"; then
                                            model_name+="warm_up_${warm_up[@]}_"
                                        fi

                                        if diff_lists "${lux_taxes[*]}" "0"; then
                                            model_name+="lux_tax_${lux_tax[@]}_"
                                        fi                                        

                                        if [ "$repeats" -gt 1 ]; then
                                            model_name+="repeat_${repeat}_"
                                        fi

                                        # Create directory
                                        dir_name="runs/${model_name:0:-1}"

                                        mkdir -p $dir_name

                                        # Create configuration files
                                        config_file="$dir_name/config.cfg"
                                        echo "training_data_path=\"$training_data_path\"" >> "$config_file"
                                        echo "species=\"$species\"" >> "$config_file"
                                        echo "dimensions=\"$dimensions\"" >> "$config_file"
                                        echo "epsilon=\"$epsilon\"" >> "$config_file"
                                        echo "points=\"$point\"" >> "$config_file"
                                        echo "diff_coeffs=\"${diff_coeffs[*]}\"" >> "$config_file"
                                        echo "duplicates=\"$duplicate\"" >> "$config_file"
                                        echo "degree=\"$degree\"" >> "$config_file"
                                        echo "pde_weight=\"$pde_weight\"" >> "$config_file"
                                        echo "l0_weight=\"$l0_weight\"" >> "$config_file"
                                        echo "warm_up=\"$warm_up\"" >> "$config_file"
                                        echo "lux_tax=\"$lux_tax\"" >> "$config_file"
                                        echo "param_bounds=\"$param_bound\"" >> "$config_file"

                                        # sbatch -p l40-gpu -N 1 -n 1 --mem=48g --qos gpu_access --gres=gpu:1 -t 1-00:00:00 --output="./$dir_name/slurm-%j.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/binn_eql/train_binn_eql_net_fine_tune.py $dir_name"
                                        sbatch -p volta-gpu -N 1 -n 1 --mem=48g --qos gpu_access --gres=gpu:1 -t 2-00:00:00 --output="./$dir_name/slurm-%j.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/binn_eql/train_binn_eql_net_fine_tune.py $dir_name"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done