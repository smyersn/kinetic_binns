#!/bin/bash

source ~/.bashrc
conda activate binns

# Define parameter values
training_data_path="/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/turing_type_200/du_0.1_dv_0.5_a_5.0_b_5.0.npz"
reaction="turing_type"
params=("0.1" "0.5" "5" "5")

dimensions=2
species=2

density_weights=("0")

uv_layers=("3") 
uv_neurons=("128")
f_layers=("3")
f_neurons=("32")

epsilons=("0")
points=("0")

diffusion="False"

# Create configuration files
dir_name="."
config_file="$dir_name/config.cfg"

echo "training_data_path=\"$training_data_path\"" >> "$config_file"
echo "reaction=\"$reaction\"" >> "$config_file"
echo "params=\"${params[@]}\"" >> "$config_file"
echo "dimensions=\"$dimensions\"" >> "$config_file"
echo "species=\"$species\"" >> "$config_file"
echo "density_weight=\"$density_weights\"" >> "$config_file"
echo "uv_layers=$uv_layers" >> "$config_file"
echo "uv_neurons=$uv_neurons" >> "$config_file"
echo "f_layers=$f_layers" >> "$config_file"
echo "f_neurons=$f_neurons" >> "$config_file"
echo "epsilon=$epsilons" >> "$config_file"
echo "points=$points" >> "$config_file"
echo "diffusion=$diffusion" >> "$config_file"

# Submit the second job with dependency on the first job
sbatch -p general -N 1 -n 64 --mem=16g -t 6:00:00 --output="./$dir_name/slurm-%j.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/analysis/analysis.py $dir_name"