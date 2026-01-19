#!/bin/bash

# --- Configuration Generator Function ---
generate_cfg() {
    local fileprefix="$1"
    local directory="$2"
    local tstop="$3"
    local samplingrate="$4"
    local n_Cdc42="$5"
    local n_BemGEF="$6"
    local n_FarGEF="$7"
    local length="$8"
    local random_seed="$9"

    local cfg_name="${directory}/${fileprefix}.cfg"
    local xyz_name="${fileprefix}.xyz"

    # Physics Parameters
    local rho=0.05
    local dt=0.0001
    
    # Calculate rates (using bc for floating point arithmetic)
    local k1a=10
    local k1b=40
    local P2a=$(echo "5.3 * $dt" | bc -l)
    local k2b=0.35
    local P3=$(echo "180 * $dt" | bc -l)
    local P2c=$P3
    local P4a=$(echo "9.6 * $dt" | bc -l)
    local k4b=40
    local k5a=36
    local k5b=13
    local P7=$(echo "256 * $dt" | bc -l)
    local P8a=$(echo "300 * $dt" | bc -l)
    local k8b=0.11
    local P9=$(echo "0.025 * $dt" | bc -l)
    local k10=0.002
    local Dm=0.0025
    local Dc=15
    local diff_complex=0.0001

    # Start writing configuration file
    cat > "$cfg_name" <<EOF
random_seed $random_seed
variable rho = $rho
variable rho_eps = 0.00001
variable L = $length
variable k1a = $k1a
variable k1b = $k1b
variable P2a = $P2a
variable P2c = $P2c
variable k2b = $k2b
variable P3 = $P3
variable P4a = $P4a
variable k4b = $k4b
variable k5a = $k5a
variable k5b = $k5b
variable P7  = $P7
variable P8a = $P8a
variable P9 = $P9
variable k8b = $k8b
variable k10 = $k10
dim 2
boundaries x -0.2 L+0.2
boundaries y -0.2 L+0.2
species Cdc42T Cdc42Dc Cdc42Dm BemGEFc BemGEFm BemGEF42 complex_Cdc42Dm_BemGEF42 complex_Cdc42Dm_BemGEFm
species RaGEF Ram Ric FarGEF complex_Cdc42D_Ra complex_Ra_Cdc42T
difc Cdc42T $Dm
difc Cdc42Dc $Dc
difc Cdc42Dm $Dm
difc BemGEF42 $Dm
difc BemGEFc $Dc
difc BemGEFm $Dm
difc complex_Cdc42Dm_BemGEF42 $Dm
difc complex_Cdc42Dm_BemGEFm $Dm
difc RaGEF $diff_complex
difc Ram $diff_complex
difc Ric $Dc
difc FarGEF $Dc
difc complex_Cdc42D_Ra $diff_complex
difc complex_Ra_Cdc42T $diff_complex
molecule_lists list1 list2 list3 list4 list5 list6 list7 list8 list9 list10
mol_list Cdc42T list1
mol_list BemGEF42 list2
mol_list Cdc42Dm list3
mol_list Cdc42Dc list4
mol_list BemGEFm list5
mol_list BemGEFc list6
mol_list Ram list7
mol_list Ric list8
mol_list RaGEF list9
mol_list FarGEF list10
start_surface inner_walls
action both all jump
polygon both edge
panel rect +x 0 0 L r1
panel rect -x L 0 L r2
panel rect +y 0 0 L r3
panel rect -y 0 L L r4
jump r1 front <-> r2 front
jump r3 front <-> r4 front
end_surface
start_compartment full_domain
surface inner_walls
point L/2 L/2
end_compartment
reaction BemGEF_cmtransition BemGEFc <-> BemGEFm k1a k1b
reaction Cdc42_cmtransition Cdc42Dc <-> Cdc42Dm k5a k5b
reaction Ra_cmtransition Ram -> Ric k10
reaction Cdc42Dm_2_T_bindTo_BemGEFm Cdc42Dm + BemGEFm -> complex_Cdc42Dm_BemGEFm
reaction Cdc42Dm_2_T_catBy_BemGEFm complex_Cdc42Dm_BemGEFm -> Cdc42T + BemGEFm
reaction_probability Cdc42Dm_2_T_bindTo_BemGEFm P2a
binding_radius Cdc42Dm_2_T_bindTo_BemGEFm rho
reaction_probability Cdc42Dm_2_T_catBy_BemGEFm 1
product_placement Cdc42Dm_2_T_catBy_BemGEFm unbindrad rho+rho_eps
reaction Cdc42T_2_Cdc42Dm Cdc42T -> Cdc42Dm k2b
reaction Cdc42Dm_2_T_bindTo_BemGEF42 Cdc42Dm + BemGEF42 -> complex_Cdc42Dm_BemGEF42
reaction Cdc42Dm_2_T_catBy_BemGEF42 complex_Cdc42Dm_BemGEF42 -> Cdc42T + BemGEF42
reaction_probability Cdc42Dm_2_T_bindTo_BemGEF42 P3
binding_radius Cdc42Dm_2_T_bindTo_BemGEF42 rho
reaction_probability Cdc42Dm_2_T_catBy_BemGEF42 1
product_placement Cdc42Dm_2_T_catBy_BemGEF42 unbindrad rho+rho_eps
reaction make_BemGEF42_fromm BemGEFm + Cdc42T <-> BemGEF42
reaction_probability make_BemGEF42_frommfwd P4a
binding_radius make_BemGEF42_frommfwd rho
reaction_rate make_BemGEF42_frommrev k4b
product_placement make_BemGEF42_frommrev unbindrad rho+rho_eps
reaction make_BemGEF42_fromc BemGEFc + Cdc42T -> BemGEF42
reaction_probability make_BemGEF42_fromc P7
binding_radius make_BemGEF42_fromc rho
reaction Ric_recruitBy_Cdc42T Ric + Cdc42T -> complex_Ra_Cdc42T
reaction Ric_2_Ram complex_Ra_Cdc42T -> Ram + Cdc42T
reaction_probability Ric_recruitBy_Cdc42T P9
reaction_probability Ric_2_Ram 1
binding_radius Ric_recruitBy_Cdc42T rho
product_placement Ric_2_Ram unbindrad rho+rho_eps
reaction FarGEF_bindTo_Ra Ram + FarGEF -> RaGEF
reaction_probability FarGEF_bindTo_Ra P8a
binding_radius FarGEF_bindTo_Ra rho
reaction RaGEF_dissociation RaGEF -> Ram + FarGEF k8b
product_placement RaGEF_dissociation unbindrad rho+rho_eps
reaction RaGEF_endocytosis RaGEF -> Ric + FarGEF k10
product_placement RaGEF_endocytosis unbindrad rho+rho_eps
reaction Cdc42D_2_T_bindTo_RaGEF  Cdc42Dm + RaGEF -> complex_Cdc42D_Ra
reaction Cdc42D_2_T_catBy_RaGEF complex_Cdc42D_Ra -> Cdc42T + RaGEF
reaction_probability Cdc42D_2_T_bindTo_RaGEF P2c
binding_radius Cdc42D_2_T_bindTo_RaGEF rho
reaction_probability Cdc42D_2_T_catBy_RaGEF 1
product_placement Cdc42D_2_T_catBy_RaGEF unbindrad rho+rho_eps
time_start 0
time_stop $tstop
time_step $dt
compartment_mol $n_Cdc42 Cdc42Dc full_domain
compartment_mol $n_BemGEF BemGEFc full_domain
compartment_mol $n_FarGEF FarGEF full_domain
compartment_mol 500 Ric full_domain
compartment_mol 2000 Ram full_domain
output_files $xyz_name
cmd N $samplingrate molpos Cdc42T $xyz_name
cmd N $samplingrate molpos Cdc42Dm $xyz_name
cmd N $samplingrate molpos Cdc42Dc $xyz_name
EOF
}

# --- Main Script ---

dir_name="second_go"
directory="runs/${dir_name}"
mkdir -p "$directory"

# Simulation parameters
tstop=1000
samplingrate=50000
k_l=8.8623

# Concentrations
c_cdc42=$(echo "3000 / ($k_l^2)" | bc -l)
c_BemGEF=$(echo "400 / ($k_l^2)" | bc -l)
c_FarGEF=$(echo "30 / ($k_l^2)" | bc -l)

# Smoldyn Executable Path
smoldyn_exec="/nas/longleaf/home/smyersn/software/smoldyn_install/bin/smoldyn"

# Ranges
# Seeds: 1 to 5
# Lengths: 20 to 50 step 10
# Scales: 1 to 3

for j in {1..5}; do
    for l in {20..30..10}; do
        for scale in {1..3}; do
            curr_seed=$j
            curr_length=$l
            curr_fileprefix="length_${curr_length}_scale_${scale}-seeds_${curr_seed}"
            
            # Calculate molecule counts
            l_sq=$(echo "$l * $l" | bc)
            
            # Cdc42
            raw_n_Cdc42=$(echo "$c_cdc42 * $l_sq" | bc -l)
            rounded_n_Cdc42=$(printf "%.0f" "$raw_n_Cdc42")
            n_Cdc42=$(( rounded_n_Cdc42 * scale ))

            # BemGEF
            raw_n_BemGEF=$(echo "$c_BemGEF * $l_sq" | bc -l)
            rounded_n_BemGEF=$(printf "%.0f" "$raw_n_BemGEF")
            n_BemGEF=$(( rounded_n_BemGEF * scale ))

            # FarGEF
            raw_n_FarGEF=$(echo "$c_FarGEF * $l_sq" | bc -l)
            rounded_n_FarGEF=$(printf "%.0f" "$raw_n_FarGEF")
            n_FarGEF=$(( rounded_n_FarGEF * scale ))

            # 1. Generate the config file
            generate_cfg \
                "$curr_fileprefix" \
                "$directory" \
                "$tstop" \
                "$samplingrate" \
                "$n_Cdc42" \
                "$n_BemGEF" \
                "$n_FarGEF" \
                "$l" \
                "$curr_seed"

            # 2. Submit the job immediately
            cfg_file="${curr_fileprefix}.cfg"
            
            echo "Submitting job for $cfg_file..."
            
            # -D sets working directory to $directory so xyz outputs go there
            # -J sets job name
            sbatch -p general -N 1 -J Smoldyn -t 264:00:00 --mem=5g -D "$directory" --wrap="$smoldyn_exec $cfg_file"
            
        done
    done
done