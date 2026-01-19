import os
import math

def smoldyn_cfg_variable_size(fileprefix, directory, tstop, samplingrate, n_Cdc42, n_BemGEF, n_FarGEF, length, random_seed):
    """
    Generates a Smoldyn configuration file.
    Adapted from "make_smoldyn_cfg.m"
    """
    # Define file paths
    # Note: .cfg is created inside the directory
    cfg_name = os.path.join(directory, f"{fileprefix}.cfg")
    # Output file name for the simulation to write to (relative path)
    xyz_name = f"{fileprefix}.xyz"

    with open(cfg_name, 'w') as fid:
        # Parameters for the simulation
        rho = 0.05
        dt = 0.0001
        k1a = 10
        k1b = 40
        P2a = 5.3 * dt
        k2b = 0.35
        P3 = 180 * dt
        P2c = P3
        P4a = 9.6 * dt
        k4b = 40
        k5a = 36
        k5b = 13
        P7 = 256 * dt
        P8a = 300 * dt
        k8b = 0.11
        P9 = 0.025 * dt
        k10 = 0.002
        Dm = 0.0025
        Dc = 15

        # Define a random number generator
        fid.write(f'random_seed {int(random_seed)}\n')
        # Reaction radius rho
        fid.write(f'variable rho = {rho:g}\n')
        # A really small number which helps separate molecules
        fid.write('variable rho_eps = 0.00001\n')
        # Domain length
        fid.write(f'variable L = {length:g}\n')
        
        # Reaction rates
        fid.write(f'variable k1a = {k1a:g}\n')
        fid.write(f'variable k1b = {k1b:g}\n')
        fid.write(f'variable P2a = {P2a:g}\n')
        fid.write(f'variable P2c = {P2c:g}\n')
        fid.write(f'variable k2b = {k2b:g}\n')
        fid.write(f'variable P3 = {P3:g}\n')
        fid.write(f'variable P4a = {P4a:g}\n')
        fid.write(f'variable k4b = {k4b:g}\n')
        fid.write(f'variable k5a = {k5a:g}\n')
        fid.write(f'variable k5b = {k5b:g}\n')
        fid.write(f'variable P7  = {P7:g}\n')
        fid.write(f'variable P8a = {P8a:g}\n')
        fid.write(f'variable P9 = {P9:g}\n')
        fid.write(f'variable k8b = {k8b:g}\n')
        fid.write(f'variable k10 = {k10:g}\n')

        # Define domain boundaries
        fid.write('dim 2\n')
        fid.write('boundaries x -0.2 L+0.2\n')
        fid.write('boundaries y -0.2 L+0.2\n')

        # Define species
        fid.write('species Cdc42T Cdc42Dc Cdc42Dm BemGEFc BemGEFm BemGEF42 complex_Cdc42Dm_BemGEF42 complex_Cdc42Dm_BemGEFm\n')
        fid.write('species RaGEF Ram Ric FarGEF complex_Cdc42D_Ra complex_Ra_Cdc42T\n')

        # Define diffusion rates.
        fid.write(f'difc Cdc42T {Dm:g}\n')
        fid.write(f'difc Cdc42Dc {Dc:g}\n')
        fid.write(f'difc Cdc42Dm {Dm:g}\n')
        fid.write(f'difc BemGEF42 {Dm:g}\n')
        fid.write(f'difc BemGEFc {Dc:g}\n')
        fid.write(f'difc BemGEFm {Dm:g}\n')
        fid.write(f'difc complex_Cdc42Dm_BemGEF42 {Dm:g}\n')
        fid.write(f'difc complex_Cdc42Dm_BemGEFm {Dm:g}\n')
        fid.write(f'difc RaGEF {0.0001:g}\n')
        fid.write(f'difc Ram {0.0001:g}\n')
        fid.write(f'difc Ric {Dc:g}\n')
        fid.write(f'difc FarGEF {Dc:g}\n')
        fid.write(f'difc complex_Cdc42D_Ra {0.0001:g}\n')
        fid.write(f'difc complex_Ra_Cdc42T {0.0001:g}\n')

        # Set up lists to store molecular coordinates
        fid.write('molecule_lists list1 list2 list3 list4 list5 list6 list7 list8 list9 list10\n')
        fid.write('mol_list Cdc42T list1\n')
        fid.write('mol_list BemGEF42 list2\n')
        fid.write('mol_list Cdc42Dm list3\n')
        fid.write('mol_list Cdc42Dc list4\n')
        fid.write('mol_list BemGEFm list5\n')
        fid.write('mol_list BemGEFc list6\n')
        fid.write('mol_list Ram list7\n')
        fid.write('mol_list Ric list8\n')
        fid.write('mol_list RaGEF list9\n')
        fid.write('mol_list FarGEF list10\n')

        # Define a domain. The boundary condition is periodic.
        fid.write('start_surface inner_walls\n')
        fid.write('action both all jump\n')
        fid.write('polygon both edge\n')
        fid.write('panel rect +x 0 0 L r1\n')
        fid.write('panel rect -x L 0 L r2\n')
        fid.write('panel rect +y 0 0 L r3\n')
        fid.write('panel rect -y 0 L L r4\n')
        fid.write('jump r1 front <-> r2 front\n')
        fid.write('jump r3 front <-> r4 front\n')
        fid.write('end_surface\n')

        # Define the compartment which is required by Smoldyn to input molecules.
        fid.write('start_compartment full_domain\n')
        fid.write('surface inner_walls\n')
        fid.write('point L/2 L/2\n')
        fid.write('end_compartment\n')

        # Define reactions
        # Bem1-GEF associates and dissociates from the membrane
        fid.write('reaction BemGEF_cmtransition BemGEFc <-> BemGEFm k1a k1b\n')
        # Cdc42-GDP associates and dissociates from the membrane
        fid.write('reaction Cdc42_cmtransition Cdc42Dc <-> Cdc42Dm k5a k5b\n')
        # Receptors dissociates from the membrane
        fid.write('reaction Ra_cmtransition Ram -> Ric k10\n')
        
        # Bem1-GEFm + Cdc42Dm -> Bem1-GEFm + Cdc42T
        fid.write('reaction Cdc42Dm_2_T_bindTo_BemGEFm Cdc42Dm + BemGEFm -> complex_Cdc42Dm_BemGEFm\n')
        fid.write('reaction Cdc42Dm_2_T_catBy_BemGEFm complex_Cdc42Dm_BemGEFm -> Cdc42T + BemGEFm\n')
        fid.write('reaction_probability Cdc42Dm_2_T_bindTo_BemGEFm P2a\n')
        fid.write('binding_radius Cdc42Dm_2_T_bindTo_BemGEFm rho\n')
        fid.write('reaction_probability Cdc42Dm_2_T_catBy_BemGEFm 1\n')
        # Place molecules beyond their reactive radius
        fid.write('product_placement Cdc42Dm_2_T_catBy_BemGEFm unbindrad rho+rho_eps\n')
        
        # Cdc42T -> Cdc42Dm
        fid.write('reaction Cdc42T_2_Cdc42Dm Cdc42T -> Cdc42Dm k2b\n')
        
        # Cdc42T-Bem1-GEF + Cdc42Dm -> Cdc42T-Bem1-GEF + Cdc42T
        fid.write('reaction Cdc42Dm_2_T_bindTo_BemGEF42 Cdc42Dm + BemGEF42 -> complex_Cdc42Dm_BemGEF42\n')
        fid.write('reaction Cdc42Dm_2_T_catBy_BemGEF42 complex_Cdc42Dm_BemGEF42 -> Cdc42T + BemGEF42\n')
        fid.write('reaction_probability Cdc42Dm_2_T_bindTo_BemGEF42 P3\n')
        fid.write('binding_radius Cdc42Dm_2_T_bindTo_BemGEF42 rho\n')
        fid.write('reaction_probability Cdc42Dm_2_T_catBy_BemGEF42 1\n')
        fid.write('product_placement Cdc42Dm_2_T_catBy_BemGEF42 unbindrad rho+rho_eps\n')
        
        # Cdc42T + Bem1-GEFm -> Cdc42T-Bem1-GEF
        fid.write('reaction make_BemGEF42_fromm BemGEFm + Cdc42T <-> BemGEF42\n')
        fid.write('reaction_probability make_BemGEF42_frommfwd P4a\n')
        fid.write('binding_radius make_BemGEF42_frommfwd rho\n')
        fid.write('reaction_rate make_BemGEF42_frommrev k4b\n')
        fid.write('product_placement make_BemGEF42_frommrev unbindrad rho+rho_eps\n')
        
        # Cdc42T + BemGEFc -> Cdc42T-Bem1-GEF
        fid.write('reaction make_BemGEF42_fromc BemGEFc + Cdc42T -> BemGEF42\n')
        fid.write('reaction_probability make_BemGEF42_fromc P7\n')
        fid.write('binding_radius make_BemGEF42_fromc rho\n')
        
        # Ric + Cdc42T -> Ram + Cdc42T
        fid.write('reaction Ric_recruitBy_Cdc42T Ric + Cdc42T -> complex_Ra_Cdc42T\n')
        fid.write('reaction Ric_2_Ram complex_Ra_Cdc42T -> Ram + Cdc42T\n')
        fid.write('reaction_probability Ric_recruitBy_Cdc42T P9\n')
        fid.write('reaction_probability Ric_2_Ram 1\n')
        fid.write('binding_radius Ric_recruitBy_Cdc42T rho\n')
        fid.write('product_placement Ric_2_Ram unbindrad rho+rho_eps\n')
        
        # Far1-GEF + Ram -> RaGEF
        fid.write('reaction FarGEF_bindTo_Ra Ram + FarGEF -> RaGEF\n')
        fid.write('reaction_probability FarGEF_bindTo_Ra P8a\n')
        fid.write('binding_radius FarGEF_bindTo_Ra rho\n')
        
        # RaGEF -> Far1-GEF + Ram 
        fid.write('reaction RaGEF_dissociation RaGEF -> Ram + FarGEF k8b\n')
        fid.write('product_placement RaGEF_dissociation unbindrad rho+rho_eps\n')
        
        # RaGEF -> Ric + Far1-GEF
        fid.write('reaction RaGEF_endocytosis RaGEF -> Ric + FarGEF k10\n')
        fid.write('product_placement RaGEF_endocytosis unbindrad rho+rho_eps\n')
        
        # Cdc42Dm + RaGEF -> Cdc42T + RaGEF
        fid.write('reaction Cdc42D_2_T_bindTo_RaGEF  Cdc42Dm + RaGEF -> complex_Cdc42D_Ra\n')
        fid.write('reaction Cdc42D_2_T_catBy_RaGEF complex_Cdc42D_Ra -> Cdc42T + RaGEF\n')
        fid.write('reaction_probability Cdc42D_2_T_bindTo_RaGEF P2c\n')
        fid.write('binding_radius Cdc42D_2_T_bindTo_RaGEF rho\n')
        fid.write('reaction_probability Cdc42D_2_T_catBy_RaGEF 1\n')
        fid.write('product_placement Cdc42D_2_T_catBy_RaGEF unbindrad rho+rho_eps\n')

        # Define the starting and ending point and time step
        fid.write('time_start 0\n')
        fid.write(f'time_stop {int(tstop)}\n')
        fid.write(f'time_step {dt:g}\n')

        # Input initial conditions
        fid.write(f'compartment_mol {int(n_Cdc42)} Cdc42Dc full_domain\n')
        fid.write(f'compartment_mol {int(n_BemGEF)} BemGEFc full_domain\n')
        fid.write(f'compartment_mol {int(n_FarGEF)} FarGEF full_domain\n')
        fid.write('compartment_mol 500 Ric full_domain\n')
        fid.write('compartment_mol 2000 Ram full_domain\n')

        # Output coordinates of molecules
        fid.write(f'output_files {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos Cdc42T {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos BemGEF42 {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos Cdc42Dm {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos Cdc42Dc {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos BemGEFm {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos BemGEFc {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos Ram {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos Ric {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos RaGEF {xyz_name}\n')
        # fid.write(f'cmd N {int(samplingrate)} molpos FarGEF {xyz_name}\n')
        fid.write(f'cmd N {int(samplingrate)} molpos Cdc42T {xyz_name}\n')
        fid.write(f'cmd N {int(samplingrate)} molpos Cdc42Dm {xyz_name}\n')
        fid.write(f'cmd N {int(samplingrate)} molpos Cdc42Dc {xyz_name}\n')

# --- Main Script ---

# Create the directory that stores the simulations
dir_name = 'first_go'
directory = f'runs/{dir_name}'
if not os.path.exists(directory):
    os.makedirs(directory)

# Simulation parameters
tstop = 1000
samplingrate = 50000
k_l = 8.8623
c_cdc42 = 3000 / (k_l**2)
c_BemGEF = 400 / (k_l**2)
c_FarGEF = 30 / (k_l**2)

# Range of seeds and lengths
random_seeds = range(1, 11)  # 1 to 10
lengths = range(25, 51, 5)    # 5 to 50
scales = range(1, 4, 1)

# Create the shell script for SLURM
run_script_path = os.path.join(directory, 'run.sh')
with open(run_script_path, 'w') as fid:
    fid.write('#!/bin/bash\n\n')
    
    for j in random_seeds:
        for l in lengths:
            for scale in scales: 
                curr_seed = j
                curr_length = l
                curr_fileprefix = f'length_{curr_length}_scale_{scale}-seeds_{curr_seed}'
                
                # Rounding to match MATLAB's integer molecule counts
                n_Cdc42 = round(c_cdc42 * (l**2)) * scale
                n_BemGEF = round(c_BemGEF * (l**2)) * scale
                n_FarGEF = round(c_FarGEF * (l**2)) * scale
                
                # Call the function to generate .cfg
                smoldyn_cfg_variable_size(
                    curr_fileprefix, 
                    directory, 
                    tstop, 
                    samplingrate, 
                    n_Cdc42, 
                    n_BemGEF, 
                    n_FarGEF, 
                    l, 
                    curr_seed
                )
                
                # Define the config file name
                cfg_name = f'{curr_fileprefix}.cfg'
                
                # Write the sbatch command
                # Note: I replaced the slightly ambiguous "cd" logic from the original 
                # with a cleaner execution path. Adjust the path to smoldyn executable as needed.
                smoldyn_exec = "/nas/longleaf/home/smyersn/rotations/elston/smoldyn/smoldyn-2.61/cmake/smoldyn"
                
                # Using --wrap requires the command to be a single string. 
                # This runs the executable on the specific config file.
                cmd = f'sbatch -p general -N 1 -J Smoldyn -t 264:00:00 --mem=5g --wrap="{smoldyn_exec} {cfg_name}"\n'
                fid.write(cmd)