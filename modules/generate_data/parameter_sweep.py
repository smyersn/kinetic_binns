import sys, importlib, os
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{file_dir}/../../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from modules.loaders.visualize_training_data import plot_animation, animate_data
from modules.utils.numpy_torch_conversion import *
from modules.loaders.format_data import format_data_general
from modules.generate_data.simulate_system import *


if __name__ == '__main__':    
    # Define initial conditions
    u0 = 1
    v0 = 1.0246
    
    # Define parameters
    N = 200
    L = 10
    T = 100
    dim = 2
    species = 2
    
    # Define reaction
    reaction_fn = wave_pinning
    Du, Dv = float(sys.argv[1]), float(sys.argv[2])
    params = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
    
    # Calculate initial conditions for grid
    save_name = f'/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/wave_pinning/du_{Du}_dv_{Dv}_a_{params[0]}_b_{params[1]}_k_{params[2]}'
    
    u0, v0 = generate_initial_conditions(u0, v0, N, dim, random=True)
    
    # Simulate
    simulate(u0, v0, L, N, T, dim, reaction_fn, params, Du, Dv, save_name=save_name)
    
    sim_formatted = format_data_general(dim, species, file=f'{save_name}.npz')

    animate_data(sim_formatted, dim, 2, name=save_name)