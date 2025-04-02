import sys, importlib, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.utils.numpy_torch_conversion import *
from modules.loaders.format_data import format_data_general
from modules.loaders.visualize_training_data import animate_data
from modules.generate_data.simulate_system import generate_initial_conditions, simulate
import time
    
def simulate_surface(model, device, dimensions, species, params, model_dir, training_data_path):
    # Load in data
    training_data = format_data_general(dimensions, species, training_data_path)
    xt = training_data[:, :dimensions+1]
    outputs = training_data[:, dimensions+1:]
    
    # Get initial conditions from training data 
    L = np.max(xt[:, 0])
    T = np.max(xt[:, dimensions])

    points = len(np.unique(training_data[:, 0]))
    ic = training_data[training_data[:, dimensions] == 0]

    u0 = np.reshape(ic[:, dimensions+1], (points,)*dimensions)
    v0 = np.reshape(ic[:, dimensions+2], (points,)*dimensions)
    
    # Load model
    model.load(f'{model_dir}/binn_best_val_model', device=device)

    # Simulate surface from training data initial conditions and animate
    x_array, u_array, v_array, t_array = simulate(u0, v0, L, points, T, 
                                                  dimensions, species, 
                                                  Du=params[0], Dv=params[1],
                                                  nn=model.model, 
                                                  early_stop=False)
        
    sim_formatted = format_data_general(dimensions, species, x_array=x_array, 
                        t_array=t_array, u_array=u_array, v_array=v_array)
    
    animate_data(sim_formatted, dimensions, species, name=f'{model_dir}/f_mlp_animation_training_data_ic')

    # Make sure u and v don't go negative
    print(f'Minimum u-value during simulation: {np.min(u_array)}')
    print(f'Minimum v-value during simulation: {np.min(v_array)}')
    
    # Generate random initial conditions
    u0, v0 = generate_initial_conditions(1, 1.0246, points, dimensions, random=True)
    
    # Simulate surface from random initial conditions and animate
    x_array, u_array, v_array, t_array = simulate(u0, v0, L, points, T, 
                                                  dimensions, species,
                                                  Du=params[0], Dv=params[1],
                                                  nn=model.model, 
                                                  early_stop=False)
    
    sim_formatted = format_data_general(dimensions, species, x_array=x_array, 
                    t_array=t_array, u_array=u_array, v_array=v_array)

    animate_data(sim_formatted, dimensions, species, name=f'{model_dir}/f_mlp_animation_random_ic')

    # Make sure u and v don't go negative
    print(f'Minimum u-value during simulation: {np.min(u_array)}')
    print(f'Minimum v-value during simulation: {np.min(v_array)}')

    # Create kymograph
    if dimensions == 1:
        # Reshape data for kymograph, instantiate figure
        u_kymograph = np.reshape(outputs[:, 0], (int(2 * T + 1), points), order='F')

        fig = plt.figure(figsize=(5, 5), facecolor='w')
        fig.subplots_adjust(wspace=0.2)

        # Create axes
        ax1 = fig.add_axes([0.1, 0.2, 0.3, 0.3])
        ax2 = fig.add_axes([0.55, 0.2, 0.3, 0.3])
        # Create extra axis for colorbar
        ax3 = fig.add_axes([0.9, 0.2, 0.2, 0.3])
        ax3.axis('off')

        # Plot Data
        kymograph_sim = ax1.imshow(u_kymograph.T, aspect='auto', cmap='viridis', extent=[0,24.5,0,10]) # simulated
        kymograph_learned = ax2.imshow(u_array.T, aspect='auto', cmap='viridis', extent=[0,24.5,0,10]) # learned

        # Show colorbar
        cbar = fig.colorbar(kymograph_sim, ax=ax3, location='left', ticklocation='bottom')
        plt.text(0, 0.33, '[A] (uM)', rotation=270)

        # Format plots
        ax2.set_yticklabels([])
        ax1.set_ylabel('Space (uM)')
        ax1.set_xlabel('Time (s)')
        ax2.set_ylabel('Space (uM)')
        ax2.set_xlabel('Time (s)')
        ax1.title.set_text('Solution w/ F(A, B)')
        ax2.title.set_text('Solution w/ F*(A, B)')

        # Show the plot
        plt.savefig(f'{model_dir}/f_mlp_kymograph.png')
        plt.show()
