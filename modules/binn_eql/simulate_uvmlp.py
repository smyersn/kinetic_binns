import sys, importlib, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.utils.numpy_torch_conversion import *
from modules.loaders.format_data import format_data_general
from modules.loaders.visualize_training_data import animate_data
from modules.generate_data.simulate_system import generate_initial_conditions, simulate
from modules.binn_eql.model_wrapper_2d import model_wrapper
from modules.binn_eql.build_binn_eql_net import BINN
from modules.loaders.format_data import format_data_general, format_data_torch
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split

def simulate_uvmlp(training_data, model):
    # --- Initial Conditions ---
    dimensions = model.model.dimensions
    device = 'cuda'
    
    xt = training_data[:, :dimensions+1]
    times = torch.unique(xt[:, dimensions])
    points = torch.unique(xt[:, 0])
    
    grid_x, grid_y = torch.meshgrid(points, points, indexing='xy')
    spatial_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # --- Storage ---
    u_array = torch.zeros((len(times), len(points), len(points), 2))

    # --- Simulation loop ---
    for n, t in enumerate(times):
        # We create a tensor of shape (40000, 1) filled with the scalar t
        t_column = torch.full((spatial_coords.shape[0], 1), t)
        
        # [x, y] + [t] -> [x, y, t]
        input_tensor = torch.cat([spatial_coords, t_column], dim=1).to(device)

        # Calculate surface
        with torch.no_grad():
            uv = model.model(input_tensor).view(len(points), len(points), 2)
            
        u_array[n] = uv

    return u_array, times 

def animate_uvmlp(u_array, times, name=None, titles=("u", "v")):
    """
    Plots two arrays side-by-side.
    Assumes input shape: (Time, Channel, Height, Width)
    """    
    # 1. Setup Figure: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2. Determine Shared Color Limits
    # This ensures the colors mean the same thing in both plots
    umin, umax = u_array[:, :, :, 0].min(), u_array[:, :, :, 0].max()
    vmin, vmax = u_array[:, :, :, 1].min(), u_array[:, :, :, 1].max()

    # 3. Initialize Plots
    # Plot 1
    im1 = axes[0].imshow(u_array[0, :, :, 0], cmap='viridis', vmin=umin, vmax=umax)
    axes[0].set_title(titles[0])
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot 2
    im2 = axes[1].imshow(u_array[0, :, :, 1], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1])
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Shared Main Title (for Time)
    main_title = fig.suptitle(f'T = {times[0]:.1f}', fontsize=16)

    # 4. Define Update Function
    def animate(frame):
        # Update data for both plots
        im1.set_array(u_array[frame, :, :, 0])
        im2.set_array(u_array[frame, :, :, 1])
        
        # Update time text
        main_title.set_text(f'T = {times[frame]:.2f}')
        
        return im1, im2, main_title

    # 5. Create Animation
    anim = animation.FuncAnimation(fig, animate, frames=len(times), interval=200)
    
    # Save if name provided
    if name:
        writergif = animation.PillowWriter(fps=10)
        anim.save(f'{name}.gif', writer=writergif)

    return anim