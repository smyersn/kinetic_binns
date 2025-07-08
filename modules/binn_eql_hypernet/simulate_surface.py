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
    
def simulate_surface(training_data, model):
    # --- Initial Conditions ---
    dimensions = model.model.dimensions
    device = 'cuda'
    
    xt = training_data[:, :dimensions+1]
    points = int(torch.unique(xt[:, 0]).numel())
    ic = training_data[training_data[:, dimensions] == 0]
    
    u0 = ic[:, dimensions + 1]                      # shape: (points**dimensions,)
    v0 = ic[:, dimensions + 2]

    grid_shape = [points] * dimensions
    u = u0.reshape(*grid_shape).float().half().to(device)  # (points, points, ..., points)
    v = v0.reshape(*grid_shape).float().half().to(device)

    # --- Parameters ---
    T = float(xt[:, dimensions].max().item())       # max time
    L = float(xt[:, 0].max().item())                # max x-coordinate

    nx, ny = u.shape                 # grid size
    dx, dy = L / nx, L / ny  
    dt = torch.tensor(0.0001, device=device, dtype=torch.half)          # time step
    nits = int(T / dt.item())                                   # number of time steps
    du, dv = model.model.diff_coeffs                     # diffusion rates

    # --- Laplacian kernel (5-point stencil) ---
    laplace_kernel = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)

    conv = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding=1,
        padding_mode='circular',
        bias=False)

    conv.weight.data = laplace_kernel
    conv.weight.requires_grad = False
    conv = conv.half().to(device)

    # --- Neural network for reaction ---
    reaction = model.model.reaction.eql_layer.eval()

    # --- Storage ---
    half_sec_nits = int(0.5 / dt)
    half_secs = int(nits / half_sec_nits) + 1

    u_array = torch.zeros((half_secs, 2, nx, ny))
    t_array = torch.arange(0, T + 0.5, 0.5)

    # Track time point in storage array
    i = 0

    # --- Simulation loop ---
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for t in range(nits):
            # Update storage every half second
            if t % half_sec_nits == 0:
                u_array[i] = torch.stack([u, v], dim=0)
                i += 1

            # Compute Laplacian (diffusion)
            lap_u = conv(u[None, None, :, :]).squeeze() / dx**2
            lap_v = conv(v[None, None, :, :]).squeeze() / dx**2

            # Reaction term
            uv = torch.column_stack((u.flatten(), v.flatten()))
            ruv = reaction(uv, inference=True)[0].view(nx, ny)
            
            # Euler update
            u = u + dt * (du * lap_u + ruv)
            v = v + dt * (dv * lap_v - ruv)        

            if t % (nits // 10) == 0:
                print(f"Progress: {(t / nits) * 100}%", flush=True)  
            
    return u_array, t_array   
            
def animate_sim(u_array, t_array, name=None): 
    fig, ax = plt.subplots()
    u_plot = ax.imshow(u_array[0, 0, :, :], cmap='viridis')
    u_plot.set_clim(vmin=u_array[:, 0, :, :].min(),
                    vmax=u_array[:, 0, :, :].max())
    cbar = plt.colorbar(u_plot, ax=ax)
    
    # Define update function
    def animate(frame):
        u_plot.set_array(u_array[frame, 0, :, :])
        ax.set_title(f'T = {t_array[frame]}')
        
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=range(len(u_array)), repeat=True)
    
    if name:
        writergif = animation.PillowWriter(fps=5)
        anim.save(f'{name}.gif', writer=writergif)

    return anim