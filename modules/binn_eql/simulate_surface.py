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
    
# def simulate_surface(training_data, dimensions, species, reaction, Du, Dv, dir_name):
#     # Load in data
#     xt = training_data[:, :dimensions+1]
#     outputs = training_data[:, dimensions+1:]
    
#     # Get initial conditions from training data 
#     L = np.max(xt[:, 0])
#     T = np.max(xt[:, dimensions])

#     points = len(np.unique(training_data[:, 0]))
#     ic = training_data[training_data[:, dimensions] == 0]

#     u0 = np.reshape(ic[:, dimensions+1], (points,)*dimensions)
#     v0 = np.reshape(ic[:, dimensions+2], (points,)*dimensions)
    
#     # Simulate surface from training data initial conditions and animate
#     x_array, u_array, v_array, t_array = simulate(u0, v0, L, points, T, 
#                                                   dimensions, reaction, Du=Du, 
#                                                   Dv=Dv, early_stop=False)
            
#     sim_formatted = format_data_general(dimensions, species, x_array=x_array, 
#                         t_array=t_array, u_array=u_array, v_array=v_array)
    
#     animate_data(sim_formatted, dimensions, species, name=f'{dir_name}/f_mlp_animation_training_data_ic')

#     # Make sure u and v don't go negative
#     print(f'Minimum u-value during simulation: {np.min(u_array)}')
#     print(f'Minimum v-value during simulation: {np.min(v_array)}')
    
#     # Generate random initial conditions
#     u0, v0 = generate_initial_conditions(1, 1.0246, points, dimensions, random=True)
    
#     # Simulate surface from random initial conditions and animate
#     x_array, u_array, v_array, t_array = simulate(u0, v0, L, points, T, 
#                                                   dimensions, reaction, Du=Du, 
#                                                   Dv=Dv, early_stop=False)
        
#     sim_formatted = format_data_general(dimensions, species, x_array=x_array, 
#                     t_array=t_array, u_array=u_array, v_array=v_array)

#     animate_data(sim_formatted, dimensions, species, name=f'{dir_name}/f_mlp_animation_random_ic')

#     # Make sure u and v don't go negative
#     print(f'Minimum u-value during simulation: {np.min(u_array)}')
#     print(f'Minimum v-value during simulation: {np.min(v_array)}')

#     # Create kymograph
#     if dimensions == 1:
#         # Reshape data for kymograph, instantiate figure
#         u_kymograph = np.reshape(outputs[:, 0], (int(2 * T + 1), points), order='F')

#         fig = plt.figure(figsize=(5, 5), facecolor='w')
#         fig.subplots_adjust(wspace=0.2)

#         # Create axes
#         ax1 = fig.add_axes([0.1, 0.2, 0.3, 0.3])
#         ax2 = fig.add_axes([0.55, 0.2, 0.3, 0.3])
#         # Create extra axis for colorbar
#         ax3 = fig.add_axes([0.9, 0.2, 0.2, 0.3])
#         ax3.axis('off')

#         # Plot Data
#         kymograph_sim = ax1.imshow(u_kymograph.T, aspect='auto', cmap='viridis', extent=[0,24.5,0,10]) # simulated
#         kymograph_learned = ax2.imshow(u_array.T, aspect='auto', cmap='viridis', extent=[0,24.5,0,10]) # learned

#         # Show colorbar
#         cbar = fig.colorbar(kymograph_sim, ax=ax3, location='left', ticklocation='bottom')
#         plt.text(0, 0.33, '[A] (uM)', rotation=270)

#         # Format plots
#         ax2.set_yticklabels([])
#         ax1.set_ylabel('Space (uM)')
#         ax1.set_xlabel('Time (s)')
#         ax2.set_ylabel('Space (uM)')
#         ax2.set_xlabel('Time (s)')
#         ax1.title.set_text('Solution w/ F(A, B)')
#         ax2.title.set_text('Solution w/ F*(A, B)')

#         # Show the plot
#         plt.savefig(f'{dir_name}/f_mlp_kymograph.png')
#         plt.show()

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
    u = u0.reshape(*grid_shape).float().to(device)  # (points, points, ..., points)
    v = v0.reshape(*grid_shape).float().to(device)

    # --- Parameters ---
    T = float(xt[:, dimensions].max().item())       # max time
    L = float(xt[:, 0].max().item())                # max x-coordinate

    nx, ny = u.shape                 # grid size
    dx, dy = L / nx, L / ny  
    dt = 0.0001                       # time step
    nits = int(T / dt)                # number of time steps
    du, dv = model.model.diff_coeffs  # diffusion rates

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
    conv = conv.to(device)

    # --- Neural network for reaction ---
    reaction = model.model.reaction.eval()

    # --- Storage ---
    half_sec_nits = int(0.5 / dt)
    half_secs = int(nits / half_sec_nits) + 1

    u_array = torch.zeros((half_secs, 2, nx, ny))
    t_array = torch.arange(0, T + 0.5, 0.5)

    # Track time point in storage array
    i = 0

    # --- Simulation loop ---
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
        with torch.no_grad():
            ruv = reaction(uv).view(nx, ny)
        
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