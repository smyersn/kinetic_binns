import sys, importlib, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

import matplotlib.animation as animation
from modules.utils.imports import *
from modules.utils.numpy_torch_conversion import *
from modules.loaders.format_data import format_data_general
from modules.loaders.visualize_training_data import animate_data
from modules.generate_data.simulate_system import wave_pinning, traveling_wave
    
def simulate_surface(reaction, params=(1, 1, 0.01), diff_coeffs=(0.01, 1), early_stop=True):
    # --- Set Parameters --- 
    dim = 2
    species = 2
    N = 200                           # grid points
    L = 10                            # domain length
    ss_tolerance = 0.025
    device = 'cuda'
    
    T = 100
    dx = L / N
    dt = 0.0001                       # time step
    nits = int(T / dt)                # number of time steps
    du, dv = diff_coeffs                  # diffusion rates

    # --- Initial Conditions ---
    u0, v0 = 1, 1.0246
    u = ((torch.rand(*(N,) * dim) + 0.5) * u0).to(device)
    v = (torch.ones((N,) * dim) * v0).to(device)
    # u = (torch.rand(*(N,) * dim) * 3).to(device)
    # v = (torch.rand(*(N,) * dim) * 3).to(device)

    # --- Laplacian kernel (5-point stencil) ---
    laplace_kernel = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

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

    # --- Storage ---
    half_sec_nits = int(0.5 / dt)
    half_secs = int(nits / half_sec_nits) + 1
    
    u_array = torch.zeros((half_secs, species, N, N))
    x_array = torch.linspace(0, L, steps=200)
    t_array = torch.arange(0, T + 0.5, 0.5)

    # Track how many half secs have passed
    i = 0

    # --- Simulation loop ---
    for t in range(nits+1):
        
        # Update storage every half second
        if t % half_sec_nits == 0:
            u_array[i] = torch.stack([u, v], dim=0)
            
            # Stop simulation if it reaches steady state
            if early_stop and t > 0:
                current_save = u_array[i]
                last_save = u_array[i-1]
                diff = torch.max(torch.abs(current_save - last_save))
                
                if diff < ss_tolerance:
                    print(f'Steady state at t = {t * dt}', flush=True)
                    break
            # Increase half sec counter    
            i += 1
            
        # Compute Laplacian (diffusion)
        lap_u = conv(u[None, None, :, :]).squeeze() / dx**2
        lap_v = conv(v[None, None, :, :]).squeeze() / dx**2

        # Reaction term
        uv = torch.column_stack((u.flatten(), v.flatten()))
        ruv = reaction(uv, params).view(N, N)
        
        # Euler update
        u = u + dt * (du * lap_u + ruv)
        v = v + dt * (dv * lap_v - ruv)        

        if t % (nits // 10) == 0:
            print(f"Progress: {(t / nits) * 100}%", flush=True)  
            
    return u_array, x_array, t_array   
            
def animate_new_sim(u_array, t_array, save_name=None): 
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
    
    if save_name:
        writergif = animation.PillowWriter(fps=5)
        anim.save(save_name, writer=writergif)

    return anim

if __name__ == '__main__':  
    a = float(sys.argv[1]) 
    diff_coeffs = [float(sys.argv[2]), float(sys.argv[3])]

    params = [0.1, 1, 0.6, 0.01]

    # save_name = f'/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/pos_feedback_new_ic/a_{a}_b_1_k_0.01'
    save_name = f'/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/traveling_wave/a_0.1_b_1_c_0.6_k_0.01' 
    u_array, x_array, t_array = simulate_surface(traveling_wave, params, diff_coeffs)
    torch.save({'u_array': u_array, 'x_array': x_array, 't_array': t_array}, f'{save_name}.pt')
    
    anim = animate_new_sim(u_array, t_array, f'{save_name}.gif')