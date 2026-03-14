import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from numba import njit, prange
from modules.utils.imports import *
from modules.simulation.animation import animate_u_array
from modules.simulation.reaction_functions import (wave_pinning, turing_type,
                                                   custom_equation)
from modules.utils.format_data import format_u_array_to_training_data

def simulate_reaction(reaction, params, diff_coeffs, early_stop=True):
    # --- Set Parameters --- 
    dim = 2
    species = 2
    N = 200                           # grid points
    L = 10                            # domain length
    ss_tolerance = 0.025
    device = 'cuda'
    
    T = 50
    dx = L / N
    dt = 0.0001                       # time step
    nits = int(T / dt)                # number of time steps
    du, dv = diff_coeffs                  # diffusion rates

    # --- Initial Conditions ---
    u0, v0 = 1, 1.0246
    u = ((torch.rand(*(N,) * dim) + 0.5) * u0).to(device)
    v = (torch.ones((N,) * dim) * v0).to(device)

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
    
    u_array = torch.zeros((half_secs, N, N, species))
    x_array = torch.linspace(0, L, steps=200)
    t_array = torch.arange(0, T + 0.5, 0.5)

    # Track how many half secs have passed
    i = 0

    # --- Simulation loop ---
    for t in range(nits+1):
        
        # Update storage every half second
        if t % half_sec_nits == 0:
            u_array[i] = torch.stack([u, v], dim=-1)
            
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
            
    return u_array.cpu(), x_array.cpu(), t_array.cpu()

@njit(parallel=True)
def fast_laplacian(mat, out_lap):
    """Computes the 5-point stencil Laplacian with periodic boundaries."""
    rows, cols = mat.shape
    for i in prange(rows):
        for j in range(cols):
            ip = (i + 1) % rows
            im = (i - 1) % rows
            jp = (j + 1) % cols
            jm = (j - 1) % cols
            out_lap[i, j] = mat[ip, j] + mat[im, j] + mat[i, jp] + mat[i, jm] - 4.0 * mat[i, j]

def simulate_reaction_cpu(reaction, params, diff_coeffs, early_stop=True):
    # --- Set Parameters (Matching original function) --- 
    dim = 2
    species = 2
    N = 200                           # grid points
    L = 10                            # domain length
    ss_tolerance = 0.025
    
    T = 50                            # total time
    dx = L / N
    dt = 0.0001                       # time step
    nits = int(T / dt)                # number of time steps
    du, dv = diff_coeffs              # diffusion rates
    inv_dx2 = 1.0 / (dx**2)

    # --- Initial Conditions ---
    u0, v0 = 1.0, 1.0246
    # np.random.rand returns [0, 1), so +0.5 perfectly matches torch.rand + 0.5
    u = (np.random.rand(N, N) + 0.5) * u0
    v = np.ones((N, N)) * v0

    # --- Storage ---
    half_sec_nits = int(0.5 / dt)
    half_secs = int(nits / half_sec_nits) + 1
    
    # Pre-allocate Animator-friendly shape: (Time, Y, X, Species)
    u_array = np.zeros((half_secs, N, N, species), dtype=np.float32)
    x_array = np.linspace(0, L, N, dtype=np.float32)
    t_array = np.arange(0, T + 0.5, 0.5, dtype=np.float32)

    # Pre-allocate arrays for Numba to avoid memory reallocation in the loop
    lap_u = np.zeros((N, N), dtype=np.float64)
    lap_v = np.zeros((N, N), dtype=np.float64)

    i = 0

    # --- Simulation loop ---
    for t in range(nits + 1):
        
        # Update storage every half second
        if t % half_sec_nits == 0:
            u_array[i, :, :, 0] = u
            u_array[i, :, :, 1] = v
            
            # Stop simulation if it reaches steady state
            if early_stop and t > 0:
                diff = np.max(np.abs(u_array[i] - u_array[i-1]))
                
                if diff < ss_tolerance:
                    print(f'Steady state at t = {t * dt:.2f}', flush=True)
                    # Trim excess zeros if we stopped early
                    u_array = u_array[:i+1]
                    t_array = t_array[:i+1]
                    break
            i += 1
            
        if t == nits:
            break # Reached the end, no need to step physics

        # Compute Laplacian using the Numba kernel
        fast_laplacian(u, lap_u)
        fast_laplacian(v, lap_v)

        # Reaction term
        uv = np.column_stack((u.flatten(), v.flatten()))
        
        # Call the user-provided reaction function
        ruv = reaction(uv, params)
        
        # Failsafe: If the reaction function happens to return a PyTorch tensor, 
        # instantly cast it back to a NumPy array so the CPU math doesn't break.
        if hasattr(ruv, 'detach'):
            ruv = ruv.detach().cpu().numpy()
            
        ruv = ruv.reshape(N, N)
        
        # Euler update
        u = u + dt * (du * lap_u * inv_dx2 + ruv)
        v = v + dt * (dv * lap_v * inv_dx2 - ruv)        

        if t % (nits // 10) == 0 and t > 0:
            print(f"Progress: {(t / nits) * 100:.0f}%", flush=True)  
            
    return u_array, x_array, t_array

def simulate_uvmlp(training_data, model):
    # --- Initial Conditions ---
    dimensions = model.model.dimensions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    xt = training_data[:, :dimensions+1]
    t_array = torch.unique(xt[:, dimensions])
    x_array = torch.unique(xt[:, 0])
    
    grid_x, grid_y = torch.meshgrid(x_array, x_array, indexing='xy')
    spatial_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # --- Storage ---
    u_array = torch.zeros((len(t_array), len(x_array), len(x_array), 2))

    # --- Simulation loop ---
    for n, t in enumerate(t_array):
        # We create a tensor of shape (40000, 1) filled with the scalar t
        t_column = torch.full((spatial_coords.shape[0], 1), t)
        
        # [x, y] + [t] -> [x, y, t]
        input_tensor = torch.cat([spatial_coords, t_column], dim=1).to(device)

        # Calculate surface
        with torch.no_grad():
            uv = model.model(input_tensor).view(len(x_array), len(x_array), 2)
            
        u_array[n] = uv

    return u_array.cpu(), x_array.cpu(), t_array.cpu()

def simulate_feql(training_data, model):
    # --- Initial Conditions ---
    dimensions = model.model.dimensions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Get Model Scale
    # Shape: (1, 2) -> [[s_u_max, s_v_max]]
    max_scale = model.model.max_scale.to(device)
    
    xt = training_data[:, :dimensions+1]
    t_array = torch.unique(xt[:, dimensions])
    x_array = torch.unique(xt[:, 0])
    points = len(x_array)
    
    ic = training_data[training_data[:, dimensions] == 0]
    
    u0 = ic[:, dimensions + 1]
    v0 = ic[:, dimensions + 2]

    # stack u and v: Shape becomes (1, 2, H, W)
    grid_shape = [points] * dimensions
    uv_grid = torch.stack([u0, v0], dim=0).reshape(1, 2, *grid_shape).float().to(device)

    # --- Parameters ---
    T = float(xt[:, dimensions].max().item())
    L = float(xt[:, 0].max().item())

    nx, ny = points, points
    dx = L / nx
    dt = 0.0001
    nits = int(T / dt)
    
    # Get diffusion coefficients
    if model.model.diff_coeffs:
        du, dv = model.model.diff_coeffs
    else:        
        with torch.no_grad():
            D_vals = model.model.diffusion_fitter()
            du, dv = float(D_vals[0]), float(D_vals[1])
            
    # Create a tensor for diffusion coeffs to broadcast: shape (1, 2, 1, 1)
    D_tensor = torch.tensor([du, dv], device=device).view(1, 2, 1, 1)

    # --- Laplacian kernel (Grouped Conv2d) ---
    laplace_kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32, device=device)
    
    # Reshape for Conv2d: (Out=2, In/Groups=1, K=3, K=3)
    weights = laplace_kernel.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)

    conv = nn.Conv2d(
        in_channels=2,   # u and v
        out_channels=2,  # u and v
        kernel_size=3,
        padding=1,
        groups=2,        # Independent convolution for each channel
        padding_mode='circular',
        bias=False)

    conv.weight.data = weights
    conv.weight.requires_grad = False
    conv = conv.to(device)

    # --- Neural network for reaction ---
    reaction = model.model.reaction.eval()

    # --- Storage ---
    half_sec_nits = int(0.5 / dt)
    u_array = torch.zeros((len(t_array), nx, ny, 2), device=device)
    storage_idx = 0

    @torch.compile 
    def physics_step(current_state):
        # 1. Diffusion
        lap_uv = conv(current_state) / (dx**2)
        
        # 2. Reaction 
        # Permute to (Batch, H, W, Channels) -> Flatten
        state_permuted = current_state.permute(0, 2, 3, 1).contiguous()
        uv_flat_phys = state_permuted.view(-1, 2)
        r_flat_phys = reaction(uv_flat_phys) 
                
        # Reshape R back to grid: (1, 1, H, W)
        r_grid = r_flat_phys.view(1, 1, nx, ny) 
        
        # 3. Construct Reaction Update: [ +R, -R ]
        reaction_term = torch.cat([r_grid, -r_grid], dim=1) 
        
        # Euler Step
        new_state = current_state + dt * (D_tensor * lap_uv + reaction_term)
        return new_state      
    
    # --- Progress Tracking Variables ---
    print_interval = nits // 10  # 10%
    last_time = time.time()
    
    # Ensure no gradients are tracked for the loop (saves memory/speed)
    with torch.no_grad():
        for t in range(nits):
            
            # Save state
            if t % half_sec_nits == 0 and storage_idx < len(t_array):
                # Permute to (H, W, 2) for storage
                u_array[storage_idx] = uv_grid.squeeze(0).permute(1, 2, 0)
                storage_idx += 1

            # Run Physics
            uv_grid = physics_step(uv_grid)

            # Print Progress and Time per 5% segment
            if t > 0 and t % print_interval == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                last_time = current_time
                
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                print(f"Progress: {(t / nits) * 100:.0f}% | Segment took: {mins}m {secs}s", flush=True)
            elif t == 0:
                print("Progress: 0%", flush=True)

    return u_array.cpu(), x_array.cpu(), t_array.cpu()

# if __name__ == '__main__':        
#     # Load parameters
#     save_path = str(sys.argv[1])
#     Du, Dv = float(sys.argv[2]), float(sys.argv[3])
#     diff_coeffs = (Du, Dv)
#     params = [float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])]
#     reaction_fn = wave_pinning
    
#     # Create save name
#     save_name = f'{save_path}/du_{Du}_dv_{Dv}_a_{params[0]}_b_{params[1]}_k_{params[2]}'
    
#     # Simulate and animate
#     u_array, x_array, t_array = simulate_reaction(reaction_fn, params, diff_coeffs, early_stop=True)
#     animate_u_array(u_array, t_array, name=f'{save_name}.gif', titles=("u", "v"))
    
#     # Reformat to training data and save
#     training_data = format_u_array_to_training_data(u_array, x_array, t_array)
#     torch.save({'training_data': training_data}, f'{save_name}.pt')

if __name__ == '__main__':        
    import json
    
    # 1. Load the single JSON config argument
    config_path = str(sys.argv[1])
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # 2. Extract parameters from the dictionary
    save_path = config["save_path"]
    Du, Dv = float(config["du"]), float(config["dv"])
    diff_coeffs = (Du, Dv)
    
    a, b, k = float(config["a"]), float(config["b"]), float(config["k"])
    params = [a, b, k]
    
    # 3. Dynamically map the reaction string to the imported function
    reaction_str = config["reaction"]
    if reaction_str == "wave_pinning":
        reaction_fn = wave_pinning
    elif reaction_str == "turing_type":
        reaction_fn = turing_type
    elif reaction_str == "custom_equation":
        reaction_fn = custom_equation
    else:
        raise ValueError(f"Unknown reaction function specified: {reaction_str}")
    
    # 4. Create save name (matching your original format)
    save_name = f'{save_path}/du_{Du}_dv_{Dv}_a_{a}_b_{b}_k_{k}'
    
    print(f"Starting simulation for: {save_name}", flush=True)
    
    # 5. Simulate and animate
    u_array, x_array, t_array = simulate_reaction_cpu(reaction_fn, params, diff_coeffs, early_stop=False)
    animate_u_array(u_array, t_array, name=f'{save_name}.gif', titles=("u", "v"))
    
    # 6. Reformat to training data and save
    training_data = format_u_array_to_training_data(u_array, x_array, t_array)
    torch.save({'training_data': training_data}, f'{save_name}.pt')
    
    print(f"Successfully saved data to {save_name}.pt", flush=True)    