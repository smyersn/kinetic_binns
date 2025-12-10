import sys, importlib, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.utils.numpy_torch_conversion import *

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
            # 1. NORMALIZE INPUTS (Using BINN helper)
            input_norm = model.model.scale_inputs(input_tensor)
            
            # 2. PREDICT (Returns Dimensionless [0, 1])
            uv_hat = model.model.surface_fitter(input_norm)
            
            # 3. UNSCALE OUTPUT (To Physical)
            uv_phys = uv_hat * model.model.max_scale
            
            # Reshape for storage
            uv = uv_phys.view(len(points), len(points), 2)
            
        u_array[n] = uv

    return u_array, times 

####
# OG BUT SLOW
####
    
# def simulate_feql(training_data, model):
#     # --- Initial Conditions ---
#     dimensions = model.model.dimensions
#     device = 'cuda'
    
#     xt = training_data[:, :dimensions+1]
#     times = torch.unique(xt[:, dimensions])
#     points = torch.unique(xt[:, 0])
#     ic = training_data[training_data[:, dimensions] == 0]
    
#     u0 = ic[:, dimensions + 1]                      # shape: (points**dimensions,)
#     v0 = ic[:, dimensions + 2]

#     grid_shape = [len(points)] * dimensions
#     u = u0.reshape(*grid_shape).float().to(device)  # (points, points, ..., points)
#     v = v0.reshape(*grid_shape).float().to(device)

#     # --- Parameters ---
#     T = float(xt[:, dimensions].max().item())       # max time
#     L = float(xt[:, 0].max().item())                # max x-coordinate

#     nx, ny = u.shape                 # grid size
#     dx, dy = L / nx, L / ny  
#     dt = 0.0001                       # time step
#     nits = int(T / dt)                # number of time steps
    
#     if model.model.diff_coeffs:
#         du, dv = model.model.diff_coeffs  # diffusion rates
#     else:        
#         with torch.no_grad():
#             du, dv = model.model.diffusion_fitter()

#     # --- Laplacian kernel (5-point stencil) ---
#     laplace_kernel = torch.tensor([[0, 1, 0],
#                                 [1, -4, 1],
#                                 [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

#     # conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)

#     conv = nn.Conv2d(
#         in_channels=1,
#         out_channels=1,
#         kernel_size=3,
#         padding=1,
#         padding_mode='circular',
#         bias=False)

#     conv.weight.data = laplace_kernel
#     conv.weight.requires_grad = False
#     conv = conv.to(device)

#     # --- Neural network for reaction ---
#     reaction = model.model.reaction.eval()

#     # --- Storage ---
#     half_sec_nits = int(0.5 / dt)
#     half_secs = int(nits / half_sec_nits) + 1

#     u_array = torch.zeros((len(times), len(points), len(points), 2))

#     # Track time point in storage array
#     i = 0

#     # --- Simulation loop ---
#     for t in range(nits):
        
#         # Update storage every half second
#         if t % half_sec_nits == 0:
#             u_array[i] = torch.stack([u, v], dim=-1)
#             i += 1

#         # Compute Laplacian (diffusion)
#         lap_u = conv(u[None, None, :, :]).squeeze() / dx**2
#         lap_v = conv(v[None, None, :, :]).squeeze() / dx**2

#         # Reaction term
#         uv = torch.column_stack((u.flatten(), v.flatten()))
#         with torch.no_grad():
#             ruv = reaction(uv).view(nx, ny)
        
#         # Euler update
#         u = u + dt * (du * lap_u + ruv)
#         v = v + dt * (dv * lap_v - ruv)        

#         if t % (nits // 10) == 0:
#             print(f"Progress: {(t / nits) * 100}%", flush=True)  
            
#     return u_array, times 


####
# NEW ATTEMPT 1
####

def simulate_feql(training_data, model):
    # --- Initial Conditions ---
    dimensions = model.model.dimensions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    xt = training_data[:, :dimensions+1]
    times = torch.unique(xt[:, dimensions])   
    points_raw = torch.unique(xt[:, 0])
    points_len = len(points_raw)
    
    ic = training_data[training_data[:, dimensions] == 0]
    
    u0 = ic[:, dimensions + 1]
    v0 = ic[:, dimensions + 2]

    # stack u and v: Shape becomes (1, 2, H, W)
    grid_shape = [points_len] * dimensions
    uv_grid = torch.stack([u0, v0], dim=0).reshape(1, 2, *grid_shape).float().to(device)

    # --- Parameters ---
    T = float(xt[:, dimensions].max().item())
    # T = 10
    L = float(xt[:, 0].max().item())

    nx, ny = points_len, points_len
    dx = L / nx
    dt = 0.0001
    nits = int(T / dt)
    
    # Get Scales from Model (Buffers)
    u_scale = model.model.max_scale # Shape [1, 2]
    
    # Get diffusion coefficients
    if model.model.diff_coeffs:
        du, dv = model.model.diff_coeffs
    else:        
        with torch.no_grad():
            du, dv = model.model.diffusion_fitter()
            
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
    u_array = torch.zeros((len(times), nx, ny, 2), device=device)
    storage_idx = 0

    # This fuses the physics operations into fewer kernels
    @torch.compile 
    def physics_step(current_state):
        # 1. Diffusion (Physical Units)
        # current_state is Physical. laplace kernel is unitless. dx is Physical.
        # So lap_uv is Physical concentration / length^2.
        lap_uv = conv(current_state) / (dx**2)
        
        # 2. Reaction 
        state_permuted = current_state.permute(0, 2, 3, 1).contiguous()
        uv_flat_phys = state_permuted.view(-1, 2) # Physical Units
        
        # --- SCALING FIX START ---
        # A. Normalize Input: Physical -> [0, 1]
        uv_flat_norm = uv_flat_phys / u_scale
        
        # B. Run Network: [0, 1] -> Dimensionless Rate
        r_flat_hat = reaction(uv_flat_norm) 
        
        # C. Scale Output: Dimensionless Rate -> Physical Rate
        # Recall: F_phys = F_hat * s_u_max
        r_flat_phys = r_flat_hat * u_scale[0, 0] 
        # --- SCALING FIX END ---
        
        # Reshape R back to grid
        r_grid = r_flat_phys.view(1, 1, nx, ny) 
        
        # 3. Construct Reaction Update
        reaction_term = torch.cat([r_grid, -r_grid], dim=1) 
        
        # Euler Step (All Physical)
        new_state = current_state + dt * (D_tensor * lap_uv + reaction_term)
        return new_state
        
    # --- Progress Tracking Variables ---
    print_interval = nits // 10  # 10%
    last_time = time.time()
    
    # Ensure no gradients are tracked for the loop (saves memory/speed)
    with torch.no_grad():
        for t in range(nits):
            
            # Save state
            if t % half_sec_nits == 0 and storage_idx < len(times):
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

    return u_array.cpu(), times  

def format_training_data_for_animation(training_data):
    # 1. Extract raw columns
    x_raw = training_data[:, 0]
    y_raw = training_data[:, 1]
    t_raw = training_data[:, 2]
    
    # 2. Identify Unique Dimensions
    x_vals = np.unique(x_raw)
    y_vals = np.unique(y_raw)
    times = np.unique(t_raw)
    
    N_x = len(x_vals) # Width
    N_y = len(y_vals) # Height
    N_t = len(times) # Time
    
    # 3. Sort Data
    sort_indices = np.lexsort((x_raw, y_raw, t_raw))
    data_sorted = training_data[sort_indices]

    # 4. Extract and Reshape Species
    u_shaped = data_sorted[:, 3].reshape(N_t, N_y, N_x)
    v_shaped = data_sorted[:, 4].reshape(N_t, N_y, N_x)

    # 5. Stack Channels
    u_array = np.stack([u_shaped, v_shaped], axis=-1)
    
    return u_array, times

def animate_uarray(u_array, times, name=None, titles=("u", "v")):
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

def animate_residuals(pred_array, true_array, times, name=None, 
                      titles=("Pred u - True u", "Pred v - True v")):
    """
    Plots the difference (Pred - True) side-by-side for u and v.
    Assumes inputs are (Time, Height, Width, 2).
    Red: Model > Truth (Overestimation).
    Blue: Model < Truth (Underestimation).
    """
    # 1. Shape Handling: Ensure (Time, Height, Width, Channel)
    # Check if inputs are PyTorch Tensors (CPU or GPU) and convert
    if hasattr(pred_array, 'detach'): 
        pred_array = pred_array.detach().cpu().numpy()
    
    if hasattr(true_array, 'detach'): 
        true_array = true_array.detach().cpu().numpy()

    if pred_array.ndim == 4 and pred_array.shape[1] == 2: 
        pred_array = np.transpose(pred_array, (0, 2, 3, 1))
        
    if true_array.ndim == 4 and true_array.shape[1] == 2:
        true_array = np.transpose(true_array, (0, 2, 3, 1))
        
    # 2. Compute Residuals (Predicted - Ground Truth)
    residuals = pred_array - true_array
    
    # 3. Determine Scale (Symmetric for diverging colormap)
    # We define the range as [-max_error, +max_error] so 0 is always white/centered
    # Use slicing [..., 0] for u channel and [..., 1] for v channel
    max_err_u = np.max(np.abs(residuals[..., 0]))
    max_err_v = np.max(np.abs(residuals[..., 1]))
    
    # Avoid div by zero if perfect match
    max_err_u = max(max_err_u, 1e-6)
    max_err_v = max(max_err_v, 1e-6)

    # 4. Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Residual U (Channel 0)
    im1 = axes[0].imshow(residuals[0, :, :, 0], cmap='seismic', vmin=-max_err_u, vmax=max_err_u)
    axes[0].set_title(f"{titles[0]}\n(Range: +/- {max_err_u:.3f})")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot 2: Residual V (Channel 1)
    im2 = axes[1].imshow(residuals[0, :, :, 1], cmap='seismic', vmin=-max_err_v, vmax=max_err_v)
    axes[1].set_title(f"{titles[1]}\n(Range: +/- {max_err_v:.3f})")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Shared Main Title
    main_title = fig.suptitle(f'T = {times[0]:.2f}', fontsize=16)

    # 5. Define Update Function
    def animate(frame):
        im1.set_array(residuals[frame, :, :, 0])
        im2.set_array(residuals[frame, :, :, 1])
        main_title.set_text(f'Residuals at T = {times[frame]:.2f}')
        return im1, im2, main_title

    # 6. Create Animation
    anim = animation.FuncAnimation(fig, animate, frames=len(times), interval=200)
    
    # Save if name provided
    if name:
        writergif = animation.PillowWriter(fps=10)
        anim.save(f'{name}.gif', writer=writergif)
        
    return anim