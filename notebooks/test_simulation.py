import sys, os
import importlib
from IPython.display import HTML
# from torchdiffeq import odeint
repo_start = f'../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn_eql.model_wrapper_2d import model_wrapper
from modules.binn_eql.build_binn_eql_net import BINN
from modules.loaders.format_data import format_data_general
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split
from modules.generate_data.simulate_system import wave_pinning
from modules.symbolic_net.visualize_surface import visualize_surface
from modules.binn_eql.simulate_surface import simulate_surface
from modules.generate_data.simulate_system import generate_initial_conditions, simulate
from modules.loaders.visualize_training_data import animate_data

def animate_new_sim(u_array, t_array, name=None): 
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

# load params from configuration file
dir_name = '/work/users/s/m/smyersn/elston/projects/kinetics_binns/development/binn_eql_net/runs/debugging/23_longer_fine_tuning/binn_eql_gls_1_pde_1_repeat_12'
save_name = '12'
config = {}
exec(Path(f'{dir_name}/config.cfg').read_text(encoding="utf8"), {}, config)

# Training data params
training_data_path = config['training_data_path']
species = int(config['species'])
dimensions = int(config['dimensions'])
epsilon = float(config['epsilon'])
points = int(config['points'])

# BINN params
diff_coeffs = [float(x) for x in config['diff_coeffs'].strip("()").split()]

# Symbolic Net params
duplicates = int(config['duplicates'])
degree = int(config['degree'])
gls_weight=float(config['gls_weight'])
pde_weight=float(config['pde_weight'])
l05_weight = float(config['l05_weight'])
l1_weight = float(config['l1_weight'])
param_bounds = float(config['param_bounds'])

# Set training hyperparameters
epochs = int(1e6)
rel_save_thresh = 0.01
device = 'cuda'

# Load training data (columns: x*dimensions, t, species concentrations)
training_data = format_data_general(dimensions, species, file=training_data_path)

# Add noise to training data if specified in config file
if epsilon != 0 or points != 0:
    training_data = noise_and_interpolate(training_data, points, epsilon, dimensions, species)
    
# Create triangle mesh from min and max uv vals seen in training data
u_triangle_mesh, v_triangle_mesh = lltriangle(training_data[:, -2:], 
                                                training_data[:, -1:])
# Create 1d arrays from meshes
u_triangle, v_triangle = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)

# Create separate variables for arrays containing and not containing nans
uv_nans = np.stack((u_triangle, v_triangle), axis=1)
mask = ~np.isnan(uv_nans).any(axis=1)
uv = torch.from_numpy(uv_nans[mask]).to(device)

# Split training data
train_data, val_data = training_test_split(training_data, device)

# initialize model and compile
binn = BINN(
    dimensions=dimensions,
    species=species, 
    train_data=train_data, 
    duplicates=duplicates,
    diff_coeffs=diff_coeffs,
    degree=degree,
    gls_weight=gls_weight,
    pde_weight=pde_weight,
    l05_weight=l05_weight,
    param_bounds=param_bounds)

binn.to(device)

parameters = binn.parameters()

opt = torch.optim.Adam(parameters, lr=0.001)

model = model_wrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    dir_name=dir_name,
    save_name=f'{dir_name}/binn')

# model.load(f"{dir_name}/binn_best_val_model", device=device)
model.load(f"{dir_name}/binn_best_val_fine_tuned_model", device=device)

# --- Initial Conditions ---
xt = training_data[:, :dimensions+1]
points = len(np.unique(training_data[:, 0]))
ic = training_data[training_data[:, dimensions] == 0]

u = torch.tensor(np.reshape(ic[:, dimensions+1], (points,)*dimensions)).float().to(device)
v = torch.tensor(np.reshape(ic[:, dimensions+2], (points,)*dimensions)).float().to(device)

# --- Parameters ---
T = np.max(xt[:, dimensions])
L = np.max(xt[:, 0])

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
        
anim = animate_new_sim(u_array, t_array, save_name)
