import torch
print(torch.__version__)           # Should end with +cu118
print(torch.cuda.is_available())   # Should be True (on a GPU node)

import torchdiffeq
import sys, os
import time
import importlib
from IPython.display import HTML
from torchdiffeq import odeint
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

# load params from configuration file
dir_name = '/work/users/s/m/smyersn/elston/projects/kinetics_binns/development/binn_eql_net/runs/debugging/17_fine_tuning/binn_eql_gls_1_pde_1_l05_0.01_repeat_33'
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

# Get GPU
# device = 'cpu'
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
x_train, y_train, x_val, y_val = training_test_split(training_data, dimensions, device)

# initialize model and compile
binn = BINN(
    dimensions=dimensions,
    species=species, 
    duplicates=duplicates,
    data=x_train.cpu(), 
    diff_coeffs=diff_coeffs,
    degree=degree,
    gls_weight=gls_weight,
    pde_weight=pde_weight,
    l05_weight=l05_weight,
    l1_weight=l1_weight,
    param_bounds=param_bounds)

binn.to(device)

parameters = binn.parameters()

opt = torch.optim.Adam(parameters, lr=0.001)

model = model_wrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    augmentation=None,
    save_name=f'{dir_name}/binn')

model.load(f"{dir_name}/binn_best_val_fine_tuned_model", device=device)
model.model.remove_insignificant_terms(uv)
model.model.fix_cheating_hill_functions(uv)

# Format initial conditions
ic = training_data[training_data[:, dimensions] == 0]

x_vals = ic[:, 0]
y_vals = ic[:, 1]
u_vals = ic[:, 3]
v_vals = ic[:, 4]

# Identify unique sorted grid points
x_unique = np.sort(np.unique(x_vals))
y_unique = np.sort(np.unique(y_vals))

H, W = len(y_unique), len(x_unique)  # H: rows (y), W: cols (x)

# Build mapping from (x, y) to grid indices
x_to_idx = {x: i for i, x in enumerate(x_unique)}
y_to_idx = {y: i for i, y in enumerate(y_unique)}

# Initialize empty grid tensors
u_grid = np.zeros((H, W))
v_grid = np.zeros((H, W))

# Fill in the grid
for row in ic:
    x, y, _, u, v = row
    i, j = y_to_idx[y], x_to_idx[x]
    u_grid[i, j] = u
    v_grid[i, j] = v

# Stack channels and add batch dimension
u0 = np.stack([u_grid, v_grid], axis=0)         # shape: (2, H, W)
u0 = torch.tensor(u0, dtype=torch.float32).unsqueeze(0)  # shape: (1, 2, H, W)

# Calculate dx
x_vals = np.sort(np.unique(ic[:, 0]))
dx = np.min(np.diff(x_vals))

class RDESystem(torch.nn.Module):
    def __init__(self, reaction_model, diff_coeffs, dx):
        super().__init__()
        self.reaction = reaction_model
        self.diff_coeffs = torch.tensor(diff_coeffs, device=u0.device).view(1, 2, 1, 1)
        self.dx2 = dx ** 2

    def forward(self, t, u):
        # --- Diffusion ---
        lap_kernel = torch.tensor([[[[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]]]], dtype=torch.float32)
        lap_kernel = lap_kernel.expand(2, 1, 3, 3)  # (out_channels, in_channels/groups, H, W)

        laplace_u = self.diff_coeffs * (torch.nn.functional.conv2d(u, lap_kernel, padding=1, groups=2) / self.dx2)

        # --- Reaction ---
        u_flat = u0.squeeze(0).permute(1, 2, 0).reshape(-1, 2)  # shape: (H*W, 2)
        reaction_u_flat = self.reaction(u_flat)  # shape: (H*W, 1)
        reaction_u = reaction_u_flat.reshape(H, W, 1).permute(2, 0, 1).unsqueeze(0)  # shape: (1, 1, H, W)
        reaction_u = torch.cat([reaction_u, -reaction_u], dim=1)  # shape: (1, 2, H, W)
class RDESystem(torch.nn.Module):
    def __init__(self, reaction_model, diff_coeffs, dx):
        super().__init__()
        self.reaction = reaction_model
        self.diff_coeffs = torch.tensor(diff_coeffs, device=u0.device).view(1, 2, 1, 1)
        self.dx2 = dx ** 2

    def forward(self, t, u):
        # --- Diffusion ---
        lap_kernel = torch.tensor([[[[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]]]], dtype=torch.float32)
        lap_kernel = lap_kernel.expand(2, 1, 3, 3)  # (out_channels, in_channels/groups, H, W)

        laplace_u = self.diff_coeffs * (torch.nn.functional.conv2d(u, lap_kernel, padding=1, groups=2) / self.dx2)

        # --- Reaction ---
        u_flat = u0.squeeze(0).permute(1, 2, 0).reshape(-1, 2)  # shape: (H*W, 2)
        reaction_u_flat = self.reaction(u_flat)  # shape: (H*W, 1)
        reaction_u = reaction_u_flat.reshape(H, W, 1).permute(2, 0, 1).unsqueeze(0)  # shape: (1, 1, H, W)
        reaction_u = torch.cat([reaction_u, -reaction_u], dim=1)  # shape: (1, 2, H, W)

        return laplace_u + reaction_u
    
reaction = model.model.reaction
diff_coeffs = model.model.diff_coeffs

T = 10
dt = 0.0001
t = torch.arange(0, T + dt, dt)

start = time.time()

solution = odeint(RDESystem(reaction, diff_coeffs, dx), u0, t, method='rk4')

end = time.time()

print(f"Elapsed time: {((end - start)/60):.4f} minutes")