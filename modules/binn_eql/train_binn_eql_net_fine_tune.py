import sys, os, re
import importlib
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn_eql.model_wrapper_2d import model_wrapper
from modules.binn_eql.build_binn_eql_net import BINN
from modules.loaders.format_data import format_data_general, format_data_torch
from modules.loaders.visualize_training_data import animate_data
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split
from modules.analysis.generate_loss_curves import generate_loss_curves
from modules.binn_eql.visualize_surface import visualize_surface
from modules.binn_eql.simulate_surface import simulate_surface, animate_sim
from modules.generate_data.simulate_system import wave_pinning

# load params from configuration file
config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

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
param_bounds = float(config['param_bounds'])
prune_thresh = float(config['prune_thresh'])

dir_name = sys.argv[1]

# Set training hyperparameters
epochs = int(1e6)
# epochs = 350
rel_save_thresh = 0.01

# Get GPU
device = 'cuda'

# Load training data (columns: x*dimensions, t, species concentrations)
data = torch.load(training_data_path)
u_array, x_array, t_array = data['u_array'], data['x_array'], data['t_array']
training_data = format_data_torch(u_array, x_array, t_array)

# Add noise to training data if specified in config file
if epsilon != 0 or points != 0:
    training_data = noise_and_interpolate(training_data, points, epsilon, 
                                          dimensions, species, 
                                          multiplicative_noise=False)

animate_data(training_data, dimensions, species, name=f'{dir_name}/training_data')



########
# DO TRIANGLE BS IN PYTORCH
########




# Create triangle mesh from min and max uv vals seen in training data
u_triangle_mesh, v_triangle_mesh = lltriangle(to_numpy(training_data[:, -2:]), 
                                                to_numpy(training_data[:, -1:]))
# Create 1d arrays from meshes
u_triangle, v_triangle = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)

# Create separate variables for arrays containing and not containing nans
uv_nans = np.stack((u_triangle, v_triangle), axis=1)
mask = ~np.isnan(uv_nans).any(axis=1)
uv = torch.from_numpy(uv_nans[mask]).to(device)

# Get params for calculating true surface
filename = os.path.basename(training_data_path)
numbers = re.findall(r'(?<=_)(?:\d*\.\d+|\d+)', filename)

if len(numbers) == 3:
    params = list(map(float, numbers))
else:
    params = 1, 1, 0.01
    
# Generate true surface
F_true = wave_pinning(uv_nans, params).reshape(501, 501)

# Split training data
batch_size=int(0.1*len(training_data))
# train_loader, val_loader = training_test_split(training_data, batch_size, species)
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

# train jointly
train_loss_dict, val_loss_dict = model.fit(
    train_data=train_data,
    val_data=val_data,
    prune_thresh=prune_thresh,
    batch_size=batch_size,
    epochs=epochs,
    early_stopping=2500,
    rel_save_thresh=rel_save_thresh)

generate_loss_curves(train_loss_dict, val_loss_dict, dir_name, 20, 'training_loss_curves')

# Load and prune final model
model.load(f"{dir_name}/binn_best_val_model", device=device)

# Print final equation
fn = f'{dir_name}/equation.txt'
file = open(fn, 'a')

file.write(f'Final equation:\n')
for term in model.model.generate_equation():
    file.write(f'{term}\n')

if not diff_coeffs:
    file.write(f'\nDiff. coeffs. after fine tuning:\n')
    
    file.write(f'{[D.item() for D in model.model.diffusion_fitter()]}\n')
        
file.close()

# Generate learned learned surface after initial training
F_mlp_unformatted = model.model.reaction(torch.tensor(uv_nans).float().to(device))
F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)

# Visualize surfaces
visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                F_true, F_mlp, 'f_mlp_surfaces')

# Simulate surface
u_array, t_array = simulate_surface(training_data, model)
animate_sim(u_array, t_array, f'{dir_name}/f_mlp_animation_training_data_ic')