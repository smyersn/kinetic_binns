import sys, os
import importlib
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn_eql_hypernet.model_wrapper_2d import model_wrapper
from modules.binn_eql_hypernet.build_binn_eql_net import BINN
from modules.loaders.format_data import format_data_general
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split
from modules.generate_data.simulate_system import wave_pinning
from modules.symbolic_net.visualize_surface import visualize_surface
from modules.binn_eql_hypernet.simulate_surface import simulate_surface


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
l1_weight = float(config['l1_weight'])
param_bounds = float(config['param_bounds'])

dir_name = sys.argv[1]

# Set device
device = 'cpu'

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

model.load(f"{dir_name}/binn_best_val_model", device=device)
        
# Generate true reaction surface
params = 1, 1, 0.01
F_true = wave_pinning(uv_nans, params).reshape(501, 501)

# Generate learned reaction surface
F_mlp_unformatted = model.model.reaction(torch.tensor(uv_nans).float().to(device))
F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)

# Visualize surfaces
visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                F_true, F_mlp, 'f_mlp_surface_unrefined')

# Refine and print equation       
fn = f'{dir_name}/equation.txt'
file = open(fn, 'w')

file.write(f'Original Equation:\n')
for term in model.model.generate_equation():
    file.write(f'{term}\n')

file.write(f'\nNo insignificant terms:\n')
model.model.remove_insignificant_terms(uv)
for term in model.model.generate_equation():
    file.write(f'{term}\n')

file.write(f'\nNo cheating Hill functions:\n')
model.model.fix_cheating_hill_functions(uv)
for term in model.model.generate_equation():
    file.write(f'{term}\n')

if not diff_coeffs:
    # Write the variables to the file
    file.write(f"\nDiffusion coeffs = {model.model.diffusion_fitter()}")

file.close()

# Generate refined learned surface
F_mlp_unformatted_refined = model.model.reaction(torch.tensor(uv_nans).float().to(device))
F_mlp_refined = F_mlp_unformatted_refined.cpu().detach().numpy().reshape(501, 501)
    
# Visualize surfaces
visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                F_true, F_mlp_refined, 'f_mlp_surface_refined')

# Simulate
reaction = model.model.reaction.eql_layer

if not model.model.diff_coeffs:
    D = model.model.diffusion_fitter()
    Du = D[0].detach().numpy()
    Dv = D[1].detach().numpy()
    
else:
    Du, Dv = 0.01, 1

simulate_surface(training_data, 2, 2, reaction, Du, Dv, dir_name)
        
# analyze_model(model, dir_name, training_data, device)