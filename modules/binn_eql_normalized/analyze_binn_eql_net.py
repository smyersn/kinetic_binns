import sys, os
import importlib
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn_eql_normalized.model_wrapper_2d import model_wrapper
from modules.binn_eql_normalized.build_binn_eql_net import BINN
from modules.binn_eql_normalized.analyze_model import analyze_model
from modules.loaders.format_data import format_data_general
from modules.loaders.visualize_training_data import animate_data
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split
from modules.utils.normalization import normalize_sets

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
l05_reg = float(config['l05_reg'])
l1_reg = float(config['l1_reg'])
param_bounds = float(config['param_bounds'])

dir_name = sys.argv[1]

# Set training hyperparameters
epochs = int(1e6)
rel_save_thresh = 0.01

# Get GPU
device = 'cpu'

# Load training data (columns: x*dimensions, t, species concentrations)
training_data = format_data_general(dimensions, species, file=training_data_path)

# Add noise to training data if specified in config file
if epsilon != 0 or points != 0:
    training_data = noise_and_interpolate(training_data, points, epsilon, dimensions, species)

animate_data(training_data, dimensions, species, name=f'{dir_name}/training_data')

# initialize model and compile
binn = BINN(
    dimensions=dimensions,
    species=species, 
    duplicates=duplicates,
    mu=torch.zeros((dimensions+species+1)).float(),
    sigma=torch.zeros((dimensions+species+1)).float(),
    diff_coeffs=diff_coeffs,
    degree=degree,
    l05_reg=l05_reg,
    l1_reg=l1_reg,
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
        
# Analysis
model.load(f"{dir_name}/binn_best_val_model", device=device)

if not diff_coeffs:
    # Create file to save diffusion coefficients
    with open(f'{dir_name}/diffusion.txt', 'w') as file:
        # Write the variables to the file
        file.write(f"Diffusion coeffs = {model.model.diffusion_fitter()}")
        
analyze_model(model, dir_name, training_data, device)