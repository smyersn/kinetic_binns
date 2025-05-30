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
from modules.analysis.generate_loss_curves import generate_loss_curves

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
# epochs = int(1e6)
epochs = 40_000
rel_save_thresh = 0.01

# Get GPU
device = 'cuda'

# Load training data (columns: x*dimensions, t, species concentrations)
training_data = format_data_general(dimensions, species, file=training_data_path)

# Add noise to training data if specified in config file
if epsilon != 0 or points != 0:
    training_data = noise_and_interpolate(training_data, points, epsilon, dimensions, species)

animate_data(training_data, dimensions, species, name=f'{dir_name}/training_data')

# Split and normalize data
x_train_unnorm, y_train_unnorm, x_val_unnorm, y_val_unnorm = training_test_split(
    training_data, dimensions, device)

x_train, y_train, x_val, y_val, mu, sigma = normalize_sets(
    x_train_unnorm, y_train_unnorm, x_val_unnorm, y_val_unnorm)

# Get unnormalized extrema for PDE loss
x_min = torch.min(x_train_unnorm, dim=0)[0].cpu()
x_max = torch.max(x_train_unnorm, dim=0)[0].cpu()

# initialize model and compile
binn = BINN(
    dimensions=dimensions,
    species=species, 
    input_extrema=(x_min, x_max),
    mu=mu,
    sigma=sigma,
    duplicates=duplicates,
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

# train jointly
train_loss_dict, val_loss_dict = model.fit(
    x=x_train,
    y=y_train,
    batch_size=int(0.05*len(training_data)),
    epochs=epochs,
    callbacks=None,
    verbose=1,
    validation_data=[x_val, y_val],
    early_stopping=1000,
    rel_save_thresh=rel_save_thresh)

generate_loss_curves(train_loss_dict, val_loss_dict, dir_name)