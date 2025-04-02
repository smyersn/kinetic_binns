import sys, os
import importlib
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn.model_wrapper_2d import model_wrapper
from modules.binn.build_binns import BINN
from modules.loaders.format_data import format_data_general
from modules.loaders.visualize_training_data import animate_data
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split
from modules.analysis.generate_loss_curves import generate_loss_curves
import modules.generate_data.simulate_system

# load params from configuration file
config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

training_data_path = config['training_data_path']

reaction = getattr(modules.generate_data.simulate_system,
                   config['reaction'])
params = [float(param) for param in config['params'].replace(',', ' ').split()]

dimensions = int(config['dimensions'])
species = int(config['species'])

density_weight = int(config['density_weight'])

uv_layers = int(config['uv_layers'])
uv_neurons = int(config['uv_neurons'])
f_layers = int(config['f_layers'])
f_neurons = int(config['f_neurons'])

epsilon = float(config['epsilon'])
points = int(config['points'])

diffusion = bool(config['diffusion'])

dir_name = sys.argv[1]

uv_arch = uv_layers * [uv_neurons] + [2]
f_arch = f_layers * [f_neurons] + [1]

# Set training hyperparameters
epochs = int(1e6)
# epochs = 100
rel_save_thresh = 0.05

# Get GPU
device = 'cuda'

# Load training data (columns: x*dimensions, t, species concentrations)
training_data = format_data_general(dimensions, species, file=training_data_path)

# Add noise to training data if specified in config file
if epsilon != 0 or points != 0:
    training_data = noise_and_interpolate(training_data, points, epsilon, dimensions, species)

animate_data(training_data, dimensions, species, name=f'{dir_name}/training_data')

# Split training data

x_train, y_train, x_val, y_val = training_test_split(training_data, dimensions, device)

# initialize model and compile
binn = BINN(
    data=x_train.cpu(), 
    species=species, 
    dimensions=dimensions,
    uv_arch=uv_arch, 
    f_arch=f_arch,
    diff_coeffs=params[:2],
    diff=diffusion)

binn.to(device)

parameters = binn.parameters()

opt = torch.optim.Adam(parameters, lr=0.001)

model = model_wrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    augmentation=None,
    save_name=f'{dir_name}/binn')

# generate histogram for trainind data density if specified in config file
hist = None
edges = None

if density_weight != 0:
    u = y_train[:, 0].flatten()
    v = y_train[:, 1].flatten()

    hist_uncorrected = torchist.normalize(torchist.histogramdd(y_train, bins=10, 
                                low=[min(u).item(), min(v).item()], 
                                upp=[max(u).item(), max(v).item()]))[0]

    # change zero values in histogram so reciprocal density is not inf
    hist = torch.where(hist_uncorrected == 0, 0.001, hist_uncorrected)

    edges = torchist.histogramdd_edges(y_train, bins=10, 
                                    low=[min(u).item(), min(v).item()], 
                                    upp=[max(u).item(), max(v).item()])
    edges[0] = edges[0].to(device)
    edges[1] = edges[1].to(device)

# train jointly
train_loss_dict, val_loss_dict = model.fit(
    x=x_train,
    y=y_train,
    batch_size=int(0.05*len(training_data)),
    epochs=epochs,
    callbacks=None,
    verbose=1,
    validation_data=[x_val, y_val],
    early_stopping=10000,
    rel_save_thresh=rel_save_thresh,
    density_weight=density_weight,
    hist=hist,
    edges=edges)

generate_loss_curves(train_loss_dict, val_loss_dict, dir_name)

if diffusion == True:
    # Load learned model parameters
    model.load(f"{dir_name}/binn_best_val_model", device=device)
    # Create file to save diffusion coefficients
    with open(f'{dir_name}/diffusion.txt', 'w') as file:
        # Write the variables to the file
        file.write(f"Diffusion coeffs = {model.model.diffusion_fitter()}")