import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns import BINN
from modules.analysis.visualize_surface import visualize_surface
from modules.analysis.simulate_surface import simulate_surface
import modules.generate_data.simulate_system

def to_torch(ndarray):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

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

# initialize model
device = torch.device('cpu')

binn = BINN(
    species=species, 
    dimensions=dimensions,
    uv_arch=uv_arch, 
    f_arch=f_arch,
    diff_coeffs=params[:2],
    diff=diffusion)

binn.to(device)

opt = torch.optim.Adam(list(binn.parameters()), lr=1e-3)
model = model_wrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    augmentation=None,
    save_name=f'{dir_name}/binn')

model.load(f"{dir_name}/binn_best_val_model", device=device)

# Analyze results
visualize_surface(model, dimensions, species, reaction, params, dir_name, training_data_path)
simulate_surface(model, device, dimensions, species, params, dir_name, training_data_path)