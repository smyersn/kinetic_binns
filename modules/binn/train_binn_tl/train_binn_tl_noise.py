import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns import BINN
from modules.loaders.format_data import format_data
from modules.loaders.visualize_training_data import plot_animation, animate_training_data
from modules.generate_data.simulate_system import simulate
from modules.ewc.utils import *

def to_torch(ndarray):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

# load params from configuration file
config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

training_data_path = config['training_data_path']
epsilon = config['epsilon']
num_points = config['num_points']
rounds_of_training = config['rounds_of_training']
dir_name = sys.argv[1]

epochs = int(1e6)
#epochs = 100
batch_size = 5000 # 25% of split
rel_save_thresh = 0.05

# Get GPU
device = torch.device('cuda')

# initialize model
device = torch.device('cuda')
binn = BINN(uv_layers=[128, 128, 128, 128, 128, 128, 128, 128, 2], f_layers=[32, 32, 32, 1])
binn.to(device)

# load data
inputs, output_u, output_v, shape_u, shape_v = format_data(training_data_path)
training_data = np.column_stack((inputs, output_u, output_v))

# training loop
for i in range(rounds_of_training):
    model_name = f'{dir_name}/model_{i+1}'

    # modify and split data for each round of training
    training_data_noise = noise_and_interpolate(training_data, num_points, epsilon)

    # shuffle data
    N = len(training_data_noise)
    p = np.random.permutation(N)
    training_data_shuffled = training_data_noise[p[:]]

    # split into train/val and convert to torch
    M = len(training_data_shuffled)
    split = int(0.8*M)
    x_train = to_torch(training_data_shuffled[:split, :2])
    y_train = to_torch(training_data_shuffled[:split, 2:])
    x_val = to_torch(training_data_shuffled[split:, :2])
    y_val = to_torch(training_data_shuffled[split:, 2:])
    animate_training_data(training_data_shuffled[:, :2], training_data_shuffled[:, 2:], save=True, name=f'{dir_name}/training_data')

    if i == 0: 
        parameters = binn.parameters()
        ewc = None
        
    else:
        prior_model_name = f"{dir_name}/model_{i}_best_val_model" 
        prior_params = torch.load(prior_model_name, map_location=device)
        prior_reaction_dict = {key: val for key, val in prior_params.items() if 'reaction' in key}
        binn.load_state_dict(prior_reaction_dict, strict=False)
        ewc = EWC(binn, x_train_prev, y_train_prev)

    opt = torch.optim.Adam(list(binn.parameters()), lr=1e-3)
    model = model_wrapper(
        model=binn,
        optimizer=opt,
        loss=binn.loss,
        augmentation=None,
        save_name=model_name)

    # train jointly
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=None,
        verbose=1,
        validation_data=[x_val, y_val],
        early_stopping=5000,
        rel_save_thresh=rel_save_thresh)

    # store previous model's training data for EWC
    x_train_prev = x_train
    y_train_prev = y_train