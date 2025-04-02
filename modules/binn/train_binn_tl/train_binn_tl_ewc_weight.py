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
from modules.analysis.visualize_tl_surfaces import visualize_tl_surfaces
from modules.analysis.simulate_tl_surfaces import simulate_tl_surfaces
from modules.ewc.utils import *

def to_torch(ndarray):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

# initialize model
device = torch.device('cuda')
binn = BINN()
binn.to(device)

# define training data
training_data_path = '/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/'

# load model parameters
weight = sys.argv[1]
training_data_string = "".join(sys.argv[2:])
training_data_list = training_data_string.split()
model_names = [os.path.splitext(file)[0] for file in training_data_list]
epochs = int(1e6)
#epochs = 100
batch_size = 5000 # 25% of split
rel_save_thresh = 0.05

# make directory for runs
run_dir = 'runs/'
os.makedirs(run_dir, exist_ok=True)
# make a directory for model
model_dir = f'{run_dir}/{weight}'
os.makedirs(model_dir, exist_ok=True)

# training loop
for i in range(len(training_data_list)):
    model_name = model_dir+'/'+model_names[i]
    
    # load in training data
    inputs, output_u, output_v, shape_u, shape_v = format_data(training_data_path+training_data_list[i])
    outputs = np.concatenate((output_u, output_v), axis=1)

    # split into train/val and convert to torch
    N = len(inputs)
    split = int(0.8*N)
    p = np.random.permutation(N)
    x_train = to_torch(inputs[p[:split]])
    y_train = to_torch(outputs[p[:split]])
    x_val = to_torch(inputs[p[split:]])
    y_val = to_torch(outputs[p[split:]])
    inputs = to_torch(inputs)
    outputs = to_torch(outputs)

    # compile
    if i == 0: 
        parameters = binn.parameters()
        ewc = None
    else:
        prior_model_name = f"{model_dir}/{model_names[i-1]}_best_val_model" 
        prior_params = torch.load(prior_model_name, map_location=device)
        prior_reaction_dict = {key: val for key, val in prior_params.items() if 'reaction' in key}
        binn.load_state_dict(prior_reaction_dict, strict=False)
        ewc = EWC(binn, x_train_prev, y_train_prev, weight=weight)

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
        rel_save_thresh=rel_save_thresh,
        ewc=ewc)

    # store previous model's training data for EWC
    x_train_prev = x_train
    y_train_prev = y_train