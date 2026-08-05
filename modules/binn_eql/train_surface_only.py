import sys, os, json, time
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

import torch
from pathlib import Path
from modules.utils.imports import *
from modules.utils.time_remaining import *
from modules.utils.format_data import format_training_data_to_u_array
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split
from modules.utils.gradient import gradient
from modules.binn_eql.build_binn_eql_net import BINN
from modules.simulation.simulation import simulate_uvmlp          # NEW
from modules.simulation.animation import animate_u_array, animate_residuals  # NEW

dir_name = sys.argv[1]
config_path = Path(dir_name) / 'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

training_data_path = config['training_data_path']
species = config['species']
dimensions = config['dimensions']
batch_size = config['batch_size']
fourier_scale = config['fourier_scale']
fourier_mapping_size = config['fourier_mapping_size']
time_scale = config.get('time_scale', 2.0)
max_early_weight = config.get('max_early_weight', 10.0)
epsilon = config.get('epsilon', 0)
points = config.get('points', 0)

device = 'cuda'

training_data = torch.load(training_data_path)['training_data']

if epsilon != 0 or points != 0:
    training_data = noise_and_interpolate(training_data, points, epsilon,
                                          dimensions, species,
                                          multiplicative_noise=False)

u_array, x_array, t_array = format_training_data_to_u_array(training_data)
train_data, val_data = training_test_split(training_data, device)

total_data_points = len(training_data)
target_total_steps = 1_000_000
steps_per_epoch = max(1, total_data_points // batch_size)
full_epochs = int(target_total_steps // steps_per_epoch)
phase1_epochs = int(0.2 * full_epochs)

print(f"total_data_points={total_data_points} (epsilon={epsilon}, points={points})", flush=True)
print(f"Derived full_epochs={full_epochs}, phase1_epochs={phase1_epochs} "
      f"(matches main pipeline for this batch_size/dataset size)", flush=True)

binn = BINN(
    dimensions=dimensions, species=species, train_data=train_data,
    duplicates=1, diff_coeffs=[0.0, 0.0],
    degree=2, param_bounds=10,
    fourier_scale=fourier_scale, fourier_mapping_size=fourier_mapping_size)
binn.to(device)

opt = torch.optim.AdamW(binn.surface_fitter.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=1e-2, total_steps=phase1_epochs, pct_start=0.3, div_factor=25, final_div_factor=1e4)

binn.train()
start_time = time.time()

for epoch in range(phase1_epochs):
    epoch_start_time = time.time()

    perm = torch.randperm(train_data.size(0))
    for i in range(0, len(train_data), batch_size):
        idx = perm[i:i+batch_size]
        x_true = train_data[idx, :-species].detach().clone().requires_grad_(True)
        y_true = train_data[idx, -species:].detach().clone()

        opt.zero_grad()
        y_pred = binn(x_true)
        gls = binn.gls_loss_time_weighted(y_pred, y_true,
                                    time_scale=time_scale,
                                    max_weight=max_early_weight)
        loss = gls
        loss.backward()
        torch.nn.utils.clip_grad_norm_(binn.surface_fitter.parameters(), max_norm=1.0)
        opt.step()
    scheduler.step()

    elapsed, remaining, ms = time_remaining(
        current_iter=epoch+1,
        total_iter=phase1_epochs,
        start_time=start_time,
        previous_time=epoch_start_time,
        ops_per_iter=batch_size)

    if epoch % 200 == 0 or epoch == phase1_epochs - 1:
        p = 'Epoch {0}/{1}'.format(epoch, phase1_epochs)
        p += ' | GLS = {0:1.4e}'.format(gls.item())
        p += ' | Elapsed = ' + elapsed
        p += ' | Remaining = ' + remaining + '           '
        print(p, flush=True)

# --- Evaluate: derivative diagnostic, summarize into headline numbers ---
binn.eval()
grid_x, grid_y = torch.meshgrid(x_array, x_array, indexing='xy')
spatial = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
N = len(x_array)
dx = float(x_array[1] - x_array[0])
dt = float(t_array[1] - t_array[0])

up, um = torch.roll(u_array, -1, dims=1), torch.roll(u_array, 1, dims=1)
ul, ur = torch.roll(u_array, -1, dims=2), torch.roll(u_array, 1, dims=2)
lap_fd_all = (up + um + ul + ur - 4 * u_array) / dx**2

summary = {'t': [], 'lap_rel_err': [], 'ut_rel_err': []}
for n in range(1, len(t_array) - 1):
    t_col = torch.full((spatial.shape[0], 1), float(t_array[n]))
    pts = torch.cat([spatial, t_col], dim=1).to(device).requires_grad_(True)
    u_pred = binn(pts)[:, 0]
    d1 = gradient(u_pred, pts, order=1)
    ut_pred = d1[:, -1].view(N, N).detach().cpu()
    lap_pred = sum(gradient(d1[:, j], pts, order=1)[:, j] for j in range(dimensions)).view(N, N).detach().cpu()

    ut_fd = (u_array[n+1, :, :, 0] - u_array[n-1, :, :, 0]) / (2 * dt)
    lap_fd = lap_fd_all[n, :, :, 0]

    ut_err = (ut_pred - ut_fd).abs().mean().item()
    lap_err = (lap_pred - lap_fd).abs().mean().item()
    summary['t'].append(float(t_array[n]))
    summary['ut_rel_err'].append(ut_err / (ut_fd.abs().mean().item() + 1e-8))
    summary['lap_rel_err'].append(lap_err / (lap_fd.abs().mean().item() + 1e-8))

summary['lap_rel_err_first'] = summary['lap_rel_err'][0]
summary['lap_rel_err_mean'] = sum(summary['lap_rel_err']) / len(summary['lap_rel_err'])
summary['ut_rel_err_first'] = summary['ut_rel_err'][0]
summary['ut_rel_err_mean'] = sum(summary['ut_rel_err']) / len(summary['ut_rel_err'])
summary['fourier_scale'] = fourier_scale
summary['fourier_mapping_size'] = fourier_mapping_size
summary['phase1_epochs'] = phase1_epochs

with open(f'{dir_name}/surface_fit_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"DONE scale={fourier_scale} mapping_size={fourier_mapping_size} "
      f"lap_rel_err_first={summary['lap_rel_err_first']:.3f} "
      f"lap_rel_err_mean={summary['lap_rel_err_mean']:.3f} "
      f"ut_rel_err_first={summary['ut_rel_err_first']:.3f} "
      f"ut_rel_err_mean={summary['ut_rel_err_mean']:.3f}", flush=True)

# --- NEW: Simulate the fitted surface directly and generate residual animations,
# same as the main training script's uvmlp_sim/uvmlp_residuals output.
# simulate_uvmlp expects a model_wrapper-shaped object (accesses model.model.*),
# so wrap the raw BINN in a minimal shim rather than restructuring this script
# around the full model_wrapper class.
class _ModelShim:
    def __init__(self, model):
        self.model = model

shim = _ModelShim(binn)

print("Simulating uvmlp surface for residual comparison...", flush=True)
uvmlp_u_array, uvmlp_x_array, uvmlp_times = simulate_uvmlp(training_data, shim)

animate_u_array(uvmlp_u_array, uvmlp_times, f'{dir_name}/uvmlp_sim.gif')
animate_residuals(uvmlp_u_array, u_array, uvmlp_times,
                   name=f'{dir_name}/uvmlp_residuals.gif')

print(f"Saved uvmlp_sim.gif and uvmlp_residuals.gif to {dir_name}", flush=True)