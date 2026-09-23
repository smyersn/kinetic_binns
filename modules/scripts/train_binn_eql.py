"""
Train one BINN-EQL model and write its results.

    python scripts/train_binn_eql.py <run_dir>

<run_dir>/config.json is written by a sweep script. All outputs (logs,
checkpoints, equations, figures, animations) go to <run_dir>. An
interrupted run resumes from <run_dir>/latest_checkpoint.pt.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Repository root: the nearest ancestor containing the `modules` package, so this
# works whether scripts/ sits at the repo root or inside modules/.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "modules").is_dir())
sys.path.append(str(REPO_ROOT))

import torch

from modules.analysis.generate_loss_curves import generate_loss_curves
from modules.analysis.plot_param_history import plot_param_history
from modules.analysis.visualize_surface import compare_surfaces_over_training_domain
from modules.binn_eql.equations.format import write_equations
from modules.binn_eql.equations.prune import fine_tune_eql
from modules.binn_eql.model.binn import BINN
from modules.binn_eql.physics.losses import BINNLoss
from modules.binn_eql.training.trainer import Trainer
from modules.simulation.animation import animate_residuals, animate_u_array
from modules.simulation.reaction_library import REACTION_REGISTRY
from modules.simulation.reaction_registry import library_size
from modules.simulation.simulation import simulate_feql, simulate_uvmlp
from modules.utils.format_data import format_training_data_to_u_array
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split

# Full FP32 matmuls: TF32 loses precision in the autograd derivatives of the surface.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ---------------------------------------------------------------------------
# Training settings shared by every run
# ---------------------------------------------------------------------------
DEVICE = 'cuda'
EPOCHS = 100_000
PHASE_ENDS = (5_000, 20_000, 30_000)    # ends of the surface, physics and L0-ramp phases
# DEVICE = 'cpu'
# EPOCHS = 50
# PHASE_ENDS = (10, 20, 30)    # ends of the surface, physics and L0-ramp phases
EARLY_STOPPING = int(0.05 * EPOCHS)     # Phase 4 patience
LEARNING_RATES = {'surface': 1e-2, 'reaction': 1e-3, 'diffusion': 1e-3}
SURFACE_WEIGHT_DECAY = 1e-3

# Post-training pruning (see equations/prune.py)
PRUNE_THRESHOLD = 0.01
HILL_MERGE_EPSILON = 0.1


@dataclass
class RunConfig:
    """Contents of config.json."""
    training_data_path: str
    reaction: str                 # key into REACTION_REGISTRY (ground truth, for comparison plots)
    params: list                  # ground-truth reaction parameters
    batch_size: int
    species: int
    dimensions: int
    epsilon: float                # multiplicative noise level
    points: int                   # spatial interpolation grid (0 = none)
    diff_coeffs: list             # fixed D per species; [] to learn D
    duplicates: int
    degree: int
    l0_weight: float
    param_bounds: float
    mcas: bool = False
    l0_reference_gates: Optional[int] = None
    include_poly: bool = True
    include_increasing_hill: bool = True
    include_decreasing_hill: bool = True
    gls_max_weight: float = 50.0  # 1.0 = no GLS activity weighting

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def load_training_data(cfg):
    """Rows of [x..., t, u...], with noise and interpolation applied if configured."""
    data = torch.load(cfg.training_data_path)['training_data']
    if cfg.epsilon != 0 or cfg.points != 0:
        data = noise_and_interpolate(data, cfg.points, cfg.epsilon, cfg.dimensions,
                                     cfg.species, multiplicative_noise=True)
    return data


def build_binn(cfg, train_data):
    return BINN(
        dimensions=cfg.dimensions, species=cfg.species, train_data=train_data,
        duplicates=cfg.duplicates, diff_coeffs=cfg.diff_coeffs, degree=cfg.degree,
        param_bounds=cfg.param_bounds, mcas=cfg.mcas,
        l0_reference_gates=cfg.l0_reference_gates,
        include_poly=cfg.include_poly,
        include_increasing_hill=cfg.include_increasing_hill,
        include_decreasing_hill=cfg.include_decreasing_hill,
        gls_max_weight=cfg.gls_max_weight)


def check_library_size(cfg, binn):
    """Warn if library_size() (used to set l0_reference_gates) disagrees with the model."""
    expected = library_size(
        species=cfg.species, degree=cfg.degree, duplicates=cfg.duplicates, mcas=cfg.mcas,
        include_poly=cfg.include_poly,
        include_increasing_hill=cfg.include_increasing_hill,
        include_decreasing_hill=cfg.include_decreasing_hill)
    actual = binn.reaction.eql_layer.n_gates
    if expected != actual:
        print(f"WARNING: library_size says {expected} gates, model has {actual}. "
              f"The two have drifted -- l0_reference_gates is unreliable.", flush=True)


def build_optimizer(binn):
    """AdamW with one param group per sub-network; OneCycleLR over Phase 1."""
    groups = [
        {'params': binn.surface_fitter.parameters(), 'name': 'surface',
         'lr': LEARNING_RATES['surface'], 'weight_decay': SURFACE_WEIGHT_DECAY},
        {'params': binn.reaction.parameters(), 'name': 'reaction',
         'lr': LEARNING_RATES['reaction'], 'weight_decay': 0.0},
    ]
    if binn.learns_diffusion:
        groups.append({'params': binn.diffusion_fitter.parameters(), 'name': 'diffusion',
                       'lr': LEARNING_RATES['diffusion'], 'weight_decay': 0.0})

    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g['lr'] for g in groups], total_steps=PHASE_ENDS[0],
        pct_start=0.3, div_factor=25, final_div_factor=1e4)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def report_equations(trainer, cfg, training_data, run_dir):
    """Write the best-validation equation before and after pruning, with surface comparisons."""
    binn = trainer.model
    reaction_fn = REACTION_REGISTRY[cfg.reaction]['fn']
    eq_path = run_dir / 'equation.txt'

    trainer.load(run_dir / 'binn_best_val_model', device=DEVICE)
    write_equations(eq_path, binn, 'Final equation:')
    if binn.learns_diffusion:
        with open(eq_path, 'a') as f:
            f.write(f'\nDiff. coeffs:\n{[D.item() for D in binn.diffusion_fitter()]}\n')
    compare_surfaces_over_training_domain(training_data, trainer, DEVICE, reaction_fn,
                                          cfg.params, str(run_dir), 'feql_surface_untuned')

    fine_tune_eql(binn, threshold=PRUNE_THRESHOLD, epsilon=HILL_MERGE_EPSILON)
    write_equations(eq_path, binn, '\nFine tuned final equation:')
    compare_surfaces_over_training_domain(training_data, trainer, DEVICE, reaction_fn,
                                          cfg.params, str(run_dir), 'feql_surface_tuned')


def animate_simulations(trainer, training_data, u_array, run_dir):
    """Simulate the surface fitter and the learned PDE; animate them and their residuals."""
    for name, simulate in (('uvmlp', simulate_uvmlp), ('feql', simulate_feql)):
        sim_u, _, sim_t = simulate(training_data, trainer)
        animate_u_array(sim_u, sim_t, f'{run_dir}/{name}_sim.gif')
        animate_residuals(sim_u, u_array, sim_t, name=f'{run_dir}/{name}_residuals.gif')


# ---------------------------------------------------------------------------
def main(run_dir):
    run_dir = Path(run_dir)
    cfg = RunConfig.load(run_dir / 'config.json')

    # --- Data ---
    training_data = load_training_data(cfg)
    u_array, _, t_array = format_training_data_to_u_array(training_data)
    animate_u_array(u_array, t_array, f'{run_dir}/training_data.gif')
    train_data, val_data = training_test_split(training_data, DEVICE)

    # --- Model, objective, optimizer ---
    binn = build_binn(cfg, train_data).to(DEVICE)
    check_library_size(cfg, binn)
    optimizer, scheduler = build_optimizer(binn)
    trainer = Trainer(binn, BINNLoss(binn), optimizer, scheduler, out_dir=str(run_dir))

    initial_epoch = 0
    checkpoint = run_dir / 'latest_checkpoint.pt'
    if checkpoint.exists():
        print(f"\nFound existing checkpoint. Resuming from {checkpoint}...", flush=True)
        initial_epoch = trainer.resume(checkpoint, device=DEVICE)

    # --- Train ---
    print(f"l0_weight from config: {cfg.l0_weight!r} (type {type(cfg.l0_weight).__name__})", flush=True)
    param_history, train_losses, val_losses = trainer.fit(
        train_data, val_data, epochs=EPOCHS, batch_size=cfg.batch_size,
        l0_weight=cfg.l0_weight, phase_ends=PHASE_ENDS, initial_epoch=initial_epoch,
        early_stopping=EARLY_STOPPING)

    # --- Results ---
    generate_loss_curves(train_losses, val_losses, str(run_dir), 40, 'training_loss_curves.png')
    plot_param_history(binn, param_history, f"{run_dir}/param_history.png")
    report_equations(trainer, cfg, training_data, run_dir)
    animate_simulations(trainer, training_data, u_array, run_dir)


if __name__ == '__main__':
    main(sys.argv[1])