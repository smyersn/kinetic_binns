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
from modules.utils.format_data import (format_training_data_to_u_array,
                                       format_u_array_to_training_data)
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split

# Full FP32 matmuls: TF32 loses precision in the autograd derivatives of the surface.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ---------------------------------------------------------------------------
# Training settings shared by every run
# ---------------------------------------------------------------------------
DEVICE = 'cuda'
EPOCHS = 200_000
PHASE_ENDS = (5_000, 40_000, 75_000)    # default ends of the surface, physics and L0-ramp phases
# DEVICE = 'cpu'
# EPOCHS = 50
# PHASE_ENDS = (10, 20, 30)    # default ends of the surface, physics and L0-ramp phases
EARLY_STOPPING = int(0.05 * EPOCHS)     # Phase 4 patience
LEARNING_RATES = {'surface': 1e-2, 'reaction': 1e-3, 'diffusion': 1e-3}
SURFACE_WEIGHT_DECAY = 1e-2

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
    include_constant: bool = False   # constant term in the polynomial library (feed/source rates)
    # Surface-fit resolution and cost
    fourier_scale: float = 1.0       # spread of Fourier frequencies; raise for fine/fast patterns
    fourier_mapping_size: int = 64   # number of Fourier frequencies
    physics_steps_per_epoch: Optional[int] = None   # fixed Phase 2-4 steps; None = one per data batch
    phase_ends: Optional[list] = None               # (p1, p2, p3) end epochs; None = PHASE_ENDS
    gls_max_weight: float = 50.0  # 1.0 = no GLS activity weighting
    validation: str = "sampled"   # Phase 2-4 validation / model selection: "sampled" or "full_cache"
    l0_pricing: str = "auto"      # gate pricing: "auto" (global for mcas, else per_species),
                                  # or force "global" / "per_species"
 
    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))
 
 
# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def load_training_data(cfg):
    """
    (raw, training): rows of [x..., t, u...] at the simulation's native
    resolution, and the same after the configured noise and interpolation.
    The raw data are kept so the learned PDE can be simulated at the
    resolution the data were generated at (see animate_simulations).
    """
    data = torch.load(cfg.training_data_path)['training_data']
 
    # Rows are [x_1..x_d, t, u_1..u_S]. A config/data mismatch otherwise surfaces
    # as an unrelated reshape error deep inside noise_and_interpolate.
    expected = cfg.dimensions + 1 + cfg.species
    if data.ndim != 2 or data.shape[1] != expected:
        raise ValueError(
            f"{cfg.training_data_path} has shape {tuple(data.shape)}, but config.json "
            f"(dimensions={cfg.dimensions}, species={cfg.species}) expects "
            f"{expected} columns: [x...(dimensions), t, u...(species)].")
 
    raw = data
    if cfg.epsilon != 0 or cfg.points != 0:
        data = noise_and_interpolate(data, cfg.points, cfg.epsilon, cfg.dimensions,
                                     cfg.species, multiplicative_noise=True)
    return raw, data
 
 
def build_binn(cfg, train_data):
    return BINN(
        dimensions=cfg.dimensions, species=cfg.species, train_data=train_data,
        duplicates=cfg.duplicates, diff_coeffs=cfg.diff_coeffs, degree=cfg.degree,
        param_bounds=cfg.param_bounds, mcas=cfg.mcas,
        l0_reference_gates=cfg.l0_reference_gates,
        include_poly=cfg.include_poly,
        include_increasing_hill=cfg.include_increasing_hill,
        include_decreasing_hill=cfg.include_decreasing_hill,
        include_constant=cfg.include_constant,
        fourier_scale=cfg.fourier_scale,
        fourier_mapping_size=cfg.fourier_mapping_size,
        gls_max_weight=cfg.gls_max_weight)
 
 
def check_library_size(cfg, binn):
    """Warn if library_size() (used to set l0_reference_gates) disagrees with the model."""
    expected = library_size(
        species=cfg.species, degree=cfg.degree, duplicates=cfg.duplicates, mcas=cfg.mcas,
        include_poly=cfg.include_poly,
        include_increasing_hill=cfg.include_increasing_hill,
        include_decreasing_hill=cfg.include_decreasing_hill,
        include_constant=cfg.include_constant)
    actual = binn.reaction.eql_layer.n_gates
    if expected != actual:
        print(f"WARNING: library_size says {expected} gates, model has {actual}. "
              f"The two have drifted -- l0_reference_gates is unreliable.", flush=True)
 
 
def build_optimizer(binn, phase_ends):
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
        optimizer, max_lr=[g['lr'] for g in groups], total_steps=phase_ends[0],
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
 
 
def to_training_grid(sim_u, sim_x, sim_t, cfg):
    """
    Bring a native-resolution simulation down to the training grid by the
    SAME path the training data took: rows -> noise_and_interpolate (with no
    noise) -> grid. Residuals then compare like with like.
    """
    if cfg.points == 0:
        return sim_u                     # training data are already at native resolution
    rows = format_u_array_to_training_data(sim_u, sim_x, sim_t)
    rows = noise_and_interpolate(rows, cfg.points, 0.0, cfg.dimensions, cfg.species,
                                 multiplicative_noise=True)
    coarse, _, _ = format_training_data_to_u_array(rows)
    return coarse
 
 
def animate_simulations(trainer, training_data, raw_data, u_array, run_dir, cfg, dt_cap):
    """
    Simulate the surface fitter and the learned PDE; animate them and their
    residuals.

    The learned PDE is integrated on the RAW data's grid (the resolution the
    data were generated at, 200 x 200), then brought down to the training
    grid (points x points) by the same interpolation the training data took.
    feql_sim.gif and feql_residuals.gif are both written at the training
    grid, so they line up frame for frame with training_data.gif.

    Integrating on the coarse training grid directly would under-resolve the
    pattern: the 5-point Laplacian under-diffuses features only a few cells
    wide and is anisotropic, which shifts pattern selection toward smaller,
    grid-aligned spots that the true dynamics would not produce. So the
    simulation stays at native resolution and only the output is downsampled.

    dt_cap is the reaction's own timestep ceiling from REACTION_SPECS; the
    CFL limit, which shrinks with the finer grid, is applied on top.
    """
    sim_u, _, sim_t = simulate_uvmlp(training_data, trainer)
    animate_u_array(sim_u, sim_t, f'{run_dir}/uvmlp_sim.gif')
    animate_residuals(sim_u, u_array, sim_t, name=f'{run_dir}/uvmlp_residuals.gif')
 
    # A learned equation that blows up cannot be simulated, but everything
    # else (equations, loss curves, surface comparisons) is already written,
    # so report it and let the run finish rather than failing at the last step.
    try:
        sim_u, sim_x, sim_t = simulate_feql(raw_data, trainer, dt_cap=dt_cap)
    except FloatingPointError as exc:
        print(f"\nSkipping the learned-PDE animation: {exc}\n", flush=True)
        return

    # Simulated at native resolution; animated at the training grid.
    coarse = to_training_grid(sim_u, sim_x, sim_t, cfg)
    animate_u_array(coarse, sim_t, f'{run_dir}/feql_sim.gif')

    # Both start from the same initial condition via the same interpolation,
    # so frame 0 should agree to rounding (exactly, when epsilon = 0). A large
    # value here means the two grids are laid out differently.
    mismatch = (coarse[0] - u_array[0]).abs().max().item()
    print(f"Learned-PDE sim vs training data at t=0: max |difference| = {mismatch:.3e}"
          + ("" if cfg.epsilon else "  (should be ~0)"), flush=True)
    animate_residuals(coarse, u_array, sim_t, name=f'{run_dir}/feql_residuals.gif')
 
 
# ---------------------------------------------------------------------------
def main(run_dir):
    run_dir = Path(run_dir)
    cfg = RunConfig.load(run_dir / 'config.json')
 
    # --- Data ---
    raw_data, training_data = load_training_data(cfg)
    u_array, _, t_array = format_training_data_to_u_array(training_data)
    animate_u_array(u_array, t_array, f'{run_dir}/training_data.gif')
    train_data, val_data = training_test_split(training_data, DEVICE)
 
    # --- Model, objective, optimizer ---
    binn = build_binn(cfg, train_data).to(DEVICE)
    check_library_size(cfg, binn)
    phase_ends = tuple(cfg.phase_ends) if cfg.phase_ends else PHASE_ENDS
    optimizer, scheduler = build_optimizer(binn, phase_ends)
    loss_fn = BINNLoss(binn, l0_pricing=cfg.l0_pricing)
    trainer = Trainer(binn, loss_fn, optimizer, scheduler, out_dir=str(run_dir),
                      validation=cfg.validation,
                      physics_steps_per_epoch=cfg.physics_steps_per_epoch)
 
    initial_epoch = 0
    checkpoint = run_dir / 'latest_checkpoint.pt'
    if checkpoint.exists():
        print(f"\nFound existing checkpoint. Resuming from {checkpoint}...", flush=True)
        initial_epoch = trainer.resume(checkpoint, device=DEVICE)
 
    # --- Train ---
    print(f"l0_weight from config: {cfg.l0_weight!r} (type {type(cfg.l0_weight).__name__})", flush=True)
    param_history, train_losses, val_losses = trainer.fit(
        train_data, val_data, epochs=EPOCHS, batch_size=cfg.batch_size,
        l0_weight=cfg.l0_weight, phase_ends=phase_ends, initial_epoch=initial_epoch,
        early_stopping=EARLY_STOPPING)
 
    # --- Results ---
    generate_loss_curves(train_losses, val_losses, str(run_dir), 40, 'training_loss_curves.png')
    plot_param_history(binn, param_history, f"{run_dir}/param_history.png")
    report_equations(trainer, cfg, training_data, run_dir)
    animate_simulations(trainer, training_data, raw_data, u_array, run_dir, cfg,
                        dt_cap=REACTION_REGISTRY[cfg.reaction]['dt_cap'])
 
 
if __name__ == '__main__':
    main(sys.argv[1])