"""
Train one BINN-EQL model and write its results.

    python scripts/train_binn_eql.py <run_dir>

<run_dir>/config.json is written by a sweep script. All outputs (logs,
checkpoints, equations, figures, animations) go to <run_dir>. An
interrupted run resumes from <run_dir>/latest_checkpoint.pt.
"""
import json
import sys
from dataclasses import dataclass, field
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
SURFACE_WEIGHT_DECAY = 1e-3             # default; override per run with "surface_weight_decay"

# Post-training pruning (see equations/prune.py)
PRUNE_THRESHOLD = 0.01
HILL_MERGE_EPSILON = 0.1


@dataclass
class RunConfig:
    """Contents of config.json."""
    training_data_path: str
    reaction: Optional[str]       # key into REACTION_REGISTRY (ground truth, for comparison plots);
                                  # null for data with no known equations (e.g. particle simulations):
                                  # the true-vs-learned surface and parity plots are then skipped
    params: list                  # ground-truth reaction parameters ([] when reaction is null)
    batch_size: int
    species: int
    dimensions: int
    epsilon: float                # noise level (meaning set by noise_model)
    points: int                   # spatial interpolation grid (0 = none)
    diff_coeffs: list             # fixed D per species; [] to learn D
    duplicates: int
    degree: int
    l0_weight: float              # with l0_divide_by_equations: the BASE weight
    param_bounds: float
    mcas: bool = False
    mass_conserved: bool = False  # total of all species conserved by construction: learn S-1
                                  # equations, last = -(sum of the others); 2 species = mcas
    include_poly: bool = True
    include_increasing_hill: bool = True
    include_decreasing_hill: bool = True
    include_constant: bool = False   # constant term in the polynomial library (feed/source rates)
    noise_model: str = "multiplicative"   # "multiplicative" (u * (1 + eps N)), "additive_std"
                                          # (u + eps * std(u) * N: the SINDy/PDE-FIND convention)
                                          # or "additive" (u + eps N); see noise_and_interpolate
    noise_seed: Optional[int] = None      # seed for the noise draw; set it to the repeat number
                                          # to match the baselines (sweep_baselines seed = repeat)
    gate_per_term: bool = False      # True: the duplicate copies of each library term share one
                                     # L0 gate, so L0 counts terms, not copies (see eql_layer.py)
    # Surface-fit resolution and cost
    fourier_scale: float = 1.0       # spread of Fourier frequencies; raise for fine/fast patterns
    fourier_mapping_size: int = 64   # number of Fourier frequencies
    surface_weight_decay: float = SURFACE_WEIGHT_DECAY   # raise for noisy data (stops the surface
                                                         # fitting the noise); keep low when noise-free
    surface_lr_schedule: str = "onecycle"   # Phase 1 surface LR: "onecycle" (warm up, then anneal
                                            # to ~0) or "constant" (base LR throughout -- how the
                                            # surface trained before the OneCycle fix)
    surface_ema_epochs: Optional[float] = None   # Phase 1 surface weight averaging over ~this many
                                                 # epochs (None = off); validation, best-surface
                                                 # selection and the restored surface use the average
    physics_steps_per_epoch: Optional[int] = None   # fixed Phase 2-4 steps; None = one per data batch
    phase_ends: Optional[list] = None               # (p1, p2, p3) end epochs; None = PHASE_ENDS
    gls_max_weight: float = 50.0  # 1.0 = no GLS activity weighting
    hill_kd_min_frac: float = 0.15   # Hill soft wall: each half-saturation K^(-1/n) is kept above
                                     # this x (99th-percentile |input|). Lower it (e.g. 0.01) when
                                     # the true switch is far below the typical concentration
                                     # (Oregonator: half-saturation q = 0.01)
    validation: str = "sampled"   # Phase 2-4 validation / model selection: "sampled" or "full_cache"
    l0_pricing: str = "auto"      # gate pricing: "auto" (global for mcas, else per_species),
                                  # or force "global" / "per_species"
    l0_divide_by_equations: bool = False   # True: train with l0_weight / (number of learned
                                           # equations): 1 for mcas, else one per species
    # Rule-based smoothing of image data (e.g. particle simulations, microscopy), applied to
    # the loaded data before anything else. Each species s is smoothed (periodic Gaussian)
    # to a TOTAL width
    #     sigma_s = max(smoothing_sigma_min, smoothing_rule_c * sqrt(2 * D_s * dt))
    # where dt is the frame spacing: below its per-frame diffusion length a species' field
    # cannot be followed from frame to frame, and D lap(u) is counting noise. The same rule
    # and constants apply to every species; only D_s differs.
    smoothing_rule_c: Optional[float] = None     # None = off (data used as loaded)
    smoothing_sigma_min: float = 0.0   # floor on the total width (counting-noise limit), same units as x
    image_sigma: float = 0.0           # blur already in the data (the converter's --sigma); only
                                       # the remainder sqrt(sigma_s^2 - image_sigma^2) is added
    smoothing_diff_coeffs: Optional[list] = None  # D used by the rule; None = diff_coeffs
                                                  # (required when diff_coeffs is [], i.e. D is learned)
    sim_start_time: float = 0.0      # start the learned-PDE simulation (feql_sim.gif) from the
                                     # first data frame at or after this time instead of t = 0.
                                     # For data that start uniform (all-GDP PBS): a deterministic
                                     # PDE cannot break symmetry from a uniform field, so start
                                     # once the data show fluctuations (e.g. 1.0 s)
    # When sim_start_time > 0, three learned-PDE simulations are written instead of one:
    # feql_sim_from_t0.gif, feql_sim_from_t<start>.gif, feql_sim_from_noise.gif (uniform
    # state + noise_amplitude relative noise, run for noise_sim_time)
    noise_amplitude: float = 0.05
    noise_sim_time: float = 100.0
    noise_sim_seed: int = 0
    dt_cap: Optional[float] = None   # timestep ceiling for simulating the learned PDE; None = the
                                     # reaction's REACTION_REGISTRY dt_cap (required when reaction
                                     # is null). The diffusion CFL limit is applied on top.
    # Set by load(): True if config.json still has the retired "l0_reference_gates"
    # key, i.e. its l0_weight is on the old scale (old weight = new weight x 51).
    legacy_l0_scale: bool = field(default=False, init=False, repr=False)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        # Run folders made before the reference gate count was removed still
        # carry the key. Drop it so they load for analysis (resimulation,
        # reaction checks); main() refuses to TRAIN them -- see there.
        legacy = 'l0_reference_gates' in data
        data.pop('l0_reference_gates', None)
        data.pop('surface_positive', None)   # retired option (surface is always positive)
        data.pop('pde_species_weights', None)   # retired option (every equation counts fully)
        cfg = cls(**data)
        cfg.legacy_l0_scale = legacy
        return cfg
 
 
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
 
    if cfg.smoothing_rule_c is not None:
        data = smooth_by_diffusion_rule(data, cfg)
    raw = data
    if cfg.epsilon != 0 or cfg.points != 0:
        data = noise_and_interpolate(data, cfg.points, cfg.epsilon, cfg.dimensions,
                                     cfg.species, noise_model=cfg.noise_model,
                                     seed=cfg.noise_seed)
    return raw, data
 
 
def smooth_by_diffusion_rule(data, cfg):
    """
    Smooth each species of gridded rows [x..., t, u...] to its rule width (see RunConfig:
    smoothing_rule_c). Periodic Gaussian applied in Fourier space, frame by frame.
    Returns new rows in the same order; the chosen widths are stored in cfg.smoothing_report.
    """
    d, S = cfg.dimensions, cfg.species
    D = cfg.smoothing_diff_coeffs if cfg.smoothing_diff_coeffs is not None else cfg.diff_coeffs
    if not D or len(D) != S:
        raise ValueError("smoothing_rule_c needs one diffusion coefficient per species: set "
                         "diff_coeffs, or smoothing_diff_coeffs when D is learned")
    axes = [torch.unique(data[:, i]) for i in range(d + 1)]          # x..., t
    idx = [torch.searchsorted(a, data[:, i].contiguous()) for i, a in enumerate(axes)]
    shape = [len(a) for a in axes]
    if int(torch.tensor(shape).prod()) != len(data):
        raise ValueError("smoothing_rule_c needs data on a full regular grid "
                         f"({' x '.join(map(str, shape))} grid points vs {len(data)} rows)")
    spacing = [float(a[1] - a[0]) for a in axes[:d]]
    times = axes[d].double()
    dt = float(torch.median(times[1:] - times[:-1]))

    report = {'frame_spacing': dt, 'grid_spacing': spacing, 'rule_c': cfg.smoothing_rule_c,
              'sigma_min': cfg.smoothing_sigma_min, 'image_sigma': cfg.image_sigma,
              'diff_coeffs': list(D), 'sigma_total': [], 'sigma_added': []}
    print(f"\n--- Rule-based smoothing: sigma_s = max({cfg.smoothing_sigma_min:g}, "
          f"{cfg.smoothing_rule_c:g} * sqrt(2 D_s dt)), dt = {dt:.4g}, "
          f"image blur {cfg.image_sigma:g} ---")
    out = data.clone()
    for s in range(S):
        diffusion_length = (2.0 * D[s] * dt) ** 0.5
        total = max(cfg.smoothing_sigma_min, cfg.smoothing_rule_c * diffusion_length)
        added = max(total ** 2 - cfg.image_sigma ** 2, 0.0) ** 0.5
        report['sigma_total'].append(total); report['sigma_added'].append(added)
        print(f"  species {s}: D = {D[s]:.4g}, diffusion length per frame "
              f"{diffusion_length:.3g} -> total sigma {total:.3g} (adding {added:.3g})")
        if added == 0.0:
            continue
        field = torch.zeros(shape, dtype=torch.float64)
        field[tuple(idx)] = data[:, d + 1 + s].double()
        # Gaussian transfer function exp(-sigma^2 k^2 / 2) over the spatial axes.
        kernel = torch.ones(shape[:d], dtype=torch.float64)
        for i in range(d):
            k = 2 * torch.pi * torch.fft.fftfreq(shape[i], d=spacing[i]).double()
            view = [1] * d; view[i] = -1
            kernel = kernel * torch.exp(-0.5 * (added * k) ** 2).view(view)
        spatial = list(range(d))
        smoothed = torch.fft.ifftn(torch.fft.fftn(field, dim=spatial) * kernel.unsqueeze(-1),
                                   dim=spatial).real
        out[:, d + 1 + s] = smoothed[tuple(idx)].to(data.dtype)
    print()
    cfg.smoothing_report = report
    return out


def build_binn(cfg, train_data):
    return BINN(
        dimensions=cfg.dimensions, species=cfg.species, train_data=train_data,
        duplicates=cfg.duplicates, diff_coeffs=cfg.diff_coeffs, degree=cfg.degree,
        param_bounds=cfg.param_bounds, mcas=cfg.mcas, mass_conserved=cfg.mass_conserved,
        include_poly=cfg.include_poly,
        include_increasing_hill=cfg.include_increasing_hill,
        include_decreasing_hill=cfg.include_decreasing_hill,
        include_constant=cfg.include_constant,
        gate_per_term=cfg.gate_per_term,
        fourier_scale=cfg.fourier_scale,
        fourier_mapping_size=cfg.fourier_mapping_size,
        gls_max_weight=cfg.gls_max_weight)
 
 
def build_optimizer(binn, phase_ends, surface_weight_decay=SURFACE_WEIGHT_DECAY,
                    lr_schedule="onecycle"):
    """
    AdamW with one param group per sub-network, plus the Phase 1 schedule:
    OneCycleLR over Phase 1 ("onecycle"), or none ("constant"), in which case
    the trainer holds every trained group at its base LR.
    """
    groups = [
        {'params': binn.surface_fitter.parameters(), 'name': 'surface',
         'lr': LEARNING_RATES['surface'], 'weight_decay': surface_weight_decay},
        {'params': binn.reaction.parameters(), 'name': 'reaction',
         'lr': LEARNING_RATES['reaction'], 'weight_decay': 0.0},
    ]
    if binn.learns_diffusion:
        groups.append({'params': binn.diffusion_fitter.parameters(), 'name': 'diffusion',
                       'lr': LEARNING_RATES['diffusion'], 'weight_decay': 0.0})
 
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    if lr_schedule == "constant":
        return optimizer, None
    if lr_schedule != "onecycle":
        raise ValueError(f'surface_lr_schedule must be "onecycle" or "constant", got {lr_schedule!r}')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g['lr'] for g in groups], total_steps=phase_ends[0],
        pct_start=0.3, div_factor=25, final_div_factor=1e4)
    return optimizer, scheduler
 
 
# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def report_equations(trainer, cfg, training_data, run_dir):
    """Write the best-validation equation before and after pruning, with surface comparisons
    (the comparisons need the true reaction, so they are skipped when cfg.reaction is null)."""
    binn = trainer.model
    reaction_fn = REACTION_REGISTRY[cfg.reaction]['fn'] if cfg.reaction else None
    if reaction_fn is None:
        print("No true reaction (reaction is null): skipping the true-vs-learned surface "
              "and parity plots.", flush=True)
    eq_path = run_dir / 'equation.txt'

    trainer.load(run_dir / 'binn_best_val_model', device=DEVICE)
    write_equations(eq_path, binn, 'Final equation:')
    if binn.learns_diffusion:
        with open(eq_path, 'a') as f:
            f.write(f'\nDiff. coeffs:\n{[D.item() for D in binn.diffusion_fitter()]}\n')
    if reaction_fn is not None:
        compare_surfaces_over_training_domain(training_data, trainer, DEVICE, reaction_fn,
                                              cfg.params, str(run_dir), 'feql_surface_untuned')

    fine_tune_eql(binn, threshold=PRUNE_THRESHOLD, epsilon=HILL_MERGE_EPSILON)
    write_equations(eq_path, binn, '\nFine tuned final equation:')
    if reaction_fn is not None:
        compare_surfaces_over_training_domain(training_data, trainer, DEVICE, reaction_fn,
                                              cfg.params, str(run_dir), 'feql_surface_tuned')


def simulation_dt_cap(cfg):
    """Timestep ceiling for the learned-PDE simulation: config.json's dt_cap, else the registry's."""
    if cfg.dt_cap is not None:
        return cfg.dt_cap
    if not cfg.reaction:
        raise ValueError('config.json has "reaction": null, so it must give "dt_cap" '
                         '(the timestep ceiling for simulating the learned PDE).')
    return REACTION_REGISTRY[cfg.reaction]['dt_cap']


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
 
 
def animate_simulations(trainer, training_data, raw_data, u_array, run_dir, cfg, dt_cap,
                        suffix='', start_time=None, with_surface=True):
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
    if with_surface:
        sim_u, _, sim_t = simulate_uvmlp(training_data, trainer)
        animate_u_array(sim_u, sim_t, f'{run_dir}/uvmlp_sim.gif')
        animate_residuals(sim_u, u_array, sim_t, name=f'{run_dir}/uvmlp_residuals.gif')
    if start_time is None:
        start_time = cfg.sim_start_time
 
    # A learned equation that blows up cannot be simulated, but everything
    # else (equations, loss curves, surface comparisons) is already written,
    # so report it and let the run finish rather than failing at the last step.
    # Optionally start later than t = 0: drop the earlier frames and shift time so the
    # chosen frame is t = 0 for the integrator, then shift back for the animation.
    d = cfg.dimensions
    t_start = 0.0
    if start_time > 0:
        raw_times = torch.unique(raw_data[:, d])
        later = raw_times[raw_times >= start_time - 1e-9]
        if len(later) < 2:
            raise ValueError(f"start time {start_time} leaves fewer than 2 frames")
        t_start = float(later[0])
        raw_data = raw_data[raw_data[:, d] >= t_start].clone()
        raw_data[:, d] -= t_start
        train_times = torch.unique(training_data[:, d])
        u_array = u_array[train_times >= t_start - 1e-9]
        print(f"Learned-PDE simulation starts from the data frame at t = {t_start:.4g}", flush=True)
    try:
        sim_u, sim_x, sim_t = simulate_feql(raw_data, trainer, dt_cap=dt_cap)
    except FloatingPointError as exc:
        print(f"\nSkipping the learned-PDE animation: {exc}\n", flush=True)
        return
    sim_t = sim_t + t_start

    # Simulated at native resolution; animated at the training grid.
    coarse = to_training_grid(sim_u, sim_x, sim_t, cfg)
    animate_u_array(coarse, sim_t, f'{run_dir}/feql_sim{suffix}.gif')

    # Both start from the same initial condition via the same interpolation,
    # so frame 0 should agree to rounding (exactly, when epsilon = 0). A large
    # value here means the two grids are laid out differently.
    mismatch = (coarse[0] - u_array[0]).abs().max().item()
    print(f"Learned-PDE sim vs training data at its first frame: max |difference| = {mismatch:.3e}"
          + ("" if cfg.epsilon else "  (should be ~0)"), flush=True)
    animate_residuals(coarse, u_array, sim_t, name=f'{run_dir}/feql_residuals{suffix}.gif')


def simulate_from_noise(trainer, raw_data, run_dir, cfg, dt_cap, suffix='_from_noise'):
    """
    The strictest polarization test: start the learned PDE from a UNIFORM state with
    small random noise, so no pattern from the data is handed to it.

    Uniform state: each species' spatial mean in the data's LAST frame (the
    patterned, near-steady state the system settles into). Noise: +-noise_amplitude
    (relative, Gaussian, independent per grid point) on species 0; species 1 gets the
    opposite change, so the local total (u + v) is exactly the data's mean. Run for
    noise_sim_time (longer than the data: growth from small noise is slower than
    from the lumps already present at t = 1 s). Writes feql_sim_from_noise.gif
    (no residuals: there is no matching data).
    """
    d, S = cfg.dimensions, cfg.species
    times = torch.unique(raw_data[:, d])
    ic = raw_data[raw_data[:, d] == times[0]].clone()
    last = raw_data[raw_data[:, d] == times[-1]]
    means = last[:, d + 1:].mean(dim=0)
    gen = torch.Generator().manual_seed(cfg.noise_sim_seed)
    xi = torch.randn(len(ic), generator=gen, dtype=ic.dtype)
    ic[:, d + 1:] = means
    ic[:, d + 1] = means[0] * (1 + cfg.noise_amplitude * xi)
    if S > 1 and cfg.mcas:
        ic[:, d + 2] = means[1] - means[0] * cfg.noise_amplitude * xi
    ic[:, d + 1:] = ic[:, d + 1:].clamp(min=0)
    # Frames: the IC repeated at the save times (simulate_feql reads only the t = 0
    # rows and the frame times).
    n_frames = 101
    save_times = torch.linspace(0, cfg.noise_sim_time, n_frames, dtype=ic.dtype)
    rows = ic.repeat(n_frames, 1)
    rows[:, d] = save_times.repeat_interleave(len(ic))
    print(f"Learned PDE from uniform + {cfg.noise_amplitude:.0%} noise "
          f"(means {[round(float(m), 4) for m in means]}), {cfg.noise_sim_time:g} time units",
          flush=True)
    try:
        sim_u, sim_x, sim_t = simulate_feql(rows, trainer, dt_cap=dt_cap)
    except FloatingPointError as exc:
        print(f"\nSkipping the from-noise simulation: {exc}\n", flush=True)
        return
    coarse = to_training_grid(sim_u, sim_x, sim_t, cfg)
    animate_u_array(coarse, sim_t, f'{run_dir}/feql_sim{suffix}.gif')
    first, final = coarse[0, ..., 0], coarse[-1, ..., 0]
    print(f"  species 0 spatial s.d.: {float(first.std()):.4g} at start -> "
          f"{float(final.std()):.4g} at the end "
          f"({'pattern grew' if final.std() > 2 * first.std() else 'no pattern grew'})",
          flush=True)


def polarization_tests(trainer, training_data, raw_data, u_array, run_dir, cfg, dt_cap):
    """Three learned-PDE simulations: from t = 0, from sim_start_time, from uniform + noise."""
    animate_simulations(trainer, training_data, raw_data, u_array, run_dir, cfg, dt_cap,
                        suffix='_from_t0', start_time=0.0, with_surface=False)
    animate_simulations(trainer, training_data, raw_data, u_array, run_dir, cfg, dt_cap,
                        suffix=f'_from_t{cfg.sim_start_time:g}', start_time=cfg.sim_start_time,
                        with_surface=False)
    simulate_from_noise(trainer, raw_data, run_dir, cfg, dt_cap)
 
 
# ---------------------------------------------------------------------------
def main(run_dir, resimulate_start=None):
    run_dir = Path(run_dir)
    cfg = RunConfig.load(run_dir / 'config.json')
    if resimulate_start is not None:
        cfg.sim_start_time = resimulate_start
    if cfg.legacy_l0_scale:
        # The price per gate used to be divided by the reference count (51), and
        # a checkpoint stores the price it calibrated, so resuming or retraining
        # this config would silently change the penalty by 51x.
        raise SystemExit(
            f"{run_dir}/config.json uses the retired \"l0_reference_gates\" key, so its "
            f"l0_weight ({cfg.l0_weight}) is on the old scale. Start a new run folder with "
            f"l0_weight = old / reference count (e.g. 50 / 51 = 0.98) and no such key.")
 
    simulation_dt_cap(cfg)          # fail now, not after a day of training, if dt_cap is missing

    # --- Data ---
    raw_data, training_data = load_training_data(cfg)
    if getattr(cfg, 'smoothing_report', None) is not None:
        with open(run_dir / 'smoothing.json', 'w') as f:
            json.dump(cfg.smoothing_report, f, indent=4)
    # The surface fitter's Softplus output is always positive. Slightly negative
    # values from unclipped additive noise are harmless; a variable that is
    # genuinely negative over much of the data cannot be fitted.
    negative_fraction = (training_data[:, cfg.dimensions + 1:] < 0).float().mean().item()
    if negative_fraction > 0.01:
        print(f"\nWARNING: {negative_fraction:.0%} of the training values are negative, but the "
              f"surface fitter's output is always positive, so it cannot fit them and the "
              f"learned equation will be wrong.\n", flush=True)
    u_array, _, t_array = format_training_data_to_u_array(training_data)
    if resimulate_start is None:
        animate_u_array(u_array, t_array, f'{run_dir}/training_data.gif')
    train_data, val_data = training_test_split(training_data, DEVICE)
 
    # --- Model, objective, optimizer ---
    binn = build_binn(cfg, train_data).to(DEVICE)
    phase_ends = tuple(cfg.phase_ends) if cfg.phase_ends else PHASE_ENDS
    optimizer, scheduler = build_optimizer(binn, phase_ends, cfg.surface_weight_decay,
                                           cfg.surface_lr_schedule)
    print(f"Surface weight decay: {cfg.surface_weight_decay:g}; "
          f"Phase 1 LR schedule: {cfg.surface_lr_schedule}", flush=True)
    eql = binn.reaction.eql_layer
    print(f"L0 gates: {'one per term' if cfg.gate_per_term else 'one per copy'} -- "
          f"{eql.n_gates} gate(s) over {eql.total_features} column(s) x {eql.n_free} row(s)",
          flush=True)
    loss_fn = BINNLoss(binn, l0_pricing=cfg.l0_pricing, soft_wall_epsilon=cfg.hill_kd_min_frac)
    trainer = Trainer(binn, loss_fn, optimizer, scheduler, out_dir=str(run_dir),
                      validation=cfg.validation,
                      physics_steps_per_epoch=cfg.physics_steps_per_epoch,
                      surface_ema_epochs=cfg.surface_ema_epochs)
 
    if resimulate_start is not None:
        # Re-run only the learned-PDE simulation of a finished run: the same pruned
        # ("fine tuned") equation as equation.txt, nothing retrained or overwritten.
        trainer.load(run_dir / 'binn_best_val_model', device=DEVICE)
        fine_tune_eql(binn, threshold=PRUNE_THRESHOLD, epsilon=HILL_MERGE_EPSILON)
        for line in binn.equations_as_strings():
            print(line, flush=True)
        polarization_tests(trainer, training_data, raw_data, u_array, run_dir, cfg,
                           dt_cap=simulation_dt_cap(cfg))
        return

    initial_epoch = 0
    checkpoint = run_dir / 'latest_checkpoint.pt'
    if checkpoint.exists():
        print(f"\nFound existing checkpoint. Resuming from {checkpoint}...", flush=True)
        initial_epoch = trainer.resume(checkpoint, device=DEVICE)
 
    # --- Train ---
    l0_weight = cfg.l0_weight
    if cfg.l0_divide_by_equations:
        l0_weight = cfg.l0_weight / binn.n_learned_equations
        print(f"l0_weight: base {cfg.l0_weight!r} / {binn.n_learned_equations} learned equation(s) "
              f"= {l0_weight:g}", flush=True)
    else:
        print(f"l0_weight from config: {cfg.l0_weight!r}", flush=True)
    param_history, train_losses, val_losses = trainer.fit(
        train_data, val_data, epochs=EPOCHS, batch_size=cfg.batch_size,
        l0_weight=l0_weight, phase_ends=phase_ends, initial_epoch=initial_epoch,
        early_stopping=EARLY_STOPPING)
 
    # --- Results ---
    generate_loss_curves(train_losses, val_losses, str(run_dir), 40, 'training_loss_curves.png')
    plot_param_history(binn, param_history, f"{run_dir}/param_history.png")
    report_equations(trainer, cfg, training_data, run_dir)
    if cfg.sim_start_time > 0:
        animate_simulations(trainer, training_data, raw_data, u_array, run_dir, cfg,
                            dt_cap=simulation_dt_cap(cfg), suffix=f'_from_t{cfg.sim_start_time:g}')
        animate_simulations(trainer, training_data, raw_data, u_array, run_dir, cfg,
                            dt_cap=simulation_dt_cap(cfg), suffix='_from_t0', start_time=0.0,
                            with_surface=False)
        simulate_from_noise(trainer, raw_data, run_dir, cfg, dt_cap=simulation_dt_cap(cfg))
    else:
        animate_simulations(trainer, training_data, raw_data, u_array, run_dir, cfg,
                            dt_cap=simulation_dt_cap(cfg))
 
 
if __name__ == '__main__':
    # Train:                       python train_binn_eql.py <run_dir>
    # Re-simulate a finished run:  python train_binn_eql.py <run_dir> --resimulate 1.0
    #   (the three polarization tests for the run's pruned equation: from t = 0, from the
    #    data frame at t >= 1.0, and from uniform + noise -> feql_sim_from_t0.gif,
    #    feql_sim_from_t1.gif (+ residuals), feql_sim_from_noise.gif; nothing else changes)
    if len(sys.argv) > 2 and sys.argv[2] == '--resimulate':
        main(sys.argv[1], resimulate_start=float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
    else:
        main(sys.argv[1])