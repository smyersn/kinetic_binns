"""
Re-simulate a trained model's learned PDE at the data's native resolution.

    python resimulate_feql.py <run_dir> [<run_dir> ...] [--no-prune] [--residuals]
                              [--dt-cap DT] [--grid N] [--on-training-grid]
                              [--true-reaction]
                              [--set SPECIES:TERM=VALUE ...]

For each run directory: loads config.json and binn_best_val_model, prunes the
equation the same way training does (unless --no-prune), integrates the
learned PDE from the raw data's initial condition -- before noise, at the
resolution the data were generated at -- and writes feql_sim_new.gif there.

With --residuals it also brings the simulation down to the training grid and
writes feql_residuals_new.gif, printing the t=0 agreement check first.

--true-reaction replaces the learned reaction with the TRUE one from
REACTION_REGISTRY (same parameters the data were generated with), keeping
everything else -- initial condition, grid, diffusion, timestep, integrator --
identical, and writes feql_sim_true_reaction.gif. If the true reaction
misbehaves the same way, the problem is the simulation, not the learned
equation.

--set overrides single coefficients of the (pruned) learned equation before
simulating, e.g. to test which coefficient error changes the dynamics:

    --set 'u:1=1.0'           constant term of du/dt
    --set 'u:u^2*v=1.0'       the u^2 v term of du/dt (quote: ^ and * are shell-special)

Repeat --set for several overrides. A term that was pruned is switched back on;
setting a value of 0 prunes it. Polynomial terms only. Writes
feql_sim_set_<overrides>.gif.

--dt-cap overrides the reaction's timestep ceiling from REACTION_SPECS. The
diffusion CFL limit is still applied on top, so this can only loosen the step
as far as the grid allows. If the step is too large the simulation diverges,
and simulate_feql retries with smaller steps automatically.

--grid N simulates on an N x N grid instead of the native one: the raw
initial condition is brought down to N x N by the same interpolation the
training data take (without noise), and the learned PDE is integrated there.
A coarser grid is much cheaper: 4x fewer points per step at half the native
resolution, and the diffusion CFL limit loosens with dx^2, so pass a larger
--dt-cap as well to use it (the default cap is often the binding limit).
The cost is accuracy: too coarse a grid under-resolves the pattern and the
5-point Laplacian shifts it toward small, grid-aligned spots, so compare
against the training data (--residuals). Output names get a _gridN suffix.
--residuals needs N at least the training grid size (config "points").

--on-training-grid brings the simulation down to the training grid (config
"points"), by the same interpolation the training data took, before writing
the feql_sim animation, so it can be compared frame for frame with the
training data. It is still SIMULATED at the native grid or at --grid N; only
the animation is downsampled. Output names get an _onP suffix (P = points).

Keep this file next to train_binn_eql.py: it reuses that script's config,
data loading and model construction, so each model is rebuilt exactly as it
was trained.
"""
import argparse
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

import train_binn_eql as train          # also puts the repo root on sys.path
from modules.binn_eql.equations.extract import gate_values
from modules.binn_eql.equations.prune import fine_tune_eql
from modules.simulation.animation import animate_residuals, animate_u_array
from modules.simulation.reaction_library import REACTION_REGISTRY
from modules.simulation.simulation import simulate_feql
from modules.utils.format_data import format_training_data_to_u_array
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split

WEIGHTS_NAME = 'binn_best_val_model'

# simulate_feql integrates on CUDA when it is available, so the model must live
# on the same device.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrueReaction(nn.Module):
    """The ground-truth reaction from REACTION_REGISTRY, in the learned reaction's
    interface: (n_points, species) in, (n_points, species) out."""

    def __init__(self, fn, params):
        super().__init__()
        self.fn, self.params = fn, params

    def forward(self, uv):
        return torch.stack(self.fn(uv, self.params), dim=1)


def parse_term(term, names):
    """'1' -> (0, 0); 'u' -> (1, 0); 'u^2*v' -> (2, 1), for species names (u, v)."""
    powers = [0] * len(names)
    if term.strip() == '1':
        return tuple(powers)
    for factor in term.split('*'):
        name, _, k = factor.strip().partition('^')
        if name not in names:
            raise ValueError(f"unknown species {name!r} in term {term!r}; species are {names}")
        powers[names.index(name)] += int(k) if k else 1
    return tuple(powers)


def parse_override(text):
    """'u:u^2*v=1.0' -> ('u', 'u^2*v', 1.0)."""
    try:
        target, value = text.rsplit('=', 1)
        species, term = target.split(':', 1)
        return species.strip(), term.strip(), float(value)
    except ValueError:
        raise ValueError(f"--set expects SPECIES:TERM=VALUE, e.g. 'u:1=1.0'; got {text!r}")


@torch.no_grad()
def set_coefficient(binn, species, term, value):
    """
    Make the effective coefficient of one polynomial term exactly `value`.
    Duplicate copies of the monomial are cleared so the value lives in one
    column; the gate is opened for a nonzero value and closed for zero.
    """
    eql = binn.reaction.eql_layer
    names = binn.species_names
    if eql.poly is None:
        raise ValueError("--set needs a polynomial library")
    if species not in names:
        raise ValueError(f"unknown species {species!r}; species are {names}")

    role, row = eql.species_role[names.index(species)]
    if role != 'free':
        raise ValueError(f"{species} mirrors another species (mcas); set that one instead")

    powers = parse_term(term, names)
    if powers not in eql.poly.powers:
        raise ValueError(f"term {term!r} is not in the library (degree {binn.degree}"
                         + ("" if eql.include_constant else ", no constant") + ")")
    i = eql.poly.powers.index(powers)
    n_single = len(eql.poly.powers)

    for d in range(eql.duplicates):
        eql.fc.weight[row, i + d * n_single] = 0.0
    log_alpha = eql.l0_gates[row].log_alpha
    log_alpha[i] = 10.0 if value != 0 else -10.0
    gate = gate_values(eql.l0_gates[row])[i].item()
    eql.fc.weight[row, i] = value / gate if value != 0 else 0.0


def load_model(run_dir, cfg, training_data):
    """Rebuild the BINN exactly as training did and load its best-validation weights."""
    train_data, _ = training_test_split(training_data, DEVICE)
    binn = train.build_binn(cfg, train_data).to(DEVICE)
    binn.load_state_dict(torch.load(run_dir / WEIGHTS_NAME, map_location=DEVICE))
    binn.eval()
    return binn


def native_points(data, dimensions):
    """Grid points per side of rows [x..., t, u...] (number of distinct x values)."""
    return len(torch.unique(data[:, 0]))


def resimulate(run_dir, prune=True, residuals=False, dt_cap=None, true_reaction=False,
               overrides=(), grid=None, on_training_grid=False):
    run_dir = Path(run_dir)
    if not (run_dir / WEIGHTS_NAME).exists():
        raise FileNotFoundError(f"no {WEIGHTS_NAME} in {run_dir}")

    cfg = train.RunConfig.load(run_dir / 'config.json')
    raw_data, training_data = train.load_training_data(cfg)
    binn = load_model(run_dir, cfg, training_data)

    # The grid the learned PDE is integrated on: native unless --grid asks for less.
    native = native_points(raw_data, cfg.dimensions)
    sim_data, suffix = raw_data, ''
    if grid is not None and grid < native:
        if (residuals or on_training_grid) and cfg.points and grid < cfg.points:
            raise ValueError(f"--residuals and --on-training-grid need --grid >= the "
                             f"training grid ({cfg.points}); got {grid}")
        # Same interpolation path as the training data, with no noise.
        sim_data = noise_and_interpolate(raw_data, grid, 0.0, cfg.dimensions, cfg.species,
                                         multiplicative_noise=True)
        suffix = f'_grid{grid}'
        print(f"Simulating on a {grid} x {grid} grid (native {native} x {native}).", flush=True)
    elif grid is not None:
        print(f"--grid {grid} is not below the native {native} x {native}; "
              f"simulating at native resolution.", flush=True)

    if true_reaction:
        # Keep the model's grid, scales and diffusion; swap only the reaction.
        spec = REACTION_REGISTRY[cfg.reaction]
        binn.reaction = TrueReaction(spec['fn'], cfg.params)
        print(f"\nSimulating the TRUE {cfg.reaction} reaction, params {cfg.params}\n", flush=True)
        gif_name = 'feql_sim_true_reaction.gif'
    else:
        # binn_best_val_model is saved before post-hoc pruning; training
        # simulates the pruned equation, so do the same unless asked not to.
        if prune:
            fine_tune_eql(binn, threshold=train.PRUNE_THRESHOLD, epsilon=train.HILL_MERGE_EPSILON)
        for species, term, value in overrides:
            set_coefficient(binn, species, term, value)
        label = ('pruned' if prune else 'unpruned') + (', with overrides' if overrides else '')
        print(f"\nEquation simulated ({label}):")
        for line in binn.equations_as_strings():
            print(f"  {line}")
        print(flush=True)
        gif_name = 'feql_sim_new.gif'
        if overrides:
            slug = '__'.join(f"{sp}-{t}={v:g}" for sp, t, v in overrides)
            gif_name = f"feql_sim_set_{re.sub(r'[^A-Za-z0-9.=_-]+', '', slug.replace('^', 'p'))}.gif"

    # simulate_feql only reads model.model, so a bare namespace stands in for
    # the Trainer it is normally given.
    model = types.SimpleNamespace(model=binn)
    if dt_cap is None:
        dt_cap = REACTION_REGISTRY[cfg.reaction]['dt_cap']
    sim_u, sim_x, sim_t = simulate_feql(sim_data, model, dt_cap=dt_cap)

    # The simulation on the training grid, for the animation and/or residuals.
    coarse = None
    if residuals or on_training_grid:
        coarse = train.to_training_grid(sim_u, sim_x, sim_t, cfg)

    if on_training_grid:
        side = coarse.shape[1]
        out = run_dir / gif_name.replace('.gif', f'{suffix}_on{side}.gif')
        animate_u_array(coarse, sim_t, str(out))
        print(f"Wrote {out} (downsampled to the {side} x {side} training grid)", flush=True)
    else:
        out = run_dir / gif_name.replace('.gif', f'{suffix}.gif')
        animate_u_array(sim_u, sim_t, str(out))
        print(f"Wrote {out}", flush=True)

    if residuals:
        u_array, _, _ = format_training_data_to_u_array(training_data)
        mismatch = (coarse[0] - u_array[0]).abs().max().item()
        expect = ("" if cfg.epsilon else
                  "  (small: two interpolation steps instead of one)" if suffix else
                  "  (should be ~0)")
        print(f"Sim vs training data at t=0: max |difference| = {mismatch:.3e}{expect}",
              flush=True)
        out = run_dir / f'feql_residuals_new{suffix}.gif'
        animate_residuals(coarse, u_array, sim_t, name=str(out))
        print(f"Wrote {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('run_dirs', nargs='+',
                        help=f'run directories, each containing config.json and {WEIGHTS_NAME}')
    parser.add_argument('--no-prune', action='store_true',
                        help='simulate the equation as trained, before post-hoc pruning')
    parser.add_argument('--residuals', action='store_true',
                        help='also write feql_residuals_new.gif against the training data')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='SPECIES:TERM=VALUE',
                        help="override one learned coefficient, e.g. 'u:1=1.0'; repeatable")
    parser.add_argument('--true-reaction', action='store_true',
                        help='simulate the ground-truth reaction instead of the learned one')
    parser.add_argument('--grid', type=int, metavar='N',
                        help='simulate on an N x N grid instead of the native one '
                             '(faster, less accurate; combine with a larger --dt-cap)')
    parser.add_argument('--on-training-grid', action='store_true',
                        help='write the feql_sim animation downsampled to the training grid '
                             '(config "points") instead of at the simulation grid')
    parser.add_argument('--dt-cap', type=float,
                        help="timestep ceiling (default: the reaction's dt_cap from "
                             "REACTION_SPECS); the diffusion CFL limit still applies")
    args = parser.parse_args()
    overrides = [parse_override(o) for o in args.overrides]
    if overrides and args.true_reaction:
        parser.error("--set and --true-reaction cannot be combined")

    print(f"Device: {DEVICE}"
          + (f", {torch.get_num_threads()} CPU thread(s)" if DEVICE == 'cpu' else ""),
          flush=True)

    failed = []
    for run_dir in args.run_dirs:
        print("=" * 90 + f"\n{run_dir}", flush=True)
        try:
            resimulate(run_dir, prune=not args.no_prune, residuals=args.residuals,
                       dt_cap=args.dt_cap, true_reaction=args.true_reaction,
                       overrides=overrides, grid=args.grid,
                       on_training_grid=args.on_training_grid)
        except FloatingPointError as exc:
            # The learned equation blows up; report it and move on.
            print(f"Learned PDE could not be simulated: {exc}", flush=True)
            failed.append(run_dir)
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed.append(run_dir)

    if failed:
        print(f"\n{len(failed)} of {len(args.run_dirs)} run(s) did not complete:")
        for run_dir in failed:
            print(f"  {run_dir}")
        sys.exit(1)


if __name__ == '__main__':
    main()