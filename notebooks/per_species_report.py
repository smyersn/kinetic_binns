"""
Per-species breakdown of what a trained reaction term is doing.

For each species it reports the PDE residual with and without that species'
reaction, the share of the explainable range it accounts for, and how many
gates are still open on its row. That separates the two ways an equation
ends up zeroed:

    explainable range ~ 0    its terms could never help: u_t is dominated by
                             diffusion or by surface error (check pde_scale)
    range healthy, no gates  the L0 penalty, priced from the total across
                             species, exceeded what its terms earn alone

As a library:

    from per_species_report import per_species_report
    per_species_report(binn, train_data)     # binn already holds the weights

As a script, over every run under one or more sweep directories:

    python per_species_report.py <sweep_dir> [<sweep_dir> ...]
    python per_species_report.py nonconserved/ --cache-size 50000 --repeats 1

Runs are matched to DATASETS below by their config's training_data_path;
pass --all-datasets to report on every run found instead. The model is read
from binn_best_val_model as trained, i.e. before post-hoc pruning.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "modules").is_dir())
sys.path.append(str(REPO_ROOT))

from modules.binn_eql.equations.extract import effective_weights  # noqa: E402
from modules.binn_eql.model.binn import BINN  # noqa: E402
from modules.binn_eql.physics.collocation import CollocationCache  # noqa: E402
from modules.binn_eql.physics.losses import BINNLoss  # noqa: E402
from modules.utils.noise_and_interpolate import noise_and_interpolate  # noqa: E402
from modules.utils.training_test_split import training_test_split  # noqa: E402

DATASETS = [
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/brusselator_du_0.02_dv_0.16_a_1.0_b_3.0.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/gray_scott_du_0.0004_dv_0.0002_feed_0.018_kill_0.051.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/gray_scott_du_0.0004_dv_0.0002_feed_0.029_kill_0.057.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/gray_scott_du_0.0004_dv_0.0002_feed_0.03_kill_0.062.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/nonconserved/schnakenberg_du_0.01_dv_0.1_a_0.1_b_0.8.pt",
]

WEIGHTS_NAME = 'binn_best_val_model'


# ----------------------------------------------------------------------
# Report for one model
# ----------------------------------------------------------------------
@torch.no_grad()
def _residual_per_species(binn, batch, zero_reaction=False):
    """Smooth-L1 PDE residual of each species, as a list of floats."""
    eql = binn.reaction.eql_layer
    saved = eql.fc.weight.data.clone()
    if zero_reaction:
        eql.fc.weight.data.zero_()
    try:
        reaction = binn.reaction(batch.outputs)
        D = binn.diffusion_coefficients()
        losses = []
        for s in range(binn.species):
            laplacian = D[s] * batch.u_xx[s].sum(dim=1, keepdim=True)
            residual = batch.u_t[:, s:s + 1] - (laplacian + reaction[:, s:s + 1])
            residual = residual / torch.sqrt(binn.pde_scale[s])
            losses.append(F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0).item())
        return losses
    finally:
        eql.fc.weight.data.copy_(saved)


def species_breakdown(binn, train_data, cache_size=200_000):
    """
    Per-species diagnostics for a model that already holds its weights.
    Returns a list of dicts, one per species.
    """
    loss_fn = BINNLoss(binn)
    # Build the cache first so calibrate_physics only measures the scales;
    # a smaller pool keeps a sweep-wide report tractable.
    loss_fn.cache = CollocationCache(binn, binn.collocation_t_min(),
                                     loss_fn.mass_t_cutoff, size=cache_size)
    loss_fn.calibrate_physics(train_data)

    batch = loss_fn.cache.data
    fitted = _residual_per_species(binn, batch)
    null = _residual_per_species(binn, batch, zero_reaction=True)
    explainable = [n - f for n, f in zip(null, fitted)]
    total = sum(e for e in explainable if e > 0) or 1.0

    eql = binn.reaction.eql_layer
    rows = []
    for s in range(binn.species):
        role, _ = eql.species_role[s]
        _, gates, effective = effective_weights(eql, s)
        effective = effective.cpu().numpy()
        rows.append({
            'species': binn.species_names[s] + ('' if role == 'free' else ' (mirror)'),
            'null': null[s],
            'fitted': fitted[s],
            'explainable': explainable[s],
            'share': explainable[s] / total,
            'open_gates': int((gates.cpu().numpy() > 0.5).sum()),
            'max_coeff': float(np.abs(effective).max()),
            'pde_scale': binn.pde_scale[s].item(),
        })
    return rows


def print_breakdown(rows):
    print(f"{'species':>12} {'PDE null':>12} {'PDE fitted':>12} {'explained':>12} "
          f"{'share':>8} {'gates':>7} {'max |w*z|':>11} {'pde_scale':>11}")
    for r in rows:
        print(f"{r['species']:>12} {r['null']:12.4e} {r['fitted']:12.4e} "
              f"{r['explainable']:12.4e} {100 * r['share']:7.1f}% {r['open_gates']:7d} "
              f"{r['max_coeff']:11.3e} {r['pde_scale']:11.3e}")


def per_species_report(binn, train_data, cache_size=200_000):
    """Print the breakdown for one already-loaded model."""
    rows = species_breakdown(binn, train_data, cache_size)
    print_breakdown(rows)
    print("\nNear-zero explained: the species' u_t is dominated by diffusion or surface "
          "error, so its reaction was never worth fitting (compare pde_scale across species).")
    print("Explained but no open gates: the L0 price, set from the total across species, "
          "exceeded what this row's terms earn on their own.\n")
    return rows


# ----------------------------------------------------------------------
# Batch mode
# ----------------------------------------------------------------------
def find_runs(roots, dataset_paths=None, repeats=None, stats=None):
    """
    Run directories under roots holding both a config.json and trained weights.
    stats: optional dict, filled with per-stage counts so a caller can explain
    an empty result.
    """
    wanted = None if dataset_paths is None else {Path(p).stem for p in dataset_paths}
    stats = {} if stats is None else stats
    stats.update(missing_roots=[], configs=0, with_weights=0, datasets_seen=set())
    runs = []

    for root in roots:
        if not Path(root).is_dir():
            stats['missing_roots'].append(str(root))
            continue
        for config_path in sorted(Path(root).rglob('config.json')):
            stats['configs'] += 1
            run_dir = config_path.parent
            if not (run_dir / WEIGHTS_NAME).exists():
                continue
            stats['with_weights'] += 1
            config = json.loads(config_path.read_text())
            stem = Path(config['training_data_path']).stem
            stats['datasets_seen'].add(stem)
            if wanted is not None and stem not in wanted:
                continue
            if repeats is not None and not run_dir.name.endswith(
                    tuple(f'repeat_{r}' for r in repeats)):
                continue
            runs.append((run_dir, config, stem))
    return runs


def _explain_empty(roots, stats):
    """Which stage of the search came up empty, so the fix is obvious."""
    if stats['missing_roots']:
        return (f"These directories do not exist (relative paths are resolved from the "
                f"current working directory, {Path.cwd()}):\n  "
                + "\n  ".join(stats['missing_roots']))
    if stats['configs'] == 0:
        return (f"No config.json anywhere under {', '.join(map(str, roots))}. Point this at "
                f"the sweep output directory, the one holding <dataset>/binn_eql_.../ runs.")
    if stats['with_weights'] == 0:
        return (f"Found {stats['configs']} run director(ies), but none contains "
                f"{WEIGHTS_NAME}: those runs have not reached Phase 4 yet, or they failed.")
    return ("Runs were found, but none matched the filters. Datasets present:\n  "
            + "\n  ".join(sorted(stats['datasets_seen']))
            + "\nUse --all-datasets to include them, or edit DATASETS in this file. "
              "If you passed --repeats, check those repeat numbers exist.")


def load_training_data(config, cache):
    """Training data exactly as the run saw it (same noise and interpolation)."""
    key = (config['training_data_path'], config['points'], config['epsilon'])
    if key not in cache:
        data = torch.load(config['training_data_path'])['training_data']
        if config['epsilon'] != 0 or config['points'] != 0:
            data = noise_and_interpolate(data, config['points'], config['epsilon'],
                                         config['dimensions'], config['species'],
                                         multiplicative_noise=True)
        cache[key] = data
    return cache[key]


def build_model(run_dir, config, train_data, device):
    binn = BINN(
        dimensions=config['dimensions'], species=config['species'], train_data=train_data,
        duplicates=config['duplicates'], diff_coeffs=config['diff_coeffs'],
        degree=config['degree'], param_bounds=config['param_bounds'],
        mcas=config.get('mcas', False),
        l0_reference_gates=config.get('l0_reference_gates'),
        gls_max_weight=config.get('gls_max_weight', 50.0),
        include_poly=config.get('include_poly', True),
        include_increasing_hill=config.get('include_increasing_hill', True),
        include_decreasing_hill=config.get('include_decreasing_hill', True)).to(device)

    binn.load_state_dict(torch.load(run_dir / WEIGHTS_NAME, map_location=device))
    binn.eval()
    return binn


def main(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = None if args.all_datasets else DATASETS
    stats = {}
    runs = find_runs(args.roots, datasets, args.repeats, stats)
    if not runs:
        raise SystemExit(_explain_empty(args.roots, stats))

    print(f"{len(runs)} run(s) on {device}, {args.cache_size:,} collocation points each\n")
    data_cache, summary = {}, []

    for run_dir, config, stem in runs:
        print("=" * 100)
        print(f"{stem}  |  {run_dir.name}")
        try:
            data = load_training_data(config, data_cache)
            train_data, _ = training_test_split(data, device)
            binn = build_model(run_dir, config, train_data, device)
            rows = species_breakdown(binn, train_data, args.cache_size)
            print_breakdown(rows)
            summary.append((stem, run_dir.name, rows))
        except Exception as exc:                      # keep going through the sweep
            print(f"  SKIPPED: {type(exc).__name__}: {exc}")
        finally:
            if device == 'cuda':
                torch.cuda.empty_cache()

    # One line per species per run, grouped by dataset, for scanning at the end.
    print("\n" + "=" * 100)
    print("SUMMARY: share of the explainable PDE range, and open gates, per species\n")
    print(f"{'dataset':>46} {'run':>34} {'species':>10} {'share':>8} {'gates':>7}")
    for stem, run_name, rows in summary:
        for r in rows:
            print(f"{stem[-46:]:>46} {run_name[-34:]:>34} {r['species']:>10} "
                  f"{100 * r['share']:7.1f}% {r['open_gates']:7d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('roots', nargs='+', help='sweep directories to search for runs')
    parser.add_argument('--all-datasets', action='store_true',
                        help='report every run found, not just the DATASETS listed in this file')
    parser.add_argument('--repeats', type=int, nargs='+',
                        help='only these repeat numbers (default: all)')
    parser.add_argument('--cache-size', type=int, default=200_000,
                        help='collocation points per model (default 200000, as in training)')
    parser.add_argument('--device', help="'cuda' or 'cpu' (default: cuda if available)")
    main(parser.parse_args())