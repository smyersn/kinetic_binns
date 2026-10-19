"""
Put baseline runs and BINN-EQL runs into one table, scored the same way.

    python collect_benchmark.py <dir> [<dir> ...] [--binn] [--simulate] [--out benchmark]
                                [--device cpu]

Walks every <dir> for run folders (a folder with config.json):

  baselines  read from result.json (written by run_baseline.py)
  BINN runs  only with --binn: the model is rebuilt from config.json and
             binn_best_val_model (as resimulate_feql.py does), pruned the same way
             training prunes it, and scored with common.score_reaction. Its
             polynomial coefficients are read off by projecting the pruned
             reaction onto the monomials, which is exact for polynomial-only
             libraries; with Hill terms only the reaction-function error is
             reported. --simulate also integrates the learned PDE (slow; GPU
             recommended) and writes feql_sim_benchmark.gif in the run folder.
             Scores are cached in <run dir>/benchmark_score.json.

Writes <out>_runs.csv (one row per run) and <out>_summary.csv (mean and spread
over repeats for each dataset x method x noise x grid), and prints the summary.

For SINDy methods, 'path_has_true_support' says whether ANY threshold on the
saved path gave the true support for every species: if not, no selection rule
could have found it, and the failure is the method's, not the rule's.

Keep this file next to train_binn_eql.py (modules/scripts/).
"""
import argparse
import json
import re
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "modules").is_dir())
sys.path.append(str(REPO_ROOT))

from modules.benchmarks.common import (SPECIES_NAMES, BaselineConfig,  # noqa: E402
                                       coefficients_to_dicts, load_grid_data, library_powers,
                                       monomials, save_json, score_reaction, simulate_model,
                                       term_name, true_coefficients)

BINN_WEIGHTS = 'binn_best_val_model'


# ---------------------------------------------------------------------------
# BINN runs
# ---------------------------------------------------------------------------
def score_binn(run_dir, device, simulate):
    """The same result fields run_baseline.py writes, for a trained BINN-EQL run."""
    import torch
    import train_binn_eql as train
    from modules.binn_eql.equations.prune import fine_tune_eql
    from modules.utils.training_test_split import training_test_split

    cache = run_dir / 'benchmark_score.json'
    if cache.exists():
        with open(cache) as f:
            cached = json.load(f)
        if 'simulation' in cached or not simulate:
            return cached

    tcfg = train.RunConfig.load(run_dir / 'config.json')
    seed = int(re.search(r'repeat_(\d+)', run_dir.name).group(1)) if 'repeat_' in run_dir.name else 0
    cfg = BaselineConfig(method='binn_eql', training_data_path=tcfg.training_data_path,
                         reaction=tcfg.reaction, params=tcfg.params, epsilon=tcfg.epsilon,
                         points=tcfg.points, diff_coeffs=list(tcfg.diff_coeffs or []),
                         degree=tcfg.degree, include_constant=tcfg.include_constant,
                         mcas=tcfg.mcas, species=tcfg.species, dimensions=tcfg.dimensions,
                         seed=seed)
    grid = load_grid_data(cfg)

    # Rebuild exactly as trained, load the best-validation weights, prune.
    _, training_data = train.load_training_data(tcfg)
    train_data, _ = training_test_split(training_data, device)
    binn = train.build_binn(tcfg, train_data).to(device)
    binn.load_state_dict(torch.load(run_dir / BINN_WEIGHTS, map_location=device))
    binn.eval()
    fine_tune_eql(binn, threshold=train.PRUNE_THRESHOLD, epsilon=train.HILL_MERGE_EPSILON)

    states = grid.reference_states
    with torch.no_grad():
        F_hat = binn.reaction(torch.tensor(states, dtype=torch.float32, device=device))
    F_hat = F_hat.double().cpu().numpy()

    D = None
    if binn.learns_diffusion:
        with torch.no_grad():
            D = [float(d) for d in binn.diffusion_fitter()]

    poly_only = not (tcfg.include_increasing_hill or tcfg.include_decreasing_hill)
    est = None
    coefficients = None
    if poly_only and tcfg.include_poly:
        powers = library_powers(cfg)
        Phi = monomials(states, powers)
        coef = np.linalg.lstsq(Phi, F_hat, rcond=None)[0].T            # (species, terms)
        # Pruned terms come back as rounding noise: drop a term whose contribution
        # is below 1e-6 of its species' reaction.
        contrib = np.abs(coef) * np.sqrt(np.mean(Phi ** 2, axis=0))[None, :]
        scale = np.sqrt(np.mean(F_hat ** 2, axis=0))[:, None]
        coef[contrib < 1e-6 * scale] = 0.0
        est = coefficients_to_dicts(coef, powers)
        coefficients = {SPECIES_NAMES[s]: {term_name(p): c for p, c in d.items()}
                        for s, d in enumerate(est)}

    if D is not None:
        # True D is not in a BINN config; read it from the data file name if it is there.
        m = re.search(r'du_([0-9.eE+-]+?)_dv_([0-9.eE+-]+?)(?:_|\.pt$)',
                      Path(tcfg.training_data_path).name)
        cfg.true_diff_coeffs = [float(m.group(1)), float(m.group(2))] if m else []
    result = {'method': 'binn_eql', 'config': vars(cfg).copy(), 'coefficients': coefficients,
              'diffusion': D if D is not None else cfg.diff_coeffs,
              'metrics': score_reaction(est, cfg, states, D, F_hat=F_hat)}

    if simulate:
        result['simulation'] = simulate_model(types.SimpleNamespace(model=binn), cfg, grid,
                                              f'{run_dir}/feql_sim_benchmark.gif')
    save_json(cache, result)
    return result


# ---------------------------------------------------------------------------
# One table row per run
# ---------------------------------------------------------------------------
def path_has_true_support(result):
    """Whether any threshold on a saved STLSQ path gives the true support everywhere."""
    cfg = result['config']
    truth = true_coefficients(cfg['reaction'], cfg['params'])
    if truth is None or 'path' not in result:
        return None
    want = {SPECIES_NAMES[s]: {term_name(p) for p in d} for s, d in enumerate(truth)}
    for entry in result['path']:
        if all({k for k in entry['coef'][n] if not k.startswith('lap(')} == want[n]
               for n in SPECIES_NAMES):
            return True
    return False


def variant_name(run_dir):
    """The method name as it appears in the folder name: 'pdefind_sg', 'binn_eql', ..."""
    m = re.match(r'(.+?)_(?:batch_|l0_|epsilon_|points_|degree_|repeat_)', run_dir.name)
    return m.group(1) if m else run_dir.name


def row(run_dir, result):
    cfg = result['config']
    m = result['metrics']
    sim = result.get('simulation') or {}
    rep = re.search(r'repeat_(\d+)', run_dir.name)
    r = {
        'run_dir': str(run_dir),
        'dataset': Path(cfg['training_data_path']).stem,
        'reaction': cfg['reaction'],
        'method': variant_name(run_dir),
        'epsilon': cfg['epsilon'],
        'points': cfg['points'],
        'learn_D': not cfg['diff_coeffs'],
        'repeat': int(rep.group(1)) if rep else None,
        'precision': m.get('precision'),
        'recall': m.get('recall'),
        'exact_support': m.get('exact_support'),
        'path_has_true_support': path_has_true_support(result),
        'runtime_s': result.get('runtime_s'),
        'sim_status': sim.get('status'),
    }
    for n in SPECIES_NAMES:
        s = m['species'][n]
        r[f'F_err_{n}'] = s['F_rel_error']
        r[f'n_terms_{n}'] = s['n_terms']
        r[f'coef_err_mean_{n}'] = s.get('coef_rel_error_mean')
        r[f'missing_{n}'] = ' '.join(s.get('missing_terms', []))
        r[f'spurious_{n}'] = ' '.join(s.get('spurious_terms', []))
        if sim.get('status') == 'ok':
            r[f'sim_final_err_{n}'] = sim['final_frame_rel_error'][n]
            r[f'wavelength_learned_{n}'] = sim['learned_final'][n]['wavelength']
            r[f'wavelength_true_{n}'] = sim['true_final'][n]['wavelength']
    if isinstance(m.get('diffusion'), dict):
        for n, d in m['diffusion'].items():
            r[f'D_learned_{n}'] = d['learned']
            r[f'D_rel_error_{n}'] = d['rel_error']
    return r


def find_runs(dirs):
    for d in dirs:
        for cfg_path in sorted(Path(d).rglob('config.json')):
            yield cfg_path.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--binn', action='store_true', help='also score BINN-EQL runs')
    ap.add_argument('--simulate', action='store_true', help='simulate BINN runs too')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='benchmark')
    args = ap.parse_args()

    import pandas as pd

    rows, skipped = [], []
    for run_dir in find_runs(args.dirs):
        try:
            if (run_dir / 'result.json').exists():
                with open(run_dir / 'result.json') as f:
                    result = json.load(f)
            elif args.binn and (run_dir / BINN_WEIGHTS).exists():
                print(f"Scoring BINN run {run_dir}", flush=True)
                result = score_binn(run_dir, args.device, args.simulate)
            else:
                skipped.append((run_dir, 'no result.json' + ('' if args.binn else
                                                             ' (BINN runs need --binn)')))
                continue
            rows.append(row(run_dir, result))
        except Exception as exc:                    # one broken run should not stop the table
            skipped.append((run_dir, f"{type(exc).__name__}: {exc}"))

    for run_dir, why in skipped:
        print(f"skipped {run_dir}: {why}")
    if not rows:
        print("No scored runs found.")
        return

    runs = pd.DataFrame(rows)
    runs.to_csv(f'{args.out}_runs.csv', index=False)

    group = ['dataset', 'method', 'epsilon', 'points', 'learn_D']
    metrics = [c for c in ['precision', 'recall', 'exact_support', 'path_has_true_support',
                           'F_err_u', 'F_err_v', 'coef_err_mean_u', 'coef_err_mean_v',
                           'sim_final_err_u', 'sim_final_err_v', 'runtime_s'] if c in runs]
    numeric = runs[group + metrics].copy()
    for c in metrics:
        numeric[c] = pd.to_numeric(numeric[c].astype(float), errors='coerce')
    summary = numeric.groupby(group).agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(f'{args.out}_summary.csv')

    show = [c for c in ['precision_mean', 'recall_mean', 'exact_support_mean',
                        'F_err_u_mean', 'F_err_v_mean', 'sim_final_err_u_mean',
                        'precision_count'] if c in summary]
    with pd.option_context('display.max_rows', 500, 'display.max_columns', 20,
                           'display.width', 250,
                           'display.float_format', '{:.3g}'.format):
        print(summary[show])
    print(f"\nWrote {args.out}_runs.csv ({len(runs)} runs) and {args.out}_summary.csv")


if __name__ == '__main__':
    main()
