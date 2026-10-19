"""
Run one equation-learning baseline (PDE-FIND, weak SINDy or DeepMoD) and score it.

    python run_baseline.py <run_dir>

<run_dir>/config.json is written by sweep_baselines.py (fields: see
modules/benchmarks/common.py, BaselineConfig). Writes to <run_dir>:

    result.json     coefficients, metrics, runtime, the STLSQ path (SINDy methods),
                    and, once the learned PDE has been simulated, 'simulation'
    equation.txt    the selected equations
    feql_sim.gif    the learned PDE integrated from the raw initial condition
    deepmod_model.pt, deepmod_tensorboard/   (DeepMoD only)

The fit is written to result.json before the simulation starts, so a killed or
diverging simulation never loses the fit. Rerunning a directory that already has
a simulated result.json does nothing; one without a simulation only simulates.
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "modules").is_dir())
sys.path.append(str(REPO_ROOT))

from modules.benchmarks.common import (BaselineConfig, load_grid_data,  # noqa: E402
                                       path_to_json, save_json, score_reaction,
                                       simulate_and_score, write_result)


def fit(cfg, grid, run_dir):
    if cfg.method == 'pdefind':
        from modules.benchmarks import pdefind
        return pdefind.run(cfg, grid)
    if cfg.method == 'weak_sindy':
        from modules.benchmarks import weak_sindy
        return weak_sindy.run(cfg, grid)
    if cfg.method == 'deepmod':
        from modules.benchmarks import deepmod
        return deepmod.run(cfg, grid, run_dir)
    raise ValueError(f"unknown method {cfg.method!r}; use pdefind, weak_sindy or deepmod")


def reaction_from_json(result):
    """The selected reaction back as {exponent tuple: value} dicts (for a resumed simulation)."""
    from modules.benchmarks.common import SPECIES_NAMES, library_powers, term_name
    cfg = BaselineConfig(**result['config'])
    by_name = {term_name(p): p for p in library_powers(cfg)}
    return [{by_name[k]: v for k, v in result['coefficients'][s].items()} for s in SPECIES_NAMES]


def main(run_dir):
    run_dir = Path(run_dir)
    cfg = BaselineConfig.load(run_dir / 'config.json')
    result_path = run_dir / 'result.json'

    result = None
    if result_path.exists():
        with open(result_path) as f:
            result = json.load(f)
        if 'simulation' in result or not cfg.simulate:
            print(f"{result_path} is complete; nothing to do.", flush=True)
            return
        print("Found a fit without a simulation; simulating only.", flush=True)

    grid = load_grid_data(cfg)

    if result is None:
        started = time.time()
        coefs, diffusion, extra = fit(cfg, grid, run_dir)
        extra['metrics'] = score_reaction(coefs, cfg, grid.reference_states, diffusion)
        extra['pde_t_cutoff'] = grid.t_cutoff
        if 'path' in extra:
            extra['path'] = path_to_json(extra['path'], extra['feature_names'])
        result = write_result(run_dir, cfg, coefs, diffusion, extra, started)
        m = extra['metrics']
        if 'precision' in m:
            print(f"precision {m['precision']:.2f}, recall {m['recall']:.2f}, "
                  f"exact support {m['exact_support']}", flush=True)
        print("F relative error: " + ", ".join(
            f"{k} {v['F_rel_error']:.3g}" for k, v in m['species'].items()), flush=True)
    else:
        coefs, diffusion = reaction_from_json(result), result['diffusion']

    if cfg.simulate:
        sim = simulate_and_score(coefs, diffusion, cfg, grid, run_dir)
        result['simulation'] = sim
        save_json(result_path, result)
        print(f"Simulation: {sim['status']}"
              + (f", final-frame relative error {sim['final_frame_rel_error']}"
                 if sim['status'] == 'ok' else f" ({sim.get('reason', '')})"), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
