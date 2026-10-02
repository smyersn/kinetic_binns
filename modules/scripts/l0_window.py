"""
Which l0_weight can separate the true reaction terms from spurious ones?

    python l0_window.py <sweep_dir> [--repeats 1] [--cache-size 50000]
                        [--device cuda] [--ref 51] [--seed 0]

For every trained run under <sweep_dir> (only the first --repeats repeats of
each configuration), this evaluates the frozen surface fitter on a fixed
collocation pool, takes the PDE targets

    y_s = du_s/dt - D_s lap(u_s)       (what the reaction must explain)

and, by unpenalized least-squares fits over the run's polynomial library
(one copy of each monomial), measures how much each candidate term is worth.
All losses are the training PDE loss: smooth-L1 of the residual divided by
sqrt(pde_scale[s]).

The L0 pricing rule (see physics/calibration.py):

    price per open gate = l0_weight x (PDE_null - PDE_floor) / ref
    a term is kept  <=>  its share of the explainable range > l0_weight / ref

so every share converts directly into an l0_weight bound:

    true term     share = loss increase when it is DROPPED from the true support
                  -> kept only if  l0_weight < ref x share
    spurious term share = loss decrease when it is ADDED to the true support
                  -> excluded only if  l0_weight > ref x share

The window is  ref x (largest spurious share) < l0_weight < ref x (smallest
true share).  If the lower end exceeds the upper, no l0_weight selects the
true equation for that dataset: a spurious term is worth more to the fit than
a true one.

Swap check (identifiability)
----------------------------
Gray-Scott-style lookalikes REPLACE a true term rather than sit beside it
(0.03(1-u) learned as 0.012(1-u^3)). For each true term, the script also
finds the best single replacement from the rest of the library and reports
the swap cost: the loss increase, as a share of the explainable range. A swap
cost far below the true term's own share means the data barely distinguish
the true term from its replacement, whatever l0_weight is.

Both pricing modes are reported: per_species (the default for non-conserved
runs: each species' range prices its own gates) and global (one range, the
sum over species). Polynomial terms only; runs with Hill terms are analysed
over their polynomial part. The PDE "floor" is the full-library least-squares
fit, which stands in for the best Phase-2 residual used in training.

Keep this file next to train_binn_eql.py (modules/scripts/).
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import train_binn_eql as train          # also puts the repo root on sys.path
from modules.binn_eql.physics.collocation import CollocationCache
from modules.utils.training_test_split import training_test_split

WEIGHTS_NAME = 'binn_best_val_model'

# True supports as exponent tuples (power of u, power of v), per species.
TRUE_SUPPORT = {
    'brusselator':  [{(0, 0), (1, 0), (2, 1)}, {(1, 0), (2, 1)}],
    'schnakenberg': [{(0, 0), (1, 0), (2, 1)}, {(0, 0), (2, 1)}],
    'gray_scott':   [{(0, 0), (1, 0), (1, 2)}, {(1, 2), (0, 1)}],
}


# ----------------------------------------------------------------------
# Numerical core (NumPy only)
# ----------------------------------------------------------------------
def smooth_l1(r):
    """Mean smooth-L1 (beta = 1), as in the training PDE loss."""
    a = np.abs(r)
    return float(np.mean(np.where(a < 1.0, 0.5 * a * a, a - 0.5)))


def monomials(u, powers):
    """(n, len(powers)) matrix of u_0^p0 * u_1^p1 * ..."""
    return np.stack([np.prod(u ** np.array(p, dtype=float), axis=1) for p in powers], axis=1)


class SubsetFitter:
    """Least-squares fits of y on column subsets of Phi, scored by smooth-L1, memoized."""

    def __init__(self, Phi, y, sigma):
        self.Phi, self.y, self.sigma = Phi, y, sigma
        self.norms = np.linalg.norm(Phi, axis=0) + 1e-300
        self._cache = {}

    def loss(self, cols):
        key = frozenset(cols)
        if key not in self._cache:
            cols = sorted(key)
            if not cols:
                residual = self.y
            else:
                A = self.Phi[:, cols] / self.norms[cols]
                coef, *_ = np.linalg.lstsq(A, self.y, rcond=None)
                residual = self.y - A @ coef
            self._cache[key] = smooth_l1(residual / self.sigma)
        return self._cache[key]

    def coefficients(self, cols):
        cols = sorted(cols)
        A = self.Phi[:, cols] / self.norms[cols]
        coef, *_ = np.linalg.lstsq(A, self.y, rcond=None)
        return dict(zip(cols, coef / self.norms[cols]))


def analyse_species(fitter, true_cols, n_terms):
    """Loss landmarks and term shares (in loss units) for one species."""
    all_cols = list(range(n_terms))
    null = fitter.loss([])
    floor = fitter.loss(all_cols)
    base = fitter.loss(true_cols)

    true_drop = {t: fitter.loss(set(true_cols) - {t}) - base for t in true_cols}
    spurious_add = {j: base - fitter.loss(set(true_cols) | {j})
                    for j in all_cols if j not in true_cols}
    swaps = {}
    for t in true_cols:
        others = [j for j in all_cols if j not in true_cols]
        if not others:
            continue
        costs = {j: fitter.loss((set(true_cols) - {t}) | {j}) - base for j in others}
        best = min(costs, key=costs.get)
        swaps[t] = (best, costs[best])
    return dict(null=null, floor=floor, base=base, true_drop=true_drop,
                spurious_add=spurious_add, swaps=swaps,
                coefficients=fitter.coefficients(true_cols) if true_cols else {})


def window(result, rng_value, ref):
    """(lower, upper, worst spurious col, weakest true col) l0_weight bounds for a given range."""
    rng_value = max(rng_value, 1e-300)
    weakest = min(result['true_drop'], key=result['true_drop'].get)
    upper = ref * result['true_drop'][weakest] / rng_value
    if result['spurious_add']:
        worst = max(result['spurious_add'], key=result['spurious_add'].get)
        lower = ref * max(result['spurious_add'][worst], 0.0) / rng_value
    else:
        worst, lower = None, 0.0
    return lower, upper, worst, weakest


# ----------------------------------------------------------------------
# Model side
# ----------------------------------------------------------------------
def find_runs(sweep_dir, repeats):
    """{dataset name: [(config stem, run_dir), ...]}, first `repeats` repeats per config."""
    groups = defaultdict(list)
    for cfg_path in sorted(Path(sweep_dir).glob('*/*/config.json')):
        run_dir = cfg_path.parent
        match = re.match(r'(.*)_repeat_(\d+)$', run_dir.name)
        stem, k = (match.group(1), int(match.group(2))) if match else (run_dir.name, 1)
        groups[(run_dir.parent.name, stem)].append((k, run_dir))

    runs = defaultdict(list)
    for (dataset, stem), members in sorted(groups.items()):
        with_weights = [(k, d) for k, d in sorted(members) if (d / WEIGHTS_NAME).exists()]
        for _, run_dir in with_weights[:repeats]:
            runs[dataset].append((stem, run_dir))
    return runs


def pde_targets(run_dir, cfg, device, cache_size, seed):
    """(u, y, sigma, library powers, species names) for one trained run, as NumPy arrays."""
    _, training_data = train.load_training_data(cfg)
    train_data, _ = training_test_split(training_data, device)
    binn = train.build_binn(cfg, train_data).to(device)
    binn.load_state_dict(torch.load(run_dir / WEIGHTS_NAME, map_location=device))
    binn.eval()

    devices = [device] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.enable_grad():
        torch.manual_seed(seed)
        cache = CollocationCache(binn, binn.collocation_t_min(), mass_t_cutoff=0.0,
                                 size=cache_size)
    data = cache.data
    with torch.no_grad():
        D = binn.diffusion_coefficients().reshape(-1)
        y = torch.stack([data.u_t[:, s] - D[s] * data.u_xx[s].sum(dim=1)
                         for s in range(binn.species)], dim=1)
    eql = binn.reaction.eql_layer
    powers = list(eql.poly.powers) if eql.poly is not None else []
    return (data.outputs.detach().double().cpu().numpy(), y.double().cpu().numpy(),
            np.sqrt(binn.pde_scale.double().cpu().numpy().clip(min=1e-300)),
            powers, list(binn.species_names))


def term_name(powers, names):
    factors = [n if k == 1 else f"{n}^{k}" for n, k in zip(names, powers) if k]
    return '*'.join(factors) or '1'


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def analyse_run(run_dir, device, cache_size, seed, ref):
    cfg = train.RunConfig.load(run_dir / 'config.json')
    if cfg.reaction not in TRUE_SUPPORT:
        print(f"  no TRUE_SUPPORT entry for {cfg.reaction!r}; skipped")
        return None
    if cfg.mcas:
        print("  mcas run (one shared F); this script handles non-conserved runs only; skipped")
        return None

    u, y, sigma, powers, names = pde_targets(run_dir, cfg, device, cache_size, seed)
    if not powers:
        print("  no polynomial library in this run; skipped")
        return None
    Phi = monomials(u, powers)

    per_species = []
    for s, support in enumerate(TRUE_SUPPORT[cfg.reaction]):
        missing = [p for p in support if p not in powers]
        if missing:
            print(f"  {names[s]}: true term(s) {[term_name(p, names) for p in missing]} are not "
                  f"in the library (include_constant / degree?); species skipped")
            per_species.append(None)
            continue
        true_cols = [powers.index(p) for p in support]
        fitter = SubsetFitter(Phi, y[:, s], sigma[s])
        per_species.append(analyse_species(fitter, true_cols, len(powers)))

    valid = [r for r in per_species if r is not None]
    global_range = sum(r['null'] - r['floor'] for r in valid)
    windows = {'per_species': [], 'global': []}

    for s, r in enumerate(per_species):
        if r is None:
            continue
        name = lambda c: term_name(powers[c], names)        # noqa: E731
        explainable = r['null'] - r['floor']
        print(f"\n  {names[s]}: PDE null {r['null']:.4g}, floor (full library) {r['floor']:.4g}, "
              f"explainable {explainable:.4g}")
        captured = (r['null'] - r['base']) / max(explainable, 1e-300)
        coefs = ', '.join(f"{name(c)} {v:+.4g}" for c, v in sorted(r['coefficients'].items()))
        print(f"     true support alone: loss {r['base']:.4g} ({100 * captured:.1f}% of range); "
              f"refit coefficients: {coefs}")

        print(f"     {'true term':>10}  {'share':>8}  {'kept if l0 <':>12}   "
              f"{'best stand-in':>13}  {'swap cost':>9}")
        for c, drop in sorted(r['true_drop'].items(), key=lambda kv: kv[1]):
            share = drop / max(explainable, 1e-300)
            stand_in, cost = r['swaps'].get(c, (None, float('nan')))
            flag = '   <-- barely distinguishable' if share > 0 and cost < 0.1 * drop else ''
            print(f"     {name(c):>10}  {share:8.2%}  {ref * share:12.3g}   "
                  f"{name(stand_in) if stand_in is not None else '-':>13}  "
                  f"{cost / max(explainable, 1e-300):9.2%}{flag}")

        ranked = sorted(r['spurious_add'].items(), key=lambda kv: -kv[1])[:3]
        if ranked:
            print(f"     {'spurious':>10}  {'share':>8}  {'dropped if l0 >':>15}")
            for c, gain in ranked:
                share = gain / max(explainable, 1e-300)
                print(f"     {name(c):>10}  {share:8.2%}  {ref * share:15.3g}")

        for mode, rng_value in (('per_species', explainable), ('global', global_range)):
            lower, upper, worst, weakest = window(r, rng_value, ref)
            windows[mode].append((lower, upper))
            verdict = (f"{lower:.3g} < l0_weight < {upper:.3g}" if lower < upper else
                       f"NONE: spurious {name(worst)} is worth more than true {name(weakest)}")
            print(f"     window ({mode:>11} pricing): {verdict}")

    combined = {}
    for mode, spans in windows.items():
        if spans:
            combined[mode] = (max(lo for lo, _ in spans), min(hi for _, hi in spans))
    for mode, (lo, hi) in combined.items():
        print(f"  run window ({mode} pricing, all species): "
              + (f"{lo:.3g} < l0_weight < {hi:.3g}" if lo < hi else f"NONE ({lo:.3g} > {hi:.3g})"))
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('sweep_dir')
    parser.add_argument('--repeats', type=int, default=1,
                        help='repeats per configuration to analyse (default 1)')
    parser.add_argument('--cache-size', type=int, default=50_000,
                        help='collocation points per run (default 50000)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ref', type=float, default=51.0,
                        help='l0_reference_gates used in training (default 51)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    runs = find_runs(args.sweep_dir, args.repeats)
    if not runs:
        print(f"No trained runs found under {args.sweep_dir} "
              f"(looked for */*/config.json with {WEIGHTS_NAME}).")
        sys.exit(1)
    print(f"Device: {args.device}; {sum(len(v) for v in runs.values())} run(s) "
          f"in {len(runs)} dataset(s)")

    summary = defaultdict(dict)
    for dataset, members in runs.items():
        for stem, run_dir in members:
            print("\n" + "=" * 90 + f"\n{dataset}\n  {run_dir.name}")
            try:
                result = analyse_run(run_dir, args.device, args.cache_size, args.seed, args.ref)
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                continue
            if result:
                for mode, span in result.items():
                    summary[mode].setdefault(dataset, []).append(span)

    print("\n" + "=" * 90 + "\nSummary: l0_weight windows (lower, upper)")
    for mode, datasets in summary.items():
        print(f"\n  {mode} pricing")
        overall_lo, overall_hi = 0.0, float('inf')
        for dataset, spans in datasets.items():
            lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
            overall_lo, overall_hi = max(overall_lo, lo), min(overall_hi, hi)
            print(f"    {dataset:60} " + (f"{lo:9.3g} .. {hi:<9.3g}" if lo < hi else
                                          f"NONE ({lo:.3g} > {hi:.3g})"))
        print(f"    {'ALL DATASETS':60} " + (f"{overall_lo:9.3g} .. {overall_hi:<9.3g}"
                                             if overall_lo < overall_hi else
                                             f"NONE ({overall_lo:.3g} > {overall_hi:.3g})"))


if __name__ == '__main__':
    main()