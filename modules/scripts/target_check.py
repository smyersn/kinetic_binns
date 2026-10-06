"""
Are the PDE targets from the surface the right size?

    python target_check.py <run_dir> [<run_dir> ...] [--no-prune]
                           [--cache-size 50000] [--device cuda] [--seed 0]
                           [--csv results.csv]

The reaction is trained to match, at collocation points on the fitted surface,

    target_s = du_s/dt - D_s lap(u_s)

If the surface is smoother than the data, u_t and lap(u) are attenuated, and
so is every coefficient the reaction learns. This script compares three
quantities at the same points, per species:

    true     the ground-truth reaction (REACTION_REGISTRY) at the surface's (u, v)
    target   what the surface says the reaction must be
    learned  the trained reaction (pruned, as in equation.txt, unless --no-prune)

and reports least-squares slopes through the origin and correlations:

    target vs true     ~1: the surface's derivatives are the right size.
                       well below 1: the surface flattens the dynamics, and
                       no amount of reaction training can recover the truth.
    learned vs true    the end-to-end result (e.g. ~0.5 = coefficients halved).
    learned / target   (learned vs true) / (target vs true): the extra scaling
                       added by reaction training (L0 shrinkage and the like).

Both "vs true" slopes regress on the noise-free true reaction, so noise in
the targets does not bias them. (Regressing the learned reaction directly on
the noisy targets would: noise in the regressor drags the slope toward zero.)
For an mcas run the shared F fits both species' targets at once, so compare
its "learned vs true" with the average of the two species' "target vs true".

So: target-vs-true low  -> fix the surface (resolution, smoothing, LR schedule);
    target-vs-true ~1 but learned / target low -> the reaction training or L0.

With several runs it ends with a summary table, one row per (sweep, epsilon,
points), averaged over repeats. "Sweep" is the first folder in the run's path
named like 19_smart_sweep. --csv also writes every run's numbers to a file.

Keep this file next to train_binn_eql.py (modules/scripts/).
"""
import csv
import re
from collections import defaultdict
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import train_binn_eql as train          # also puts the repo root on sys.path
from modules.binn_eql.equations.prune import fine_tune_eql
from modules.binn_eql.physics.collocation import CollocationCache
from modules.simulation.reaction_library import REACTION_REGISTRY
from modules.utils.training_test_split import training_test_split

WEIGHTS_NAME = 'binn_best_val_model'


def slope_and_corr(y, x):
    """Least-squares slope of y on x through the origin, and Pearson correlation."""
    denom = float(np.dot(x, x))
    slope = float(np.dot(x, y)) / denom if denom > 0 else float('nan')
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float('nan')
    return slope, corr


def check(run_dir, prune, cache_size, device, seed):
    run_dir = Path(run_dir)
    cfg = train.RunConfig.load(run_dir / 'config.json')
    _, training_data = train.load_training_data(cfg)
    train_data, _ = training_test_split(training_data, device)
    binn = train.build_binn(cfg, train_data).to(device)
    binn.load_state_dict(torch.load(run_dir / WEIGHTS_NAME, map_location=device))
    binn.eval()
    if prune:
        fine_tune_eql(binn, threshold=train.PRUNE_THRESHOLD, epsilon=train.HILL_MERGE_EPSILON)

    devices = [device] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.enable_grad():
        torch.manual_seed(seed)
        cache = CollocationCache(binn, binn.collocation_t_min(), mass_t_cutoff=0.0,
                                 size=cache_size)
    data = cache.data
    with torch.no_grad():
        D = binn.diffusion_coefficients().reshape(-1)
        target = torch.stack([data.u_t[:, s] - D[s] * data.u_xx[s].sum(dim=1)
                              for s in range(binn.species)], dim=1)
        learned = binn.reaction(data.outputs)
    uv = data.outputs.detach().double().cpu().numpy()
    target = target.double().cpu().numpy()
    learned = learned.double().cpu().numpy()

    true_fn = REACTION_REGISTRY[cfg.reaction]['fn']
    true = np.stack([np.asarray(F, dtype=float) for F in true_fn(uv, cfg.params)], axis=1)

    print(f"\n{run_dir}")
    print(f"  {cfg.reaction}, points={cfg.points}, epsilon={cfg.epsilon}, "
          f"{len(uv)} collocation points, t >= {binn.collocation_t_min():g}"
          + ("" if prune else "  (unpruned)"))
    print(f"  {'species':>8}  {'target vs true':>18}  {'learned vs true':>18}  "
          f"{'learned / target':>16}")
    names = list(binn.species_names)
    zeroed = bool(np.abs(learned).max() < 1e-8)
    row = {'run': str(run_dir), 'sweep': sweep_label(run_dir), 'epsilon': cfg.epsilon,
           'points': cfg.points, 'zeroed': zeroed}
    for s in range(binn.species):
        t_slope, t_corr = slope_and_corr(target[:, s], true[:, s])
        l_slope, l_corr = slope_and_corr(learned[:, s], true[:, s])
        ratio = l_slope / t_slope if t_slope else float('nan')
        print(f"  {names[s]:>8}  {t_slope:6.3f} (r={t_corr:5.3f})  "
              f"{l_slope:6.3f} (r={l_corr:5.3f})  {ratio:16.3f}")
        row.update({f'target_slope_{names[s]}': t_slope, f'target_r_{names[s]}': t_corr,
                    f'learned_slope_{names[s]}': l_slope, f'learned_r_{names[s]}': l_corr})
    if zeroed:
        print("  learned reaction is zero everywhere (pruned to nothing)")
    return row


def sweep_label(run_dir):
    """The first folder in the path named like '19_smart_sweep', else the grandparent."""
    for part in Path(run_dir).resolve().parts:
        if re.match(r'^\d+_[A-Za-z]', part):
            return part
    return Path(run_dir).resolve().parent.parent.name


def summarize(rows):
    """Mean of each number over repeats, per (sweep, epsilon, points)."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['sweep'], row['epsilon'], row['points'])].append(row)
    names = sorted({k[len('target_slope_'):] for row in rows for k in row
                    if k.startswith('target_slope_')})

    def mean(values):
        values = [v for v in values if v == v]          # drop NaN
        return sum(values) / len(values) if values else float('nan')

    header = (f"\n{'sweep':<22} {'eps':>5} {'pts':>4} {'runs':>4} {'zeroed':>6}  "
              + "  ".join(f"{'target ' + n:>9} {'r':>5}" for n in names)
              + "  " + "  ".join(f"{'learned ' + n:>10}" for n in names))
    print("\n" + "=" * 90 + "\nSummary (means over repeats; slopes vs the true reaction)")
    print(header)
    for (sweep, eps, pts), members in sorted(groups.items(),
                                             key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        cells = []
        for n in names:
            cells.append(f"{mean([m.get(f'target_slope_{n}') for m in members]):9.3f} "
                         f"{mean([m.get(f'target_r_{n}') for m in members]):5.2f}")
        learned = [f"{mean([m.get(f'learned_slope_{n}') for m in members]):10.3f}"
                   for n in names]
        zeroed = sum(m['zeroed'] for m in members)
        print(f"{sweep:<22} {eps:>5g} {pts:>4} {len(members):>4} {zeroed:>6}  "
              + "  ".join(cells) + "  " + "  ".join(learned))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('run_dirs', nargs='+')
    parser.add_argument('--no-prune', action='store_true',
                        help='use the trained reaction before pruning')
    parser.add_argument('--cache-size', type=int, default=50_000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--csv', help='also write every run\'s numbers to this CSV file')
    args = parser.parse_args()
    rows = []
    started = time.time()
    total = len(args.run_dirs)
    for i, run_dir in enumerate(args.run_dirs, start=1):
        elapsed = time.time() - started
        eta = (f", about {elapsed / (i - 1) * (total - i + 1) / 60:.0f} min left"
               if i > 1 else "")
        print(f"\n[{i}/{total}] {sweep_label(run_dir)}: {Path(run_dir).name}"
              f"  ({elapsed / 60:.0f} min elapsed{eta})", flush=True)
        try:
            rows.append(check(run_dir, not args.no_prune, args.cache_size, args.device,
                              args.seed))
        except Exception as exc:
            print(f"\n{run_dir}\n  FAILED: {type(exc).__name__}: {exc}", flush=True)
    if len(rows) > 1:
        summarize(rows)
    if args.csv and rows:
        fields = sorted({k for row in rows for k in row},
                        key=lambda k: (k not in ('sweep', 'epsilon', 'points', 'zeroed', 'run'), k))
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == '__main__':
    main()