#!/usr/bin/env python
"""
Print the per-frame GLS weights a dataset would get.

Shows the activity-based weights (what BINN now uses), the legacy exponential
for comparison, and per-frame diagnostics -- so you can see at a glance whether
the peak lands where the pattern actually forms.

Runs on CPU in seconds. Read-only; builds no model, loads no checkpoint.

Usage
-----
    # from a run directory (reads config.json, matches training preprocessing)
    python print_gls_weights.py <run_dir>

    # or straight from a dataset
    python print_gls_weights.py path/to/data.pt --dimensions 2 --species 2

Options
-------
    --max-weight W    peak weight (default 50)
    --factor F        outlier threshold for the unresolved-IC check (default 5)
    --time-scale S    legacy exponential decay constant in ABSOLUTE time units
                      (default 5.0, the current hardcoded value)
    --time-frac F     legacy exponential decay as a FRACTION of the window;
                      overrides --time-scale. 0.1 reproduces time_scale=5 on a
                      T=50 dataset and scales correctly to other T.
    --every N         print every Nth frame (default: auto, ~30 rows)
    --raw             skip noise_and_interpolate even if the config asks for it
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{file_dir}/../../')
sys.path.append(f'{file_dir}/../')
sys.path.append(f'{file_dir}')

from modules.binn_eql.build_binn_eql_net import frame_stats, frame_activity_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', help='run directory with config.json, or a .pt dataset')
    ap.add_argument('--dimensions', type=int, default=2)
    ap.add_argument('--species', type=int, default=2)
    ap.add_argument('--max-weight', type=float, default=50.0)
    ap.add_argument('--factor', type=float, default=5.0)
    ap.add_argument('--time-scale', type=float, default=5.0)
    ap.add_argument('--time-frac', type=float, default=None)
    ap.add_argument('--every', type=int, default=None)
    ap.add_argument('--raw', action='store_true')
    args = ap.parse_args()

    target = Path(args.target)
    dimensions, species = args.dimensions, args.species
    epsilon, points = 0, 0

    # --- resolve the dataset, matching training preprocessing when we can ---
    if target.is_dir():
        with open(target / 'config.json') as f:
            config = json.load(f)
        data_path = config['training_data_path']
        dimensions = config['dimensions']
        species = config['species']
        epsilon, points = config['epsilon'], config['points']
        print(f"run     : {target.name}")
    else:
        data_path = str(target)

    print(f"dataset : {Path(data_path).stem}")
    train_data = torch.load(data_path, map_location='cpu')['training_data'].float()

    if not args.raw and (epsilon != 0 or points != 0):
        from modules.utils.noise_and_interpolate import noise_and_interpolate
        train_data = noise_and_interpolate(train_data, points, epsilon, dimensions,
                                           species, multiplicative_noise=True)
        print(f"preproc : noise_and_interpolate(points={points}, epsilon={epsilon})")

    # --- weights ---
    w, ts = frame_activity_weights(train_data, dimensions, species,
                                   max_weight=args.max_weight, factor=args.factor)
    std, _ = frame_stats(train_data, dimensions, species)
    nf = len(ts)
    t_min, t_max = ts[0].item(), ts[-1].item()

    # legacy exponential, for comparison
    scale = args.time_frac * (t_max - t_min) if args.time_frac else args.time_scale
    w_exp = 1.0 + (args.max_weight - 1.0) * torch.exp(-(ts - ts[0]) / scale)

    dt = (ts[1] - ts[0]).item() if nf > 1 else 0.0
    print(f"frames  : {nf} over t in [{t_min:.4g}, {t_max:.4g}], dt={dt:.4g}")
    print(f"exp ref : time_scale={scale:.4g} "
          f"({'fractional' if args.time_frac else 'absolute'}) "
          f"-> covers {100*scale/(t_max-t_min):.1f}% of the window")
    if scale < 2 * dt:
        print(f"  WARNING: the exponential's decay constant is under 2 frames. "
              f"It is a delta function on the IC here -- this is the absolute-"
              f"vs-fractional time_scale bug.")

    every = args.every or max(1, nf // 30)
    bar_w = 28

    print(f"\n{'frame':>6}{'t':>10}{'std(u)':>10}{'activity':>11}"
          f"{'w_act':>8}{'w_exp':>8}  activity weight")
    print('-' * 100)
    for i in range(0, nf, every):
        # recover the activity that produced the weight (inverse of the map)
        act_frac = (w[i].item() - 1.0) / max(args.max_weight - 1.0, 1e-12)
        bar = '#' * int(round(act_frac * bar_w))
        print(f"{i:>6}{ts[i].item():>10.4g}{std[i, 0].item():>10.4f}"
              f"{act_frac:>11.3f}{w[i].item():>8.1f}{w_exp[i].item():>8.1f}  {bar}")

    peak = int(w.argmax())
    print('-' * 100)
    print(f"peak activity weight {w.max().item():.1f} at frame {peak} "
          f"(t={ts[peak].item():.4g}); exponential's weight there: "
          f"{w_exp[peak].item():.1f}")
    print(f"mean w_act {w.mean().item():.1f} | frames at the floor (w<2): "
          f"{(w < 2).sum().item()}/{nf}")

    # how many leading frames were zeroed as unresolved
    zeroed = int((w[:10] <= 1.0 + 1e-6).sum()) if nf > 10 else 0
    lead = 0
    while lead < nf and w[lead].item() <= 1.0 + 1e-6:
        lead += 1
    if lead:
        print(f"leading frames zeroed as unresolved IC: {lead} "
              f"(t < {ts[lead].item():.4g})")

    if peak <= 1:
        print("\n  WARNING: the peak weight is on the first resolvable frame. If the "
              "\n  initial condition is a noise seed, the unresolved-IC zeroing may "
              "\n  not be wide enough -- check that frame_activity_weights zeroes "
              "\n  act[:k+1], not act[:k] (a central difference at frame k still "
              "\n  straddles the unresolved frame).")
    print()


if __name__ == '__main__':
    main()