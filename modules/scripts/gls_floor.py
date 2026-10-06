"""
How close is each run's best Phase-1 validation GLS to the noise floor?

    python gls_floor.py <run_dir> [<run_dir> ...] [--device cuda] [--seed 0]

The validation GLS compares the surface with NOISY held-out data, so even a
perfect surface (the true, noise-free concentrations) cannot score below the
noise itself. For each run this reports, all with the run's own scaling
(mean_scale) and frame weights, exactly as in the training GLS:

  floor       GLS of the noise-free data against the noisy data: the best
              any surface could score. 0 when epsilon = 0.
  val GLS     the restored best Phase-1 validation GLS, read from training.log
              ("Restored best Phase-1 surface ... (val GLS = ...)").
  vs clean    GLS of the trained surface against the NOISE-FREE data: how far
              the surface is from the truth at the data points. This is the
              real headroom.

Because the noise is independent of the surface, val GLS ~ floor + vs clean.
So if "vs clean" is small next to "floor", the minimum is already close to
the best achievable and there is little to gain; if it is comparable or
larger, a better surface (e.g. weight averaging) has room to help.

The noise is redrawn here (same level, new random numbers), so "floor" is the
expected floor rather than the exact one for the training noise; with tens of
thousands of points the difference is negligible.

Keep this file next to train_binn_eql.py (modules/scripts/).
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch

import train_binn_eql as train          # also puts the repo root on sys.path
from modules.utils.noise_and_interpolate import noise_and_interpolate
from modules.utils.training_test_split import training_test_split

WEIGHTS_NAME = 'binn_best_val_model'
RESTORED = re.compile(r'Restored best Phase-1 surface from epoch (\d+) \(val GLS = ([0-9.eE+-]+)\)')


def gls(binn, rows, values):
    """Training GLS of `values` against the data values in `rows` ([x..., t, u...])."""
    d = binn.dimensions
    residual = ((values - rows[:, d + 1:]) / binn.mean_scale) ** 2
    frame = torch.bucketize(rows[:, d].contiguous(), binn.gls_frame_edges)
    weights = binn.gls_frame_weights[frame].unsqueeze(1)
    return float(torch.mean(residual * weights))


def logged_val_gls(run_dir):
    """(epoch, val GLS) of the restored best Phase-1 surface, or (None, None)."""
    log = Path(run_dir) / 'training.log'
    if not log.exists():
        return None, None
    found = RESTORED.findall(log.read_text(errors='replace'))
    if not found:
        return None, None
    epoch, value = found[-1]            # the last one, in case the run was resumed
    return int(epoch), float(value)


def sweep_label(run_dir):
    for part in Path(run_dir).resolve().parts:
        if re.match(r'^\d+_[A-Za-z]', part):
            return part
    return Path(run_dir).resolve().parent.parent.name


def check(run_dir, device, seed):
    run_dir = Path(run_dir)
    cfg = train.RunConfig.load(run_dir / 'config.json')

    torch.manual_seed(seed)
    raw, noisy = train.load_training_data(cfg)
    clean = (noise_and_interpolate(raw, cfg.points, 0.0, cfg.dimensions, cfg.species,
                                   multiplicative_noise=True)
             if cfg.points else raw)
    if clean.shape != noisy.shape:
        raise ValueError(f"noise-free and noisy data differ in shape "
                         f"({tuple(clean.shape)} vs {tuple(noisy.shape)})")

    train_data, _ = training_test_split(noisy, device)
    binn = train.build_binn(cfg, train_data).to(device)
    # The saved weights include the run's own mean_scale and GLS frame weights.
    binn.load_state_dict(torch.load(run_dir / WEIGHTS_NAME, map_location=device))
    binn.eval()

    clean, noisy = clean.to(device).float(), noisy.to(device).float()
    d = cfg.dimensions
    floor = gls(binn, noisy, clean[:, d + 1:])

    with torch.no_grad():
        surface = torch.cat([binn(chunk[:, :d + 1]) for chunk in torch.split(clean, 50_000)])
    vs_clean = gls(binn, clean, surface)

    epoch, val = logged_val_gls(run_dir)
    return {'run': str(run_dir), 'sweep': sweep_label(run_dir), 'epsilon': cfg.epsilon,
            'points': cfg.points, 'floor': floor, 'val_gls': val, 'best_epoch': epoch,
            'vs_clean': vs_clean}


def fmt(x, spec='9.3e'):
    return format(x, spec) if x is not None else ' ' * (int(spec.split('.')[0]) - 1) + '-'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('run_dirs', nargs='+')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rows = []
    for i, run_dir in enumerate(args.run_dirs, start=1):
        print(f"[{i}/{len(args.run_dirs)}] {sweep_label(run_dir)}: {Path(run_dir).name}",
              flush=True)
        try:
            row = check(run_dir, args.device, args.seed)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(row)
        print(f"  floor {fmt(row['floor'])}   val GLS {fmt(row['val_gls'])} "
              f"(epoch {row['best_epoch'] if row['best_epoch'] is not None else '-'})   "
              f"vs clean {fmt(row['vs_clean'])}", flush=True)

    if len(rows) > 1:
        groups = defaultdict(list)
        for row in rows:
            groups[(row['sweep'], row['epsilon'], row['points'])].append(row)
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None       # noqa: E731
        print("\n" + "=" * 90 + "\nSummary (means over repeats)")
        print(f"{'sweep':<24} {'eps':>5} {'pts':>4} {'runs':>4}  {'floor':>9}  {'val GLS':>9}  "
              f"{'vs clean':>9}  {'vs clean / floor':>16}")
        for (sweep, eps, pts), members in sorted(groups.items(),
                                                 key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
            floor = mean([m['floor'] for m in members])
            val = mean([m['val_gls'] for m in members if m['val_gls'] is not None])
            vs = mean([m['vs_clean'] for m in members])
            ratio = f"{vs / floor:16.2f}" if floor else f"{'(no noise)':>16}"
            print(f"{sweep:<24} {eps:>5g} {pts:>4} {len(members):>4}  {fmt(floor)}  {fmt(val)}  "
                  f"{fmt(vs)}  {ratio}")


if __name__ == '__main__':
    main()