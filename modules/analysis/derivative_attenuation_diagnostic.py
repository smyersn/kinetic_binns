#!/usr/bin/env python
"""
Surface-derivative diagnostic
=============================

Compares the trained surface fitter's derivatives against finite differences
on the training grid, and solves the mass-conservation least squares

    sum_s u_s,t  =  sum_s D_s * lap(u_s)

both ways. The reaction cancels out of that equation under conservation
(F1 = -F2), so the comparison is independent of the EQL library, the gates,
and everything Phases 2-4 train.

Runs on CPU in a few minutes. Read-only; no training, no GPU.

Version history -- both fixes matter for reading past output correctly:

v2  Grid rebuilt by scattering through unique-value inverse indices instead
    of a positional reshape. v1 assumed rows arrived t-major, then y, then x
    -- true of the raw simulation files but NOT of what noise_and_interpolate
    returns, so the finite-difference reference was silently permuted
    (matching rms, near-zero correlation). Section 0 now hard-checks
    alignment before anything downstream is reported.

v3  The FD reference now samples EVERY interior time frame and subsamples
    spatially, rather than 8 frames at full spatial density. w_t is a small
    residual of two nearly-cancelling, ~95% anti-correlated terms, so a
    handful of late frames gives a rank-deficient solve (R2 ~ 0.07 and
    0.0015 in v2) even though the same solve over full time coverage
    recovers the true coefficients. Also adds section 1b, a per-time-bin
    breakdown of u_t, which separates "the surface mis-fits the late-time
    dynamics" (a gls time-weighting problem) from "the surface can't resolve
    u_t anywhere" (a Fourier-encoding problem).

Usage
-----
    python derivative_attenuation_diagnostic.py <run_dir> [--true-d 0.05 0.1]

Options
-------
    --true-d DU DV    true coefficients, for the report (optional)
    --target-points N approx. sample size (default 20000)
    --chunk N         autograd chunk size (default 2000)
    --checkpoint NAME weights to load (default binn_best_phase1_model)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{file_dir}/../../')
sys.path.append(f'{file_dir}/../')
sys.path.append(f'{file_dir}')

from modules.binn_eql.build_binn_eql_net import BINN
from modules.utils.noise_and_interpolate import noise_and_interpolate


def banner(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def to_grid(data, dimensions, species):
    """
    Rebuild the (n_frames, species, ny, nx) field array from the flat
    training_data table, whose columns are [x * dimensions, t, concentrations].

    Scatters through unique-value inverse indices rather than reshaping, so
    the result is correct whatever order the rows are in. A positional
    reshape is a silent correctness bug here: a wrong assumption permutes the
    fields, which preserves rms magnitudes (the Laplacian is symmetric under
    transpose) while destroying the pointwise correspondence the whole
    diagnostic rests on.
    """
    xs, xi = torch.unique(data[:, 0], return_inverse=True)
    if dimensions == 2:
        ys, yi = torch.unique(data[:, 1], return_inverse=True)
    else:
        ys = torch.zeros(1)
        yi = torch.zeros(len(data), dtype=torch.long)
    ts, ti = torch.unique(data[:, dimensions], return_inverse=True)

    nx, ny, nf = len(xs), len(ys), len(ts)
    if nx * ny * nf != len(data):
        raise RuntimeError(
            f"Not a complete regular grid: nx={nx} ny={ny} nf={nf} implies "
            f"{nx * ny * nf} rows, data has {len(data)}.")

    fields = torch.full((nf, species, ny, nx), float('nan'))
    conc = data[:, -species:]
    for s in range(species):
        fields[ti, s, yi, xi] = conc[:, s]
    if torch.isnan(fields).any():
        raise RuntimeError("Grid has holes after scatter -- duplicate or "
                           "missing (x, y, t) coordinates in the data.")

    return fields, xs, ys, ts, (xs[1] - xs[0]).item(), nx, ny


def fd_laplacian(field, dx):
    """5-point Laplacian with periodic roll. The boundary ring is trimmed by
    the caller, so wrap-around values are never used."""
    return (torch.roll(field, 1, 0) + torch.roll(field, -1, 0)
            + torch.roll(field, 1, 1) + torch.roll(field, -1, 1)
            - 4.0 * field) / dx ** 2


def lstsq_D(w_t, lap_stack):
    sol = torch.linalg.lstsq(lap_stack, w_t.unsqueeze(1)).solution.squeeze(1)
    pred = lap_stack @ sol
    ss_res = ((w_t - pred) ** 2).sum()
    ss_tot = ((w_t - w_t.mean()) ** 2).sum()
    return sol, (1 - ss_res / ss_tot).item()


def fmt_D(vec, true=None):
    s = '  '.join(f'{v:.5e}' for v in vec)
    if true is not None and len(true) == len(vec):
        s += '   (' + '  '.join(f'{100 * v / t:.1f}%' for v, t in zip(vec, true)) + ' of true)'
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--true-d', nargs='+', type=float, default=None)
    ap.add_argument('--target-points', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--checkpoint', default='binn_best_phase1_model')
    args = ap.parse_args()

    torch.manual_seed(0)
    run_dir = Path(args.run_dir)

    with open(run_dir / 'config.json') as f:
        config = json.load(f)

    dimensions = config['dimensions']
    species = config['species']
    epsilon = config['epsilon']
    points = config['points']
    diff_coeffs = config['diff_coeffs']

    banner(f"Surface derivatives: {run_dir.name}")
    print(f"  dataset      : {Path(config['training_data_path']).stem}")
    print(f"  epsilon={epsilon}  points={points}  mcas={config.get('mcas', False)}  "
          f"diff_coeffs={'LEARNED' if not diff_coeffs else diff_coeffs}")

    train_data = torch.load(config['training_data_path'],
                            map_location='cpu')['training_data']
    if epsilon != 0 or points != 0:
        # With epsilon != 0 this is a DIFFERENT noise draw than training saw.
        # The grid is reproduced exactly, which is what the derivative
        # comparison needs; the noise realisation is not.
        train_data = noise_and_interpolate(train_data, points, epsilon,
                                           dimensions, species,
                                           multiplicative_noise=True)
    train_data = train_data.float()
    print(f"  training_data: {tuple(train_data.shape)}")

    binn = BINN(dimensions=dimensions, species=species, train_data=train_data,
                duplicates=config['duplicates'], diff_coeffs=diff_coeffs,
                degree=config['degree'], param_bounds=config['param_bounds'],
                mcas=config.get('mcas', False),
                include_poly=config.get('include_poly', True),
                include_increasing_hill=config.get('include_increasing_hill', True),
                include_decreasing_hill=config.get('include_decreasing_hill', True))
    binn.to('cpu')

    ckpt = run_dir / args.checkpoint
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt}. Available:\n  " +
                         "\n  ".join(sorted(p.name for p in run_dir.glob('binn*'))))
    missing, unexpected = binn.load_state_dict(
        torch.load(ckpt, map_location='cpu'), strict=False)
    unexpected = [k for k in unexpected if k != 'pde_scale']  # registered mid-training
    if missing:
        print(f"  WARNING missing keys: {missing}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected}")
    binn.eval()
    print(f"  loaded       : {args.checkpoint}")

    D_trained = None if diff_coeffs else [d.item() for d in binn.diffusion_fitter()]
    if D_trained is not None:
        print(f"  D in this checkpoint: {fmt_D(D_trained, args.true_d)}")

    fields, xs, ys, ts, dx, nx, ny = to_grid(train_data, dimensions, species)
    nf = len(ts)
    dt = (ts[1] - ts[0]).item()
    print(f"  grid         : {nx} x {ny}, {nf} frames, dx={dx:.4f}, dt={dt:.4f}")

    # ------------------------------------------------------------------
    # Sampling: EVERY interior frame, subsampled in space.
    #
    # w_t = sum_s u_s,t is a small residual of two nearly-cancelling,
    # heavily anti-correlated Laplacian terms. A few late frames leave the
    # 2-parameter solve effectively rank-deficient; full time coverage is
    # what conditions it. Trading spatial density for temporal coverage
    # costs nothing (same point budget) and is what makes the reference
    # usable.
    # ------------------------------------------------------------------
    trim = 2
    frame_idx = np.arange(1, nf - 1)
    n_interior = nx - 2 * trim
    sp_stride = max(1, int(np.sqrt(len(frame_idx) * n_interior ** 2
                                   / max(args.target_points, 1))))

    sl = slice(trim, -trim, sp_stride)
    fd_ut, fd_lap, fd_val, coords = [], [], [], []
    for ti_ in frame_idx:
        ut_f = (fields[ti_ + 1] - fields[ti_ - 1]) / (2 * dt)
        lap_f = torch.stack([fd_laplacian(fields[ti_, s], dx) for s in range(species)])
        fd_ut.append(ut_f[:, sl, sl].reshape(species, -1).T)
        fd_lap.append(lap_f[:, sl, sl].reshape(species, -1).T)
        fd_val.append(fields[ti_][:, sl, sl].reshape(species, -1).T)
        yy, xx = torch.meshgrid(ys[sl], xs[sl], indexing='ij')
        coords.append(torch.stack([xx.reshape(-1), yy.reshape(-1),
                                   torch.full((xx.numel(),), ts[ti_].item())], dim=1))

    fd_ut, fd_lap = torch.cat(fd_ut), torch.cat(fd_lap)
    fd_val, coords = torch.cat(fd_val), torch.cat(coords)
    N = len(coords)
    print(f"  sample       : {N} points, all {len(frame_idx)} interior frames "
          f"(t={ts[1]:.2f}..{ts[-2]:.2f}), spatial stride {sp_stride}")

    mlp_val, mlp_ut, mlp_lap = [], [], []
    for i in range(0, N, args.chunk):
        pts = coords[i:i + args.chunk].clone().requires_grad_(True)
        out = binn.surface_fitter(binn.normalize(pts))
        ut_c, uxx_c = binn.compute_field_derivatives(pts, out)
        mlp_val.append(out.detach())
        mlp_ut.append(ut_c.detach())
        mlp_lap.append(uxx_c.sum(dim=2).T.detach())
        del pts, out, ut_c, uxx_c
    mlp_val = torch.cat(mlp_val)
    mlp_ut, mlp_lap = torch.cat(mlp_ut), torch.cat(mlp_lap)

    names = binn.species_names
    rms = lambda a: a.pow(2).mean().sqrt().item()

    def corr(a, b):
        if a.numel() < 2 or a.std() < 1e-12 or b.std() < 1e-12:
            return float('nan')
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    # ------------------------------------------------------------------
    banner("0. Alignment check (must pass before anything below means anything)")
    val_corrs = [corr(mlp_val[:, s], fd_val[:, s]) for s in range(species)]
    for s in range(species):
        print(f"  corr(MLP {names[s]}, data {names[s]}) = {val_corrs[s]:+.4f}"
              f"   rms MLP {rms(mlp_val[:, s]):.4f} vs data {rms(fd_val[:, s]):.4f}")
    if min(val_corrs) < 0.9:
        print("\n  FAIL: the surface fitter's own values do not track the data at "
              "these\n  coordinates. Either the grid reconstruction is misaligned "
              "or this\n  checkpoint never fit the data. Stop here.")
        return
    print("\n  PASS: values align, so the derivative comparisons are like-for-like.")

    # ------------------------------------------------------------------
    banner("1. Derivative magnitudes: MLP vs finite differences")
    print(f"{'quantity':<14}{'MLP':>14}{'finite diff':>14}{'ratio':>10}{'corr':>10}")
    for s in range(species):
        for label, m, f in [(f'lap {names[s]}', mlp_lap[:, s], fd_lap[:, s]),
                            (f'{names[s]}_t', mlp_ut[:, s], fd_ut[:, s])]:
            print(f"{label:<14}{rms(m):>14.4f}{rms(f):>14.4f}"
                  f"{rms(m) / max(rms(f), 1e-12):>10.2f}{corr(m, f):>10.4f}")

    # ------------------------------------------------------------------
    banner("1b. u_t fidelity by time bin")
    print("  The PDE residual's left-hand side is u_t. If it degrades only at")
    print("  late times, the gls time weighting (1 + 49*exp(-(t-t0)/5), so ~50x")
    print("  at t=0 and ~1.3x by t=25) is starving the epochs that carry the")
    print("  dynamics -- a config fix. If it is bad everywhere, the surface")
    print("  cannot resolve u_t at all and the Fourier time encoding is the")
    print("  suspect instead.\n")

    t_col = coords[:, -1]
    t_lo, t_hi = ts[1].item(), ts[-2].item()
    edges = np.linspace(t_lo, t_hi, 6)
    hdr = f"{'t range':<16}" + ''.join(
        f"{n + '_t corr':>12}{'ratio':>9}" for n in names)
    print(hdr)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t_col >= lo) & (t_col < hi + 1e-9)
        if m.sum() < 100:
            continue
        row = f"{f'{lo:5.1f}-{hi:5.1f}':<16}"
        for s in range(species):
            c = corr(mlp_ut[m, s], fd_ut[m, s])
            r = rms(mlp_ut[m, s]) / max(rms(fd_ut[m, s]), 1e-12)
            row += f"{c:>12.3f}{r:>9.2f}"
        print(row)

    # ------------------------------------------------------------------
    banner("2. Diffusion coefficients from the mass equation")
    print("   sum_s d(u_s)/dt = sum_s D_s * lap(u_s)   (reaction cancels)\n")
    D_fd, r2_fd = lstsq_D(fd_ut.sum(dim=1), fd_lap)
    D_mlp, r2_mlp = lstsq_D(mlp_ut.sum(dim=1), mlp_lap)
    print(f"  finite differences : {fmt_D(D_fd.tolist(), args.true_d)}")
    print(f"                       R2 = {r2_fd:.4f}")
    print(f"  MLP derivatives    : {fmt_D(D_mlp.tolist(), args.true_d)}")
    print(f"                       R2 = {r2_mlp:.4f}")
    if D_trained is not None:
        print(f"  trained checkpoint : {fmt_D(D_trained, args.true_d)}")

    ref_ok = r2_fd >= 0.3
    if not ref_ok:
        print(f"\n  WARNING: the finite-difference solve has R2 = {r2_fd:.4f}, so the")
        print("  reference is unreliable. Try --target-points 50000, or check the")
        print("  boundary conditions against the periodic-roll stencil used here.")

    atten = [(m / f).item() if abs(f) > 1e-12 else float('nan')
             for m, f in zip(D_mlp, D_fd)]
    if ref_ok:
        print(f"\n  attenuation (MLP / FD): "
              + '  '.join(f'{names[s]}: {a:.3f}' for s, a in enumerate(atten)))

    # ------------------------------------------------------------------
    banner("3. Verdict")
    lap_corr = min(corr(mlp_lap[:, s], fd_lap[:, s]) for s in range(species))
    ut_corr = min(corr(mlp_ut[:, s], fd_ut[:, s]) for s in range(species))
    ut_ratio = min(rms(mlp_ut[:, s]) / max(rms(fd_ut[:, s]), 1e-12)
                   for s in range(species))

    if lap_corr >= 0.9 and (ut_corr < 0.7 or ut_ratio < 0.7):
        print(f"  LHS PROBLEM: the Laplacians are faithful (corr {lap_corr:.3f}) but")
        print(f"  u_t is not (corr {ut_corr:.3f}, rms ratio {ut_ratio:.2f}). Phase 2")
        print("  solves u_t = D*lap(u) + F with a left-hand side that is too small,")
        print("  and the only way to balance that is to shrink D and F together --")
        print("  exactly the observed collapse. Read section 1b to decide whether")
        print("  this is the gls time weighting or the Fourier time encoding.")
    elif lap_corr < 0.7:
        print(f"  RHS PROBLEM: the MLP's Laplacian correlates only {lap_corr:.3f} with")
        print("  finite differences, so Phase 2 regresses on a corrupted regressor.")
        print("  That biases D toward zero regardless of optimizer or L0 settings.")
    elif ref_ok and float(np.nanmean(atten)) < 0.5:
        print(f"  ATTENUATION: derivatives look broadly right, but the mass solve on")
        print(f"  them still shrinks D by {float(np.nanmean(atten)):.2f}x.")
    else:
        print(f"  DERIVATIVES CLEAN: lap corr {lap_corr:.3f}, u_t corr {ut_corr:.3f},")
        print(f"  u_t ratio {ut_ratio:.2f}. The surface is not the problem. Next")
        print("  suspect: pde_scale (the 90th percentile of u_t^2) dividing a")
        print("  diffusion-dominated residual, making small D the cheap minimum.")
    print("\n  Run this on a dataset that WORKED as well -- the contrast between")
    print("  the two is more informative than either set of numbers alone.\n")


if __name__ == '__main__':
    main()