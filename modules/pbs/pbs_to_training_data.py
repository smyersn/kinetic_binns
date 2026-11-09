"""
Convert a Smoldyn particle simulation (.xyz from `cmd N ... molpos`) into
BINN-EQL training data: rows of [x, y, t, active_Cdc42, inactive_Cdc42] in
molecules / um^2, saved as {'training_data': tensor, 'meta': {...}}.

Species are lumped into two pools (see POOLS below):
    active   = Cdc42T + BemGEF42 (+ transient complexes, if written)
    inactive = Cdc42Dm + Cdc42Dc (+ transient complexes, if written)
Cdc42Dm <-> Cdc42Dc exchange (~0.02 s) is far faster than the frame spacing,
so the inactive pool diffuses with the time-averaged coefficient
    D_eff = sum_s f_s D_s,   f_s = fraction of the pool in species s.
D for each pool is measured from the data (frames with t > 0) and, for the
inactive pool, compared with the equilibrium value from the cfg's
Cdc42Dc <-> Cdc42Dm rates. The measured values are written into
meta['diff_coeffs'] and a .json next to the .pt -- copy them into config.json.

The .cfg next to the .xyz (same prefix) supplies the domain length, the
diffusion coefficients, the rates and the ORDER of the molpos lines, so
none of these are hard-coded here.

Usage (from anywhere; output defaults to <repo>/data/particle_based/<run folder>/):
    python pbs_to_training_data.py runs/second_go/length_30_scale_3-seeds_1.xyz --t-max 250
    python pbs_to_training_data.py runs/second_go/*.xyz --grid-res 100 --sigma 0.3 --t-max 250
    python pbs_to_training_data.py runs/third_go/length_20_scale_1-seeds_1.xyz --gif

--gif also writes <output>.gif next to the .pt: particles (top) and densities
(bottom) for active, inactive-membrane and inactive-cytosol Cdc42.
run_sims.sh / generate_cfg.py call this script with --gif at the end of every
Smoldyn job, so each finished run comes with its .pt, .json and .gif.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch


# Pool name -> {Smoldyn species: number of Cdc42 molecules it carries in that pool}.
# Species not written by molpos are skipped (with a warning if REQUIRED).
POOLS = {
    'active': {'Cdc42T': 1, 'BemGEF42': 1, 'complex_Ra_Cdc42T': 1,
               'complex_Cdc42Dm_BemGEF42': 1},          # its BemGEF42 part carries a Cdc42T
    'inactive': {'Cdc42Dm': 1, 'Cdc42Dc': 1, 'complex_Cdc42Dm_BemGEF42': 1,  # its Cdc42Dm part
                 'complex_Cdc42Dm_BemGEFm': 1, 'complex_Cdc42D_Ra': 1},
}
# Missing these changes the answer (BemGEF42 holds much of the active Cdc42).
# The complexes last one time step (catalysis probability 1), so they are negligible.
REQUIRED = {'Cdc42T', 'BemGEF42', 'Cdc42Dm', 'Cdc42Dc'}
# The fast membrane <-> cytosol exchange used for the theoretical D_eff.
CYTOSOLIC, MEMBRANE = 'Cdc42Dc', 'Cdc42Dm'

# --gif panels: (title, species). One column per panel: particles on top,
# density (molecules / um^2, same smoothing as the training data) below.
GIF_PANELS = [
    ('Active Cdc42', ['Cdc42T', 'BemGEF42', 'complex_Ra_Cdc42T', 'complex_Cdc42Dm_BemGEF42']),
    ('Inactive Cdc42 (membrane)', ['Cdc42Dm', 'complex_Cdc42Dm_BemGEFm', 'complex_Cdc42D_Ra']),
    ('Inactive Cdc42 (cytosol)', ['Cdc42Dc']),
    # Same species as the 'inactive' pool in POOLS, so this density is exactly
    # the second species of the training data.
    ('Inactive Cdc42 (total, trained on)', list(POOLS['inactive'])),
]


# --------------------------------------------------------------------------- cfg
def _eval(expr, variables):
    """Evaluate a Smoldyn numeric expression such as 'L+0.2' or 'k5a'."""
    names = {k: v for k, v in variables.items()}
    names.update(math=math)
    try:
        return float(eval(expr, {'__builtins__': {}}, names))
    except Exception as err:
        raise ValueError(f"cannot evaluate cfg expression {expr!r}: {err}") from None


def parse_cfg(cfg_path):
    """Variables, difc, molpos order, reactions, time step from a Smoldyn cfg."""
    variables, difc, molpos, reactions = {}, {}, [], {}
    time_step = None
    for raw in Path(cfg_path).read_text().splitlines():
        line = raw.split('#')[0].strip()
        if not line:
            continue
        tok = line.split()
        key = tok[0]
        if key == 'variable':                            # variable L = 30
            m = re.match(r'variable\s+(\w+)\s*=\s*(.+)', line)
            variables[m.group(1)] = _eval(m.group(2), variables)
        elif key == 'difc':                              # difc Cdc42Dc 15
            difc[tok[1]] = _eval(tok[2], variables)
        elif key == 'time_step':
            time_step = _eval(tok[1], variables)
        elif key == 'cmd' and 'molpos' in tok:           # cmd N 50000 molpos Cdc42T file.xyz
            molpos.append(tok[tok.index('molpos') + 1])
        elif key == 'reaction':                          # reaction name A <-> B kf kb
            m = re.match(r'reaction\s+(\S+)\s+(.+?)\s+(<->|->)\s+(.+)$', line)
            if m:
                name, lhs, arrow, rest = m.groups()
                rest = rest.split()
                n_rates = 2 if arrow == '<->' else 1
                # Trailing tokens are rates only if they evaluate to numbers.
                rates = []
                for t in rest[-n_rates:]:
                    try:
                        rates.append(_eval(t, variables))
                    except ValueError:
                        rates = []
                        break
                products = rest[:-n_rates] if len(rates) == n_rates else rest
                reactions[name] = dict(lhs=lhs.split(' + '), rhs=' '.join(products).split(' + '),
                                       arrow=arrow, rates=rates)
    if 'L' not in variables:
        raise ValueError(f"{cfg_path}: no 'variable L = ...' line, cannot get the domain length")
    if not molpos:
        raise ValueError(f"{cfg_path}: no 'cmd ... molpos' lines")
    return dict(L=variables['L'], variables=variables, difc=difc, molpos=molpos,
                reactions=reactions, time_step=time_step)


def exchange_fraction_cytosolic(cfg):
    """Equilibrium fraction of inactive Cdc42 in the cytosol from Dc <-> Dm rates."""
    for r in cfg['reactions'].values():
        if r['arrow'] == '<->' and r['lhs'] == [CYTOSOLIC] and r['rhs'] == [MEMBRANE] \
                and len(r['rates']) == 2:
            k_on, k_off = r['rates']                    # Dc -> Dm, Dm -> Dc
            return k_off / (k_on + k_off), k_on + k_off
    return None, None


# --------------------------------------------------------------------------- xyz
def read_frames(xyz_path, species_order, L):
    """
    Yield (time, {species: (N, 2) positions}) per output time.
    Each molpos line is: time x1 y1 x2 y2 ... ; lines cycle through
    species_order (the order of the molpos commands in the cfg).
    """
    n_species = len(species_order)
    block, block_time = {}, None
    with open(xyz_path) as f:
        k = 0
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            vals = np.array(raw.split(), dtype=np.float64)
            t, coords = vals[0], vals[1:]
            if coords.size % 2:
                raise ValueError(f"{xyz_path}:{lineno}: odd number of coordinates "
                                 f"({coords.size}); expected 'time x y x y ...' for dim 2")
            s = species_order[k % n_species]
            if k % n_species == 0:
                block, block_time = {}, t
            elif not math.isclose(t, block_time, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"{xyz_path}:{lineno}: time {t} for {s} but {block_time} for "
                                 f"{species_order[0]}; molpos lines are not in cfg order")
            block[s] = np.mod(coords.reshape(-1, 2), L)   # periodic domain [0, L)
            k += 1
            if k % n_species == 0:
                yield block_time, block
    if k % n_species:
        print(f"  note: dropped a partial last frame ({k % n_species}/{n_species} species lines; "
              f"simulation probably still running or killed)", file=sys.stderr)


# --------------------------------------------------------------------------- density
def density_grid(coords, L, grid_res, sigma):
    """
    Molecules / um^2 on a grid_res x grid_res periodic grid, indexed [y, x].
    sigma > 0: periodic Gaussian kernel density (std sigma, um) via FFT,
    which conserves the total exactly. sigma = 0: plain histogram.
    """
    edges = np.linspace(0.0, L, grid_res + 1)
    counts, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[edges, edges])
    counts = counts.T                                   # histogram2d gives [x, y]
    dx = L / grid_res
    if sigma > 0:
        k = np.fft.fftfreq(grid_res, d=dx)
        kernel = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
        counts = np.real(np.fft.ifft2(np.fft.fft2(counts) * kernel))
        counts = np.clip(counts, 0.0, None)             # FFT round-off
    return counts / dx ** 2


# --------------------------------------------------------------------------- main
def find_repo_root(start):
    for p in [start, *start.parents]:
        if (p / 'modules').is_dir():
            return p
    return None


def convert(xyz_path, args):
    xyz_path = Path(xyz_path).resolve()
    cfg_path = Path(args.cfg).resolve() if args.cfg else xyz_path.with_suffix('.cfg')
    if not cfg_path.exists():
        raise FileNotFoundError(f"no cfg at {cfg_path} (pass --cfg)")
    cfg = parse_cfg(cfg_path)
    L, order = cfg['L'], cfg['molpos']
    print(f"\n{xyz_path.name}: L = {L:g}, molpos order = {order}")

    present = set(order)
    missing = sorted(REQUIRED - present)
    if missing:
        print(f"  WARNING: not written by molpos: {missing}. Their Cdc42 is invisible, so the "
              f"pools will not conserve mass. Add 'cmd N <rate> molpos <species> <file>' "
              f"lines for them and rerun the simulation.")
    pools = {p: {s: w for s, w in members.items() if s in present} for p, members in POOLS.items()}
    for p, members in pools.items():
        if not members:
            raise ValueError(f"pool {p!r} has no species in this file")
        for s in members:
            if s not in cfg['difc']:
                raise ValueError(f"no difc for {s} in {cfg_path}")
    pool_names = list(pools)
    sigma_inactive = args.sigma if args.sigma_inactive is None else args.sigma_inactive
    pool_sigma = {p: (sigma_inactive if p == 'inactive' else args.sigma) for p in pool_names}

    t_list, grids = [], []
    sum_ND = {p: 0.0 for p in pool_names}                # for measured D_eff
    sum_N = {p: 0.0 for p in pool_names}
    totals = []
    check = []                                           # (t, cyto, mem, active) for the depletion check
    panels = [(title, [s for s in members if s in present]) for title, members in GIF_PANELS]
    panels = [(title, members) for title, members in panels if members]
    gif_frames = []                                      # (t, [coords per panel], [density per panel])
    rng = np.random.default_rng(0)
    kept = 0
    for i, (t, block) in enumerate(read_frames(xyz_path, order, L)):
        if t < args.t_min - 1e-9 or (args.t_max is not None and t > args.t_max + 1e-9):
            continue
        if kept % args.stride:
            kept += 1
            continue
        kept += 1
        frame = []
        total = 0
        for p in pool_names:
            g = np.zeros((args.grid_res, args.grid_res))
            for s, w in pools[p].items():
                n = len(block[s])
                total += w * n
                if t > 0:
                    sum_ND[p] += w * n * cfg['difc'][s]
                    sum_N[p] += w * n
                if n:
                    g += w * density_grid(block[s], L, args.grid_res, pool_sigma[p])
            frame.append(g)
        totals.append(total)
        grids.append(np.stack(frame, axis=-1))
        t_list.append(t)
        if CYTOSOLIC in present and MEMBRANE in present:
            check.append((t, density_grid(block[CYTOSOLIC], L, args.grid_res, 0),
                          density_grid(block[MEMBRANE], L, args.grid_res, 0), frame[0]))
        if args.gif:
            coords, dens = [], []
            for title, members in panels:
                c = np.concatenate([block[s] for s in members], axis=0)
                sig = sigma_inactive if title.startswith('Inactive') else args.sigma
                dens.append(density_grid(c, L, args.grid_res, sig) if len(c)
                            else np.zeros((args.grid_res, args.grid_res)))
                if len(c) > args.gif_max_points:          # keeps the scatter plots fast
                    c = c[rng.choice(len(c), args.gif_max_points, replace=False)]
                coords.append(c.astype(np.float32))
            gif_frames.append((t, coords, dens))

    if len(t_list) < 2:
        raise ValueError(f"only {len(t_list)} frame(s) in [{args.t_min}, {args.t_max}]")
    times = np.array(t_list)
    u_array = np.stack(grids, axis=0)                    # (T, y, x, pools)
    t_offset = times[0]
    times = times - t_offset                             # simulate_feql needs a t == 0 frame

    # ---- diffusion coefficients
    D_measured = [sum_ND[p] / sum_N[p] if sum_N[p] else float('nan') for p in pool_names]
    fc_theory, k_exchange = exchange_fraction_cytosolic(cfg)
    print(f"  frames: {len(times)} (t = {t_offset:g} .. {t_offset + times[-1]:g}, "
          f"spacing {np.diff(times).min():g}), grid {args.grid_res}x{args.grid_res}, "
          f"dx = {L / args.grid_res:.3g} um, sigma = {args.sigma:g} um"
          + (f" (inactive: {sigma_inactive:g} um)" if sigma_inactive != args.sigma else ""))
    for p, D in zip(pool_names, D_measured):
        mix = ', '.join(f"{s} {cfg['difc'][s]:g}" for s in pools[p])
        print(f"  D[{p}] = {D:.4g} um^2/s  (measured; members' difc: {mix})")
    if fc_theory is not None:
        D_theory = fc_theory * cfg['difc'][CYTOSOLIC] + (1 - fc_theory) * cfg['difc'][MEMBRANE]
        print(f"  D[inactive] equilibrium theory = {D_theory:.4g}  (cytosolic fraction "
              f"{fc_theory:.3f}, exchange time {1 / k_exchange:.3g} s vs frame spacing "
              f"{np.diff(times).min():g} s; rebinding length "
              f"{math.sqrt(cfg['difc'][CYTOSOLIC] / k_exchange):.2g} um vs dx "
              f"{L / args.grid_res:.2g} um)")

    # ---- mass conservation of the visible pools
    totals = np.array(totals, dtype=float)
    drift = (totals.max() - totals.min()) / totals.mean()
    print(f"  total Cdc42 in pools: {totals.min():.0f} .. {totals.max():.0f} "
          f"(relative spread {drift:.2%}){'  <-- not conserved' if drift > 0.01 else ''}")

    # ---- does the spot deplete the membrane share of the inactive pool?
    # Pool the last third of the kept frames (or t >= --check-after) to beat counting noise.
    if check:
        t_from = args.check_after if args.check_after is not None else check[len(check) * 2 // 3][0]
        late = [c for c in check if c[0] >= t_from] or check[-1:]
        n_late = len(late)
        late_cyto, late_mem, late_active = (sum(c[i] for c in late) for i in (1, 2, 3))
        thresh = np.percentile(late_active, 99)
        spot = late_active >= thresh
        fm = lambda mask: late_mem[mask].sum() / max((late_mem[mask] + late_cyto[mask]).sum(), 1)
        theory = f", theory {1 - fc_theory:.3f}" if fc_theory is not None else ''
        print(f"  membrane fraction of inactive pool (t >= {t_from:g}, {n_late} frames): "
              f"spot {fm(spot):.3f}, background {fm(~spot):.3f}{theory}")

    # ---- training rows [x, y, t, pools...]; x, y are bin centres
    dx = L / args.grid_res
    x_array = torch.linspace(dx / 2, L - dx / 2, args.grid_res, dtype=torch.float32)
    sys.path.insert(0, str(find_repo_root(Path(__file__).resolve()) or ''))
    from modules.utils.format_data import format_u_array_to_training_data
    training_data = format_u_array_to_training_data(
        torch.tensor(u_array, dtype=torch.float32), x_array,
        torch.tensor(times, dtype=torch.float32))

    meta = dict(source_xyz=str(xyz_path), source_cfg=str(cfg_path), L=L,
                grid_res=args.grid_res, sigma=args.sigma, sigma_inactive=sigma_inactive,
                units='molecules/um^2',
                species_names=pool_names, pool_members={p: list(m) for p, m in pools.items()},
                diff_coeffs=[float(D) for D in D_measured],
                diff_coeffs_theory_inactive=None if fc_theory is None else float(D_theory),
                t_offset=float(t_offset), t_min=args.t_min, t_max=args.t_max, stride=args.stride,
                n_frames=int(len(times)), total_molecules_spread=float(drift),
                missing_required=missing)

    if args.out and len(args.xyz) == 1:
        out = Path(args.out)
    else:
        root = find_repo_root(Path(__file__).resolve()) or Path.cwd()
        out_dir = Path(args.out_dir) if args.out_dir else root / 'data' / 'particle_based' / xyz_path.parent.name
        tag = f"grid_{args.grid_res}_sigma_{args.sigma:g}"
        if sigma_inactive != args.sigma:
            tag += f"_sigmav_{sigma_inactive:g}"
        tag += f"_t_{args.t_min:g}_{'end' if args.t_max is None else f'{args.t_max:g}'}"
        out = out_dir / f"{xyz_path.stem}_{tag}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'training_data': training_data, 'meta': meta}, out)
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(f"  saved {tuple(training_data.shape)} -> {out}")
    print(f"  config.json: \"training_data_path\": \"{out}\", \"species\": 2, \"dimensions\": 2, "
          f"\"diff_coeffs\": [{D_measured[0]:.4g}, {D_measured[1]:.4g}]")
    if args.gif:
        gif = out.with_suffix('.gif')
        # matplotlib holds every GIF frame in memory as a full image until the end
        # (~10 MB per frame at full size), so animate at most --gif-max-frames
        # evenly spaced frames at --gif-dpi. The training data keep every frame.
        n = len(gif_frames)
        keep = sorted(set(np.linspace(0, n - 1, min(n, args.gif_max_frames)).round().astype(int)))
        write_gif([gif_frames[i] for i in keep], [title for title, _ in panels], L, gif,
                  args.gif_fps, f"{xyz_path.parent.name}/{xyz_path.stem}", dpi=args.gif_dpi)
        print(f"  animation -> {gif} ({len(keep)} of {n} frames)")
    return out


def write_gif(gif_frames, titles, L, path, fps, name, dpi=60):
    """Particles (top row) and densities (bottom row), one column per panel."""
    import matplotlib
    matplotlib.use('Agg')                                # no display on compute nodes
    import matplotlib.pyplot as plt
    from matplotlib import animation

    n = len(titles)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10), squeeze=False)
    # One colour scale per column for the whole movie, so frames are comparable.
    vmaxs = [max(max(f[2][k].max() for f in gif_frames), 1e-12) for k in range(n)]
    scats, ims = [], []
    for k, title in enumerate(titles):
        ax = axes[0, k]
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal')
        ax.set_title(f"{title}: particles")
        scats.append(ax.scatter(*gif_frames[0][1][k].T, s=1, alpha=0.5, c='C0'))
        ax = axes[1, k]
        ax.set_title(f"{title}: density (molecules/µm²)")
        im = ax.imshow(gif_frames[0][2][k], cmap='viridis', vmin=0, vmax=vmaxs[k],
                       origin='lower', extent=[0, L, 0, L])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)
    suptitle = fig.suptitle('', fontsize=16)

    def draw(i):
        t, coords, dens = gif_frames[i]
        for k in range(n):
            scats[k].set_offsets(coords[k] if len(coords[k]) else np.zeros((0, 2)))
            ims[k].set_array(dens[k])
        suptitle.set_text(f"{name}    t = {t:.0f} s")
        return scats + ims + [suptitle]

    anim = animation.FuncAnimation(fig, draw, frames=len(gif_frames), blit=False)
    anim.save(str(path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xyz', nargs='+', help='Smoldyn .xyz file(s); the .cfg with the same prefix is read too')
    ap.add_argument('--cfg', help='cfg path (only with a single .xyz; default: same prefix as the .xyz)')
    ap.add_argument('--grid-res', type=int, default=100, help='bins per side (default 100)')
    ap.add_argument('--sigma', type=float, default=0.3,
                    help='Gaussian kernel std in um; 0 = plain histogram (default 0.3)')
    ap.add_argument('--sigma-inactive', type=float, default=None,
                    help='kernel std (um) for the INACTIVE pool only (default: same as --sigma). '
                         'The inactive pool diffuses fast (D ~ 4), so its true profile varies only '
                         'over several um; with the default 0.3 its Laplacian, times D, is mostly '
                         'counting noise (~30x its real rate of change). 1.0 removes it.')
    ap.add_argument('--t-min', type=float, default=0.0, help='first time kept (shifted to t = 0)')
    ap.add_argument('--t-max', type=float, default=None, help='last time kept (default: all)')
    ap.add_argument('--stride', type=int, default=1, help='keep every n-th frame in the window')
    ap.add_argument('--check-after', type=float, default=None,
                    help='frames from this time on are pooled for the spot-depletion check '
                         '(default: last third of the kept frames)')
    ap.add_argument('--gif', action='store_true',
                    help='also write an animation (<output>.gif) of the kept frames')
    ap.add_argument('--gif-max-points', type=int, default=20000,
                    help='particles drawn per scatter panel per frame (random subsample; default 20000)')
    ap.add_argument('--gif-fps', type=int, default=10, help='animation frames per second (default 10)')
    ap.add_argument('--gif-max-frames', type=int, default=101,
                    help='animate at most this many evenly spaced frames (default 101; memory '
                         'grows with every frame the GIF holds)')
    ap.add_argument('--gif-dpi', type=int, default=60,
                    help='animation resolution (default 60, i.e. 1440 x 600 for 4 panels; '
                         'memory grows with dpi squared)')
    ap.add_argument('--out', help='output .pt (single input only)')
    ap.add_argument('--out-dir', help='output folder (default <repo>/data/particle_based/<run folder>)')
    args = ap.parse_args()
    if args.cfg and len(args.xyz) > 1:
        ap.error('--cfg only works with a single .xyz')
    for path in args.xyz:
        convert(path, args)


if __name__ == '__main__':
    main()