"""
PDE-FIND (Rudy et al., Science Advances 2017) for 2-species reaction-diffusion data.

    1. derivatives straight from the gridded data: u_t, and lap(u) = u_xx + u_yy
       - finite differences (sg_window_t = sg_window_x = 0), or
       - Savitzky-Golay smoothing-differentiation along each axis, the usual
         PDE-FIND treatment of noisy data (local polynomial fits, as in Rudy's
         polynomial-interpolation differentiation)
    2. targets  y_s = u_s,t - D_s lap(u_s)      (D fixed)
                y_s = u_s,t, with lap(u), lap(v) as extra candidates   (D learned)
    3. sequentially thresholded least squares (PySINDy STLSQ) over a path of
       thresholds; one model is selected by a fixed rule (common.select_from_path)

Options (config.json "options"):
    sg_window_t, sg_window_x   odd Savitzky-Golay window lengths in frames / grid
                               cells; 0 = plain finite differences (default 0, 0)
    sg_order                   Savitzky-Golay polynomial order (default 3)
    max_rows                   random subsample of grid points used (default 1,000,000)
    thresholds, alpha, selection ('share' | 'knee' | 'fixed'), threshold,
               selection_tol: see common.select_from_path (default 'share', 0.01)
"""
import numpy as np
from scipy.signal import savgol_filter

from modules.benchmarks.common import (SPECIES_NAMES, library_powers, monomials,
                                       select_from_path, split_coefficients, stlsq_path,
                                       term_name, uniform_spacing)


def _second_difference(U, dx, axis):
    """Second-order central second derivative along `axis`; the two edge slices are NaN."""
    out = np.full_like(U, np.nan)
    n = U.shape[axis]
    centre = [slice(None)] * U.ndim
    plus, minus = list(centre), list(centre)
    centre[axis], plus[axis], minus[axis] = slice(1, n - 1), slice(2, n), slice(0, n - 2)
    out[tuple(centre)] = (U[tuple(plus)] - 2.0 * U[tuple(centre)] + U[tuple(minus)]) / dx ** 2
    return out


def derivatives(U, x, t, sg_window_t=0, sg_window_x=0, sg_order=3):
    """
    (U_smooth, U_t, lap U, margins) for U of shape (T, Y, X, species).

    margins = (frames, cells): how many slices at each end of the time and space
    axes are unreliable (filter edge effects) and must be dropped. The grid is not
    treated as periodic: interpolated grids span [x_min, x_max] inclusive, so their
    wrap-around spacing differs from the interior spacing.
    """
    dt = uniform_spacing(t, 't')
    dx = uniform_spacing(x, 'x')
    for w, name in ((sg_window_t, 'sg_window_t'), (sg_window_x, 'sg_window_x')):
        if w and (w % 2 == 0 or w <= sg_order):
            raise ValueError(f"{name} must be odd and larger than sg_order ({sg_order}); got {w}")

    if sg_window_t:
        U_t = savgol_filter(U, sg_window_t, sg_order, deriv=1, delta=dt, axis=0)
        Us = savgol_filter(U, sg_window_t, sg_order, axis=0)
    else:
        U_t = np.gradient(U, dt, axis=0)
        Us = U

    if sg_window_x:
        U_yy = savgol_filter(Us, sg_window_x, sg_order, deriv=2, delta=dx, axis=1)
        U_xx = savgol_filter(Us, sg_window_x, sg_order, deriv=2, delta=dx, axis=2)
        U_yy = savgol_filter(U_yy, sg_window_x, sg_order, axis=2)     # smooth across, too
        U_xx = savgol_filter(U_xx, sg_window_x, sg_order, axis=1)
        Us = savgol_filter(savgol_filter(Us, sg_window_x, sg_order, axis=1),
                           sg_window_x, sg_order, axis=2)
    else:
        U_yy = _second_difference(Us, dx, axis=1)
        U_xx = _second_difference(Us, dx, axis=2)

    margins = (max(1, sg_window_t // 2), max(1, sg_window_x // 2))
    return Us, U_t, U_xx + U_yy, margins


def run(cfg, grid):
    """Fit PDE-FIND; returns (reaction dicts, learned D or None, extra result fields)."""
    opts = cfg.options
    powers = library_powers(cfg)

    Us, U_t, lap, (mt, mx) = derivatives(grid.U, grid.x, grid.t,
                                         opts.get('sg_window_t', 0), opts.get('sg_window_x', 0),
                                         opts.get('sg_order', 3))

    # Rows: frames past the cutoff and away from every filter edge.
    T, Y, X, S = grid.U.shape
    frames = np.where(grid.t >= grid.t_cutoff)[0]
    frames = frames[(frames >= mt) & (frames < T - mt)]
    if len(frames) == 0:
        raise ValueError("no frames left after the t cutoff and the time-filter margin")
    sl = (frames, slice(mx, Y - mx), slice(mx, X - mx))
    states = Us[sl].reshape(-1, S)
    dudt = U_t[sl].reshape(-1, S)
    laps = lap[sl].reshape(-1, S)

    rng = np.random.default_rng(cfg.seed)
    max_rows = int(opts.get('max_rows', 1_000_000))
    if len(states) > max_rows:
        keep = rng.choice(len(states), max_rows, replace=False)
        states, dudt, laps = states[keep], dudt[keep], laps[keep]
    print(f"PDE-FIND: {len(frames)} frames x {Y - 2 * mx} x {X - 2 * mx} cells -> "
          f"{len(states):,} rows (sg windows t={opts.get('sg_window_t', 0)}, "
          f"x={opts.get('sg_window_x', 0)})", flush=True)

    Theta = monomials(states, powers)
    names = [term_name(p) for p in powers]
    diffusion_names = None
    if cfg.learns_diffusion:
        Y_target = dudt
        diffusion_names = [f"lap({n})" for n in SPECIES_NAMES]
        Theta = np.hstack([Theta, laps])
        names += diffusion_names
    else:
        Y_target = dudt - laps * np.asarray(cfg.diff_coeffs, dtype=float)[None, :]

    path = stlsq_path(Theta, Y_target, names, mcas=cfg.mcas,
                      thresholds=opts.get('thresholds'), alpha=opts.get('alpha', 1e-5),
                      seed=cfg.seed)
    coef, chosen = select_from_path(path, opts.get('selection', 'share'),
                                    opts.get('threshold'), opts.get('selection_tol', 0.01))
    reaction, D, cross = split_coefficients(coef, names, powers, diffusion_names)
    return reaction, D, {'path': path, 'feature_names': names, 'selected_threshold': chosen,
                         'cross_diffusion': cross, 'n_rows': len(states)}
