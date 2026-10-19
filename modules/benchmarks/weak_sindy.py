"""
Weak SINDy for PDEs (Messenger & Bortz, J. Comput. Phys. 2021), via PySINDy's
WeakPDELibrary, for 2-species reaction-diffusion data.

Instead of differentiating the data, every term is integrated against K compactly
supported test functions phi on random space-time subdomains, and derivatives are
moved onto phi by integration by parts:

    integral phi u_t   = - integral phi_t u
    integral phi lap u =   integral u lap(phi)

so noise is averaged rather than amplified. The regression is then the same as
PDE-FIND's: targets  y_s = weak(u_s,t) - D_s weak(lap u_s)  (D fixed), or weak(u_s,t)
with weak(lap u) and weak(lap v) as extra candidates (D learned), and the same STLSQ
path and selection rule (common.stlsq_path / select_from_path).

Options (config.json "options"):
    K          number of test-function subdomains (default 1000)
    H_frac     subdomain half-widths as fractions of the (x, y, t) extents of the
               data used (default [0.1, 0.1, 0.1])
    p          test-function smoothness exponent, phi ~ (1 - s^2)^p (default 4)
    thresholds, alpha, selection ('share' | 'knee' | 'fixed'), threshold,
               selection_tol: see common.select_from_path (default 'share', 0.01)

Memory grows with K x points per subdomain; at a 200 x 200 grid the defaults need
a few GB.
"""
import numpy as np

from modules.benchmarks.common import (SPECIES_NAMES, library_powers, select_from_path,
                                       split_coefficients, stlsq_path, term_name,
                                       uniform_spacing)


def _parse_poly_name(name, names=SPECIES_NAMES):
    """PySINDy polynomial feature name ('1', 'u', 'u^2 v') -> exponent tuple."""
    powers = [0] * len(names)
    if name.strip() == '1':
        return tuple(powers)
    for factor in name.split():
        base, _, k = factor.partition('^')
        if base not in names:
            return None                     # not a polynomial feature (a derivative)
        powers[names.index(base)] += int(k) if k else 1
    return tuple(powers)


def weak_features(cfg, grid):
    """
    (Theta_poly (K, n_terms) in library_powers order, weak Laplacians (K, species),
    weak time derivatives (K, species)).
    """
    import pysindy as ps

    opts = cfg.options
    frames = grid.t >= grid.t_cutoff
    U = grid.U[frames]                                   # (T, Y, X, S)
    t = grid.t[frames]
    uniform_spacing(t, 't')
    uniform_spacing(grid.x, 'x')

    # PySINDy wants (x, y, t, species) data on a (x, y, t, 3) coordinate grid.
    U_xyt = np.ascontiguousarray(np.transpose(U, (2, 1, 0, 3)))
    X, Y, T = np.meshgrid(grid.x, grid.x, t, indexing='ij')
    st_grid = np.stack([X, Y, T], axis=-1)

    extents = [grid.x[-1] - grid.x[0], grid.x[-1] - grid.x[0], t[-1] - t[0]]
    H_frac = opts.get('H_frac', [0.1, 0.1, 0.1])
    H_xt = [float(f * e) for f, e in zip(H_frac, extents)]
    K = int(opts.get('K', 1000))
    p = int(opts.get('p', 4))
    print(f"Weak SINDy: {U.shape[0]} frames x {U.shape[1]} x {U.shape[2]}, K={K} test "
          f"functions, half-widths (x, y, t) = {[round(h, 4) for h in H_xt]}, p={p}", flush=True)

    # PySINDy places the subdomains with NumPy's global random state.
    np.random.seed(cfg.seed)
    lib = ps.WeakPDELibrary(
        function_library=ps.PolynomialLibrary(degree=cfg.degree,
                                              include_bias=cfg.include_constant),
        derivative_order=2, spatiotemporal_grid=st_grid, include_interaction=False,
        K=K, H_xt=H_xt, p=p)
    Theta = np.asarray(lib.fit_transform(U_xyt))
    lhs = np.asarray(lib.convert_u_dot_integral(U_xyt))
    names = lib.get_feature_names(list(SPECIES_NAMES))
    col = {n: j for j, n in enumerate(names)}

    powers = library_powers(cfg)
    by_powers = {}
    for n, j in col.items():
        pw = _parse_poly_name(n)
        if pw is not None:
            by_powers[pw] = j
    missing = [term_name(pw) for pw in powers if pw not in by_powers]
    if missing:
        raise RuntimeError(f"PySINDy's library lacks {missing}; its feature names were {names}")
    Theta_poly = Theta[:, [by_powers[pw] for pw in powers]]

    # Weak Laplacians: the u_11 + u_22 columns (index 1 = x, 2 = y).
    laps = np.stack([Theta[:, col[f'{s}_11']] + Theta[:, col[f'{s}_22']] for s in SPECIES_NAMES],
                    axis=1)
    return Theta_poly, laps, lhs, {'K': K, 'H_xt': H_xt, 'p': p}


def run(cfg, grid):
    """Fit weak SINDy; returns (reaction dicts, learned D or None, extra result fields)."""
    opts = cfg.options
    powers = library_powers(cfg)
    Theta, laps, lhs, settings = weak_features(cfg, grid)

    names = [term_name(pw) for pw in powers]
    diffusion_names = None
    if cfg.learns_diffusion:
        diffusion_names = [f"lap({n})" for n in SPECIES_NAMES]
        Theta = np.hstack([Theta, laps])
        names += diffusion_names
        target = lhs
    else:
        target = lhs - laps * np.asarray(cfg.diff_coeffs, dtype=float)[None, :]

    path = stlsq_path(Theta, target, names, mcas=cfg.mcas,
                      thresholds=opts.get('thresholds'), alpha=opts.get('alpha', 1e-5),
                      seed=cfg.seed)
    coef, chosen = select_from_path(path, opts.get('selection', 'share'),
                                    opts.get('threshold'), opts.get('selection_tol', 0.01))
    reaction, D, cross = split_coefficients(coef, names, powers, diffusion_names)
    return reaction, D, {'path': path, 'feature_names': names, 'selected_threshold': chosen,
                         'cross_diffusion': cross, 'weak_settings': settings}
