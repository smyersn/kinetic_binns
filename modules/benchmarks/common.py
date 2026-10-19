"""
Shared pieces of the equation-learning benchmark (PDE-FIND, weak SINDy, DeepMoD).

Every baseline sees exactly what a BINN-EQL run with the same config sees:

    data      the raw .pt file -> noise_and_interpolate (same points, same
              multiplicative noise level) -> a (T, Y, X, species) grid
    t cutoff  frames before unresolved_t_cutoff are left out of the regression,
              as they are left out of the BINN's PDE loss
    library   the same monomials (monomial_powers: same order, same names),
              degree and constant term
    D         fixed to the config's diff_coeffs, or learned when diff_coeffs is []

and is scored the same way (score_reaction): support precision / recall against
the true monomial coefficients, relative coefficient errors, and the error of the
learned reaction F(u, v) on the noise-free states the system visits, which is
meaningful even when the learned support is a lookalike of the true one.

Benchmarks assume 2 spatial dimensions and 2 species (the data format helpers do).
"""
import json
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from modules.binn_eql.library.polynomial import monomial_powers
from modules.binn_eql.physics.frame_statistics import unresolved_t_cutoff
from modules.utils.format_data import format_training_data_to_u_array
from modules.utils.noise_and_interpolate import noise_and_interpolate

SPECIES_NAMES = ('u', 'v')


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BaselineConfig:
    """Contents of config.json for one baseline run."""
    method: str                   # "pdefind", "weak_sindy" or "deepmod"
    training_data_path: str
    reaction: str                 # key into REACTION_REGISTRY (ground truth)
    params: list                  # ground-truth reaction parameters
    epsilon: float                # multiplicative noise level
    points: int                   # interpolation grid (0 = native)
    diff_coeffs: list             # fixed D per species; [] to learn D
    degree: int = 3
    include_constant: bool = True
    mcas: bool = False            # True: one shared F, applied as +F to u and -F to v
    species: int = 2
    dimensions: int = 2
    true_diff_coeffs: list = field(default_factory=list)   # for scoring learned D only
    seed: int = 0                 # noise realisation, row subsampling, network init
    pde_t_cutoff: Optional[float] = None   # None: detect like the BINN does
    simulate: bool = True         # integrate the learned PDE and score the pattern
    options: dict = field(default_factory=dict)   # method-specific settings

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    @property
    def learns_diffusion(self):
        return not self.diff_coeffs


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class GridData:
    U: np.ndarray            # (T, Y, X, species) training grid, after noise
    x: np.ndarray            # (X,) grid coordinates (y uses the same values)
    t: np.ndarray            # (T,) frame times
    t_cutoff: float          # frames with t < t_cutoff are left out of the regression
    raw: torch.Tensor        # raw rows [x, y, t, u, v], native resolution, no noise
    reference_states: np.ndarray   # (n, species) noise-free states with t >= t_cutoff


def load_grid_data(cfg, n_reference=200_000):
    """
    The training grid, built by the same path as train_binn_eql.load_training_data.

    The noise is seeded with cfg.seed, so repeats of a baseline differ only in
    their noise realisation (and, for DeepMoD, initialisation).
    """
    if cfg.dimensions != 2 or cfg.species != 2:
        raise ValueError("the benchmark code handles 2 dimensions and 2 species only")

    raw = torch.load(cfg.training_data_path)['training_data']
    expected = cfg.dimensions + 1 + cfg.species
    if raw.ndim != 2 or raw.shape[1] != expected:
        raise ValueError(f"{cfg.training_data_path} has shape {tuple(raw.shape)}; "
                         f"expected (N, {expected}) rows of [x, y, t, u, v]")

    torch.manual_seed(cfg.seed)
    data = raw
    if cfg.epsilon != 0 or cfg.points != 0:
        data = noise_and_interpolate(raw.clone(), cfg.points, cfg.epsilon, cfg.dimensions,
                                     cfg.species, multiplicative_noise=True)

    U, x, t = format_training_data_to_u_array(data)
    U = U.double().numpy()
    x = x.double().numpy()
    t = t.double().numpy()

    if cfg.pde_t_cutoff is not None:
        cutoff = float(cfg.pde_t_cutoff)
    else:
        cutoff = unresolved_t_cutoff(data, cfg.dimensions, cfg.species)
    print(f"Grid: {U.shape[0]} frames x {U.shape[1]} x {U.shape[2]}, noise {cfg.epsilon}; "
          f"PDE t cutoff {cutoff:.4g} ({int((t < cutoff).sum())} leading frames left out)",
          flush=True)

    # Noise-free states visited by the system, for scoring the learned reaction.
    rows = raw[raw[:, cfg.dimensions] >= cutoff, cfg.dimensions + 1:].double().numpy()
    rng = np.random.default_rng(cfg.seed)
    if len(rows) > n_reference:
        rows = rows[rng.choice(len(rows), n_reference, replace=False)]

    return GridData(U=U, x=x, t=t, t_cutoff=cutoff, raw=raw, reference_states=rows)


def uniform_spacing(values, name):
    """The spacing of an evenly spaced coordinate array; raise if it is not even."""
    steps = np.diff(values)
    if not np.allclose(steps, steps[0], rtol=1e-3, atol=0):
        raise ValueError(f"{name} is not evenly spaced (steps {steps.min():.4g} to "
                         f"{steps.max():.4g}); these baselines need a uniform grid")
    return float(steps.mean())


# ---------------------------------------------------------------------------
# Library and coefficients
# ---------------------------------------------------------------------------
def library_powers(cfg):
    """Monomial exponent tuples, in the BINN library's order."""
    return monomial_powers(cfg.species, cfg.degree, cfg.include_constant)


def term_name(powers, names=SPECIES_NAMES):
    """(0, 0) -> '1', (1, 0) -> 'u', (2, 1) -> 'u^2*v' (resimulate_feql's --set format)."""
    factors = [n if k == 1 else f"{n}^{k}" for n, k in zip(names, powers) if k]
    return '*'.join(factors) if factors else '1'


def monomials(states, powers):
    """(n, len(powers)) matrix of u^p * v^q for each exponent tuple."""
    states = np.asarray(states, dtype=float)
    return np.stack([np.prod(states ** np.array(p, dtype=float), axis=1) for p in powers],
                    axis=1)


def coefficients_to_dicts(coef, powers):
    """(species, n_terms) array -> one {exponent tuple: value} dict per species, nonzeros only."""
    return [{p: float(c) for p, c in zip(powers, row) if c != 0.0} for row in coef]


def dicts_to_json(coefs, names=SPECIES_NAMES):
    """[{(2, 1): 1.0}, ...] -> {'u': {'u^2*v': 1.0}, ...}."""
    return {names[s]: {term_name(p, names): c for p, c in d.items()} for s, d in enumerate(coefs)}


def equation_strings(coefs, diffusion, names=SPECIES_NAMES):
    """Human-readable equations, e.g. 'du/dt = 0.02 lap(u) + 1 - 4 u + u^2*v'."""
    lines = []
    for s, d in enumerate(coefs):
        terms = [f"{diffusion[s]:.4g} lap({names[s]})"] if diffusion is not None else []
        for p, c in d.items():
            name = term_name(p, names)
            terms.append(f"{c:+.4g}" if name == '1' else f"{c:+.4g} {name}")
        lines.append(f"d{names[s]}/dt = " + (" ".join(terms) if terms else "0"))
    return lines


def true_coefficients(reaction, params):
    """
    The true reaction as one {exponent tuple: coefficient} dict per species,
    or None for reactions that are not polynomial (Hill terms).
    """
    p = list(params)
    if reaction == 'brusselator':
        a, b = p
        return [{(0, 0): a, (1, 0): -(b + 1), (2, 1): 1.0}, {(1, 0): b, (2, 1): -1.0}]
    if reaction == 'schnakenberg':
        a, b = p
        return [{(0, 0): a, (1, 0): -1.0, (2, 1): 1.0}, {(0, 0): b, (2, 1): -1.0}]
    if reaction == 'gray_scott':
        feed, kill = p
        return [{(0, 0): feed, (1, 0): -feed, (1, 2): -1.0}, {(1, 2): 1.0, (0, 1): -(feed + kill)}]
    if reaction == 'fitzhugh_nagumo':
        a, b, eps = p
        return [{(0, 0): a, (1, 0): 1.0, (3, 0): -1.0, (0, 1): -1.0},
                {(1, 0): eps, (0, 1): -eps * b}]
    if reaction == 'poly_poly':
        a, b = p
        F = {(2, 1): a, (1, 0): -b}
        return [F, {k: -c for k, c in F.items()}]
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_reaction(est, cfg, reference_states, est_diffusion=None, F_hat=None):
    """
    Metrics for one learned reaction (list of {exponent tuple: coefficient}).

    Support and coefficient metrics need a polynomial ground truth (and terms the
    library can express); the reaction-function error needs only REACTION_REGISTRY.

    F_hat: the learned reaction evaluated on reference_states, (n, species), for
    models whose reaction is not a polynomial (a BINN with Hill terms). Pass
    est=None to skip the support and coefficient metrics.
    """
    from modules.simulation.reaction_library import REACTION_REGISTRY

    out = {'species': {}}
    truth = true_coefficients(cfg.reaction, cfg.params)
    powers = library_powers(cfg)

    # Reaction-function error on the noise-free states the system visits.
    fu, fv = REACTION_REGISTRY[cfg.reaction]['fn'](reference_states, cfg.params)
    F_true = np.stack([np.asarray(fu), np.asarray(fv)], axis=1)
    if F_hat is None:
        Phi = monomials(reference_states, powers)
        F_hat = np.stack([Phi @ np.array([est[s].get(p, 0.0) for p in powers])
                          for s in range(cfg.species)], axis=1)
    for s, name in enumerate(SPECIES_NAMES):
        denom = np.linalg.norm(F_true[:, s])
        out['species'][name] = {
            'F_rel_error': (float(np.linalg.norm(F_hat[:, s] - F_true[:, s]) / denom)
                            if denom else None),
            'n_terms': len(est[s]) if est is not None else None,
        }

    if truth is not None and est is not None:
        unreachable = [p for d in truth for p in d if p not in powers]
        if unreachable:
            out['note'] = (f"true term(s) {[term_name(p) for p in unreachable]} are not in "
                           f"the library (degree / include_constant)")
        tp = fp = fn = 0
        for s, name in enumerate(SPECIES_NAMES):
            t_sup, e_sup = set(truth[s]), set(est[s])
            m = out['species'][name]
            m['true_positives'] = len(t_sup & e_sup)
            m['false_positives'] = len(e_sup - t_sup)
            m['false_negatives'] = len(t_sup - e_sup)
            m['missing_terms'] = [term_name(p) for p in sorted(t_sup - e_sup)]
            m['spurious_terms'] = [term_name(p) for p in sorted(e_sup - t_sup)]
            m['exact_support'] = (e_sup == t_sup)
            # Relative error of every true coefficient (a missing term counts as 100 %).
            rel = {term_name(p): abs(est[s].get(p, 0.0) - c) / abs(c) for p, c in truth[s].items()}
            m['coef_rel_error'] = rel
            m['coef_rel_error_mean'] = float(np.mean(list(rel.values())))
            m['coef_rel_error_max'] = float(np.max(list(rel.values())))
            union = sorted(t_sup | e_sup)
            diff = [est[s].get(p, 0.0) - truth[s].get(p, 0.0) for p in union]
            m['coef_vector_rel_error'] = float(np.linalg.norm(diff) /
                                               np.linalg.norm(list(truth[s].values())))
            tp += m['true_positives']
            fp += m['false_positives']
            fn += m['false_negatives']
        out['precision'] = tp / (tp + fp) if tp + fp else 0.0
        out['recall'] = tp / (tp + fn) if tp + fn else 0.0
        out['exact_support'] = all(out['species'][n]['exact_support'] for n in SPECIES_NAMES)

    if cfg.learns_diffusion and est_diffusion is not None:
        true_D = list(cfg.true_diff_coeffs) or [None] * cfg.species
        out['diffusion'] = {
            name: {'learned': float(d), 'true': D,
                   'rel_error': abs(d - D) / D if D else None}
            for name, d, D in zip(SPECIES_NAMES, est_diffusion, true_D)}
    return out


# ---------------------------------------------------------------------------
# Sparse regression path shared by PDE-FIND and weak SINDy
# ---------------------------------------------------------------------------


def default_thresholds(Xs, Ys, n=60):
    """
    0 plus a log-spaced grid from below the smallest to above the largest
    coefficient of the dense least-squares fit (in the scaled units), so the path
    always runs from the full library to the empty model. A fixed grid can miss
    both ends: collinear monomials give dense fits with coefficients well above 1.
    """
    W, *_ = np.linalg.lstsq(Xs, Ys, rcond=None)
    mags = np.abs(W[np.abs(W) > 0])
    if len(mags) == 0:
        return [0.0]
    lo, hi = np.log10(mags.min() / 2), np.log10(mags.max() * 1.5)
    return [0.0] + list(np.logspace(lo, hi, n))


def stlsq_path(Theta, Y, feature_names, mcas=False, thresholds=None, alpha=1e-5,
               val_frac=0.2, seed=0, max_iter=20):
    """
    PySINDy STLSQ fits of Y on Theta over a path of thresholds (default: see
    default_thresholds).

    Theta (n, n_features); Y (n, species) targets. Columns and targets are scaled
    to unit RMS before fitting, so a threshold is scale-free: a term is cut when
    its RMS contribution is below `threshold` x the RMS of its target. STLSQ
    refits the surviving terms without the threshold (unbias=True).

    mcas: one shared reaction F, fitted to u's target and to minus v's target
    together (rows stacked), then reported as +F for u and -F for v.

    Returns a list with one entry per threshold:
        {'threshold', 'coef' (species, n_features) in physical units,
         'val_error' (species,) held-out MSE in target-RMS units,
         'null_error' (species,) the same for the empty model, 'nnz' (species,)}
    """
    import pysindy as ps

    Theta = np.asarray(Theta, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if mcas:
        Theta = np.vstack([Theta, Theta])
        Y = np.concatenate([Y[:, 0], -Y[:, 1]])[:, None]

    col_rms = np.sqrt(np.mean(Theta ** 2, axis=0))
    col_rms[col_rms == 0] = 1.0
    y_rms = np.sqrt(np.mean(Y ** 2, axis=0))
    y_rms[y_rms == 0] = 1.0
    Xs = Theta / col_rms
    Ys = Y / y_rms

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xs))
    n_val = int(val_frac * len(Xs))
    val, train = perm[:n_val], perm[n_val:]
    if thresholds is None:
        thresholds = default_thresholds(Xs[train], Ys[train])
    null_err = np.mean(Ys[val] ** 2, axis=0)            # error of the empty model
    if mcas:
        null_err = np.repeat(null_err, 2)

    path = []
    for th in thresholds:
        opt = ps.STLSQ(threshold=float(th), alpha=alpha, max_iter=max_iter, unbias=True,
                       normalize_columns=False)
        opt.fit(Xs[train], Ys[train])
        coef_s = np.atleast_2d(opt.coef_)                     # (targets, features)
        resid = Ys[val] - Xs[val] @ coef_s.T
        coef = coef_s * y_rms[:, None] / col_rms[None, :]
        val_err = np.mean(resid ** 2, axis=0)
        if mcas:
            coef = np.vstack([coef, -coef])
            val_err = np.repeat(val_err, 2)
        path.append({'threshold': float(th), 'coef': coef, 'val_error': val_err,
                     'null_error': null_err, 'nnz': (coef != 0).sum(axis=1)})
    print(f"STLSQ path: {len(thresholds)} thresholds, {len(train):,} fit rows, "
          f"{n_val:,} held-out rows, {len(feature_names)} candidate terms", flush=True)
    return path


def select_from_path(path, selection='share', threshold=None, tol=0.01):
    """
    One coefficient matrix (species, n_features) chosen from an STLSQ path.

    'share'  (default) per species, the sparsest entry whose held-out error exceeds
             the best on the path by at most tol x (null - best), where null is the
             empty model's error. A term is kept only if it is worth more than a
             fraction `tol` of the explainable range: the same currency the BINN's
             L0 pricing uses, so tol plays the role of the BINN's price per gate
             expressed as a share of the explainable range.
    'knee'   per species, the sparsest entry whose held-out error is within
             (1 + tol) of the best held-out error on the path
    'fixed'  the path entry whose threshold is closest to `threshold`

    Each rule is fixed across datasets and noise levels: the same rule, not a
    per-dataset tuned threshold. The whole path is saved in result.json, so other
    rules (including an oracle best-support pick) can be applied afterwards.
    """
    species = path[0]['coef'].shape[0]
    if selection == 'fixed':
        if threshold is None:
            raise ValueError("selection 'fixed' needs options.threshold")
        k = int(np.argmin([abs(e['threshold'] - threshold) for e in path]))
        return path[k]['coef'].copy(), [path[k]['threshold']] * species
    if selection not in ('share', 'knee'):
        raise ValueError(f"selection must be 'share', 'knee' or 'fixed', got {selection!r}")

    coef = np.zeros_like(path[0]['coef'])
    chosen = []
    for s in range(species):
        errs = np.array([e['val_error'][s] for e in path])
        best = errs.min()
        if selection == 'share':
            limit = best + tol * max(path[0]['null_error'][s] - best, 0.0)
        else:
            limit = (1 + tol) * best
        ok = [k for k, e in enumerate(errs) if e <= limit]
        k = min(ok, key=lambda k: (path[k]['nnz'][s], -path[k]['threshold']))
        coef[s] = path[k]['coef'][s]
        chosen.append(path[k]['threshold'])
    return coef, chosen


def split_coefficients(coef, feature_names, powers, diffusion_names=None):
    """
    Split (species, n_features) coefficients into reaction dicts and learned D.

    The first len(powers) columns are the monomials; any further columns are the
    Laplacians named in diffusion_names (learned-D runs). Own-species Laplacian
    coefficients are the learned D; cross-species ones are reported separately.
    """
    n = len(powers)
    reaction = coefficients_to_dicts(coef[:, :n], powers)
    if not diffusion_names:
        return reaction, None, {}
    D, cross = [], {}
    for s in range(coef.shape[0]):
        for j, name in enumerate(diffusion_names):
            c = float(coef[s, n + j])
            if j == s:
                D.append(c)
            elif c != 0.0:
                cross[f"{SPECIES_NAMES[s]}:{name}"] = c
    return reaction, D, cross


def path_to_json(path, feature_names):
    return [{'threshold': e['threshold'],
             'nnz': e['nnz'].tolist(),
             'val_error': e['val_error'].tolist(),
             'null_error': e['null_error'].tolist(),
             'coef': {SPECIES_NAMES[s]: {feature_names[j]: float(c)
                                          for j, c in enumerate(row) if c != 0.0}
                      for s, row in enumerate(e['coef'])}}
            for e in path]


# ---------------------------------------------------------------------------
# Learned-PDE simulation and pattern statistics
# ---------------------------------------------------------------------------
class PolynomialReaction(nn.Module):
    """A polynomial reaction in the learned reaction's interface: (n, species) -> (n, species)."""

    def __init__(self, coefs, powers):
        super().__init__()
        W = np.array([[d.get(p, 0.0) for p in powers] for d in coefs], dtype=np.float32)
        self.register_buffer('W', torch.tensor(W))                      # (species, terms)
        self.register_buffer('P', torch.tensor(np.array(powers, dtype=np.float32)))

    def forward(self, uv):
        feats = torch.prod(uv.unsqueeze(1) ** self.P, dim=2)            # (n, terms)
        return feats @ self.W.T


def radial_peak_wavelength(field, L):
    """Dominant wavelength of a 2D field from its radially averaged power spectrum."""
    f = field - field.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    n = f.shape[0]
    ky, kx = np.indices(power.shape) - n // 2
    k = np.sqrt(kx ** 2 + ky ** 2).astype(int)
    radial = np.bincount(k.ravel(), power.ravel()) / np.maximum(np.bincount(k.ravel()), 1)
    radial[0] = 0.0
    k_peak = int(np.argmax(radial[: n // 2]))
    return float(L / k_peak) if k_peak > 0 else None


def pattern_stats(U, L):
    """Spatial std, mean and dominant wavelength per species for one (Y, X, species) frame."""
    return {name: {'mean': float(U[..., s].mean()), 'std': float(U[..., s].std()),
                   'wavelength': radial_peak_wavelength(U[..., s], L)}
            for s, name in enumerate(SPECIES_NAMES)}


def simulate_model(model_ns, cfg, grid, gif_path):
    """
    Integrate a learned PDE from the raw noise-free initial condition at native
    resolution (as train_binn_eql.animate_simulations does for the BINN), save an
    animation, and compare the final frame with the true final frame.

    model_ns: anything simulate_feql accepts, i.e. an object whose .model has
    dimensions, species, diff_coeffs (or diffusion_fitter) and reaction.
    """
    from modules.simulation.animation import animate_u_array
    from modules.simulation.reaction_library import REACTION_REGISTRY
    from modules.simulation.simulation import simulate_feql

    try:
        sim_u, sim_x, sim_t = simulate_feql(grid.raw, model_ns,
                                            dt_cap=REACTION_REGISTRY[cfg.reaction]['dt_cap'])
    except FloatingPointError as exc:
        return {'status': 'diverged', 'reason': str(exc)}

    true_u, _, _ = format_training_data_to_u_array(grid.raw)
    animate_u_array(sim_u, sim_t, gif_path)
    sim_final = sim_u[-1].double().numpy()
    true_final = true_u[-1].double().numpy()
    L = float(sim_x.max())
    rel = [float(np.linalg.norm(sim_final[..., s] - true_final[..., s]) /
                 np.linalg.norm(true_final[..., s])) for s in range(cfg.species)]
    return {'status': 'ok',
            'final_frame_rel_error': dict(zip(SPECIES_NAMES, rel)),
            'learned_final': pattern_stats(sim_final, L),
            'true_final': pattern_stats(true_final, L)}


def simulate_and_score(coefs, diffusion, cfg, grid, run_dir):
    """simulate_model for a polynomial reaction found by a baseline; writes feql_sim.gif."""
    D = list(cfg.diff_coeffs) if not cfg.learns_diffusion else list(diffusion)
    if any(d <= 0 for d in D):
        return {'status': 'skipped', 'reason': f'non-positive learned diffusion {D}'}

    reaction = PolynomialReaction(coefs, library_powers(cfg))
    reaction = reaction.to('cuda' if torch.cuda.is_available() else 'cpu')
    model_ns = types.SimpleNamespace(model=types.SimpleNamespace(
        dimensions=cfg.dimensions, species=cfg.species, diff_coeffs=D, reaction=reaction))
    return simulate_model(model_ns, cfg, grid, f'{run_dir}/feql_sim.gif')


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------
def write_result(run_dir, cfg, coefs, diffusion, extra, started):
    """result.json and equation.txt for one baseline run; returns the result dict."""
    run_dir = Path(run_dir)
    lines = equation_strings(coefs, diffusion if cfg.learns_diffusion else cfg.diff_coeffs)
    result = {
        'method': cfg.method,
        'config': asdict(cfg),
        'runtime_s': time.time() - started,
        'coefficients': dicts_to_json(coefs),
        'diffusion': diffusion if cfg.learns_diffusion else list(cfg.diff_coeffs),
        'equations': lines,
        **extra,
    }
    save_json(run_dir / 'result.json', result)
    with open(run_dir / 'equation.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return result


def save_json(path, obj):
    """Write via a temporary file, so an interrupted write never leaves half a result.json."""
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)
    tmp.replace(path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")

