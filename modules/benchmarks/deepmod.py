"""
DeepMoD (Both, Choudhury, Sens & Kusters, J. Comput. Phys. 2021) for 2-species
reaction-diffusion data, built from the DeePyMoD package (pip install deepymod).

DeepMoD trains a neural surface u(t, x, y) on the data and, in the same loss,
asks the library computed from that surface by autograd to explain u_t:

    loss = MSE(surface, data) + sum_s mean( (y_s - Theta_s xi_s)^2 )

where xi_s is the least-squares solution on the currently active terms (the
LeastSquares constraint). Once the test MSE stops improving, a sparsity estimator
(LassoCV on the column-normalised library, then a threshold on the normalised
coefficients) sets which terms stay active, and is re-applied periodically
(TrainTestPeriodic). Training stops when the active coefficients stop changing.

Only the reaction-diffusion library is specific to this benchmark:

    D fixed:   y_s = u_s,t - D_s lap(u_s),   Theta = monomials of (u, v)
    D learned: y_s = u_s,t,                  Theta = monomials + [lap(u), lap(v)]

Two adaptations, both stated so they can be reported:
  - minibatches: DeePyMoD's own Loader feeds the full data set every step, which
    does not fit in GPU memory with second derivatives at 50 x 50 x 100 points or
    more. MinibatchLoader draws `batch_size` random rows per step instead; the
    training loop, constraint, estimator and schedulers are DeePyMoD's.
  - QRLeastSquares replaces DeePyMoD's LeastSquares (same QR solution) because
    torch.qr is deprecated / removed in recent PyTorch.

Frames before the PDE t cutoff are left out entirely (the BINN fits its surface
to them but leaves them out of the PDE loss).

mcas runs are fitted with independent u and v equations (DeePyMoD has no way to
tie them); the result is scored against the true +F / -F pair all the same.

Options (config.json "options"), defaults in DEFAULTS below:
    network                 'nn' (tanh MLP, DeepMoD's default), 'siren', or 'fourier'
                            (the BINN's own Fourier-feature surface: isolates the
                            surface from the sparse-selection method)
    hidden                  hidden layer widths
    first_omega_0, hidden_omega_0   SIREN frequencies
    fourier_scale, fourier_mapping_size, fourier_layers   for network 'fourier'
    lr, batch_size, max_iterations, write_iterations
    threshold               DeePyMoD Threshold on normalised coefficients
    periodicity, patience_sparsity, delta_sparsity   TrainTestPeriodic
    patience, delta         convergence of the L1 norm of the active coefficients
    n_eval                  rows used for the final least-squares coefficients
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from modules.benchmarks.common import SPECIES_NAMES, library_powers

DEFAULTS = {
    'network': 'nn',
    'hidden': [50, 50, 50, 50],
    'first_omega_0': 30.0,
    'hidden_omega_0': 30.0,
    'fourier_scale': 1.0,
    'fourier_mapping_size': 64,
    'fourier_layers': [256, 256, 256],
    'lr': 2e-3,
    'batch_size': 20_000,
    'max_iterations': 100_000,
    'write_iterations': 25,
    'threshold': 0.1,
    'periodicity': 25,
    'patience_sparsity': 200,
    'delta_sparsity': 1e-5,
    'patience': 200,
    'delta': 1e-4,
    'n_eval': 200_000,
    'n_val': 200_000,
}


# ---------------------------------------------------------------------------
# DeePyMoD building blocks for reaction-diffusion
# ---------------------------------------------------------------------------
def _monomial_matrix(pred, powers):
    """(n, n_terms) monomials of pred (n, 2), by repeated multiplication (autograd-safe at 0)."""
    max_p = max(max(p) for p in powers)
    ones = torch.ones_like(pred[:, 0])
    pows = []
    for s in range(pred.shape[1]):
        col = [ones]
        for _ in range(max_p):
            col.append(col[-1] * pred[:, s])
        pows.append(col)
    return torch.stack([pows[0][p[0]] * pows[1][p[1]] for p in powers], dim=1)


def make_library(powers, diff_coeffs, t_scale, x_scale):
    """
    A DeePyMoD Library returning, per species, the PDE target and the candidate
    matrix, in physical units. Coordinates are (t, x, y) normalised to [-1, 1];
    t_scale and x_scale are the chain-rule factors 2 / (max - min).
    """
    from deepymod.model.deepmod import Library

    class ReactionDiffusionLibrary(Library):
        def __init__(self):
            super().__init__()
            self.powers = powers
            self.diff_coeffs = diff_coeffs          # None: learn D (Laplacians as candidates)

        def library(self, input):
            prediction, coords = input
            u_t, laps = [], []
            for s in range(prediction.shape[1]):
                g = grad(prediction[:, s:s + 1], coords,
                         grad_outputs=torch.ones_like(prediction[:, s:s + 1]),
                         create_graph=True)[0]
                g_x = grad(g[:, 1:2], coords, grad_outputs=torch.ones_like(g[:, 1:2]),
                           create_graph=True)[0][:, 1:2]
                g_y = grad(g[:, 2:3], coords, grad_outputs=torch.ones_like(g[:, 2:3]),
                           create_graph=True)[0][:, 2:3]
                u_t.append(g[:, 0:1] * t_scale)
                laps.append((g_x + g_y) * x_scale ** 2)

            theta = _monomial_matrix(prediction, self.powers)
            if self.diff_coeffs is None:
                theta = torch.cat([theta] + laps, dim=1)
                targets = u_t
            else:
                targets = [u_t[s] - self.diff_coeffs[s] * laps[s]
                           for s in range(prediction.shape[1])]
            return targets, [theta] * prediction.shape[1]

    return ReactionDiffusionLibrary()


def make_constraint():
    """DeePyMoD's LeastSquares constraint, solved with torch.linalg (torch.qr is gone)."""
    from deepymod.model.deepmod import Constraint

    class QRLeastSquares(Constraint):
        def fit(self, sparse_thetas, time_derivs):
            coeffs = []
            for theta, dt in zip(sparse_thetas, time_derivs):
                if theta.shape[1] == 0:
                    coeffs.append(theta.new_zeros((0, 1)))
                    continue
                Q, R = torch.linalg.qr(theta)
                coeffs.append(torch.linalg.solve_triangular(R, Q.T @ dt, upper=True))
            return coeffs

    return QRLeastSquares()


class FourierSurface(nn.Module):
    """The BINN's Fourier-feature surface, in DeePyMoD's function-approximator interface."""

    def __init__(self, mapping_size, scale, layers, species=2):
        super().__init__()
        from modules.binn_eql.model.surface_fitter import SurfaceFitter
        self.net = SurfaceFitter(input_features=3, species=species, mapping_size=mapping_size,
                                 scale=scale, layers=list(layers) + [species])

    def forward(self, input):
        coords = input.clone().detach().requires_grad_(True)
        # DeePyMoD orders inputs (t, x, y); the BINN surface expects (x, y, t).
        return self.net(coords[:, [1, 2, 0]]), coords


def make_network(o, species=2):
    from deepymod.model.func_approx import NN, Siren
    if o['network'] == 'nn':
        return NN(3, list(o['hidden']), species)
    if o['network'] == 'siren':
        return Siren(3, list(o['hidden']), species, first_omega_0=o['first_omega_0'],
                     hidden_omega_0=o['hidden_omega_0'])
    if o['network'] == 'fourier':
        return FourierSurface(o['fourier_mapping_size'], o['fourier_scale'],
                              o['fourier_layers'], species)
    raise ValueError(f"network must be 'nn', 'siren' or 'fourier', got {o['network']!r}")


class MinibatchLoader:
    """
    Stands in for DeePyMoD's Loader: `batches` random minibatches per iteration
    (DeePyMoD's Loader returns the whole data set once per iteration).
    """

    def __init__(self, coords, targets, batch_size, batches=1, generator=None):
        self.coords, self.targets = coords, targets
        self.device = coords.device
        self.batch_size = min(batch_size, len(coords))
        self._length = batches
        self.generator = generator

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if idx >= self._length:
            raise IndexError
        if self.batch_size >= len(self.coords):
            return self.coords, self.targets
        i = torch.randint(0, len(self.coords), (self.batch_size,), device=self.device,
                          generator=self.generator)
        return self.coords[i], self.targets[i]


# ---------------------------------------------------------------------------
def final_coefficients(model, coords, n_eval, chunk=20_000):
    """
    Least-squares coefficients of the active terms on n_eval rows (in chunks: the
    library needs autograd), i.e. the LeastSquares constraint on more data than one
    minibatch. Returns (species, n_candidates) in physical units.
    """
    n = min(n_eval, len(coords))
    idx = torch.randperm(len(coords), device=coords.device)[:n]
    masks = [m.detach().cpu().numpy().astype(bool) for m in model.sparsity_masks]
    n_feat = len(masks[0])
    AtA = [np.zeros((n_feat, n_feat)) for _ in masks]
    Aty = [np.zeros(n_feat) for _ in masks]
    for start in range(0, n, chunk):
        batch = coords[idx[start:start + chunk]]
        prediction, c = model.func_approx(batch)
        targets, thetas = model.library((prediction, c))
        for s, (y, th) in enumerate(zip(targets, thetas)):
            th = th.detach().double().cpu().numpy()
            y = y.detach().double().cpu().numpy().ravel()
            AtA[s] += th.T @ th
            Aty[s] += th.T @ y
    coef = np.zeros((len(masks), n_feat))
    for s, m in enumerate(masks):
        if m.any():
            coef[s, m] = np.linalg.lstsq(AtA[s][np.ix_(m, m)], Aty[s][m], rcond=None)[0]
    return coef


def run(cfg, grid, run_dir):
    """Fit DeepMoD; returns (reaction dicts, learned D or None, extra result fields)."""
    from deepymod import DeepMoD
    from deepymod.model.sparse_estimators import Threshold
    from deepymod.training import train
    from deepymod.training.sparsity_scheduler import TrainTestPeriodic
    from sklearn.linear_model import LassoCV

    from modules.benchmarks.common import split_coefficients, term_name

    o = {**DEFAULTS, **cfg.options}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # --- Rows (t, x, y) -> (u, v), frames past the cutoff, coordinates in [-1, 1] ---
    frames = np.where(grid.t >= grid.t_cutoff)[0]
    U = grid.U[frames]                                     # (T, Y, X, S)
    t = grid.t[frames]
    T_, Y_, X_ = np.meshgrid(t, grid.x, grid.x, indexing='ij')   # matches U's (T, Y, X)
    coords = np.stack([T_.ravel(), X_.ravel(), Y_.ravel()], axis=1)
    targets = U.reshape(-1, U.shape[-1])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    coords = 2.0 * (coords - lo) / (hi - lo) - 1.0
    t_scale = 2.0 / (hi[0] - lo[0])
    x_scale = 2.0 / (hi[1] - lo[1])

    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    targets = torch.tensor(targets, dtype=torch.float32, device=device)
    perm = torch.randperm(len(coords), device=device)
    n_val = min(int(0.2 * len(coords)), o['n_val'])
    val, train_idx = perm[:n_val], perm[n_val:]
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    train_loader = MinibatchLoader(coords[train_idx], targets[train_idx], o['batch_size'],
                                   generator=gen)
    test_loader = MinibatchLoader(coords[val], targets[val], len(val))
    print(f"DeepMoD: {len(frames)} frames x {Y_.shape[1]} x {X_.shape[2]} -> "
          f"{len(train_idx):,} training / {n_val:,} test rows, batch {train_loader.batch_size}, "
          f"network {o['network']}, device {device}", flush=True)

    # --- Model ---
    powers = library_powers(cfg)
    D = None if cfg.learns_diffusion else [float(d) for d in cfg.diff_coeffs]
    model = DeepMoD(make_network(o, cfg.species),
                    make_library(powers, D, t_scale, x_scale),
                    Threshold(o['threshold'], LassoCV(cv=5, fit_intercept=False)),
                    make_constraint()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True,
                                 lr=o['lr'])
    scheduler = TrainTestPeriodic(periodicity=o['periodicity'],
                                  patience=o['patience_sparsity'], delta=o['delta_sparsity'])

    log_dir = f"{run_dir}/deepmod_tensorboard/"
    train(model, train_loader, test_loader, optimizer, scheduler,
          log_dir=log_dir, max_iterations=o['max_iterations'],
          write_iterations=o['write_iterations'], delta=o['delta'], patience=o['patience'])
    print("", flush=True)                                  # DeePyMoD's progress line uses \r

    # --- Coefficients on more rows than one minibatch ---
    names = [term_name(p) for p in powers]
    diffusion_names = None
    if D is None:
        diffusion_names = [f"lap({n})" for n in SPECIES_NAMES]
        names += diffusion_names
    coef = final_coefficients(model, coords[train_idx], o['n_eval'])
    reaction, D_learned, cross = split_coefficients(coef, names, powers, diffusion_names)

    masks = {SPECIES_NAMES[s]: [names[j] for j, on in enumerate(m.cpu().tolist()) if on]
             for s, m in enumerate(model.sparsity_masks)}
    torch.save(model.state_dict(), f"{run_dir}/deepmod_model.pt")
    with torch.no_grad():
        mse = torch.mean((model.func_approx(coords[val])[0] - targets[val]) ** 2, dim=0)
    return reaction, D_learned, {'active_terms': masks, 'cross_diffusion': cross,
                                 'surface_test_mse': mse.cpu().tolist(),
                                 'options': o, 'feature_names': names}
