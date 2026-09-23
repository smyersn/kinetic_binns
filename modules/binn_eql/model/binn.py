"""
Biologically-informed neural network (BINN) for reaction-diffusion systems

    du/dt = D lap(u) + F(u),    u = (u_1, ..., u_S).

This module only assembles the model: three sub-networks and the fixed,
data-derived buffers they need. The objective is in physics/losses.py and
the training schedule in training/.

    surface_fitter    u(x, t)  smooth interpolant of the data
    diffusion_fitter  D        learned diffusion coefficients (None if fixed)
    reaction          F(u)     sparse symbolic reaction term (EQL)

Attribute and buffer names are part of the checkpoint format; renaming them
breaks loading of existing runs.
"""
import string

import torch
import torch.nn as nn

from modules.binn_eql.equations.extract import extract_params, generate_terms
from modules.binn_eql.equations.format import equations_as_strings, generate_equation
from modules.binn_eql.equations.prune import fine_tune_eql
from modules.binn_eql.library.eql_layer import EQLLayer
from modules.binn_eql.model.diffusion import DiffusionCoefficients
from modules.binn_eql.model.surface_fitter import SurfaceFitter
from modules.binn_eql.physics.frame_statistics import (
    frame_activity_weights, unresolved_t_cutoff)


def default_species_names(species):
    """u, v, w, ... skipping t, x, y, z; numeric suffixes beyond 22 species."""
    letters = [c for c in string.ascii_lowercase if c not in 'txyz']
    start = letters.index('u')
    ordered = letters[start:] + letters[:start]
    return [ordered[i] if i < len(ordered) else f"{ordered[i % len(ordered)]}{i // len(ordered)}"
            for i in range(species)]


class ReactionTerm(nn.Module):
    """F(u). A thin wrapper so parameters are stored under `reaction.eql_layer.*`."""

    def __init__(self, **eql_kwargs):
        super().__init__()
        self.eql_layer = EQLLayer(**eql_kwargs)

    def forward(self, u):
        return self.eql_layer(u)


class BINN(nn.Module):
    """
    Parameters
    ----------
    dimensions, species   spatial dimensions and number of species
    train_data            (N, dimensions + 1 + species) rows of [x..., t, u...]
    diff_coeffs           fixed D per species, or None/[] to learn D
    duplicates, degree    library size: copies of each term, max polynomial degree
    include_*             which term families populate the library
    param_bounds          soft bound on |w| (see BINNLoss.soft_wall)
    mcas                  two-state conserved system: learn one F, apply +F/-F
                          (see library/eql_layer.py)
    l0_reference_gates    fixed gate count for the L0 scale (see calibration.l0_scale)
    pde_t_cutoff          earliest time used in the PDE residual; None to detect
                          an unresolved initial transient automatically
    gls_max_weight        maximum per-frame GLS weight
    """

    def __init__(self, dimensions, species, train_data, duplicates=1,
                 diff_coeffs=None, uv_layers=None, degree=2, param_bounds=10,
                 fourier_scale=1.0, fourier_mapping_size=64,
                 mcas=False, species_names=None, l0_reference_gates=None,
                 pde_t_cutoff=None, gls_max_weight=50.0,
                 include_poly=True, include_increasing_hill=True, include_decreasing_hill=True):
        super().__init__()
        self.dimensions = dimensions
        self.species = species
        self.train_data = train_data  # kept for post-hoc pruning
        self.duplicates = duplicates
        self.degree = degree
        self.param_bounds = param_bounds
        self.mcas = mcas
        self.l0_reference_gates = l0_reference_gates
        self.species_names = species_names or default_species_names(species)

        # mcas changes how many equations are *reported*, not how many
        # species are simulated: both species still get a PDE residual.
        self.n_equations = 1 if mcas else species
        self.conservation_groups = [[0, 1]] if mcas else []

        # Buffers are built on CPU here, whatever train_data's device, so
        # that __init__ never mixes devices; .to(device) later moves them
        # together with the parameters.
        self._register_domain(train_data)
        self._register_gls_weights(train_data, gls_max_weight)
        self.pde_t_cutoff = self._resolve_pde_t_cutoff(train_data, pde_t_cutoff)
        self._register_concentration_scales(train_data)

        # Placeholder so the key exists in every checkpoint; measured from
        # the fitted surface at Phase 2 (BINNLoss.calibrate_physics).
        self.register_buffer('pde_scale', torch.ones(species))

        self.diff_coeffs = diff_coeffs
        if diff_coeffs:
            self.register_buffer('diff_coeffs_tensor', torch.tensor(diff_coeffs, dtype=torch.float32))
        else:
            self.diff_coeffs_tensor = None

        # --- Sub-networks -------------------------------------------------
        self.diffusion_fitter = None if diff_coeffs else DiffusionCoefficients(species)
        self.surface_fitter = SurfaceFitter(
            input_features=dimensions + 1, species=species, layers=uv_layers,
            scale=fourier_scale, mapping_size=fourier_mapping_size)
        self.reaction = ReactionTerm(
            species=species, duplicates=duplicates, max_scale=self.max_scale,
            degree=degree, include_poly=include_poly,
            include_increasing_hill=include_increasing_hill,
            include_decreasing_hill=include_decreasing_hill, mcas=mcas)

    # ------------------------------------------------------------------
    # Data-derived buffers
    # ------------------------------------------------------------------
    def _register_domain(self, train_data):
        """lb, ub: (1, dimensions + 1) bounds of (x, t), used to normalize inputs to [-1, 1]."""
        x = train_data[:, :self.dimensions]
        t = train_data[:, self.dimensions]
        lb = [x.min().item()] * self.dimensions + [t.min().item()]
        ub = [x.max().item()] * self.dimensions + [t.max().item()]
        self.register_buffer('lb', torch.tensor(lb).view(1, -1))
        self.register_buffer('ub', torch.tensor(ub).view(1, -1))

    def _register_gls_weights(self, train_data, max_weight):
        """Per-frame GLS weights and the time edges between frames used to look them up."""
        weights, frame_t = frame_activity_weights(
            train_data, self.dimensions, self.species, max_weight=max_weight)
        frame_t = frame_t.cpu()
        self.register_buffer('gls_frame_weights', weights.cpu())
        self.register_buffer('gls_frame_edges', (frame_t[1:] + frame_t[:-1]) / 2)
        print(f"GLS activity weighting: peak {weights.max():.0f} at "
              f"t={frame_t[weights.argmax()].item():.3g}, mean {weights.mean():.1f}")

    def _resolve_pde_t_cutoff(self, train_data, pde_t_cutoff):
        if pde_t_cutoff is not None:
            if pde_t_cutoff > 0:
                print(f"PDE t-cutoff set explicitly to t={pde_t_cutoff:.4g}.")
            return float(pde_t_cutoff)

        cutoff = unresolved_t_cutoff(train_data, self.dimensions, self.species)
        if cutoff > 0:
            print(f"PDE t-cutoff auto-detected at t={cutoff:.4g}: the data's leading "
                  f"frame(s) change far faster than the sampling interval, so u_t there "
                  f"is not resolvable. Excluded from PDE collocation and the PDE scale.")
        else:
            print("PDE t-cutoff: 0 (data resolves its own initial transient).")
        return cutoff

    def _register_concentration_scales(self, train_data):
        """max_scale: 99th percentile of |u| per species; mean_scale: mean |u| per species."""
        conc = train_data[:, -self.species:]
        s_max = torch.stack([torch.quantile(conc[:, i].abs(), 0.99)
                             for i in range(self.species)]).cpu()
        self.register_buffer('max_scale', s_max.view(1, -1))
        self.register_buffer('mean_scale', conc.abs().mean(dim=0).cpu().view(1, -1))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def normalize(self, inputs):
        """Physical (x, t) in [lb, ub] -> [-1, 1]."""
        return 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, inputs):
        """Surface-fitter concentrations (batch, species) at physical (x, t)."""
        return self.surface_fitter(self.normalize(inputs))

    @property
    def learns_diffusion(self):
        return self.diffusion_fitter is not None

    def diffusion_coefficients(self):
        """D per species: the fixed values, or the current learned ones."""
        return self.diffusion_fitter() if self.learns_diffusion else self.diff_coeffs_tensor

    def collocation_t_min(self):
        """Earliest time at which the PDE residual is evaluated."""
        return max(self.lb[0, -1].item(), self.pde_t_cutoff)

    # ------------------------------------------------------------------
    # Equation read-out (implemented in equations/)
    # ------------------------------------------------------------------
    def generate_terms(self):
        return generate_terms(self)

    def extract_params(self, full=True):
        return extract_params(self, full=full)

    def generate_equation(self, eps=1e-12, species_names=None):
        return generate_equation(self, eps=eps, species_names=species_names)

    def equations_as_strings(self, eps=1e-12, species_names=None):
        return equations_as_strings(self, eps=eps, species_names=species_names)

    def fine_tune_eql(self, **kwargs):
        return fine_tune_eql(self, **kwargs)