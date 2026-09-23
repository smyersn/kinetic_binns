"""
Equation-learner (EQL) layer: a shared library of candidate reaction terms
with one sparse linear head per independently learned species,

    F_s(u) = sum_k  w_sk * z_sk * phi_k(u),

where phi_k are polynomial and Hill features, w are linear coefficients and
z are hard-concrete L0 gates (Louizos et al., 2018).

Species roles
-------------
mcas=False  Every species is *free*: its own row of w and its own gate.
            Needed for non-conserving systems (Gray-Scott, Brusselator, ...).
mcas=True   Two-state active/inactive systems (species == 2 only). One
            reaction F is learned for species 0; species 1 is a *mirror*
            that receives exactly -F (same weights, same gate sample), so
            u + v is conserved by construction rather than by training.
"""
import torch
import torch.nn as nn

from modules.binn_eql.library.hard_concrete_gate import HardConcreteGate
from modules.binn_eql.library.hill import DuplicateHillFeatures
from modules.binn_eql.library.polynomial import PolynomialFeatures


def resolve_species_roles(species, mcas):
    """
    Returns (species_role, free_species):
        species_role[s]  ('free', row) or ('mirror', row of its primary)
        free_species     species indices that own a row of fc.weight, in row order
    """
    if not mcas:
        return [('free', i) for i in range(species)], list(range(species))
    if species != 2:
        raise ValueError(
            f"mcas=True assumes the two-state +F/-F format (species == 2), got "
            f"species={species}. Use mcas=False for other systems.")
    return [('free', 0), ('mirror', 0)], [0]


class EQLLayer(nn.Module):

    def __init__(self, species, duplicates, max_scale, degree,
                 include_poly=True, include_increasing_hill=True,
                 include_decreasing_hill=True, mcas=False):
        super().__init__()
        if not (include_poly or include_increasing_hill or include_decreasing_hill):
            raise ValueError("EQLLayer needs at least one of include_poly, "
                             "include_increasing_hill, include_decreasing_hill.")

        self.species = species
        self.duplicates = duplicates
        self.degree = degree
        self.include_poly = include_poly
        self.include_increasing_hill = include_increasing_hill
        self.include_decreasing_hill = include_decreasing_hill
        self.mcas = mcas
        self.register_buffer('max_scale', max_scale)  # (1, species): 99th pct of |u|

        # --- Candidate library ------------------------------------------
        self.poly = PolynomialFeatures(species, duplicates, degree) if include_poly else None
        self.num_poly_features = duplicates * len(self.poly.powers) if include_poly else 0

        has_hill = include_increasing_hill or include_decreasing_hill
        self.hill = (DuplicateHillFeatures(species, duplicates,
                                           include_increasing_hill, include_decreasing_hill)
                     if has_hill else None)

        # Flat, column-ordered view of the Hill features. Plain lists (not
        # ModuleLists) so the parameters are not registered a second time.
        slots = self.hill.slots() if self.hill is not None else []
        self.hill_slots = [(form, term) for form, term, _ in slots]
        self.all_hill_funcs = [fn for _, _, fn in slots]
        self.num_hill_features = len(slots)

        self.total_features = self.num_poly_features + self.num_hill_features

        # --- One linear head and one gate per free species ---------------
        self.species_role, self.free_species = resolve_species_roles(species, mcas)
        self.n_free = len(self.free_species)
        self.fc = nn.Linear(self.total_features, self.n_free, bias=False)
        self.l0_gates = nn.ModuleList(
            [HardConcreteGate(self.total_features) for _ in range(self.n_free)])

        if self.hill is not None:
            self._initialize_K()
        nn.init.uniform_(self.fc.weight, a=-1, b=1)

    @property
    def n_gates(self):
        return self.total_features * self.n_free

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def get_features(self, x):
        features = []
        if self.poly is not None:
            features.append(self.poly(x))
        if self.hill is not None:
            features.append(self.hill(x))
        return torch.cat(features, dim=1)

    def forward(self, x):
        """Reaction term for every species, (batch, species)."""
        features = self.get_features(x)

        # One gate sample per free row, reused by its mirror, so the mirror
        # is exactly -F rather than an independent sample.
        gates = [self.l0_gates[i]() for i in range(self.n_free)]

        outputs = []
        for role, row in self.species_role:
            sign = 1.0 if role == 'free' else -1.0
            outputs.append((features * (sign * self.fc.weight[row] * gates[row]))
                           .sum(dim=1, keepdim=True))
        return torch.cat(outputs, dim=1)

    def get_species_weight_and_gate(self, s_idx):
        """
        (weight row, gate module, sign) for species s_idx. For a mirror these
        are its primary's tensors with sign -1. For reading only: mirrors
        have no row of their own to modify.
        """
        role, row = self.species_role[s_idx]
        sign = 1.0 if role == 'free' else -1.0
        return self.fc.weight[row], self.l0_gates[row], sign

    # ------------------------------------------------------------------
    # Parameter bounds
    # ------------------------------------------------------------------
    def get_physical_parameters(self, epsilon=0.2):
        """
        Quantities constrained by the soft-wall loss:
            w_phys     all head coefficients (free rows only)
            k_phys     K of every Hill feature
            k_ceiling  1 / (epsilon * u_max)^n, i.e. the half-saturation
                       concentration K^(-1/n) may not drop below epsilon * u_max
        """
        w_phys = self.fc.weight.reshape(-1)
        if self.hill is None:
            empty = torch.zeros(0, device=w_phys.device)
            return w_phys, empty, empty

        scale = self.max_scale[0]
        k_phys, k_ceiling = [], []
        for fn, (_, term) in zip(self.all_hill_funcs, self.hill_slots):
            kd_min = epsilon * (scale[term[0]] + 1e-6)
            k_phys.append(fn.K.view(1))
            k_ceiling.append((1.0 / (kd_min ** fn.n)).view(1))
        return w_phys, torch.cat(k_phys), torch.cat(k_ceiling)

    @torch.no_grad()
    def _initialize_K(self):
        """
        Place each feature's half-saturation concentration K^(-1/n) at a
        random 1-3x the 99th-percentile concentration of its input species,
        safely inside the soft-wall ceiling.
        """
        scale = self.max_scale[0]
        for fn, (_, term) in zip(self.all_hill_funcs, self.hill_slots):
            n = fn.n
            alpha = torch.empty(1, device=n.device).uniform_(1, 3)
            kd = alpha * (scale[term[0]].to(n.device) + 1e-6)
            k_target = 1.0 / (kd ** n)

            # Inverse softplus; for large targets softplus(x) ~ x.
            if k_target.item() > 20.0:
                raw = k_target.item()
            else:
                raw = torch.log(torch.exp(k_target) - 1 + 1e-9).item()
            fn.raw_K.data.fill_(raw)