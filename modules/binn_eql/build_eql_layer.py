import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.binn_eql.hard_concrete_gate import HardConcreteGate


class HillFunction(nn.Module):
    def __init__(self, param_bounds, increasing=True):
        super(HillFunction, self).__init__()
        self.param_bounds = param_bounds
        self.increasing = increasing

        # n is strictly bounded [1, 4] via Sigmoid
        self.raw_n = nn.Parameter(torch.empty(1).uniform_(-4, 4))

        # K is unbounded positive (Softplus); EQLLayer overwrites this at
        # init time via _smart_initialize_K to keep it in a sane range.
        self.raw_K = nn.Parameter(torch.empty(1).uniform_(-2, 2))

    def forward(self, x):
        n = torch.sigmoid(self.raw_n) * 3 + 1
        K = F.softplus(self.raw_K)
        x_n = x.pow(n)

        if self.increasing:
            return x_n / (1 + K * x_n)
        else:
            return 1.0 / (1 + K * x_n)


class PolynomialFeatures(nn.Module):
    def __init__(self, species, duplicates, degree):
        super(PolynomialFeatures, self).__init__()
        self.species = species
        self.duplicates = duplicates
        self.degree = degree
        self.powers = self._generate_powers()

    def _generate_powers(self):
        """Generates all combinations of powers where 1 <= sum <= degree.
        Already N-species general: iterates range(degree+1) with
        repeat=self.species, so no changes needed for stage 2."""
        powers = []
        for p in itertools.product(range(self.degree + 1), repeat=self.species):
            if 1 <= sum(p) <= self.degree:
                powers.append(p)

        powers.sort(key=lambda x: (sum(x), tuple(-power for power in x)))
        return powers

    def forward(self, x):
        features = []
        for p in self.powers:
            term = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
            for i, power in enumerate(p):
                if power > 0:
                    term = term * (x[:, i:i + 1] ** power)
            features.append(term)

        return torch.cat(features * self.duplicates, dim=1)


class HillFeatures(nn.Module):
    """
    Builds increasing and/or decreasing Hill features over `species`
    variables (raw + pairwise cross terms). Either family can be disabled
    entirely via include_increasing / include_decreasing so the EQL
    library only contains the term types you ask for.
    """
    def __init__(self, species, param_bounds, include_increasing=True, include_decreasing=True):
        super(HillFeatures, self).__init__()
        if not (include_increasing or include_decreasing):
            raise ValueError("HillFeatures needs include_increasing or include_decreasing (or both).")

        self.num_proteins = species
        self.include_increasing = include_increasing
        self.include_decreasing = include_decreasing

        if include_increasing:
            self.hill_inc_raw = nn.ModuleList(
                [HillFunction(param_bounds, increasing=True) for _ in range(species)])
            self.hill_inc_cross = nn.ModuleDict()
            for i in range(species):
                for j in range(species):
                    if i != j:
                        self.hill_inc_cross[f"{i}_{j}"] = HillFunction(param_bounds, increasing=True)

        if include_decreasing:
            self.hill_dec_raw = nn.ModuleList(
                [HillFunction(param_bounds, increasing=False) for _ in range(species)])
            self.hill_dec_cross = nn.ModuleDict()
            for i in range(species):
                for j in range(species):
                    if i != j:
                        self.hill_dec_cross[f"{i}_{j}"] = HillFunction(param_bounds, increasing=False)

    def forward(self, x):
        feats = []

        if self.include_increasing:
            for i in range(self.num_proteins):
                feats.append(self.hill_inc_raw[i](x[:, i:i + 1]))
            for i in range(self.num_proteins):
                for j in range(self.num_proteins):
                    if i != j:
                        xi, xj = x[:, i:i + 1], x[:, j:j + 1]
                        feats.append(self.hill_inc_cross[f"{i}_{j}"](xi) * xj)

        if self.include_decreasing:
            for i in range(self.num_proteins):
                feats.append(self.hill_dec_raw[i](x[:, i:i + 1]))
            for i in range(self.num_proteins):
                for j in range(self.num_proteins):
                    if i != j:
                        xi, xj = x[:, i:i + 1], x[:, j:j + 1]
                        feats.append(self.hill_dec_cross[f"{i}_{j}"](xi) * xj)

        return torch.cat(feats, dim=1)


class DuplicateHillFeatures(nn.Module):
    def __init__(self, species, param_bounds, duplicates, include_increasing=True, include_decreasing=True):
        super(DuplicateHillFeatures, self).__init__()
        self.hill_modules = nn.ModuleList([
            HillFeatures(species, param_bounds, include_increasing, include_decreasing)
            for _ in range(duplicates)
        ])

    def forward(self, x):
        features = [module(x) for module in self.hill_modules]
        return torch.cat(features, dim=1)


class EQLLayer(nn.Module):
    """
    Shared feature bank (polynomial + Hill terms, N species general) with
    ONE LINEAR HEAD PER FREE SPECIES and one L0 gate per free species.

    mcas=False (default): every species is "free" -- fully independent,
        separately-learned reaction equations. This is what non-conserving
        systems (Gray-Scott, Brusselator, FHN, ...) need.

    mcas=True: assumes the classic active/inactive two-state format --
        ONE shared reaction F(u, v), with species 0's equation being F and
        species 1's equation being architecturally forced to equal -F
        EXACTLY on every forward pass (same weight tensor, same gate
        sample -- not trained toward that, guaranteed by construction).
        Only valid for species == 2; raises otherwise, since +F/-F isn't a
        well-defined split for more than two species.

    include_poly / include_increasing_hill / include_decreasing_hill let
    you restrict which term families populate the library, e.g. set
    include_decreasing_hill=False if you know your system only has
    activating (not inhibiting) Hill kinetics.
    """
    def __init__(self, species, duplicates, param_bounds, max_scale, degree,
                 include_poly=True, include_increasing_hill=True, include_decreasing_hill=True,
                 mcas=False):
        super(EQLLayer, self).__init__()

        if not (include_poly or include_increasing_hill or include_decreasing_hill):
            raise ValueError("EQLLayer needs at least one of include_poly, "
                              "include_increasing_hill, include_decreasing_hill.")

        self.species = species
        self.duplicates = duplicates
        self.param_bounds = param_bounds
        self.degree = degree
        self.include_poly = include_poly
        self.include_increasing_hill = include_increasing_hill
        self.include_decreasing_hill = include_decreasing_hill
        self.mcas = mcas
        self.register_buffer('max_scale', max_scale)  # [1, species] tensor of max values

        # --- Feature library ---
        if include_poly:
            self.poly = PolynomialFeatures(species, duplicates, degree)
            self.num_poly_features = duplicates * len(self.poly.powers)
        else:
            self.poly = None
            self.num_poly_features = 0

        if include_increasing_hill or include_decreasing_hill:
            self.hill = DuplicateHillFeatures(
                species, param_bounds, duplicates,
                include_increasing=include_increasing_hill,
                include_decreasing=include_decreasing_hill)
            n_form = species + species * (species - 1)  # raw + cross, ONE form (inc or dec)
            n_forms_included = int(include_increasing_hill) + int(include_decreasing_hill)
            self.num_hill_features = duplicates * n_form * n_forms_included
        else:
            self.hill = None
            self.num_hill_features = 0

        self.total_features = self.num_poly_features + self.num_hill_features

        # --- Resolve which species are FREE (independently learned) vs
        # MIRROR (architecturally forced to equal -1 * the free species). ---
        self.species_role, self.free_species = self._resolve_species_roles(species, mcas)
        self.n_free = len(self.free_species)

        # --- Per-FREE-species linear head + gate ---
        # fc.weight has shape (n_free, total_features): row i is the
        # (pre-gate) reaction function for free_species[i]. Mirror species
        # have no row of their own -- see get_species_weight_and_gate().
        self.fc = nn.Linear(self.total_features, self.n_free, bias=False)
        self.l0_gates = nn.ModuleList([HardConcreteGate(self.total_features) for _ in range(self.n_free)])

        # --- Flattened Hill module list (only the forms that were built),
        # in the same order the features are concatenated, repeated once
        # per duplicate. Shared across ALL species heads (only the linear
        # coefficients differ per species; K, n live here once). ---
        self.all_hill_funcs = []
        if self.hill is not None:
            for hm in self.hill.hill_modules:
                if include_increasing_hill:
                    self.all_hill_funcs.extend(hm.hill_inc_raw)
                    for i in range(species):
                        for j in range(species):
                            if i != j:
                                self.all_hill_funcs.append(hm.hill_inc_cross[f"{i}_{j}"])
                if include_decreasing_hill:
                    self.all_hill_funcs.extend(hm.hill_dec_raw)
                    for i in range(species):
                        for j in range(species):
                            if i != j:
                                self.all_hill_funcs.append(hm.hill_dec_cross[f"{i}_{j}"])

            self._smart_initialize_K()

        nn.init.uniform_(self.fc.weight, a=-1, b=1)

    @staticmethod
    def _resolve_species_roles(species, mcas):
        """
        Returns (species_role, free_species):
            species_role[s] = ('free', free_idx) or ('mirror', primary_free_idx)
            free_species = original species indices that get their own
                learnable row, in fc.weight row order.
        """
        if mcas:
            if species != 2:
                raise ValueError(
                    f"mcas=True assumes the classic +F(u,v)/-F(u,v) two-state "
                    f"format (one shared reaction: species 0 gets +F, species "
                    f"1 gets -F) -- got species={species}. Set mcas=False for "
                    f"systems with more than 2 species, or that don't follow "
                    f"this exact conservation structure.")
            return [('free', 0), ('mirror', 0)], [0]

        return [('free', i) for i in range(species)], list(range(species))

    def get_species_weight_and_gate(self, s_idx):
        """(weight_row, gate_module, sign) for species s_idx -- for a
        mirror species, weight_row/gate_module are literally its primary's
        (same tensors, not copies), sign=-1.0. Used for reporting
        (extract_params) where reading -1*primary is exactly correct;
        NOT for mutation -- mirror species have no row of their own to
        mutate, see fine_tune_eql's use of free_species instead."""
        role, free_idx = self.species_role[s_idx]
        sign = 1.0 if role == 'free' else -1.0
        return self.fc.weight[free_idx], self.l0_gates[free_idx], sign

    def get_features(self, x):
        feats = []
        if self.include_poly:
            feats.append(self.poly(x))
        if self.hill is not None:
            feats.append(self.hill(x))
        return torch.cat(feats, dim=1)

    def forward(self, x):
        features = self.get_features(x)  # (batch, total_features)

        # One gate sample per FREE species, computed once and reused by
        # any mirrors -- this is what makes the negation exact rather
        # than "two independent samples from the same distribution".
        free_gate_vals = [self.l0_gates[i]() for i in range(self.n_free)]

        outputs = []
        for s_idx in range(self.species):
            role, free_idx = self.species_role[s_idx]
            sign = 1.0 if role == 'free' else -1.0
            w = self.fc.weight[free_idx]
            z = free_gate_vals[free_idx]
            outputs.append((features * (sign * w * z)).sum(dim=1, keepdim=True))

        return torch.cat(outputs, dim=1)  # (batch, species)

    def get_physical_parameters(self, epsilon=0.2):
        """
        w_phys: linear coefficients across every FREE species head only
            (fc.weight already excludes mirror rows -- they'd just be a
            sign-flipped duplicate of their primary's penalty, so this
            also avoids double-counting the soft-wall weight penalty).
        k_phys / k_ceilings: ONE value per Hill function in the shared
            basis (not per-species -- the Hill K/n live in the shared
            feature bank, only the weight that multiplies them differs).
        """
        w_phys = self.fc.weight.reshape(-1)

        if self.hill is None:
            empty = torch.zeros(0, device=w_phys.device)
            return w_phys, empty, empty

        k_phys_list, k_ceiling_list = [], []
        s = self.max_scale[0]
        N = self.species

        def get_k_and_ceiling(hf, u_max):
            n = torch.sigmoid(hf.raw_n) * 3 + 1
            k_phys = F.softplus(hf.raw_K)
            k_d_min = epsilon * (u_max + 1e-6)
            k_ceiling = 1.0 / (k_d_min ** n)
            return k_phys.view(1), k_ceiling.view(1)

        ptr = 0
        for _ in range(self.duplicates):
            if self.include_increasing_hill:
                for i in range(N):
                    k, ceil = get_k_and_ceiling(self.all_hill_funcs[ptr], s[i]); ptr += 1
                    k_phys_list.append(k); k_ceiling_list.append(ceil)
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            k, ceil = get_k_and_ceiling(self.all_hill_funcs[ptr], s[i]); ptr += 1
                            k_phys_list.append(k); k_ceiling_list.append(ceil)
            if self.include_decreasing_hill:
                for i in range(N):
                    k, ceil = get_k_and_ceiling(self.all_hill_funcs[ptr], s[i]); ptr += 1
                    k_phys_list.append(k); k_ceiling_list.append(ceil)
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            k, ceil = get_k_and_ceiling(self.all_hill_funcs[ptr], s[i]); ptr += 1
                            k_phys_list.append(k); k_ceiling_list.append(ceil)

        return w_phys, torch.cat(k_phys_list), torch.cat(k_ceiling_list)

    def _smart_initialize_K(self):
        """Initializes raw_K so physical K sits strictly below the dynamic ceiling."""
        s = self.max_scale[0]
        N = self.species

        def set_k(hf, scale):
            with torch.no_grad():
                n = torch.sigmoid(hf.raw_n) * 3 + 1
                # hf.raw_n is a freshly-created Parameter (CPU at this
                # point, before any .to(device) call), so alpha/kd_initial
                # must match n's device -- not scale's -- for this
                # multiplication chain to be internally consistent even if
                # scale (from self.max_scale, a buffer) ever ends up on a
                # different device than the not-yet-moved parameters here.
                alpha = torch.empty(1, device=n.device).uniform_(1, 3)
                kd_initial = alpha * (scale.to(n.device) + 1e-6)
                k_phys_target = 1.0 / (kd_initial ** n)

                if k_phys_target.item() > 20.0:
                    val = k_phys_target.item()
                else:
                    val = torch.log(torch.exp(k_phys_target) - 1 + 1e-9).item()

                hf.raw_K.data.fill_(val)

        ptr = 0
        for _ in range(self.duplicates):
            if self.include_increasing_hill:
                for i in range(N):
                    set_k(self.all_hill_funcs[ptr], s[i]); ptr += 1
                for i in range(N):
                    for j in range(N):
                        if i != j: set_k(self.all_hill_funcs[ptr], s[i]); ptr += 1
            if self.include_decreasing_hill:
                for i in range(N):
                    set_k(self.all_hill_funcs[ptr], s[i]); ptr += 1
                for i in range(N):
                    for j in range(N):
                        if i != j: set_k(self.all_hill_funcs[ptr], s[i]); ptr += 1