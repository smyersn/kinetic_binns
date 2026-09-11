import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

from modules.binn_eql.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.binn_eql.build_eql_layer import EQLLayer


def default_species_names(species):
    """
    Generic concentration names u, v, w, ... matching the original
    2-species (u, v) convention. Skips t, x, y, z since those are reserved
    for time/spatial coordinates elsewhere in this codebase. Wraps with a
    numeric suffix if species > 22 (more species than available letters).
    """
    reserved = {'t', 'x', 'y', 'z'}
    letters = [c for c in string.ascii_lowercase if c not in reserved]
    start = letters.index('u')
    ordered = letters[start:] + letters[:start]

    names = []
    for i in range(species):
        if i < len(ordered):
            names.append(ordered[i])
        else:
            names.append(f"{ordered[i % len(ordered)]}{i // len(ordered)}")
    return names


# ---------------------------------------------------------
# 1. SUB-NETWORKS
# ---------------------------------------------------------
class D_PARAMS(nn.Module):
    def __init__(self, input_features=2, base_val=0.1, noise_std=0.5):
        super().__init__()
        base_log = torch.log(torch.tensor(base_val))
        noise = torch.randn(input_features) * noise_std
        self.raw_D = nn.Parameter(base_log + noise)

    def forward(self):
        return torch.exp(self.raw_D)


class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_features, mapping_size, scale=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=True)

    def forward(self, x):
        x_proj = (2.0 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class uv_MLP(nn.Module):
    """Surface fitter. `layers` must end with an output width equal to
    `species` -- defaults to [256, 256, 256, species] if not given."""
    def __init__(self, input_features, species, mapping_size=256, scale=1.0, layers=None):
        super().__init__()
        if layers is None:
            layers = [256, 256, 256, species]
        assert layers[-1] == species, "uv_MLP's final layer width must equal `species`."

        self.encoder = FourierFeatureEncoding(in_features=input_features, mapping_size=mapping_size, scale=scale)
        encoded_features = mapping_size * 2

        self.mlp = build_mlp(
            input_features=encoded_features,
            layers=layers,
            activation=nn.GELU(),
            linear_output=False,
            output_activation=nn.Softplus())

        for module in self.mlp.MLP:
            if isinstance(module, nn.Linear):
                utils.parametrizations.weight_norm(module)

    def forward(self, inputs):
        encoded_x = self.encoder(inputs)
        return self.mlp(encoded_x)


class F_EQL(nn.Module):
    def __init__(self, species, duplicates, param_bounds, max_scale, degree,
                 include_poly=True, include_increasing_hill=True, include_decreasing_hill=True,
                 mcas=False):
        super(F_EQL, self).__init__()
        self.eql_layer = EQLLayer(
            species, duplicates, param_bounds, max_scale, degree,
            include_poly=include_poly,
            include_increasing_hill=include_increasing_hill,
            include_decreasing_hill=include_decreasing_hill,
            mcas=mcas)

    def forward(self, x):
        return self.eql_layer(x)


# ---------------------------------------------------------
# 2. BINN
# ---------------------------------------------------------
class BINN(nn.Module):
    def __init__(self, dimensions, species, train_data, duplicates=1,
                 diff_coeffs=None, uv_layers=None, degree=2, param_bounds=10,
                 fourier_scale=1.0, fourier_mapping_size=64,
                 mcas=False, species_names=None,
                 include_poly=True, include_increasing_hill=True, include_decreasing_hill=True):
        """
        mcas: single flag replacing the old conservation_groups/
            conservation_mode pair. False (default): every species gets
            its own fully independent reaction equation -- what
            non-conserving systems (Gray-Scott, Brusselator, FHN, ...)
            need. True: assumes the classic active/inactive two-state
            format (only valid for species==2) -- ONE shared reaction
            F(u, v), with species 1's equation architecturally forced to
            equal exactly -F on every forward pass (guaranteed by
            construction, not trained toward it). Downstream reporting
            changes to match: extract_params()/generate_equation() return
            ONE equation instead of `species`, and surface-comparison
            plots get one row instead of `species` rows. The soft
            mass-conservation loss (BINN.mass_loss, a diagnostic on the
            SURFACE fitter's own derivatives, separate from the reaction
            architecture) is automatically enabled for the (0, 1) pair
            when mcas=True and disabled when False -- no separate
            conservation_groups list to configure anymore.
        include_poly / include_increasing_hill / include_decreasing_hill:
            toggle which term families populate the shared EQL library.
        """
        super().__init__()
        self.dimensions = dimensions
        self.species = species
        self.train_data = train_data
        self.duplicates = duplicates
        self.param_bounds = param_bounds
        self.degree = degree
        self.mcas = mcas
        # n_equations: how many INDEPENDENT reaction equations exist, for
        # every reporting/plotting purpose (extract_params, generate_equation,
        # surface comparison). Distinct from `species` (concentration field
        # count), which the physics (PDE residuals, diffusion) always needs
        # in full regardless of mcas -- mcas collapses REPORTED equations,
        # not the number of species/PDEs being solved.
        self.n_equations = 1 if mcas else species
        self.conservation_groups = [[0, 1]] if mcas else []
        self.species_names = species_names if species_names is not None else default_species_names(species)

        # ---------------------------------------------------------
        # A. REGISTER BOUNDS & SCALES (Buffers)
        # ---------------------------------------------------------
        # NOTE: .item()/.cpu() everywhere below is deliberate, not
        # incidental. train_data may already be on GPU here (e.g. if the
        # training script moves it to `device` inside training_test_split
        # before constructing BINN) -- but every nn.Parameter created
        # during __init__ (HillFunction's raw_n/raw_K, fc.weight, etc.) is
        # CPU by default until the whole model is later moved via
        # binn.to(device). If these buffers inherited train_data's device
        # instead, EQLLayer._smart_initialize_K (which runs during this
        # same __init__, before any .to(device) call) would mix a CUDA
        # buffer with CPU parameters and crash with a device mismatch.
        # Forcing everything here to build as plain CPU tensors keeps
        # __init__ internally consistent regardless of train_data's
        # device; the training script's later binn.to(device) call moves
        # parameters and buffers together, correctly, after this point.
        x_min = torch.min(train_data[:, :dimensions]).item()
        x_max = torch.max(train_data[:, :dimensions]).item()
        t_min = torch.min(train_data[:, dimensions]).item()
        t_max = torch.max(train_data[:, dimensions]).item()

        lb_tensor = torch.cat([torch.full((dimensions,), x_min), torch.tensor([t_min])])
        ub_tensor = torch.cat([torch.full((dimensions,), x_max), torch.tensor([t_max])])
        self.register_buffer('lb', lb_tensor.view(1, -1))
        self.register_buffer('ub', ub_tensor.view(1, -1))

        # Per-species concentration scales (Physical -> Dimensionless).
        # train_data columns are [x*dimensions, t, species concentrations],
        # so the last `species` columns are the concentrations.
        conc = train_data[:, -species:]
        s_max = torch.stack([torch.quantile(conc[:, i].abs(), 0.99) for i in range(species)]).cpu()
        self.register_buffer('max_scale', s_max.view(1, -1))

        s_mean = conc.abs().mean(dim=0).cpu()
        self.register_buffer('mean_scale', s_mean.view(1, -1))

        # Diffusion coefficients: fixed physical values (one per species)
        # or None to learn them via self.diffusion_fitter.
        self.diff_coeffs = diff_coeffs
        if diff_coeffs:
            self.register_buffer('diff_coeffs_tensor', torch.tensor(diff_coeffs, dtype=torch.float32))
        else:
            self.diff_coeffs_tensor = None

        # ---------------------------------------------------------
        # B. INITIALIZE SUB-NETWORKS
        # ---------------------------------------------------------
        self.diffusion_fitter = D_PARAMS(self.species) if not self.diff_coeffs else None

        self.surface_fitter = uv_MLP(
            input_features=dimensions + 1, species=species, layers=uv_layers,
            scale=fourier_scale, mapping_size=fourier_mapping_size)

        # Reaction: shared feature bank, one linear+gate head per FREE
        # species (mirror species, when mcas=True, share the primary's
        # head exactly -- see EQLLayer).
        self.reaction = F_EQL(
            species, duplicates, self.param_bounds, self.max_scale, self.degree,
            include_poly=include_poly,
            include_increasing_hill=include_increasing_hill,
            include_decreasing_hill=include_decreasing_hill,
            mcas=mcas)

        self.num_samples = 10000
        self.name = 'Dumlp_Fmlp_Nspecies'

    def normalize(self, inputs):
        """ Maps Physical [lb, ub] -> Dimensionless [-1, 1] """
        return 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, inputs):
        """ Returns PREDICTED concentrations (batch, species) from Physical Inputs """
        self.inputs = inputs
        inputs_hat = self.normalize(inputs)
        return self.surface_fitter(inputs_hat)

    # -----------------------
    # Loss Functions
    # -----------------------
    def register_pde_scale(self, train_data, quantile_percent=0.01, chunk_size=50_000):
        """
        Locks robust per-species PDE normalization scales. Computes the
        90th-percentile of u_t^2 for EACH species from the current
        surface_fitter. Must be called once the surface has been fit
        (start of Phase 2), not at __init__ time when it's still random.
        """
        if getattr(self, 'pde_scales_locked', False):
            return

        print("\n--- Calculating Robust PDE Scales (90th percentile) ---")
        was_training = self.surface_fitter.training
        self.surface_fitter.eval()

        all_sq = [[] for _ in range(self.species)]
        total_points = len(train_data)

        for chunk_start in range(0, total_points, chunk_size):
            chunk = train_data[chunk_start:chunk_start + chunk_size].clone().requires_grad_(True)
            outputs = self.surface_fitter(self.normalize(chunk[:, :self.dimensions + 1]))

            for s_idx in range(self.species):
                u_t = gradient(outputs[:, s_idx], chunk, order=1)[:, self.dimensions]
                all_sq[s_idx].append((u_t ** 2).detach().cpu())

            del chunk, outputs

        if was_training:
            self.surface_fitter.train()

        scales = []
        for s_idx in range(self.species):
            global_sq = torch.cat(all_sq[s_idx])
            scales.append(torch.quantile(global_sq, 0.90).item() + 1e-6)

        # This runs mid-training (Phase 2), AFTER binn.to(device) already
        # moved the model to GPU -- register_buffer does NOT retroactively
        # move a newly-registered buffer to match the module's existing
        # device, so without an explicit device here this buffer would
        # silently stay on CPU while everything else (outputs, ut_array in
        # pde_loss_from_derivatives) is on GPU, crashing on the same class
        # of device-mismatch as the __init__-time bug above. self.lb is a
        # buffer that WAS moved by .to(device), so its device is the
        # correct reference.
        self.register_buffer('pde_scale', torch.tensor(scales, device=self.lb.device))
        self.pde_scales_locked = True

        print(f"Locked 90th Percentile Scales -> {[f'{s:.4e}' for s in scales]}\n")

    def register_mass_scale(self, train_data, t_cutoff=None, chunk_size=50_000):
        """
        Locks a robust normalization scale for the mass_loss term
        (analogous to register_pde_scale). Only active when self.mcas is
        True (self.conservation_groups == [[0, 1]] in that case, [] and
        this is skipped entirely otherwise). For mcas=True, this measures
        a residual that should already be ~0 from the reaction side (F1
        = -F2 architecturally), but the surface fitter's own derivatives
        don't know that, so this remains a useful, if largely redundant,
        training signal.
        """
        if getattr(self, 'mass_scale_locked', False):
            return

        if not self.conservation_groups:
            self.mass_scales = []
            self.mass_scale_locked = True
            return

        t_cutoff = t_cutoff if t_cutoff is not None else getattr(self, 'mass_t_cutoff', 2.0)
        print(f"\n--- Calculating Robust Mass Scale(s) (90th percentile, t>={t_cutoff}) ---")

        was_training = self.surface_fitter.training
        self.surface_fitter.eval()

        mask = train_data[:, self.dimensions] >= t_cutoff
        subset = train_data[mask]
        group_sq = [[] for _ in self.conservation_groups]

        for chunk_start in range(0, len(subset), chunk_size):
            chunk = subset[chunk_start:chunk_start + chunk_size].clone().requires_grad_(True)
            outputs = self.surface_fitter(self.normalize(chunk[:, :self.dimensions + 1]))

            for g_idx, group in enumerate(self.conservation_groups):
                w_t = sum(gradient(outputs[:, s_idx], chunk, order=1)[:, self.dimensions] for s_idx in group)
                group_sq[g_idx].append((w_t ** 2).detach().cpu())

            del chunk, outputs

        if was_training:
            self.surface_fitter.train()

        self.mass_scales = []
        for sqs in group_sq:
            global_wt_sq = torch.cat(sqs)
            self.mass_scales.append(torch.quantile(global_wt_sq, 0.90).item() + 1e-8)

        self.mass_scale_locked = True
        print(f"Locked Mass Scale(s) -> {[f'{s:.4e}' for s in self.mass_scales]}\n")

    @torch.no_grad()
    def _pde_loss_null(self):
        """PDE loss with every reaction term zeroed -- the residual no
        reaction can fix. Uses the full collocation cache so it's directly
        comparable to _pde_loss_current()."""
        cache = getattr(self, '_collocation_cache', None)
        if cache is None:
            return None
        eql = self.reaction.eql_layer
        saved = eql.fc.weight.data.clone()
        eql.fc.weight.data.zero_()
        loss = self.pde_loss_from_derivatives(
            cache['outputs'], cache['ut_array'], cache['uxx_array'], epoch=0).item()
        eql.fc.weight.data.copy_(saved)
        return loss

    @torch.no_grad()
    def _pde_loss_current(self):
        """PDE loss with current weights, on the full cache -- the exact
        counterpart to _pde_loss_null()."""
        cache = getattr(self, '_collocation_cache', None)
        if cache is None:
            return None
        return self.pde_loss_from_derivatives(
            cache['outputs'], cache['ut_array'], cache['uxx_array'], epoch=0).item()

    @torch.no_grad()
    def track_best_pde(self, save_weights=True):
        """
        Sample the current full-cache PDE loss and keep the best (lowest)
        seen so far. Called periodically through Phase 2 by model_wrapper.

        Why not just read val_loss_dict['pde']: that's a mean over random
        10k-point subsamples of the cache, so it carries sampling noise on
        top of the optimization oscillation -- and it isn't measured the
        same way as _pde_loss_null(), which uses the full cache. Comparing
        a noisy median against a clean instantaneous value is what made
        the floor/null verdict a coin flip. Both numbers now come from the
        same points via the same code path.

        save_weights: also snapshot the reaction weights at the best point,
            so Phase 3 can start from the best reaction Phase 2 actually
            found instead of wherever the oscillation happened to end.
        """
        current = self._pde_loss_current()
        if current is None:
            return None

        best = getattr(self, 'best_pde_floor', None)
        if best is None or current < best:
            self.best_pde_floor = current
            self.best_pde_floor_epoch = getattr(self, '_current_epoch', -1)
            if save_weights:
                self._best_reaction_state = {
                    k: v.detach().clone()
                    for k, v in self.reaction.state_dict().items()
                }
        return current

    @torch.no_grad()
    def restore_best_reaction(self):
        """Load the reaction weights that achieved best_pde_floor."""
        state = getattr(self, '_best_reaction_state', None)
        if state is None:
            return False
        self.reaction.load_state_dict(state)
        print(f"Restored best Phase-2 reaction from epoch "
              f"{getattr(self, 'best_pde_floor_epoch', '?')} "
              f"(PDE = {self.best_pde_floor:.4e})", flush=True)
        return True

    def register_l0_scale(self):
        """
        Normalize by EXPLAINABLE RANGE (how much PDE loss the reaction
        terms can actually remove), not by the PDE floor.

        The floor is the residual with all terms ALREADY active -- mostly
        irreducible surface-derivative error, which grows as the grid
        coarsens. Pricing gates against it means the same l0_weight gets
        harsher at lower resolution, which is exactly the 100x100-works /
        25x25-over-prunes behavior. Explainable range is the budget the
        terms are actually competing for, so l0_weight ~ 1 means "a term
        must claim its equal share of the achievable improvement" -- a
        resolution-independent statement.

        Uses the BEST floor seen across Phase 2 rather than the value at
        Phase 3 entry: the reaction weights oscillate by more than the
        explainable gap itself, so an instantaneous reading is decided by
        whether the measurement lands in a trough or a peak.

        pde_history / window_frac are accepted but unused -- kept so
        existing call sites don't break.
        """
        if getattr(self, 'l0_scale_locked', False):
            return

        current = self._pde_loss_current()
        best = getattr(self, 'best_pde_floor', None)
        floor = current if best is None else (min(best, current) if current is not None else best)
        null = self._pde_loss_null()

        n_features = self.reaction.eql_layer.total_features

        if null is None or floor is None:
            self.l0_scale = 1.0
            print(f"\n--- L0 scale: no collocation cache, defaulting to 1.0 ---\n")
            self.l0_scale_locked = True
            return

        explainable = null - floor
        print(f"\n--- L0 scale diagnostics ---")
        print(f"  PDE floor (best in Phase 2)  : {floor:.4e}"
              f"  (at epoch {getattr(self, 'best_pde_floor_epoch', '?')};"
              f" current = {current:.4e})")
        print(f"  PDE null  (all terms zeroed) : {null:.4e}")

        if explainable <= 0:
            # Distinct from "small but positive": the terms are making the
            # residual WORSE than F=0, which means Phase 2 never converged.
            # Silently clamping this to 1e-12 (as an earlier version did)
            # produced an l0_scale that disabled pruning entirely and
            # looked like a resolution verdict rather than a convergence
            # failure.
            self.l0_scale = 1.0
            print(f"  Explainable range:            NEGATIVE ({explainable:.4e})")
            print(f"  WARNING: the reaction terms fit WORSE than F=0. Phase 2 did not "
                  f"converge -- l0_scale is not meaningful here. Falling back to 1.0. "
                  f"Extend Phase 2 or check the reaction LR before trusting any "
                  f"pruning result from this run.")
            print()
            self.l0_scale_locked = True
            return

        self.l0_scale = explainable / n_features
        frac = explainable / max(null, 1e-12)
        print(f"  Explainable range:            {explainable:.4e}  ({100*frac:.1f}% of null)")
        print(f"  l0_scale = explainable / {n_features} features = {self.l0_scale:.4e}")
        if frac < 0.1:
            print(f"  WARNING: terms reduce PDE loss by <10%. The residual is "
                  f"dominated by surface-derivative error the reaction cannot fix "
                  f"-- likely a resolution limit, not an L0 tuning problem.")
        print()

        self.l0_scale_locked = True

    def refresh_collocation_cache(self, cache_size=200_000, mass_t_cutoff=None, chunk_size=20_000):
        """Precomputes ut_array/uxx_array for a fixed pool of collocation points."""
        t_cutoff = mass_t_cutoff if mass_t_cutoff is not None else getattr(self, 'mass_t_cutoff', 2.0)

        was_training = self.surface_fitter.training
        self.surface_fitter.eval()

        outputs_list, ut_list, uxx_list, mask_list = [], [], [], []

        for chunk_start in range(0, cache_size, chunk_size):
            n = min(chunk_size, cache_size - chunk_start)
            x = torch.empty(n, self.dimensions, device=self.lb.device).uniform_(
                self.lb[0, 0].item(), self.ub[0, 0].item())
            t = torch.empty(n, 1, device=self.lb.device).uniform_(
                self.lb[0, -1].item(), self.ub[0, -1].item())
            inputs = torch.cat([x, t], dim=1).requires_grad_(True)
            outputs = self.surface_fitter(self.normalize(inputs))
            ut_array, uxx_array = self.compute_field_derivatives(inputs, outputs)

            outputs_list.append(outputs.detach())
            ut_list.append(ut_array.detach())
            uxx_list.append(uxx_array.detach())
            mask_list.append((inputs[:, -1] >= t_cutoff).detach())

            del x, t, inputs, outputs, ut_array, uxx_array
            torch.cuda.empty_cache()

        if was_training:
            self.surface_fitter.train()

        self._collocation_cache = {
            'outputs': torch.cat(outputs_list, dim=0),
            'ut_array': torch.cat(ut_list, dim=0),
            'uxx_array': torch.cat(uxx_list, dim=1),
            'mass_mask': torch.cat(mask_list, dim=0),
        }
        print(f"Refreshed collocation cache: {cache_size} points ({chunk_size}/chunk), "
              f"{self._collocation_cache['mass_mask'].sum().item()} pass mass cutoff")

    def compute_field_derivatives(self, inputs, outputs):
        """Already N-species general -- loops over self.species."""
        points = len(inputs)
        uxx_array = torch.zeros((self.species, points, self.dimensions), device=inputs.device)
        ut_array = torch.zeros((points, self.species), device=inputs.device)
        for i in range(self.species):
            d1 = gradient(outputs[:, i], inputs, order=1)
            ut_array[:, i] = d1[:, -1]
            for j in range(self.dimensions):
                uxx_array[i, :, j] = gradient(d1[:, j], inputs, order=1)[:, j]
        return ut_array, uxx_array

    def gls_loss(self, pred, true):
        residual = ((pred - true) / self.mean_scale) ** 2
        ic_mask = self.inputs[:, -1:] == 0
        weights = torch.where(ic_mask, 10.0, 1.0)
        return torch.mean(residual * weights)

    def gls_loss_time_weighted(self, pred, true, time_scale=5.0, max_weight=50.0):
        """Smooth decay-over-time generalization of the t=0 IC weighting."""
        residual = ((pred - true) / self.mean_scale) ** 2
        t = self.inputs[:, -1:]
        t_min = self.lb[0, -1]
        weights = 1.0 + (max_weight - 1.0) * torch.exp(-(t - t_min) / time_scale)
        return torch.mean(residual * weights)

    def pde_loss_from_derivatives(self, outputs, ut_array, uxx_array, epoch):
        """
        Each species' reaction term comes from its own column of
        self.reaction(outputs) -- for hard-coupled mirror species, that
        column is already exactly -1 * its primary's (guaranteed inside
        EQLLayer.forward), so no special-casing is needed here.
        """
        F_reaction = self.reaction(outputs)

        if self.diff_coeffs:
            D = self.diff_coeffs_tensor.to(outputs.device)
        else:
            D = self.diffusion_fitter()

        total_pde_loss = torch.tensor(0.0, device=outputs.device)
        for s_idx in range(self.species):
            lap_s = D[s_idx] * torch.sum(uxx_array[s_idx, :, :], dim=1, keepdim=True)
            LHS = ut_array[:, s_idx][:, None]
            RHS = lap_s + F_reaction[:, s_idx][:, None]

            if hasattr(self, 'pde_scale'):
                scale = torch.sqrt(self.pde_scale[s_idx])
            else:
                scale = torch.tensor(1.0, device=outputs.device)

            res = (LHS - RHS) / scale
            total_pde_loss = total_pde_loss + F.smooth_l1_loss(res, torch.zeros_like(res), beta=1.0)

        return total_pde_loss

    def mass_loss_from_derivatives(self, ut_array, uxx_array, mask):
        """
        Soft penalty on the SURFACE fitter's derivatives matching
        conservation (only active when self.mcas is True).

        Returns 0 when mcas=False (no conservation assumed), AND when
        diff_coeffs are FIXED: in that case every input here -- the
        cached derivatives, the buffered D, the locked scale -- is a
        constant with respect to the reaction weights and gates, which
        are the only things trained in Phase 2+. The term contributes
        exactly zero gradient, so computing it only burns time, inflates
        the total-loss curve, and adds sampling noise to the Phase 4
        best-val comparison. It IS meaningful when diffusion is learned,
        where D is a live parameter and this is what trains it.
        """
        if not self.conservation_groups:
            return torch.tensor(0.0, device=ut_array.device)

        if self.diff_coeffs:
            return torch.tensor(0.0, device=ut_array.device)

        D = self.diffusion_fitter()

        total_mass_loss = torch.tensor(0.0, device=ut_array.device)
        for g_idx, group in enumerate(self.conservation_groups):
            w_t = sum(ut_array[mask, s_idx] for s_idx in group)
            lap_sum = sum(D[s_idx] * torch.sum(uxx_array[s_idx, mask, :], dim=1) for s_idx in group)

            scale = torch.sqrt(torch.tensor(self.mass_scales[g_idx], device=ut_array.device)) \
                if getattr(self, 'mass_scales', None) else torch.tensor(1.0, device=ut_array.device)
            residual = (w_t - lap_sum) / scale

            weights = torch.abs(w_t).detach()
            weights = weights / (weights.mean() + 1e-8)
            total_mass_loss = total_mass_loss + torch.mean(weights * residual ** 2)

        return total_mass_loss / len(self.conservation_groups)

    def reg_loss(self, epoch):
        """L0 sparsity, summed once per FREE species' gate (mirror species
        share their primary's gate exactly -- summing them again would
        double-count the same penalty, not add new information)."""
        eql = self.reaction.eql_layer
        total = torch.tensor(0.0, device=eql.fc.weight.device)
        for free_idx in range(eql.n_free):
            total = total + eql.l0_gates[free_idx].expected_l0().sum()
        return total

    def soft_wall_loss(self):
        w_phys, k_phys, k_ceilings = self.reaction.eql_layer.get_physical_parameters(epsilon=0.15)

        w_violation = torch.relu(torch.abs(w_phys) - self.param_bounds)
        w_loss = torch.sum(w_violation) * 100

        if k_phys.numel() > 0:
            k_violation = torch.relu(k_phys - k_ceilings)
            k_loss = torch.sum(k_violation) * 100
        else:
            k_loss = torch.tensor(0.0, device=w_phys.device)

        return w_loss + k_loss

    def loss(self, pred, true, epoch, phase=None):
        raw_gls = self.gls_loss_time_weighted(pred, true)
        raw_softwall = self.soft_wall_loss()
        raw_l0 = self.reg_loss(epoch)

        if phase == 1:
            zero = torch.tensor(0.0, device=pred.device)
            return raw_gls, zero, raw_l0, raw_softwall, zero

        cache = getattr(self, '_collocation_cache', None)
        if cache is not None:
            idx = torch.randint(0, cache['outputs'].shape[0], (self.num_samples,), device=pred.device)
            outputs_b = cache['outputs'][idx]
            ut_b = cache['ut_array'][idx]
            uxx_b = cache['uxx_array'][:, idx, :]
            mask_b = cache['mass_mask'][idx]
        else:
            x = torch.empty(self.num_samples, self.dimensions, device=pred.device).uniform_(
                self.lb[0, 0], self.ub[0, 0])
            t = torch.empty(self.num_samples, 1, device=pred.device).uniform_(
                self.lb[0, -1], self.ub[0, -1])
            inputs_rand = torch.cat([x, t], dim=1).requires_grad_()
            outputs_b = self.surface_fitter(self.normalize(inputs_rand))
            ut_b, uxx_b = self.compute_field_derivatives(inputs_rand, outputs_b)
            t_cutoff = getattr(self, 'mass_t_cutoff', 2.0)
            mask_b = inputs_rand[:, -1] >= t_cutoff

        raw_pde = self.pde_loss_from_derivatives(outputs_b, ut_b, uxx_b, epoch)
        raw_mass = (self.mass_loss_from_derivatives(ut_b, uxx_b, mask_b)
                    if mask_b.any() else torch.tensor(0.0, device=pred.device))

        return raw_gls, raw_pde, raw_l0, raw_softwall, raw_mass

    # -----------------------
    # Parameter Extraction
    # -----------------------
    def generate_terms(self):
        """Structural term list -- shared across all species equations, so no per-species logic needed."""
        poly_terms = self.reaction.eql_layer.poly.powers if self.reaction.eql_layer.poly is not None else []

        hill_terms = []
        for i in range(self.species):
            hill_terms.append((i,))
        for i in range(self.species):
            for j in range(self.species):
                if i != j:
                    hill_terms.append((i, j))

        return poly_terms, hill_terms

    def _extract_hill_shape_params(self):
        """n, K for each Hill function in the shared basis (independent of species head)."""
        eql = self.reaction.eql_layer
        ns_inc, Ks_inc, ns_dec, Ks_dec = [], [], [], []

        def get_vals(module):
            n = torch.sigmoid(module.raw_n) * 3 + 1
            k_phys = F.softplus(module.raw_K)
            return n.item(), k_phys.item()

        for hill_module in eql.hill.hill_modules:
            if eql.include_increasing_hill:
                for i in range(self.species):
                    n, k = get_vals(hill_module.hill_inc_raw[i]); ns_inc.append(n); Ks_inc.append(k)
                for i in range(self.species):
                    for j in range(self.species):
                        if i != j:
                            n, k = get_vals(hill_module.hill_inc_cross[f"{i}_{j}"])
                            ns_inc.append(n); Ks_inc.append(k)
            if eql.include_decreasing_hill:
                for i in range(self.species):
                    n, k = get_vals(hill_module.hill_dec_raw[i]); ns_dec.append(n); Ks_dec.append(k)
                for i in range(self.species):
                    for j in range(self.species):
                        if i != j:
                            n, k = get_vals(hill_module.hill_dec_cross[f"{i}_{j}"])
                            ns_dec.append(n); Ks_dec.append(k)

        return np.array(ns_inc), np.array(Ks_inc), np.array(ns_dec), np.array(Ks_dec)

    def extract_params(self, full=True):
        """
        Returns a LIST of dicts, length self.n_equations (1 if mcas, else
        species). When mcas=True this deliberately does NOT report
        species 1's mirrored equation separately -- there's only one
        equation to show, matching the classic +F/-F printout instead of
        printing the same information twice with a sign flip.
        """
        eql = self.reaction.eql_layer
        num_poly = eql.num_poly_features
        num_hill = eql.num_hill_features
        poly_terms, hill_terms = self.generate_terms()

        results = []
        for s_idx in range(self.n_equations):
            w_row, gate_module, sign = eql.get_species_weight_and_gate(s_idx)
            raw_w_t = (sign * w_row).detach()

            try:
                gates_t = gate_module.get_gates().detach()
            except AttributeError:
                log_alpha = gate_module.log_alpha.detach()
                gates_t = torch.sigmoid(log_alpha).clamp(0.0, 1.0)

            effective_t = raw_w_t * gates_t

            raw_w = raw_w_t.cpu().numpy().reshape(-1)
            gates = gates_t.cpu().numpy().reshape(-1)
            effective = effective_t.cpu().numpy().reshape(-1)

            eq_result = {'raw_w_unscaled': raw_w, 'effective_unscaled': effective}

            if full:
                eq_result.update({
                    'raw_w': raw_w,
                    'gates': gates,
                    'effective': effective,
                    'num_poly': num_poly,
                    'num_hill': num_hill,
                    'poly_terms': poly_terms,
                    'hill_terms': hill_terms,
                    'poly_coeffs_unscaled': effective[:num_poly] if num_poly else np.array([]),
                })

                if num_hill > 0:
                    hill_block = effective[num_poly:num_poly + num_hill]
                    n_form = self.species + self.species * (self.species - 1)
                    forms = []
                    if eql.include_increasing_hill: forms.append('inc')
                    if eql.include_decreasing_hill: forms.append('dec')

                    inc_vals, dec_vals = [], []
                    ptr = 0
                    for d in range(self.duplicates):
                        for form in forms:
                            block = hill_block[ptr: ptr + n_form]
                            ptr += n_form
                            if form == 'inc':
                                inc_vals.extend(block)
                            else:
                                dec_vals.extend(block)

                    eq_result['hill_inc_unscaled'] = np.array(inc_vals)
                    eq_result['hill_dec_unscaled'] = np.array(dec_vals)
                else:
                    eq_result['hill_inc_unscaled'] = np.array([])
                    eq_result['hill_dec_unscaled'] = np.array([])

            results.append(eq_result)

        if full and num_hill > 0:
            ns_inc, Ks_inc, ns_dec, Ks_dec = self._extract_hill_shape_params()
            for eq_result in results:
                eq_result['ns_inc'], eq_result['Ks_inc'] = ns_inc, Ks_inc
                eq_result['ns_dec'], eq_result['Ks_dec'] = ns_dec, Ks_dec
        elif full:
            for eq_result in results:
                eq_result['ns_inc'] = eq_result['Ks_inc'] = np.array([])
                eq_result['ns_dec'] = eq_result['Ks_dec'] = np.array([])

        return results

    # -----------------------
    # Fine-tuning / pruning
    # -----------------------
    def _zero_weak_terms(self, threshold):
        """Zeroes weak terms on FREE rows only -- mirror species have no
        row of their own; their reported coefficients automatically
        reflect their primary's zeroing since they're read as -1*primary."""
        eql = self.reaction.eql_layer
        all_params = self.extract_params(full=True)
        for free_idx, s_idx in enumerate(eql.free_species):
            p = all_params[s_idx]
            eff = torch.tensor(p['effective_unscaled'], device=eql.fc.weight.device)
            small_mask = torch.abs(eff) < threshold
            eql.fc.weight.data[free_idx, small_mask] = 0.0
            eql.l0_gates[free_idx].log_alpha.data[small_mask] = -10.0

    def _average_hill_params(self, idx1, idx2, weight1, weight2):
        """Weighted average of two shared Hill functions' (n, K), indexed
        directly into EQLLayer.all_hill_funcs (already built respecting
        duplicates + include flags)."""
        eql = self.reaction.eql_layer
        hf1 = eql.all_hill_funcs[idx1]
        hf2 = eql.all_hill_funcs[idx2]

        abs_w1, abs_w2 = torch.abs(weight1), torch.abs(weight2)
        total_w = abs_w1 + abs_w2
        prop1, prop2 = (0.5, 0.5) if total_w < 1e-8 else (abs_w1 / total_w, abs_w2 / total_w)

        with torch.no_grad():
            hf1.raw_n.data = hf1.raw_n.data * prop1 + hf2.raw_n.data * prop2
            hf1.raw_K.data = hf1.raw_K.data * prop1 + hf2.raw_K.data * prop2
            hf2.raw_n.data.fill_(0.0)
            hf2.raw_K.data.fill_(0.0)

    @torch.no_grad()
    def fine_tune_eql(self, threshold=0.01, epsilon=0.15, num_points=20000):
        """
        Fine-tunes the discovered EQL equations. All mutations below
        touch only FREE rows (eql.fc.weight / eql.l0_gates are indexed by
        free-row position, not species index) -- mirror species need no
        separate handling since their coefficients are always read as
        -1 * their primary's, which stays correct automatically as the
        primary's row gets pruned/merged/collapsed.

        Per-row steps (independent per free row): zero weak terms, merge
        duplicate polynomials.

        Shared-basis steps (done once, since Hill n/K live in the shared
        feature bank -- weight transfers are applied to EVERY free row
        that has a nonzero coefficient there): merge duplicate Hills,
        collapse near-monomial increasing Hills into the matching
        polynomial term.

        Uses points sampled directly from train_data instead of a uniform
        grid over concentration space -- a uniform grid scales as
        steps^species and is infeasible past ~3 species.
        """
        eql = self.reaction.eql_layer
        device = eql.fc.weight.device
        species = self.species
        dup = self.duplicates
        n_free = eql.n_free

        if eql.hill is None:
            self._zero_weak_terms(threshold)
            print("Fine-tuning committed (poly-only library, no Hill merging needed).")
            return

        # --- TASK 0: SAMPLE FROM THE TRAINING DATA DISTRIBUTION ---
        uv_data = self.train_data[:, -species:].to(device)
        if uv_data.shape[0] > num_points:
            idx = torch.randperm(uv_data.shape[0], device=device)[:num_points]
            uv_synthetic = uv_data[idx]
        else:
            uv_synthetic = uv_data

        features = eql.get_features(uv_synthetic)

        num_poly = eql.num_poly_features
        num_hill = eql.num_hill_features
        n_poly_single = num_poly // dup if dup else 0
        _, hill_terms = self.generate_terms()
        n_hill_single = len(hill_terms)

        forms = []
        if eql.include_increasing_hill: forms.append('inc')
        if eql.include_decreasing_hill: forms.append('dec')
        hill_block_size = n_hill_single * len(forms)  # per duplicate

        # --- TASK 1: ZERO WEAK TERMS (per free row) ---
        self._zero_weak_terms(threshold)

        # --- TASK 2: COMBINE DUPLICATE POLYNOMIALS (per free row) ---
        if eql.include_poly:
            for free_idx in range(n_free):
                for i in range(n_poly_single):
                    indices = [i + j * n_poly_single for j in range(dup)]
                    primary = indices[0]
                    for other in indices[1:]:
                        if torch.abs(eql.fc.weight.data[free_idx, other]) < 1e-8:
                            continue
                        eql.fc.weight.data[free_idx, primary] += eql.fc.weight.data[free_idx, other]
                        eql.fc.weight.data[free_idx, other] = 0.0
                        eql.l0_gates[free_idx].log_alpha.data[primary] = torch.max(
                            eql.l0_gates[free_idx].log_alpha.data[primary],
                            eql.l0_gates[free_idx].log_alpha.data[other])
                        eql.l0_gates[free_idx].log_alpha.data[other] = -10.0

        def any_free_row_nonzero(h_idx):
            return any(torch.abs(eql.fc.weight.data[fi, h_idx]) > 1e-8 for fi in range(n_free))

        # --- TASK 3A: MERGE DUPLICATE HILLS (shared basis, same form only) ---
        for i in range(num_hill):
            h_idx = num_poly + i
            if not any_free_row_nonzero(h_idx):
                continue

            form_id_i = i % hill_block_size
            f_hill = features[:, h_idx]
            f_hill_c = f_hill - torch.mean(f_hill)
            norm_hill_c = torch.norm(f_hill_c) + 1e-9

            for j in range(i + 1, num_hill):
                next_h_idx = num_poly + j
                if not any_free_row_nonzero(next_h_idx):
                    continue
                if j % hill_block_size != form_id_i:
                    continue

                f_other = features[:, next_h_idx]
                f_other_c = f_other - torch.mean(f_other)
                norm_other_c = torch.norm(f_other_c) + 1e-9

                correlation = torch.sum(f_hill_c * f_other_c) / (norm_hill_c * norm_other_c)
                dist = 1.0 - torch.abs(correlation)

                if dist < epsilon:
                    print(f"Merging Duplicate Hills: {h_idx} and {next_h_idx} (Dist: {dist:.4f})")

                    w_primary = max((eql.fc.weight.data[fi, h_idx] for fi in range(n_free)), key=abs)
                    w_duplicate = max((eql.fc.weight.data[fi, next_h_idx] for fi in range(n_free)), key=abs)
                    self._average_hill_params(i, j, w_primary, w_duplicate)

                    for free_idx in range(n_free):
                        eql.fc.weight.data[free_idx, h_idx] += eql.fc.weight.data[free_idx, next_h_idx]
                        eql.fc.weight.data[free_idx, next_h_idx] = 0.0
                        eql.l0_gates[free_idx].log_alpha.data[h_idx] = torch.max(
                            eql.l0_gates[free_idx].log_alpha.data[h_idx],
                            eql.l0_gates[free_idx].log_alpha.data[next_h_idx])
                        eql.l0_gates[free_idx].log_alpha.data[next_h_idx] = -10.0

        # --- TASK 3B: COLLAPSE NEAR-MONOMIAL INCREASING HILLS INTO POLYNOMIALS ---
        if eql.include_poly:
            poly_terms = self.generate_terms()[0]
            for i in range(num_hill):
                h_idx = num_poly + i
                if not any_free_row_nonzero(h_idx):
                    continue

                # Which form does this slot hold? Derive it from `forms`
                # rather than assuming inc-then-dec: when only one family
                # is included, slot 0 is that family, not necessarily inc.
                slot = i % hill_block_size
                if len(forms) == 2:
                    form = 'inc' if slot < n_hill_single else 'dec'
                else:
                    form = forms[0]

                term = hill_terms[slot % n_hill_single]

                hf = eql.all_hill_funcs[i]
                n_val = (torch.sigmoid(hf.raw_n) * 3 + 1).item()
                k_val = F.softplus(hf.raw_K).item()

                max_u = self.max_scale[0, term[0]].item()
                max_denom = 1.0 + k_val * (max_u ** n_val)
                if max_denom >= 1.1:
                    continue

                target = [0] * species
                if form == 'inc':
                    # x_i^n / (1 + K x_i^n) ~ x_i^n: needs n near-integer.
                    n_rounded = round(n_val)
                    if abs(n_val - n_rounded) >= epsilon:
                        continue
                    target[term[0]] = n_rounded
                    if len(term) == 2:
                        target[term[1]] += 1
                else:
                    # x_j / (1 + K x_i^n) ~ x_j: n is irrelevant. The raw
                    # (single-index) dec term collapses to a constant,
                    # which the poly basis has no slot for.
                    if len(term) != 2:
                        continue
                    target[term[1]] = 1

                target = tuple(target)
                if sum(target) > self.degree or target not in poly_terms:
                    continue

                best_p_idx = poly_terms.index(target)
                print(f"Collapsing Hill {h_idx} ({form}) -> Poly {best_p_idx} "
                      f"(term={target}, max_denom={max_denom:.2f}, n={n_val:.3f})")

                for free_idx in range(n_free):
                    w = eql.fc.weight.data[free_idx, h_idx]
                    if torch.abs(w) < 1e-8:
                        continue
                    eql.fc.weight.data[free_idx, best_p_idx] += w
                    eql.fc.weight.data[free_idx, h_idx] = 0.0
                    eql.l0_gates[free_idx].log_alpha.data[best_p_idx] = torch.max(
                        eql.l0_gates[free_idx].log_alpha.data[best_p_idx],
                        eql.l0_gates[free_idx].log_alpha.data[h_idx])
                    eql.l0_gates[free_idx].log_alpha.data[h_idx] = -10.0

        # --- TASK 4: FINAL ZEROING (per free row) ---
        self._zero_weak_terms(threshold)
        print("Fine-tuning committed.")

    def generate_equation(self, eps=1e-12, species_names=None):
        """
        Returns a list of dicts, length self.n_equations:
        {'species': label, 'terms': [str, ...]}. When mcas=True, label is
        "F(u, v)" (one shared reaction, not species-specific) and the list
        has exactly one entry; when mcas=False, label is each species'
        name and the list has one entry per species.
        """
        species_names = species_names or self.species_names
        all_params = self.extract_params(full=True)
        dup = self.duplicates
        all_equations = []

        for eq_idx, p in enumerate(all_params):
            poly_terms = p['poly_terms']
            hill_terms = p['hill_terms']
            terms = []
            poly_coeffs = np.asarray(p['poly_coeffs_unscaled'])

            for term, coeff in zip(poly_terms * dup, poly_coeffs):
                if abs(coeff) > eps:
                    s = f"{float(coeff):.3f}"
                    for i, power in enumerate(term):
                        if power == 1:
                            s += f" * {species_names[i]}"
                        elif power > 1:
                            s += f" * {species_names[i]}^{power}"
                    terms.append(s)

            if len(p['hill_inc_unscaled']):
                inc_b = np.asarray(p['hill_inc_unscaled'])
                Ks_inc, ns_inc = np.asarray(p['Ks_inc']), np.asarray(p['ns_inc'])
                for term, coeff, K, n in zip(hill_terms * dup, inc_b, Ks_inc, ns_inc):
                    if abs(coeff) <= eps:
                        continue
                    coeff_f, K_f, n_f = float(coeff), float(K), float(n)
                    a, b = species_names[term[0]], (species_names[term[1]] if len(term) == 2 else None)
                    if b is None:
                        s = f"{coeff_f:.3f} * {a}^{n_f:.3f} / (1 + {K_f:.3f} * {a}^{n_f:.3f})"
                    else:
                        s = f"{coeff_f:.3f} * {b} * {a}^{n_f:.3f} / (1 + {K_f:.3f} * {a}^{n_f:.3f})"
                    terms.append(s)

            if len(p['hill_dec_unscaled']):
                dec_b = np.asarray(p['hill_dec_unscaled'])
                Ks_dec, ns_dec = np.asarray(p['Ks_dec']), np.asarray(p['ns_dec'])
                for term, coeff, K, n in zip(hill_terms * dup, dec_b, Ks_dec, ns_dec):
                    if abs(coeff) <= eps:
                        continue
                    coeff_f, K_f, n_f = float(coeff), float(K), float(n)
                    a, b = species_names[term[0]], (species_names[term[1]] if len(term) == 2 else None)
                    if b is None:
                        s = f"{coeff_f:.3f} * [1 / (1 + {K_f:.3f} * {a}^{n_f:.3f})]"
                    else:
                        s = f"{coeff_f:.3f} * {b} * [1 / (1 + {K_f:.3f} * {a}^{n_f:.3f})]"
                    terms.append(s)

            if self.mcas:
                # One shared reaction, not species-specific -- label it
                # F(u, v) rather than implying it's "du/dt" specifically
                # when it's really the term that's added to species 0 and
                # subtracted from species 1.
                label = f"F({', '.join(species_names)})"
            else:
                label = species_names[eq_idx]

            all_equations.append({'species': label, 'terms': terms})

        return all_equations

    def equations_as_strings(self, eps=1e-12, species_names=None):
        """Flat list of strings, e.g. for the old `for term in model.model.generate_equation(): file.write(...)`
        training-script loop -- call this instead."""
        equations = self.generate_equation(eps=eps, species_names=species_names)
        lines = []
        for eq in equations:
            header = f"{eq['species']} =" if self.mcas else f"d{eq['species']}/dt ="
            lines.append(header)
            lines.extend(eq['terms'] if eq['terms'] else ['0'])
        return lines