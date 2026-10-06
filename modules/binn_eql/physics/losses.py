"""
Training objective for the BINN:

    L = w_gls * L_gls + w_pde * L_pde + w_l0 * L_0 + L_wall + w_mass * L_mass

    L_gls   weighted data misfit of the surface fitter
    L_pde   residual of u_t = D lap(u) + F(u) at collocation points
    L_0     expected number of active gates (sparsity)
    L_wall  soft bounds on reaction coefficients and Hill K
    L_mass  conservation of the surface fitter's derivatives (mcas only)

The phase weights w_* are set by training/phases.py. The normalization
scales used by L_pde, L_mass and L_0 are measured from the fitted surface
at Phase 2 and Phase 3 entry and held here.
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from modules.binn_eql.physics.calibration import (
    PDEFloorTracker, l0_scale, mass_scales, pde_scales, row_l0_scales)
from modules.binn_eql.physics.collocation import CollocationCache, sample_collocation


class LossTerms(NamedTuple):
    gls: torch.Tensor
    pde: torch.Tensor
    l0: torch.Tensor
    soft_wall: torch.Tensor
    mass: torch.Tensor


class BINNLoss:

    def __init__(self, model, n_collocation=10_000, mass_t_cutoff=2.0,
                 soft_wall_epsilon=0.15, soft_wall_penalty=100.0,
                 l0_pricing='auto'):
        """
        l0_pricing: how a gate's cost is set at Phase 3 entry.
            'auto'         (default) 'global' for mcas, 'per_species' otherwise.
                           The two agree for mcas anyway -- its single row is
                           charged against both species' combined range -- so
                           this routes conserved runs down the original code
                           path and its diagnostics, and nothing else changes.
            'global'       one price for every row, from the explainable range
                           summed over species. With more than one independent
                           equation, a term is charged against a budget it
                           cannot fully reach, so the species holding less of
                           the range is pruned harder.
            'per_species'  each row priced against the explainable range of the
                           species it drives, so l0_weight means the same thing
                           regardless of how many equations there are or how
                           unevenly the range is split between them.
        """
        if l0_pricing not in ('auto', 'global', 'per_species'):
            raise ValueError(f"l0_pricing must be 'auto', 'global' or 'per_species', "
                             f"got {l0_pricing!r}")
        self.model = model
        self.l0_pricing = l0_pricing
        self.n_collocation = n_collocation
        self.mass_t_cutoff = mass_t_cutoff
        self.soft_wall_epsilon = soft_wall_epsilon
        self.soft_wall_penalty = soft_wall_penalty

        # Measured at Phase 2 entry (calibrate_physics).
        self.cache = None
        self.pde_scale_locked = False
        self.mass_scales = None

        # Measured at Phase 3 entry (calibrate_l0).
        self.floor_tracker = PDEFloorTracker()
        self.l0_scale = 1.0        # scalar price the trainer multiplies l0 by
        self.row_l0_scales = None  # per-row prices, folded into l0() instead
        self.l0_scale_locked = False

    def __call__(self, inputs, pred, true, phase):
        """Unweighted loss terms for one batch of data (inputs, true) and prediction pred."""
        gls = self.gls(inputs, pred, true)
        soft_wall = self.soft_wall()
        l0 = self.l0()

        if phase == 1:
            zero = torch.tensor(0.0, device=pred.device)
            return LossTerms(gls, zero, l0, soft_wall, zero)

        if self.cache is not None:
            batch = self.cache.sample(self.n_collocation)
        else:
            batch = sample_collocation(self.model, self.n_collocation,
                                       self.model.collocation_t_min(), self.mass_t_cutoff)
        return LossTerms(gls, self.pde(batch), l0, soft_wall, self.mass(batch))

    def physics_terms(self):
        """
        Loss terms for Phases 2-4, where the surface is frozen and the GLS
        weight is 0: the data and the surface forward pass are skipped
        entirely, and gls is returned as 0.
        """
        soft_wall = self.soft_wall()
        l0 = self.l0()
        zero = torch.tensor(0.0, device=l0.device)
        if self.cache is not None:
            batch = self.cache.sample(self.n_collocation)
        else:
            batch = sample_collocation(self.model, self.n_collocation,
                                       self.model.collocation_t_min(), self.mass_t_cutoff)
        return LossTerms(zero, self.pde(batch), l0, soft_wall, self.mass(batch))

    @torch.no_grad()
    def full_cache_terms(self):
        """
        Physics loss terms on the entire collocation cache: deterministic
        (in eval mode), so suitable for model selection. None if no cache.
        """
        if self.cache is None:
            return None
        l0 = self.l0()
        zero = torch.tensor(0.0, device=l0.device)
        return LossTerms(zero, self.pde(self.cache.data), l0, self.soft_wall(),
                         self.mass(self.cache.data))

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------
    def gls(self, inputs, pred, true):
        """
        Squared misfit, normalized per species by mean |u| and weighted per
        frame by pattern activity (see frame_statistics.frame_activity_weights).
        """
        residual = ((pred - true) / self.model.mean_scale) ** 2
        frame = torch.bucketize(inputs[:, -1].contiguous(), self.model.gls_frame_edges)
        weights = self.model.gls_frame_weights[frame].unsqueeze(1)
        return torch.mean(residual * weights)

    def pde(self, batch):
        """Smooth-L1 PDE residual summed over species, each divided by sqrt(its PDE scale)."""
        model = self.model
        reaction = model.reaction(batch.outputs)
        D = model.diffusion_coefficients()

        total = torch.tensor(0.0, device=batch.outputs.device)
        for s in range(model.species):
            laplacian = D[s] * batch.u_xx[s].sum(dim=1, keepdim=True)
            residual = batch.u_t[:, s:s + 1] - (laplacian + reaction[:, s:s + 1])
            if self.pde_scale_locked:
                residual = residual / torch.sqrt(model.pde_scale[s])
            total = total + F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0)
        return total

    def mass(self, batch):
        """
        Weighted residual of d/dt(sum over group of u) = sum over group of D lap(u)
        on the surface fitter's derivatives.

        Zero unless mcas=True AND diffusion is learned: with fixed D every
        input here is constant with respect to the trainable parameters, so
        the term carries no gradient.
        """
        model = self.model
        zero = torch.tensor(0.0, device=batch.u_t.device)
        if not model.conservation_groups or not model.learns_diffusion:
            return zero

        # Points before the mass cutoff are excluded by weighting them with 0
        # rather than by boolean indexing, which would force a GPU->CPU sync
        # every step. Returns 0 when no point passes the cutoff.
        mask = batch.mass_mask.to(batch.u_t.dtype)
        count = mask.sum().clamp_min(1.0)

        D = model.diffusion_fitter()
        total = zero
        for g, group in enumerate(model.conservation_groups):
            w_t = sum(batch.u_t[:, s] for s in group)
            laplacian = sum(D[s] * batch.u_xx[s].sum(dim=1) for s in group)

            residual = w_t - laplacian
            if self.mass_scales:
                residual = residual / torch.sqrt(torch.tensor(self.mass_scales[g], device=zero.device))

            # Emphasize points where the group total is actually changing.
            weights = w_t.abs().detach()
            weights = weights / ((weights * mask).sum() / count + 1e-8)
            total = total + (mask * weights * residual ** 2).sum() / count
        return total / len(model.conservation_groups)

    def l0(self):
        """
        Expected number of open gates, summed over free rows (mirrors share
        their primary's gate). Under per-species pricing each row is weighted
        by its own price here, and the trainer's scalar l0_scale stays 1, so
        the two are never applied twice.
        """
        eql = self.model.reaction.eql_layer
        total = torch.tensor(0.0, device=eql.fc.weight.device)
        for row, gate in enumerate(eql.l0_gates):
            penalty = gate.expected_l0().sum()
            if self.row_l0_scales is not None:
                penalty = penalty * self.row_l0_scales[row]
            total = total + penalty
        return total

    def soft_wall(self):
        """Linear penalty on |w| above param_bounds and on Hill K above its ceiling."""
        eql = self.model.reaction.eql_layer
        w, k, k_ceiling = eql.get_physical_parameters(epsilon=self.soft_wall_epsilon)

        loss = torch.relu(w.abs() - self.model.param_bounds).sum() * self.soft_wall_penalty
        if k.numel() > 0:
            loss = loss + torch.relu(k - k_ceiling).sum() * self.soft_wall_penalty
        return loss

    @torch.no_grad()
    def full_cache_pde(self, zero_reaction=False):
        """
        PDE loss over the whole collocation cache. With zero_reaction=True,
        every reaction coefficient is temporarily zeroed: the residual that
        no reaction term can remove.
        """
        if self.cache is None:
            return None
        weight = self.model.reaction.eql_layer.fc.weight.data
        saved = weight.clone()
        if zero_reaction:
            weight.zero_()
        try:
            return self.pde(self.cache.data).item()
        finally:
            weight.copy_(saved)

    @property
    def resolved_l0_pricing(self):
        """'auto' resolved against the model: one equation (mcas) -> global."""
        if self.l0_pricing != 'auto':
            return self.l0_pricing
        return 'global' if self.model.mcas else 'per_species'

    @torch.no_grad()
    def full_cache_pde_per_species(self, zero_reaction=False):
        """PDE loss over the whole cache, split by species. None if no cache."""
        if self.cache is None:
            return None
        model, batch = self.model, self.cache.data
        weight = model.reaction.eql_layer.fc.weight.data
        saved = weight.clone()
        if zero_reaction:
            weight.zero_()
        try:
            reaction = model.reaction(batch.outputs)
            D = model.diffusion_coefficients()
            losses = []
            for s in range(model.species):
                laplacian = D[s] * batch.u_xx[s].sum(dim=1, keepdim=True)
                residual = batch.u_t[:, s:s + 1] - (laplacian + reaction[:, s:s + 1])
                if self.pde_scale_locked:
                    residual = residual / torch.sqrt(model.pde_scale[s])
                losses.append(
                    F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0).item())
            return losses
        finally:
            weight.copy_(saved)

    # ------------------------------------------------------------------
    # Calibration (called by the Trainer at phase boundaries; idempotent)
    # ------------------------------------------------------------------
    def calibrate_physics(self, train_data):
        """Measure PDE and mass scales and build the collocation cache from the current surface."""
        model = self.model
        t_min = model.collocation_t_min()

        if not self.pde_scale_locked:
            print(f"\n--- Calculating Robust PDE Scales (90th percentile, t>={t_min:.4g}) ---")
            scales = pde_scales(model, train_data, t_min)
            model.pde_scale.copy_(torch.tensor(scales))
            self.pde_scale_locked = True
            print(f"Locked 90th Percentile Scales -> {[f'{s:.4e}' for s in scales]}\n")

        if self.mass_scales is None:
            if model.conservation_groups:
                t_cutoff = max(self.mass_t_cutoff, t_min)
                print(f"\n--- Calculating Robust Mass Scale(s) (90th percentile, t>={t_cutoff}) ---")
                self.mass_scales = mass_scales(model, train_data, model.conservation_groups, t_cutoff)
                print(f"Locked Mass Scale(s) -> {[f'{s:.4e}' for s in self.mass_scales]}\n")
            else:
                self.mass_scales = []

        if self.cache is None:
            self.cache = CollocationCache(model, t_min, self.mass_t_cutoff)

    def track_pde_floor(self, epoch):
        """Record the current full-cache PDE loss if it is the best seen in Phase 2."""
        self.floor_tracker.update(self.full_cache_pde(), epoch, self.model.reaction)

    def calibrate_l0(self):
        """Restore the best Phase-2 reaction, then measure the L0 scale at that state."""
        if self.l0_scale_locked:
            return
        self.floor_tracker.restore(self.model.reaction)

        current = self.full_cache_pde()
        best = self.floor_tracker.best
        if best is None:
            floor = current
        else:
            floor = best if current is None else min(best, current)

        if self.resolved_l0_pricing == 'per_species':
            # The reaction now holds the best Phase-2 weights, so the current
            # per-species losses ARE that state's floor.
            self.row_l0_scales = row_l0_scales(
                null=self.full_cache_pde_per_species(zero_reaction=True),
                floor=self.full_cache_pde_per_species(),
                eql=self.model.reaction.eql_layer,
                species_names=self.model.species_names)
            self.l0_scale = 1.0
        else:
            self.l0_scale = l0_scale(
                null=self.full_cache_pde(zero_reaction=True),
                floor=floor,
                current=current,
                floor_epoch=self.floor_tracker.epoch,
                eql=self.model.reaction.eql_layer)
        self.l0_scale_locked = True

    # ------------------------------------------------------------------
    # Checkpointing (the PDE/mass scales and cache are re-measured on resume)
    # ------------------------------------------------------------------
    def state_dict(self):
        return {'l0_scale': self.l0_scale,
                'row_l0_scales': self.row_l0_scales,
                'l0_scale_locked': self.l0_scale_locked,
                'floor_tracker': self.floor_tracker.state_dict()}

    def load_state_dict(self, state):
        self.l0_scale = state['l0_scale']
        # Absent in checkpoints written before per-species pricing existed;
        # None restores the single-price behaviour they were trained with.
        self.row_l0_scales = state.get('row_l0_scales')
        self.l0_scale_locked = state['l0_scale_locked']
        self.floor_tracker.load_state_dict(state['floor_tracker'])