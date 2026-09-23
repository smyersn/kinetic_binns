"""
Normalization scales for the physics losses.

Each is measured once from the fitted surface and then held fixed, so that
loss weights mean the same thing across datasets and resolutions:

    PDE   90th percentile of u_t^2, per species
    mass  90th percentile of (sum over group of u_t)^2, per conservation group
    L0    explainable PDE range per library gate (see l0_scale)
"""
import torch

from modules.utils.gradient import gradient


def surface_time_derivatives(model, data, chunk_size=50_000):
    """du/dt of the surface fitter at the (x, t) of every data row, (N, species), on CPU."""
    was_training = model.surface_fitter.training
    model.surface_fitter.eval()

    chunks = []
    for start in range(0, len(data), chunk_size):
        inputs = data[start:start + chunk_size, :model.dimensions + 1].clone().requires_grad_(True)
        outputs = model(inputs)
        u_t = torch.stack([gradient(outputs[:, s], inputs, order=1)[:, -1]
                           for s in range(model.species)], dim=1)
        chunks.append(u_t.detach().cpu())

    model.surface_fitter.train(was_training)
    return torch.cat(chunks)


def pde_scales(model, data, t_min):
    """90th percentile of u_t^2 per species, over rows with t >= t_min."""
    subset = data[data[:, model.dimensions] >= t_min] if t_min > 0 else data
    u_t = surface_time_derivatives(model, subset)
    return [torch.quantile(u_t[:, s] ** 2, 0.90).item() + 1e-6
            for s in range(model.species)]


def mass_scales(model, data, groups, t_cutoff):
    """90th percentile of the squared total rate of change of each conservation group."""
    subset = data[data[:, model.dimensions] >= t_cutoff]
    u_t = surface_time_derivatives(model, subset)
    return [torch.quantile(sum(u_t[:, s] for s in group) ** 2, 0.90).item() + 1e-8
            for group in groups]


def l0_scale(null, floor, current, floor_epoch, eql, reference_gates=None):
    """
    Price of one active gate in PDE-loss units:

        l0_scale = (PDE_null - PDE_floor) / n_reference_gates

    PDE_null is the residual with every reaction term zeroed and PDE_floor
    the best residual reached with all terms active. Their difference is the
    improvement the library can actually deliver; normalizing by it rather
    than by the floor (mostly irreducible surface-derivative error, which
    grows as the grid coarsens) makes l0_weight resolution-independent:
    l0_weight = 1 asks each term to earn an equal share of that improvement.

    Normalizing by a fixed reference gate count, rather than this run's own,
    keeps the price per term constant as the library grows, so l0_weight
    transfers across library sizes.

    Returns 1.0 (no rescaling) when the range cannot be measured.
    """
    n_gates = eql.n_gates
    ref = reference_gates or n_gates

    if null is None or floor is None:
        print("\n--- L0 scale: no collocation cache, defaulting to 1.0 ---\n")
        return 1.0

    explainable = null - floor
    print("\n--- L0 scale diagnostics ---")
    print(f"  PDE floor (best in Phase 2)  : {floor:.4e}  "
          f"(at epoch {floor_epoch}; current = {current:.4e})")
    print(f"  PDE null  (all terms zeroed) : {null:.4e}")

    if explainable <= 0:
        # The terms fit worse than F = 0: Phase 2 did not converge. This is a
        # convergence failure, not a resolution limit.
        print(f"  Explainable range:            NEGATIVE ({explainable:.4e})")
        print("  WARNING: the reaction terms fit WORSE than F=0. Phase 2 did not "
              "converge -- l0_scale is not meaningful here. Falling back to 1.0. "
              "Extend Phase 2 or check the reaction LR before trusting any "
              "pruning result from this run.\n")
        return 1.0

    scale = explainable / ref
    fraction = explainable / max(null, 1e-12)
    print(f"  Explainable range:            {explainable:.4e}  ({100 * fraction:.1f}% of null)")
    print(f"  Gates: {n_gates} ({eql.total_features} features x {eql.n_free} free row(s))")
    if reference_gates:
        print(f"  Reference gates: {ref}  ->  library is {n_gates / ref:.2f}x reference")
    else:
        print("  Reference gates: (unset -- normalizing by this run's own gate "
              "count; l0_weight will NOT transfer across library sizes)")
    print(f"  l0_scale = explainable / {ref} = {scale:.4e}")
    if fraction < 0.1:
        print("  WARNING: terms reduce PDE loss by <10%. The residual is dominated "
              "by surface-derivative error the reaction cannot fix -- likely a "
              "resolution limit, not an L0 tuning problem.")
    print()
    return scale


class PDEFloorTracker:
    """
    Lowest full-cache PDE loss seen during Phase 2 and the reaction weights
    that achieved it. The reaction weights oscillate by more than the
    explainable range itself, so L0 is calibrated (and Phase 3 started) at
    the best point rather than wherever Phase 2 happened to end.
    """

    def __init__(self):
        self.best = None
        self.epoch = None
        self.reaction_state = None

    @torch.no_grad()
    def update(self, loss, epoch, reaction):
        if loss is None:
            return
        if self.best is None or loss < self.best:
            self.best, self.epoch = loss, epoch
            self.reaction_state = {k: v.detach().clone()
                                   for k, v in reaction.state_dict().items()}

    @torch.no_grad()
    def restore(self, reaction):
        """Load the best reaction weights into `reaction`. Returns False if none recorded."""
        if self.reaction_state is None:
            return False
        reaction.load_state_dict(self.reaction_state)
        print(f"Restored best Phase-2 reaction from epoch {self.epoch} "
              f"(PDE = {self.best:.4e})", flush=True)
        return True

    def state_dict(self):
        return {'best': self.best, 'epoch': self.epoch, 'reaction_state': self.reaction_state}

    def load_state_dict(self, state):
        self.best, self.epoch = state['best'], state['epoch']
        self.reaction_state = state['reaction_state']