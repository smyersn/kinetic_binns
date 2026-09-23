"""
Collocation points for the PDE residual.

The surface fitter is frozen after Phase 1, so its derivatives at a fixed
pool of points can be computed once and reused: the reaction network then
trains on cached (u, u_t, lap u) samples without re-differentiating the
surface every step.
"""
from typing import NamedTuple

import torch

from modules.utils.gradient import gradient


class Collocation(NamedTuple):
    outputs: torch.Tensor    # u                       (n, species)
    u_t: torch.Tensor        # du/dt                   (n, species)
    u_xx: torch.Tensor       # d2u/dx_d2, per axis d   (species, n, dimensions)
    mass_mask: torch.Tensor  # t >= mass cutoff        (n,)

    def subset(self, idx):
        return Collocation(self.outputs[idx], self.u_t[idx],
                           self.u_xx[:, idx, :], self.mass_mask[idx])


def field_derivatives(inputs, outputs, species, dimensions):
    """
    u_t and the unmixed second spatial derivatives of every species.
    `inputs` is (n, dimensions + 1) with time last and must require grad.
    """
    n = len(inputs)
    u_t = torch.zeros((n, species), device=inputs.device)
    u_xx = torch.zeros((species, n, dimensions), device=inputs.device)
    for s in range(species):
        grad_s = gradient(outputs[:, s], inputs, order=1)
        u_t[:, s] = grad_s[:, -1]
        for d in range(dimensions):
            u_xx[s, :, d] = gradient(grad_s[:, d], inputs, order=1)[:, d]
    return u_t, u_xx


def sample_collocation(model, n, t_min, mass_t_cutoff):
    """Evaluate the surface and its derivatives at n uniform points with t in [t_min, t_max]."""
    device = model.lb.device
    x = torch.empty(n, model.dimensions, device=device).uniform_(
        model.lb[0, 0].item(), model.ub[0, 0].item())
    t = torch.empty(n, 1, device=device).uniform_(t_min, model.ub[0, -1].item())
    inputs = torch.cat([x, t], dim=1).requires_grad_(True)

    outputs = model(inputs)
    u_t, u_xx = field_derivatives(inputs, outputs, model.species, model.dimensions)
    return Collocation(outputs, u_t, u_xx, inputs[:, -1] >= mass_t_cutoff)


class CollocationCache:
    """A fixed pool of collocation points, evaluated once on the frozen surface."""

    def __init__(self, model, t_min, mass_t_cutoff, size=200_000, chunk_size=20_000):
        was_training = model.surface_fitter.training
        model.surface_fitter.eval()

        chunks = []
        for start in range(0, size, chunk_size):
            n = min(chunk_size, size - start)
            chunk = sample_collocation(model, n, t_min, mass_t_cutoff)
            chunks.append(Collocation(*(v.detach() for v in chunk)))
            del chunk
            torch.cuda.empty_cache()

        model.surface_fitter.train(was_training)

        self.data = Collocation(
            outputs=torch.cat([c.outputs for c in chunks], dim=0),
            u_t=torch.cat([c.u_t for c in chunks], dim=0),
            u_xx=torch.cat([c.u_xx for c in chunks], dim=1),
            mass_mask=torch.cat([c.mass_mask for c in chunks], dim=0))

        print(f"Refreshed collocation cache: {size} points ({chunk_size}/chunk) "
              f"over t in [{t_min:.4g}, {model.ub[0, -1].item():.4g}], "
              f"{self.data.mass_mask.sum().item()} pass mass cutoff")

    def __len__(self):
        return len(self.data.outputs)

    def sample(self, n):
        """n points drawn uniformly with replacement."""
        idx = torch.randint(0, len(self), (n,), device=self.data.outputs.device)
        return self.data.subset(idx)