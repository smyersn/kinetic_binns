"""Learnable diffusion coefficients."""
import torch
import torch.nn as nn


class DiffusionCoefficients(nn.Module):
    """One positive diffusion coefficient per species, D = exp(raw_D)."""

    def __init__(self, species, base_value=0.1, noise_std=0.5):
        super().__init__()
        base = torch.log(torch.tensor(base_value))
        self.raw_D = nn.Parameter(base + noise_std * torch.randn(species))

    def forward(self):
        return torch.exp(self.raw_D)