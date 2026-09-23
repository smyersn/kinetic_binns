"""Polynomial features for the EQL reaction library."""
import itertools

import torch
import torch.nn as nn


def monomial_powers(species, degree):
    """
    All exponent tuples p with 1 <= sum(p) <= degree, ordered by total
    degree and then with higher powers of earlier species first. For two
    species and degree 2: (1,0), (0,1), (2,0), (1,1), (0,2).
    """
    powers = [p for p in itertools.product(range(degree + 1), repeat=species)
              if 1 <= sum(p) <= degree]
    powers.sort(key=lambda p: (sum(p), tuple(-k for k in p)))
    return powers


class PolynomialFeatures(nn.Module):
    """Monomials of the concentrations, repeated `duplicates` times."""

    def __init__(self, species, duplicates, degree):
        super().__init__()
        self.species = species
        self.duplicates = duplicates
        self.degree = degree
        self.powers = monomial_powers(species, degree)

    def forward(self, x):
        features = []
        for p in self.powers:
            term = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
            for i, k in enumerate(p):
                if k > 0:
                    term = term * (x[:, i:i + 1] ** k)
            features.append(term)
        return torch.cat(features * self.duplicates, dim=1)