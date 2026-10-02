"""Polynomial features for the EQL reaction library."""
import itertools

import torch
import torch.nn as nn


def monomial_powers(species, degree, include_constant=False):
    """
    All exponent tuples p with 1 <= sum(p) <= degree, ordered by total degree
    and then with higher powers of earlier species first. For two species and
    degree 2: (1,0), (0,1), (2,0), (1,1), (0,2).

    include_constant also admits the all-zero tuple, i.e. a constant term.
    Needed for any reaction with a source or feed rate -- Gray-Scott's feed,
    Brusselator's and Schnakenberg's a -- which the library otherwise has to
    approximate with whatever linear term fits least badly.
    """
    lowest = 0 if include_constant else 1
    powers = [p for p in itertools.product(range(degree + 1), repeat=species)
              if lowest <= sum(p) <= degree]
    powers.sort(key=lambda p: (sum(p), tuple(-k for k in p)))
    return powers


class PolynomialFeatures(nn.Module):
    """Monomials of the concentrations, repeated `duplicates` times."""

    def __init__(self, species, duplicates, degree, include_constant=False):
        super().__init__()
        self.species = species
        self.duplicates = duplicates
        self.degree = degree
        self.include_constant = include_constant
        self.powers = monomial_powers(species, degree, include_constant)
        # (num_monomials, species) exponent table; not part of the state_dict.
        self.register_buffer('exponents', torch.tensor(self.powers, dtype=torch.float32),
                             persistent=False)

    def forward(self, x):
        """All monomials at once: prod_i x_i^p_i for each exponent row p (x^0 = 1)."""
        monomials = torch.prod(x.unsqueeze(1) ** self.exponents.to(x.dtype), dim=2)
        return monomials.repeat(1, self.duplicates)