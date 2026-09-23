"""
Hill-function features for the EQL reaction library.

Two forms, each applied to every species (raw terms) and to every ordered
species pair (cross terms):

    increasing  h+(x) = x^n / (1 + K x^n)
    decreasing  h-(x) = 1   / (1 + K x^n)

    raw term    (i,)    ->  h(x_i)
    cross term  (i, j)  ->  h(x_i) * x_j,   i != j

Each feature has its own n in [1, 4] and K > 0. They are shared by every
species' linear head; only the coefficient multiplying them differs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def hill_terms(species):
    """Input-index tuples for one Hill form, in feature order: raw then cross."""
    raw = [(i,) for i in range(species)]
    cross = [(i, j) for i in range(species) for j in range(species) if i != j]
    return raw + cross


class HillFunction(nn.Module):
    """A single Hill function with learnable n (sigmoid-bounded) and K (softplus)."""

    def __init__(self, increasing=True):
        super().__init__()
        self.increasing = increasing
        self.raw_n = nn.Parameter(torch.empty(1).uniform_(-4, 4))
        # Overwritten by EQLLayer._initialize_K once the data scale is known.
        self.raw_K = nn.Parameter(torch.empty(1).uniform_(-2, 2))

    @property
    def n(self):
        return torch.sigmoid(self.raw_n) * 3 + 1

    @property
    def K(self):
        return F.softplus(self.raw_K)

    def forward(self, x):
        x_n = x.pow(self.n)
        denominator = 1 + self.K * x_n
        return x_n / denominator if self.increasing else 1.0 / denominator


class HillFeatures(nn.Module):
    """One copy of every included Hill form over all raw and cross terms."""

    def __init__(self, species, include_increasing=True, include_decreasing=True):
        super().__init__()
        if not (include_increasing or include_decreasing):
            raise ValueError("HillFeatures needs include_increasing or include_decreasing.")

        self.species = species
        self.include_increasing = include_increasing
        self.include_decreasing = include_decreasing
        self.terms = hill_terms(species)

        # Attribute names and "i_j" keys are part of the checkpoint format.
        if include_increasing:
            self.hill_inc_raw, self.hill_inc_cross = self._build_form(increasing=True)
        if include_decreasing:
            self.hill_dec_raw, self.hill_dec_cross = self._build_form(increasing=False)

    def _build_form(self, increasing):
        raw = nn.ModuleList([HillFunction(increasing) for _ in range(self.species)])
        cross = nn.ModuleDict({f"{t[0]}_{t[1]}": HillFunction(increasing)
                               for t in self.terms if len(t) == 2})
        return raw, cross

    def slots(self):
        """(form, term, HillFunction) for every output column, in order."""
        forms = []
        if self.include_increasing:
            forms.append(('inc', self.hill_inc_raw, self.hill_inc_cross))
        if self.include_decreasing:
            forms.append(('dec', self.hill_dec_raw, self.hill_dec_cross))

        slots = []
        for form, raw, cross in forms:
            for term in self.terms:
                fn = raw[term[0]] if len(term) == 1 else cross[f"{term[0]}_{term[1]}"]
                slots.append((form, term, fn))
        return slots

    def forward(self, x):
        features = []
        for _, term, fn in self.slots():
            f = fn(x[:, term[0]:term[0] + 1])
            if len(term) == 2:
                f = f * x[:, term[1]:term[1] + 1]
            features.append(f)
        return torch.cat(features, dim=1)


class DuplicateHillFeatures(nn.Module):
    """`duplicates` independent copies of HillFeatures, concatenated."""

    def __init__(self, species, duplicates, include_increasing=True, include_decreasing=True):
        super().__init__()
        self.hill_modules = nn.ModuleList([
            HillFeatures(species, include_increasing, include_decreasing)
            for _ in range(duplicates)])

    def slots(self):
        return [slot for module in self.hill_modules for slot in module.slots()]

    def forward(self, x):
        return torch.cat([module(x) for module in self.hill_modules], dim=1)