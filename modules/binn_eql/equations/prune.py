"""
Post-hoc simplification of a trained EQL reaction term.

Applied once after training, in order:

    1. zero terms with |w * z| < threshold
    2. merge the duplicate copies of each monomial
    3. merge Hill features whose outputs are near-collinear over the data
    4. collapse Hill features that stay in their linear regime into the
       equivalent monomial (x^n / (1 + K x^n) ~ x^n when K x^n << 1)
    5. zero weak terms again

Weights and gates exist only for *free* rows of the EQL head, so every edit
indexes free rows; mirror species follow automatically. Hill (n, K) are
shared by all rows, so Hill merges move weight on every row.
"""
import numpy as np
import torch

from modules.binn_eql.equations.extract import effective_weights

CLOSED_GATE = -10.0  # log_alpha for a gate that is effectively always closed


# ----------------------------------------------------------------------
# Primitive edits
# ----------------------------------------------------------------------
def _move_term(eql, row, src, dst):
    """Add column src's weight into dst on one row, zero src, and close its gate."""
    W = eql.fc.weight.data
    log_alpha = eql.l0_gates[row].log_alpha.data
    W[row, dst] += W[row, src]
    W[row, src] = 0.0
    log_alpha[dst] = torch.max(log_alpha[dst], log_alpha[src])
    log_alpha[src] = CLOSED_GATE


def _is_active(eql, col):
    """True if any free row has a nonzero weight on column col."""
    return any(torch.abs(eql.fc.weight.data[row, col]) > 1e-8 for row in range(eql.n_free))


def _average_hill_params(eql, i, j, w_i, w_j):
    """
    Replace Hill i's (n, K) by the |w|-weighted average of Hills i and j
    (in unconstrained parameter space) and reset Hill j.
    """
    fn_i, fn_j = eql.all_hill_funcs[i], eql.all_hill_funcs[j]
    a_i, a_j = torch.abs(w_i), torch.abs(w_j)
    total = a_i + a_j
    p_i, p_j = (0.5, 0.5) if total < 1e-8 else (a_i / total, a_j / total)

    fn_i.raw_n.data = fn_i.raw_n.data * p_i + fn_j.raw_n.data * p_j
    fn_i.raw_K.data = fn_i.raw_K.data * p_i + fn_j.raw_K.data * p_j
    fn_j.raw_n.data.fill_(0.0)
    fn_j.raw_K.data.fill_(0.0)


# ----------------------------------------------------------------------
# Pruning steps
# ----------------------------------------------------------------------
def zero_weak_terms(eql, threshold):
    """Zero every term whose effective coefficient |w * z| is below threshold."""
    for row, s_idx in enumerate(eql.free_species):
        _, _, effective = effective_weights(eql, s_idx)
        weak = torch.abs(effective) < threshold
        eql.fc.weight.data[row, weak] = 0.0
        eql.l0_gates[row].log_alpha.data[weak] = CLOSED_GATE


def merge_duplicate_polynomials(eql):
    """Sum the `duplicates` copies of each monomial into the first copy."""
    n_single = eql.num_poly_features // eql.duplicates
    for row in range(eql.n_free):
        for i in range(n_single):
            for d in range(1, eql.duplicates):
                other = i + d * n_single
                if torch.abs(eql.fc.weight.data[row, other]) >= 1e-8:
                    _move_term(eql, row, src=other, dst=i)


def merge_duplicate_hills(eql, features, epsilon):
    """
    Merge pairs of same-form, same-input Hill features whose outputs over the
    data have 1 - |corr| < epsilon. The pair's (n, K) are averaged and the
    weight moves to the earlier feature.
    """
    W = eql.fc.weight.data
    offset = eql.num_poly_features

    for i in range(eql.num_hill_features):
        col_i = offset + i
        if not _is_active(eql, col_i):
            continue
        f_i = features[:, col_i] - torch.mean(features[:, col_i])
        norm_i = torch.norm(f_i) + 1e-9

        for j in range(i + 1, eql.num_hill_features):
            col_j = offset + j
            if not _is_active(eql, col_j) or eql.hill_slots[j] != eql.hill_slots[i]:
                continue
            f_j = features[:, col_j] - torch.mean(features[:, col_j])
            norm_j = torch.norm(f_j) + 1e-9

            correlation = torch.sum(f_i * f_j) / (norm_i * norm_j)
            dist = 1.0 - torch.abs(correlation)
            if dist >= epsilon:
                continue

            print(f"Merging Duplicate Hills: {col_i} and {col_j} (Dist: {dist:.4f})")
            w_i = max((W[row, col_i] for row in range(eql.n_free)), key=abs)
            w_j = max((W[row, col_j] for row in range(eql.n_free)), key=abs)
            _average_hill_params(eql, i, j, w_i, w_j)
            for row in range(eql.n_free):
                _move_term(eql, row, src=col_j, dst=col_i)


def _monomial_equivalent(form, term, n_val, n_tol, species):
    """
    Exponent tuple of the monomial a flat Hill reduces to, or None:
        inc raw    x_i^n / (1 + K x_i^n)       ->  x_i^n      (n near an integer)
        inc cross  x_j x_i^n / (1 + K x_i^n)   ->  x_j x_i^n  (n near an integer)
        dec cross  x_j / (1 + K x_i^n)         ->  x_j
    A flat dec raw term is a constant, which the polynomial basis lacks.
    """
    powers = [0] * species
    if form == 'inc':
        n_rounded = int(round(n_val))
        if abs(n_val - n_rounded) >= n_tol:
            return None
        powers[term[0]] = n_rounded
        if len(term) == 2:
            powers[term[1]] += 1
    else:
        if len(term) != 2:
            return None
        powers[term[1]] = 1
    return tuple(powers)


def collapse_flat_hills(eql, samples, degree, n_tol, flat_denom, flat_quantile):
    """
    Move Hill features whose denominator stays near 1 over the data into the
    matching polynomial column.

    Flatness is judged at a quantile of the data (flat_quantile) rather than
    at the maximum, so a thin transient tail cannot make a term that is flat
    everywhere it matters look non-flat.

    If no matching monomial exists (degree too low), an increasing Hill is
    snapped in place instead: n rounded, K -> ~0. Its coefficient is not
    refit, so rounding n shifts the term's value slightly.
    """
    poly_terms = eql.poly.powers
    offset = eql.num_poly_features

    for i, (fn, (form, term)) in enumerate(zip(eql.all_hill_funcs, eql.hill_slots)):
        col = offset + i
        if not _is_active(eql, col):
            continue

        n_val, k_val = fn.n.item(), fn.K.item()
        denominator = 1.0 + k_val * samples[:, term[0]].pow(n_val)
        q_denom = torch.quantile(denominator, flat_quantile).item()
        if q_denom >= flat_denom:
            continue

        target = _monomial_equivalent(form, term, n_val, n_tol, eql.species)
        if target is None:
            continue

        if sum(target) > degree or target not in poly_terms:
            if form != 'inc':
                continue
            n_rounded = int(round(n_val))
            p = min(max((n_rounded - 1) / 3.0, 1e-4), 1 - 1e-4)  # invert n = 1 + 3*sigmoid(raw_n)
            fn.raw_n.data.fill_(float(np.log(p / (1 - p))))
            fn.raw_K.data.fill_(-20.0)                            # softplus(-20) ~ 2e-9
            print(f"Snapping Hill {col} -> {term}^{n_rounded} "
                  f"(no degree-{degree} monomial slot, "
                  f"q{flat_quantile:.2f}_denom={q_denom:.4f})")
            continue

        p_idx = poly_terms.index(target)
        print(f"Collapsing Hill {col} ({form}) -> Poly {p_idx} "
              f"(term={target}, q{flat_quantile:.2f}_denom={q_denom:.4f}, n={n_val:.3f})")
        for row in range(eql.n_free):
            if torch.abs(eql.fc.weight.data[row, col]) >= 1e-8:
                _move_term(eql, row, src=col, dst=p_idx)


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------
def _sample_concentrations(binn, num_points, device):
    """Up to num_points concentration vectors drawn from the training data."""
    u = binn.train_data[:, -binn.species:].to(device)
    if u.shape[0] > num_points:
        u = u[torch.randperm(u.shape[0], device=device)[:num_points]]
    return u


@torch.no_grad()
def fine_tune_eql(binn, threshold=0.01, epsilon=0.1, n_tol=0.15,
                  flat_denom=1.25, flat_quantile=0.95, num_points=20000):
    """
    Simplify binn's reaction term in place (see module docstring).

    threshold      minimum |w * z| for a term to survive
    epsilon        Hill pairs with 1 - |corr| < epsilon are merged
    n_tol          max distance of n from an integer for an increasing
                   Hill to count as a monomial
    flat_denom     a Hill is flat if its denominator stays below this ...
    flat_quantile  ... at this quantile of the training data
    num_points     training points used for the merge/flatness tests
    """
    eql = binn.reaction.eql_layer

    if eql.hill is None:
        zero_weak_terms(eql, threshold)
        print("Fine-tuning committed (poly-only library, no Hill merging needed).")
        return

    samples = _sample_concentrations(binn, num_points, eql.fc.weight.device)
    features = eql.get_features(samples)

    zero_weak_terms(eql, threshold)
    if eql.include_poly:
        merge_duplicate_polynomials(eql)
    merge_duplicate_hills(eql, features, epsilon)
    if eql.include_poly:
        collapse_flat_hills(eql, samples, binn.degree, n_tol, flat_denom, flat_quantile)
    zero_weak_terms(eql, threshold)
    print("Fine-tuning committed.")