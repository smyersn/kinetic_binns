"""Read the learned reaction coefficients and Hill parameters out of a BINN."""
import numpy as np
import torch

from modules.binn_eql.library.hill import hill_terms


def generate_terms(binn):
    """(polynomial exponent tuples, Hill input-index tuples) for one copy of the library."""
    eql = binn.reaction.eql_layer
    poly_terms = eql.poly.powers if eql.poly is not None else []
    return poly_terms, hill_terms(binn.species)


def gate_values(gate):
    """Deterministic gate openings in [0, 1]."""
    try:
        return gate.get_gates().detach()
    except AttributeError:
        return torch.sigmoid(gate.log_alpha.detach()).clamp(0.0, 1.0)


def effective_weights(eql, s_idx):
    """
    (raw w, gates z, effective w*z) for species s_idx, as detached tensors.
    Mirror species are reported as -1 x their primary.
    """
    w_row, gate, sign = eql.get_species_weight_and_gate(s_idx)
    raw = (sign * w_row).detach()
    gates = gate_values(gate)
    return raw, gates, raw * gates


def hill_shape_params(eql):
    """(n_inc, K_inc, n_dec, K_dec): Hill shape parameters per form, in library order."""
    shape = {'inc': ([], []), 'dec': ([], [])}
    for fn, (form, _) in zip(eql.all_hill_funcs, eql.hill_slots):
        shape[form][0].append(fn.n.item())
        shape[form][1].append(fn.K.item())
    return (np.array(shape['inc'][0]), np.array(shape['inc'][1]),
            np.array(shape['dec'][0]), np.array(shape['dec'][1]))


def extract_params(binn, full=True):
    """
    One dict per reported equation (1 if mcas, else one per species).

    Always contains 'raw_w_unscaled' and 'effective_unscaled' (length
    total_features). With full=True, also the gates, the library terms, the
    coefficients split into polynomial / increasing-Hill / decreasing-Hill
    blocks, and the Hill shape parameters.
    """
    eql = binn.reaction.eql_layer
    num_poly, num_hill = eql.num_poly_features, eql.num_hill_features
    poly_terms, hill_term_list = generate_terms(binn)
    forms = np.array([form for form, _ in eql.hill_slots])
    empty = np.array([])

    if full and num_hill > 0:
        ns_inc, Ks_inc, ns_dec, Ks_dec = hill_shape_params(eql)
    else:
        ns_inc = Ks_inc = ns_dec = Ks_dec = empty

    results = []
    for s_idx in range(binn.n_equations):
        raw, gates, effective = (v.cpu().numpy().reshape(-1)
                                 for v in effective_weights(eql, s_idx))
        params = {'raw_w_unscaled': raw, 'effective_unscaled': effective}

        if full:
            hill_block = effective[num_poly:num_poly + num_hill]
            params.update({
                'raw_w': raw,
                'gates': gates,
                'effective': effective,
                'num_poly': num_poly,
                'num_hill': num_hill,
                'poly_terms': poly_terms,
                'hill_terms': hill_term_list,
                'poly_coeffs_unscaled': effective[:num_poly] if num_poly else empty,
                'hill_inc_unscaled': hill_block[forms == 'inc'] if num_hill else empty,
                'hill_dec_unscaled': hill_block[forms == 'dec'] if num_hill else empty,
                'ns_inc': ns_inc, 'Ks_inc': Ks_inc,
                'ns_dec': ns_dec, 'Ks_dec': Ks_dec,
            })
        results.append(params)
    return results