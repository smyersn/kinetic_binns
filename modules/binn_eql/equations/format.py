"""Human-readable forms of the learned reaction terms."""
from modules.binn_eql.equations.extract import extract_params


def _monomial(coeff, powers, names):
    s = f"{float(coeff):.3f}"
    for name, k in zip(names, powers):
        if k == 1:
            s += f" * {name}"
        elif k > 1:
            s += f" * {name}^{k}"
    return s


def _hill(coeff, term, K, n, names, increasing):
    a = names[term[0]]
    prefix = f"{coeff:.3f}" + (f" * {names[term[1]]}" if len(term) == 2 else "")
    if increasing:
        return f"{prefix} * {a}^{n:.3f} / (1 + {K:.3f} * {a}^{n:.3f})"
    return f"{prefix} * [1 / (1 + {K:.3f} * {a}^{n:.3f})]"


def generate_equation(binn, eps=1e-12, species_names=None):
    """
    One {'species': label, 'terms': [str, ...]} per reported equation.
    Terms with |coefficient| <= eps are omitted. With mcas the single
    equation is labelled F(u, v): it is added to species 0 and subtracted
    from species 1.
    """
    names = species_names or binn.species_names
    dup = binn.duplicates
    equations = []

    for eq_idx, p in enumerate(extract_params(binn, full=True)):
        terms = [_monomial(c, powers, names)
                 for powers, c in zip(p['poly_terms'] * dup, p['poly_coeffs_unscaled'])
                 if abs(c) > eps]

        for form, increasing in (('inc', True), ('dec', False)):
            for term, c, K, n in zip(p['hill_terms'] * dup, p[f'hill_{form}_unscaled'],
                                     p[f'Ks_{form}'], p[f'ns_{form}']):
                if abs(c) > eps:
                    terms.append(_hill(float(c), term, float(K), float(n), names, increasing))

        label = f"F({', '.join(names)})" if binn.mcas else names[eq_idx]
        equations.append({'species': label, 'terms': terms})
    return equations


def equations_as_strings(binn, eps=1e-12, species_names=None):
    """Flat list of lines: a header per equation followed by its terms."""
    lines = []
    for eq in generate_equation(binn, eps=eps, species_names=species_names):
        lines.append(f"{eq['species']} =" if binn.mcas else f"d{eq['species']}/dt =")
        lines.extend(eq['terms'] or ['0'])
    return lines


def write_equations(path, binn, header):
    """Append `header` and the current equations to a text file."""
    with open(path, 'a') as f:
        f.write(f"{header}\n")
        for line in equations_as_strings(binn):
            f.write(f"{line}\n")