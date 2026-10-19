"""
All reaction functions used to generate and simulate reaction-diffusion
training data, in one place.

INTERFACE CONTRACT
-------------------
Every reaction function below has the exact same signature and return type:

    reaction_fn(uv, params) -> (F_1, ..., F_S)

`uv` is an (N, S) array of concentrations, one column per species (S = 2
unless the REACTION_SPECS entry says "species": 3); `params` is a plain list
of scalars in the order given in REACTION_REGISTRY's "param_keys" below.
One reaction term per species is always returned -- (Fu, Fv) for the
two-species systems -- so there is no separate "conserved" calling
convention to remember at the call site. Two-state mass-conserving systems
enforce Fv = -Fu directly in the function body.

ORGANIZATION
------------
    1. MASS-CONSERVING REACTIONS -- total u + v is conserved (Fv = -Fu).
    2. NON-CONSERVING REACTIONS  -- independent Fu, Fv.
    2b. THREE-SPECIES MASS-CONSERVING REACTIONS -- u1 + u2 + u3 conserved,
       one equation per species.
    3. INITIAL CONDITION BUILDERS
    4. REACTION_REGISTRY          -- full registry for compute-node code
                                      (paper_simulation.py etc.): each
                                      REACTION_SPECS entry (imported from
                                      reaction_registry.py) plus the
                                      actual "fn" and "ic_builder".

NOTE: this module imports numpy and is NOT safe to import on a login node
without the conda env active. Job-submission scripts that only need
param names/conservation status/default T should import REACTION_SPECS
from reaction_registry.py instead -- see that file's docstring.
"""

import numpy as np

from modules.simulation.reaction_registry import REACTION_SPECS


# ===========================================================
# 1. MASS-CONSERVING REACTIONS (u + v conserved: Fv = -Fu)
# ===========================================================
def hill_poly(uv, params):
    """Params: a, b, k, n."""
    u, v = uv[:, 0], uv[:, 1]
    a, b, k, n = params
    F = (a * u**n * v) / (1 + k * u**n) - b * u
    return F, -F


def poly_poly(uv, params):
    """Params: a, b."""
    u, v = uv[:, 0], uv[:, 1]
    a, b = params
    F = a * u**2 * v - b * u
    return F, -F


def hill_hill(uv, params):
    """Params: a, b, k1, n1, k2, n2."""
    u, v = uv[:, 0], uv[:, 1]
    a, b, k1, n1, k2, n2 = params
    F = (a * u**n1 * v) / (1 + k1 * u**n1) - (b * u**n2) / (1 + k2 * u**n2)
    return F, -F


def poly_hill(uv, params):
    """Params: a, b, k, n."""
    u, v = uv[:, 0], uv[:, 1]
    a, b, k, n = params
    F = a * u**2 * v - (b * u**n) / (1 + k * u**n)
    return F, -F


# ===========================================================
# 2. NON-CONSERVING REACTIONS (independent Fu, Fv)
# ===========================================================
def fitzhugh_nagumo(uv, params):
    """Excitable medium. Params: a, b, eps. Traveling pulses, spiral waves under the right stimulus."""
    u, v = uv[:, 0], uv[:, 1]
    a, b, eps = params
    Fu = u - u**3 - v + a
    Fv = eps * (u - b * v)
    return Fu, Fv


def gray_scott(uv, params):
    """Autocatalytic v; u is fed/replenished. Params: feed, kill.
    Spots, stripes, mitosis-like splitting, or chaos depending on the pair."""
    u, v = uv[:, 0], uv[:, 1]
    feed, kill = params
    uv2 = u * v**2
    Fu = -uv2 + feed * (1 - u)
    Fv = uv2 - (feed + kill) * v
    return Fu, Fv


def brusselator(uv, params):
    """Autocatalytic chemical oscillator. Params: a, b.
    Turing spots/stripes near the bifurcation, oscillatory/wave behavior further from it."""
    u, v = uv[:, 0], uv[:, 1]
    a, b = params
    u2v = u**2 * v
    Fu = a - (b + 1) * u + u2v
    Fv = b * u - u2v
    return Fu, Fv


def schnakenberg(uv, params):
    """Minimal two-parameter Turing system. Params: a, b. Stationary spots/stripes."""
    u, v = uv[:, 0], uv[:, 1]
    a, b = params
    u2v = u**2 * v
    Fu = a - u + u2v
    Fv = b - u2v
    return Fu, Fv


# ===========================================================
# 2b. THREE-SPECIES MASS-CONSERVING REACTIONS
# ===========================================================
# Sostar & Dzeroski, "Minimal mass-conserving reaction-diffusion networks
# for cell polarity", Table 2, models 1-3: the minimal four-reaction,
# second-order mass-action networks that polarize. Species are u1, u2, u3
# (columns 0, 1, 2; u, v, w in the BINN's naming). Each reaction only moves
# mass between species, so F1 + F2 + F3 = 0 identically. Rate constants keep
# the paper's numbering from its 24-reaction library (Table 1). The last
# parameter, u_total, is for the initial condition and is ignored here.
# In models 1-3, u2 is the fast-diffusing (cytosolic) component.

def mcrd3_m1(uv, params):
    """U1 -> U2 (k1), U2+U3 -> U1+U2 (k15), U2+U3 -> U1+U3 (k16), 2U1 -> U1+U3 (k20).
    Params: k1, k15, k16, k20, u_total."""
    u1, u2, u3 = uv[:, 0], uv[:, 1], uv[:, 2]
    k1, k15, k16, k20 = params[:4]
    F1 = -k1 * u1 + (k15 + k16) * u2 * u3 - k20 * u1**2
    F2 = k1 * u1 - k16 * u2 * u3
    F3 = -k15 * u2 * u3 + k20 * u1**2
    return F1, F2, F3


def mcrd3_m2(uv, params):
    """U1 -> U2 (k1), U3 -> U2 (k6), U1+U2 -> U1+U3 (k9), 2U3 -> U1+U3 (k23).
    Params: k1, k6, k9, k23, u_total."""
    u1, u2, u3 = uv[:, 0], uv[:, 1], uv[:, 2]
    k1, k6, k9, k23 = params[:4]
    F1 = k23 * u3**2 - k1 * u1
    F2 = k1 * u1 + k6 * u3 - k9 * u1 * u2
    F3 = -k23 * u3**2 - k6 * u3 + k9 * u1 * u2
    return F1, F2, F3


def mcrd3_m3(uv, params):
    """U1 -> U2 (k1), U1+U2 -> U1+U3 (k9), U3 -> U2 (k6), U1+U3 -> 2U1 (k11).
    Params: k1, k6, k9, k11, u_total."""
    u1, u2, u3 = uv[:, 0], uv[:, 1], uv[:, 2]
    k1, k6, k9, k11 = params[:4]
    F1 = k11 * u1 * u3 - k1 * u1
    F2 = k1 * u1 + k6 * u3 - k9 * u1 * u2
    F3 = k9 * u1 * u2 - k6 * u3 - k11 * u1 * u3
    return F1, F2, F3


def _numeric_jacobian(reaction, u, params, h=1e-7):
    """d F_i / d u_j of reaction at the single state u (length S), by central differences."""
    S = len(u)
    J = np.zeros((S, S))
    for j in range(S):
        step = h * max(1.0, abs(u[j]))
        up, um = u.copy(), u.copy()
        up[j] += step
        um[j] -= step
        Fp = np.array(reaction(up[None, :], params), dtype=float).reshape(S)
        Fm = np.array(reaction(um[None, :], params), dtype=float).reshape(S)
        J[:, j] = (Fp - Fm) / (2 * step)
    return J


def mass_conserving_steady_state(reaction, params, u_total, species, n_starts=200, seed=0):
    """
    The homogeneous steady state u* (F(u*) = 0, sum(u*) = u_total, all u* > 0)
    that is stable WITHOUT diffusion, as in the paper: the pattern then grows
    from a Turing (diffusion-driven) instability of a state the reaction alone
    would return to.

    Newton's method from many fixed (seeded, so reproducible) starting points,
    with the last equation replaced by the mass constraint (F sums to zero, so
    that equation is redundant). A conserved system always has one zero
    eigenvalue (shifting total mass); "stable" means every OTHER eigenvalue
    has a negative real part.

    Raises ValueError if no stable positive state exists -- with those
    parameters there is nothing for a Turing pattern to grow from.
    """
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(n_starts):
        u = rng.dirichlet(np.ones(species)) * u_total
        for _ in range(100):
            F = np.array(reaction(u[None, :], params), dtype=float).reshape(species)
            F[-1] = u.sum() - u_total
            J = _numeric_jacobian(reaction, u, params)
            J[-1, :] = 1.0
            try:
                du = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            u = u + du
            if not np.all(np.isfinite(u)):
                break
            if np.max(np.abs(du)) < 1e-13 * max(1.0, u_total):
                break
        F = np.array(reaction(u[None, :], params), dtype=float).reshape(species)
        if (np.all(np.isfinite(u)) and np.all(u > 1e-8 * u_total)
                and np.max(np.abs(F)) < 1e-9 * max(1.0, u_total)
                and not any(np.allclose(u, v, rtol=1e-6) for v in found)):
            found.append(u)

    stable = []
    for u in found:
        eig = np.sort(np.linalg.eigvals(_numeric_jacobian(reaction, u, params)).real)[::-1]
        if eig[1] < -1e-8:          # eig[0] ~ 0 is the conserved-mass direction
            stable.append(u)
    if not stable:
        raise ValueError(f"No positive steady state stable without diffusion for "
                         f"params={params}, u_total={u_total} (found {len(found)} unstable).")
    if len(stable) > 1:
        print(f"WARNING: {len(stable)} stable steady states; using the first: "
              f"{[np.round(u, 4).tolist() for u in stable]}", flush=True)
    return stable[0]


# ===========================================================
# 3. INITIAL CONDITION BUILDERS
# ===========================================================
# Each returns a tuple of NxN arrays, one per species ((u0, v0) for the
# two-species systems). Signature is uniform --
# ic_builder(N, params_dict) -- even where a given builder ignores
# params_dict, so REACTION_REGISTRY can call every entry the same way.

def ic_random_perturbation(N, u0=1.0, v0=1.0246, amplitude=0.5):
    """Original wave_pinning-style IC: uniform v, u perturbed by uniform random noise."""
    u = (np.random.rand(N, N) + amplitude) * u0
    v = np.ones((N, N)) * v0
    return u, v


def ic_localized_seed(N, background=(1.0, 0.0), seed_value=(0.5, 0.4), seed_frac=0.1, noise=0.01):
    """Gray-Scott convention: uniform background with a small central seed
    square, plus a little noise to break exact symmetry (otherwise a spot
    can stay perfectly stationary instead of splitting).

    Gray-Scott's autocatalytic term is u*v^2 (quadratic in v), so small v
    ALWAYS decays -- ignition requires v above a threshold ~ (feed+kill)/u.
    seed_value=(0.5, 0.4) clears that threshold with margin for
    feed+kill up to ~0.2 at u=0.5 (0.4*0.5=0.2); the old default (0.5, 0.25)
    only cleared ~0.125 and silently extinguished for some parameter
    combinations in REACTION_SPECS's grid instead of igniting."""
    u = np.full((N, N), background[0], dtype=np.float64)
    v = np.full((N, N), background[1], dtype=np.float64)

    r = max(1, int(N * seed_frac / 2))
    c = N // 2
    u[c - r:c + r, c - r:c + r] = seed_value[0]
    v[c - r:c + r, c - r:c + r] = seed_value[1]

    u += noise * (np.random.rand(N, N) - 0.5)
    v += noise * (np.random.rand(N, N) - 0.5)
    return u, v


def ic_square_seeds(N, background=(1.0, 0.0), seed_value=(0.5, 0.4), seed_frac=0.1,
                    spacing_frac=0.25, noise=0.01):
    """Gray-Scott, four seeds: the ic_localized_seed square repeated at the
    corners of a square centred on the domain, plus the same small noise.

    Four seeds put four times as much of the domain into the active pattern
    early on, and the seeds' fronts collide in the middle, visiting (u, v)
    states a single expanding seed never does. Both matter for learning:
    background points (u = 1, v = 0) carry no information about the reaction.

    spacing_frac is the distance between neighbouring seed centres as a
    fraction of N. Keep it well below 0.5: at exactly 0.5 the seeds are
    spaced evenly across the PERIODIC boundaries too, so the whole domain is
    four identical copies of one quadrant (apart from the noise) and holds
    no more information than a single seed on a quarter-size domain.
    seed_value and seed_frac mean the same as in ic_localized_seed."""
    u = np.full((N, N), background[0], dtype=np.float64)
    v = np.full((N, N), background[1], dtype=np.float64)

    r = max(1, int(N * seed_frac / 2))
    c = N // 2
    d = int(round(N * spacing_frac / 2))      # centre-to-seed offset along each axis
    for cy in (c - d, c + d):
        for cx in (c - d, c + d):
            u[cy - r:cy + r, cx - r:cx + r] = seed_value[0]
            v[cy - r:cy + r, cx - r:cx + r] = seed_value[1]

    u += noise * (np.random.rand(N, N) - 0.5)
    v += noise * (np.random.rand(N, N) - 0.5)
    return u, v


def ic_homogeneous_plus_noise(N, u_ss, v_ss, noise=0.01):
    """Turing convention (Brusselator, Schnakenberg): perturb the
    homogeneous fixed point with small random noise and let the
    instability grow. `noise` should be small relative to u_ss/v_ss."""
    u = u_ss + noise * (np.random.rand(N, N) - 0.5)
    v = v_ss + noise * (np.random.rand(N, N) - 0.5)
    return u, v


def ic_steady_state_plus_noise(N, reaction, params, u_total, species, noise=0.01):
    """
    Mass-conserving Turing convention (the three-species polarity models):
    the homogeneous steady state with total concentration u_total that is
    stable without diffusion (mass_conserving_steady_state), times
    (1 + noise * uniform(-1/2, 1/2)) independently at every point and for
    every species. The paper uses a 30 dB perturbation (about 3% of u*);
    noise=0.01 is gentler, and the pattern still forms by t ~ 45.

    Relative rather than absolute noise keeps every species positive however
    small its steady state is.
    """
    u_ss = mass_conserving_steady_state(reaction, params, u_total, species)
    print(f"Homogeneous steady state (u_total={u_total}): {np.round(u_ss, 5).tolist()}",
          flush=True)
    return tuple(u_ss[s] * (1 + noise * (np.random.rand(N, N) - 0.5))
                 for s in range(species))


def ic_pulse_stimulus(N, rest=(0.0, 0.0), stim_u=1.0, stim_frac=0.1):
    """FitzHugh-Nagumo convention: resting state everywhere, with a
    localized stimulus strip along one edge to trigger an inward-
    propagating pulse. Spiral waves need a cross-field / broken-wavefront
    protocol instead -- not implemented here."""
    u = np.full((N, N), rest[0], dtype=np.float64)
    v = np.full((N, N), rest[1], dtype=np.float64)

    w = max(1, int(N * stim_frac))
    u[:, :w] = stim_u
    return u, v


# ===========================================================
# 4. REGISTRY -- single source of truth for dispatching on reaction name
# ===========================================================
REACTION_REGISTRY = {
    # --- Mass-conserving ---
    "hill_poly": {
        **REACTION_SPECS["hill_poly"],
        "fn": hill_poly,
        "ic_builder": lambda N, p: ic_random_perturbation(N),
    },
    "poly_poly": {
        **REACTION_SPECS["poly_poly"],
        "fn": poly_poly,
        "ic_builder": lambda N, p: ic_random_perturbation(N),
    },
    "hill_hill": {
        **REACTION_SPECS["hill_hill"],
        "fn": hill_hill,
        "ic_builder": lambda N, p: ic_random_perturbation(N),
    },
    "poly_hill": {
        **REACTION_SPECS["poly_hill"],
        "fn": poly_hill,
        "ic_builder": lambda N, p: ic_random_perturbation(N),
    },

    # --- Non-conserving ---
    "fitzhugh_nagumo": {
        **REACTION_SPECS["fitzhugh_nagumo"],
        "fn": fitzhugh_nagumo,
        "ic_builder": lambda N, p: ic_pulse_stimulus(N),
    },
    "gray_scott": {
        **REACTION_SPECS["gray_scott"],
        "fn": gray_scott,
        # Four seeds in a square (was: ic_localized_seed, one central seed).
        # Datasets made before this change used the single seed.
        "ic_builder": lambda N, p: ic_square_seeds(N),
    },
    "brusselator": {
        **REACTION_SPECS["brusselator"],
        "fn": brusselator,
        "ic_builder": lambda N, p: ic_homogeneous_plus_noise(N, u_ss=p["a"], v_ss=p["b"] / p["a"]),
    },
    "schnakenberg": {
        **REACTION_SPECS["schnakenberg"],
        "fn": schnakenberg,
        "ic_builder": lambda N, p: ic_homogeneous_plus_noise(
            N, u_ss=p["a"] + p["b"], v_ss=p["b"] / (p["a"] + p["b"]) ** 2),
    },
}


# --- Three-species, mass-conserving ---
# The IC needs the reaction itself (to find the steady state), so these are
# added in a loop rather than written out one by one.
def _mcrd3_ic(fn, keys):
    def builder(N, p):
        return ic_steady_state_plus_noise(N, fn, [p[k] for k in keys],
                                          u_total=p["u_total"], species=3)
    return builder


for _name, _fn in (("mcrd3_m1", mcrd3_m1), ("mcrd3_m2", mcrd3_m2), ("mcrd3_m3", mcrd3_m3)):
    REACTION_REGISTRY[_name] = {
        **REACTION_SPECS[_name],
        "fn": _fn,
        "ic_builder": _mcrd3_ic(_fn, REACTION_SPECS[_name]["param_keys"]),
    }