"""
All reaction functions used to generate and simulate reaction-diffusion
training data, in one place.

INTERFACE CONTRACT
-------------------
Every reaction function below has the exact same signature and return type:

    reaction_fn(uv, params) -> (Fu, Fv)

`uv` is an (N, 2) array of [u, v] pairs; `params` is a plain list of
scalars in the order given in REACTION_REGISTRY's "param_keys" below.
Fu and Fv are always both returned -- there is no separate "conserved"
calling convention to remember at the call site. Mass-conserving systems
enforce Fv = -Fu directly in the function body.

ORGANIZATION
------------
    1. MASS-CONSERVING REACTIONS -- total u + v is conserved (Fv = -Fu).
    2. NON-CONSERVING REACTIONS  -- independent Fu, Fv.
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
# 3. INITIAL CONDITION BUILDERS
# ===========================================================
# Each returns a (u0, v0) tuple of NxN arrays. Signature is uniform --
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


def ic_homogeneous_plus_noise(N, u_ss, v_ss, noise=0.01):
    """Turing convention (Brusselator, Schnakenberg): perturb the
    homogeneous fixed point with small random noise and let the
    instability grow. `noise` should be small relative to u_ss/v_ss."""
    u = u_ss + noise * (np.random.rand(N, N) - 0.5)
    v = v_ss + noise * (np.random.rand(N, N) - 0.5)
    return u, v


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
        "ic_builder": lambda N, p: ic_localized_seed(N),
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