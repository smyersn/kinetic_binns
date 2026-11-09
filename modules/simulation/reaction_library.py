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


def barkley(uv, params):
    """Barkley excitable medium. Params: a, b, eps, k. Polynomial (degree 3),
    u in [0, 1], v >= 0; spiral waves from a broken wavefront. k is an overall
    rate (time unit): it scales every coefficient without changing the patterns."""
    u, v = uv[:, 0], uv[:, 1]
    a, b, eps, k = params
    Fu = k * u * (1 - u) * (u - (v + b) / a) / eps
    Fv = k * (u - v)
    return Fu, Fv


def oregonator(uv, params):
    """Two-variable Oregonator (Belousov-Zhabotinsky). Params: eps, f, q.
    Positive concentrations; excitable spiral waves for f > ~1 + sqrt(2)."""
    u, v = uv[:, 0], uv[:, 1]
    eps, f, q = params
    Fu = (u - u**2 - f * v * (u - q) / (u + q)) / eps
    Fv = u - v
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


def ode_limit_cycle(reaction, params, start, h=1e-3, settle=200.0, max_time=2000.0, n_samples=400):
    """
    One period of the well-mixed (no diffusion) limit cycle of a two-species
    oscillator, as (u_samples, v_samples, period), sampled at n_samples equal
    time steps starting from an upward crossing of u through start[0].
    start: the homogeneous steady state; the ODE is kicked 1% off it, run for
    `settle` time units onto the cycle, then followed for one period.
    Raises ValueError if the steady state does not oscillate.
    """
    u, v = start[0] * 1.01, start[1]

    def step(u, v):
        fu, fv = reaction(np.array([[u, v]]), params)
        return u + h * float(fu[0]), v + h * float(fv[0])

    for _ in range(int(settle / h)):
        u, v = step(u, v)
    us, vs, crossings, t, prev = [], [], [], 0.0, u
    while len(crossings) < 2:
        u, v = step(u, v)
        t += h
        if prev < start[0] <= u:
            crossings.append(t)
        if crossings:
            us.append(u)
            vs.append(v)
        prev = u
        if t > max_time:
            raise ValueError("no limit cycle: the steady state does not oscillate for these "
                             "parameters, so a spiral initial condition cannot be built "
                             "(use parameters past the Hopf bifurcation).")
    idx = np.linspace(0, len(us) - 1, n_samples).astype(int)
    return np.array(us)[idx], np.array(vs)[idx], crossings[1] - crossings[0]


def ic_spiral_pair(N, reaction, params, steady_state, separation=0.5, L=10.0):
    """
    Counter-rotating spiral pair for an oscillatory (Hopf) system: every
    point starts on the ODE limit cycle, at the phase given by the angle
    around core 1 minus the angle around core 2. The cores sit at
    (L/2 -+ separation*L/2, L/2). A periodic box needs zero net winding,
    which is why spirals come in opposite-handed pairs; with
    separation = 0.5 the cores are equally far apart directly and across
    the boundary, so the pair stays put.

    Compared with steady state + noise, which seeds a random number of
    cores at random places inside a bulk oscillation, this gives one clean,
    symmetric pair of spirals whose waves take over the whole domain.
    """
    lu, lv, period = ode_limit_cycle(reaction, params, steady_state)
    x = np.linspace(0, L, N)                      # the simulator's grid coordinates
    X, Y = np.meshgrid(x, x)
    d = separation * L / 2
    theta1 = np.arctan2(Y - L / 2, X - (L / 2 - d))
    theta2 = np.arctan2(Y - L / 2, X - (L / 2 + d))
    phase = np.mod(theta1 - theta2, 2 * np.pi) / (2 * np.pi)          # 0..1 around the cycle
    k = np.minimum((phase * len(lu)).astype(int), len(lu) - 1)
    print(f"Spiral-pair IC: limit-cycle period {period:.2f}, u on cycle in "
          f"[{lu.min():.3f}, {lu.max():.3f}], cores {2 * d:.2f} apart", flush=True)
    return lu[k].astype(np.float64), lv[k].astype(np.float64)


def ode_rest_state(reaction, params, start=(0.1, 0.1), h=0.001, max_time=1000.0, tol=1e-8):
    """Stable well-mixed rest state of a two-species excitable system, found by
    integrating the ODE from `start` until it stops moving. The default start
    avoids (0, 0), which is a (non-rest) fixed point of the Oregonator."""
    u, v = start
    for _ in range(int(max_time / h)):
        fu, fv = reaction(np.array([[u, v]]), params)
        fu, fv = float(fu[0]), float(fv[0])
        u, v = u + h * fu, v + h * fv
        if abs(fu) < tol and abs(fv) < tol:
            return u, v
    raise ValueError("no stable rest state reached: a broken-wavefront spiral needs an "
                     "EXCITABLE medium (stable rest state), not an oscillatory one.")


def ic_broken_wavefront(N, reaction, params, length_frac=0.5, width=0.4, refractory_depth=1.0,
                        L=10.0, excited_u=None, refractory_v=None):
    """
    Spiral initiation in an EXCITABLE medium (FitzHugh-Nagumo here): the
    classic broken-wavefront ("cross-field") protocol. Everything starts at
    rest except
      * an excited strip (fast variable on its other branch, u = -sign(u_rest) * 1.2)
        of the given width, spanning the middle length_frac of the domain at y = L/2;
      * a refractory band of depth refractory_depth directly below it (slow
        variable pushed toward the excited side), so the wave can only travel up.
    Both free ends of the strip curl into spirals: a counter-rotating pair
    (a periodic box needs zero net winding), in the middle of the domain, that
    keeps emitting tightly wound waves for as long as the run lasts.
    excited_u / refractory_v: the excited fast-variable level and refractory
    slow-variable level. None = the FitzHugh-Nagumo defaults (other branch of
    the cubic; slow variable pushed toward it). Positive-concentration models
    (oregonator) give their own via REACTION_SPECS[...]["wavefront_ic"].
    """
    u_r, v_r = ode_rest_state(reaction, params)
    u_ex = -np.sign(u_r) * 1.2 if excited_u is None else excited_u
    v_ref = v_r + 0.6 * (u_ex - u_r) if refractory_v is None else refractory_v
    x = np.linspace(0, L, N)                      # the simulator's grid coordinates
    X, Y = np.meshgrid(x, x)
    span = np.abs(X - L / 2) < length_frac * L / 2
    u = np.full((N, N), u_r, dtype=np.float64)
    v = np.full((N, N), v_r, dtype=np.float64)
    u[span & (Y > L / 2) & (Y < L / 2 + width)] = u_ex
    v[span & (Y <= L / 2) & (Y > L / 2 - refractory_depth)] = v_ref
    print(f"Broken-wavefront IC: rest ({u_r:.3f}, {v_r:.3f}), excited u = {u_ex:.2f}, "
          f"strip length {length_frac * L:.2f}", flush=True)
    return u, v


def ic_oregonator_rest(N, p, noise=0.001):
    """Oregonator rest state (the positive fixed point, u = v) + small noise."""
    u_r, v_r = ode_rest_state(oregonator, [p["eps"], p["f"], p["q"]])
    return ic_homogeneous_plus_noise(N, u_r, v_r, noise=noise)


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
    "barkley": {
        **REACTION_SPECS["barkley"],
        "fn": barkley,
        # Default: the rest state (0, 0) plus small POSITIVE noise (stays at rest
        # -- the medium is excitable). Use "ic": "broken_wavefront" for spirals.
        "ic_builder": lambda N, p: ic_homogeneous_plus_noise(N, 0.005, 0.005, noise=0.01),
    },
    "oregonator": {
        **REACTION_SPECS["oregonator"],
        "fn": oregonator,
        # Default: homogeneous rest state + small noise (stays at rest -- the
        # medium is excitable). Use "ic": "broken_wavefront" for spirals.
        "ic_builder": lambda N, p: ic_oregonator_rest(N, p),
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