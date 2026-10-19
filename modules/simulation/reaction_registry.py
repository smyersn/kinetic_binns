import itertools

"""
Plain-data reaction metadata -- param names/order, whether a reaction is
mass-conserving, and default sim settings (T, early_stop, dt_cap).
Deliberately has ZERO third-party imports (no numpy, torch, numba) so
job-submission scripts running on the login node -- without the conda env
activated -- can import this to validate/build configs.

The actual reaction functions and IC builders (which need numpy) live in
reaction_library.py, which imports REACTION_SPECS from here and adds
"fn"/"ic_builder" to build the full REACTION_REGISTRY used by
simulation.py on compute nodes.

dt_cap: upper bound on the integration timestep set by the REACTION
    timescale (how fast the kinetics move). This is SEPARATE from the
    diffusion CFL limit, which simulation.py computes from the diffusion
    coefficients and takes the min of. The old hardcoded dt=0.0001
    conflated the two: it was simultaneously too LOOSE for high-diffusion
    systems (Brusselator/Schnakenberg diverged to NaN) and ~20x too
    STRICT for the low-diffusion parameter sets now in use, which made
    every sweep run 20x more steps than necessary.

T and dt_cap for the four non-conserving reactions were verified by
direct simulation against the low-diffusion coefficients in
generate_training_data.py. They are NOT valid for the older textbook
diffusion values (Du~1, Dv~8-10) -- if you change diff_coeffs_pool, these
need rechecking.
"""

def library_size(species, degree, duplicates, include_poly=True,
                 include_increasing_hill=True, include_decreasing_hill=True,
                 include_constant=False, mcas=False):
    """Total L0 gate count: total_features * n_free. Mirrors EQLLayer's
    feature construction -- if you change the library there, change it here.

    include_constant admits the all-zero exponent tuple (a constant term),
    matching PolynomialFeatures' own flag."""
    n_poly = 0
    if include_poly:
        lowest = 0 if include_constant else 1
        n_poly = duplicates * sum(
            1 for p in itertools.product(range(degree + 1), repeat=species)
            if lowest <= sum(p) <= degree)

    n_forms = int(include_increasing_hill) + int(include_decreasing_hill)
    n_hill = duplicates * (species + species * (species - 1)) * n_forms

    n_free = 1 if mcas else species
    return (n_poly + n_hill) * n_free

REACTION_SPECS = {
    # --- Mass-conserving ---
    "hill_poly": {"param_keys": ["a", "b", "k", "n"], "conserved": True,
                  "T": 50, "early_stop": True, "dt_cap": 0.0001},
    "poly_poly": {"param_keys": ["a", "b"], "conserved": True,
                  "T": 50, "early_stop": True, "dt_cap": 0.0001},
    "hill_hill": {"param_keys": ["a", "b", "k1", "n1", "k2", "n2"], "conserved": True,
                  "T": 50, "early_stop": True, "dt_cap": 0.0001},
    "poly_hill": {"param_keys": ["a", "b", "k", "n"], "conserved": True,
                  "T": 50, "early_stop": True, "dt_cap": 0.0001},

    # --- Non-conserving ---
    "fitzhugh_nagumo": {"param_keys": ["a", "b", "eps"], "conserved": False,
                        "T": 200, "early_stop": False, "dt_cap": 0.01},
    "gray_scott": {"param_keys": ["feed", "kill"], "conserved": False,
                   "T": 8000, "early_stop": False, "dt_cap": 0.5},
    "brusselator": {"param_keys": ["a", "b"], "conserved": False,
                    "T": 60, "early_stop": False, "dt_cap": 0.002},
    "schnakenberg": {"param_keys": ["a", "b"], "conserved": False,
                     "T": 300, "early_stop": False, "dt_cap": 0.005},

    # --- Three-species, mass-conserving (Sostar & Dzeroski, minimal MCRD
    # networks for cell polarity, Table 2 models 1-3) ---
    # u1 + u2 + u3 is conserved, but each species has its own equation, so
    # these are NOT the two-state +F/-F (mcas) format: train with mcas=False.
    # u_total is the total concentration; it sets the initial condition (the
    # homogeneous steady state with that total) and is ignored by the
    # reaction itself. "species" defaults to 2 for every entry without it.
    # Defaults verified by direct simulation with the parameters in
    # generate_three_species_data.py: spots by t ~ 45, slow coarsening after.
    "mcrd3_m1": {"param_keys": ["k1", "k15", "k16", "k20", "u_total"], "conserved": True,
                 "species": 3, "T": 100, "early_stop": False, "dt_cap": 0.001},
    "mcrd3_m2": {"param_keys": ["k1", "k6", "k9", "k23", "u_total"], "conserved": True,
                 "species": 3, "T": 100, "early_stop": False, "dt_cap": 0.001},
    "mcrd3_m3": {"param_keys": ["k1", "k6", "k9", "k11", "u_total"], "conserved": True,
                 "species": 3, "T": 100, "early_stop": False, "dt_cap": 0.001},
}