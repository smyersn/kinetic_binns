import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

import time
from numba import njit, prange
from modules.utils.imports import *
from modules.simulation.animation import animate_u_array
from modules.simulation.paper_reaction_functions import (hill_poly, poly_poly,
                                                         hill_hill, poly_hill)
from modules.simulation.reaction_functions import (wave_pinning, turing_type,
                                                   custom_equation)
from modules.simulation.nonconserved_reaction_functions import (
    as_two_species_reaction, fitzhugh_nagumo, gray_scott, brusselator, schnakenberg,
    ic_localized_seed, ic_homogeneous_plus_noise, ic_pulse_stimulus)
from modules.utils.format_data import format_u_array_to_training_data

@njit(parallel=True)
def fast_laplacian(mat, out_lap):
    """Computes the 5-point stencil Laplacian with periodic boundaries."""
    rows, cols = mat.shape
    for i in prange(rows):
        for j in range(cols):
            ip = (i + 1) % rows
            im = (i - 1) % rows
            jp = (j + 1) % cols
            jm = (j - 1) % cols
            out_lap[i, j] = mat[ip, j] + mat[im, j] + mat[i, jp] + mat[i, jm] - 4.0 * mat[i, j]

def simulate_reaction_cpu(reaction, params, diff_coeffs, initial_condition=None,
                           T=50, early_stop=True, min_steady_state_check_frac=0.1):
    """
    reaction: callable (uv, params) -> (Fu, Fv), two independent NxN-flattened
        reaction terms. Wrap legacy single-output (mass-conserving) reaction
        functions with as_two_species_reaction() first.
    initial_condition: optional (u0, v0) tuple of NxN arrays. If None, falls
        back to the original wave_pinning-style random perturbation.
    min_steady_state_check_frac: fraction of T to wait before the early-stop
        check activates. Turing-unstable systems start almost exactly at
        their (near-)homogeneous fixed point, so the raw frame-to-frame diff
        can sit below ss_tolerance for a while BEFORE the instability has
        visibly grown -- checking from t=0 would falsely declare "steady
        state" before any pattern forms. Ignored when early_stop=False.
    """
    # --- Set Parameters (Matching original function) ---
    dim = 2
    species = 2
    N = 200                           # grid points
    L = 10                            # domain length
    ss_tolerance = 0.025

    dx = L / N
    dt = 0.0001                       # time step -- tuned for wave_pinning's
                                       # timescale. The new systems below use
                                       # classic dimensionless parameterizations
                                       # and may need a different dt/T -- do a
                                       # short test run and check the printed
                                       # progress before committing to a sweep.
    nits = int(T / dt)                # number of time steps
    du, dv = diff_coeffs              # diffusion rates
    inv_dx2 = 1.0 / (dx**2)

    # --- Initial Conditions ---
    if initial_condition is not None:
        u, v = initial_condition
        u, v = u.copy(), v.copy()
    else:
        u0, v0 = 1.0, 1.0246
        u = (np.random.rand(N, N) + 0.5) * u0
        v = np.ones((N, N)) * v0

    # --- Storage ---
    half_sec_nits = int(0.5 / dt)
    half_secs = int(nits / half_sec_nits) + 1

    # Pre-allocate Animator-friendly shape: (Time, Y, X, Species)
    u_array = np.zeros((half_secs, N, N, species), dtype=np.float32)
    x_array = np.linspace(0, L, N, dtype=np.float32)
    t_array = np.arange(0, T + 0.5, 0.5, dtype=np.float32)

    # Pre-allocate arrays for Numba to avoid memory reallocation in the loop
    lap_u = np.zeros((N, N), dtype=np.float64)
    lap_v = np.zeros((N, N), dtype=np.float64)

    i = 0
    min_steady_state_check_time = min_steady_state_check_frac * T

    # --- Simulation loop ---
    for t in range(nits + 1):

        # Update storage every half second
        if t % half_sec_nits == 0:
            u_array[i, :, :, 0] = u
            u_array[i, :, :, 1] = v

            # Stop simulation if it reaches steady state (only checked after
            # the warmup window, see docstring above)
            if early_stop and t > 0 and (t * dt) > min_steady_state_check_time:
                diff = np.max(np.abs(u_array[i] - u_array[i - 1]))

                if diff < ss_tolerance:
                    print(f'Steady state at t = {t * dt:.2f}', flush=True)
                    u_array = u_array[:i + 1]
                    t_array = t_array[:i + 1]
                    break
            i += 1

        if t == nits:
            break  # Reached the end, no need to step physics

        # Compute Laplacian using the Numba kernel
        fast_laplacian(u, lap_u)
        fast_laplacian(v, lap_v)

        # Reaction terms -- independent per species, no conservation assumed
        uv = np.column_stack((u.flatten(), v.flatten()))
        Fu, Fv = reaction(uv, params)

        if hasattr(Fu, 'detach'):
            Fu = Fu.detach().cpu().numpy()
        if hasattr(Fv, 'detach'):
            Fv = Fv.detach().cpu().numpy()

        Fu = Fu.reshape(N, N)
        Fv = Fv.reshape(N, N)

        # Euler update
        u = u + dt * (du * lap_u * inv_dx2 + Fu)
        v = v + dt * (dv * lap_v * inv_dx2 + Fv)

        if t % (nits // 10) == 0 and t > 0:
            print(f"Progress: {(t / nits) * 100:.0f}%", flush=True)

    return torch.tensor(u_array), torch.tensor(x_array), torch.tensor(t_array)


if __name__ == '__main__':
    import json

    # 1. Load the single JSON config argument
    config_path = str(sys.argv[1])

    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Extract base parameters from the dictionary
    save_path = config["save_path"]
    Du, Dv = float(config["du"]), float(config["dv"])
    diff_coeffs = (Du, Dv)
    reaction_str = config["reaction"]

    N = 200  # must match simulate_reaction_cpu's N above -- used to build ICs here

    # 3. Dynamically map the reaction string to a function, its params, its
    # initial condition, and reasonable default simulation settings. T and
    # early_stop can be overridden per-config via "T" / "early_stop" keys.
    if reaction_str == "wave_pinning":
        reaction_fn = as_two_species_reaction(wave_pinning)
        params = [config.get("a", 0), config.get("b", 0), config.get("k", 0)]
        initial_condition, T_default, early_stop_default = None, 50, True
    elif reaction_str == "turing_type":
        reaction_fn = as_two_species_reaction(turing_type)
        params = [config.get("a", 0), config.get("b", 0), config.get("k", 0)]
        initial_condition, T_default, early_stop_default = None, 50, True
    elif reaction_str == "custom_equation":
        reaction_fn = as_two_species_reaction(custom_equation)
        params = [config.get("a", 0), config.get("b", 0), config.get("k", 0)]
        initial_condition, T_default, early_stop_default = None, 50, True
    elif reaction_str == "hill_poly":
        reaction_fn = as_two_species_reaction(hill_poly)
        params = [config['a'], config['b'], config['k'], config['n']]
        initial_condition, T_default, early_stop_default = None, 50, True
    elif reaction_str == "poly_poly":
        reaction_fn = as_two_species_reaction(poly_poly)
        params = [config['a'], config['b']]
        initial_condition, T_default, early_stop_default = None, 50, True
    elif reaction_str == "hill_hill":
        reaction_fn = as_two_species_reaction(hill_hill)
        params = [config['a'], config['b'], config['k1'], config['n1'], config['k2'], config['n2']]
        initial_condition, T_default, early_stop_default = None, 50, True
    elif reaction_str == "poly_hill":
        reaction_fn = as_two_species_reaction(poly_hill)
        params = [config['a'], config['b'], config['k'], config['n']]
        initial_condition, T_default, early_stop_default = None, 50, True

    # --- Non-mass-conserving reactions ---
    elif reaction_str == "fitzhugh_nagumo":
        reaction_fn = fitzhugh_nagumo
        params = [config['a'], config['b'], config['eps']]
        initial_condition = ic_pulse_stimulus(N)
        T_default, early_stop_default = 100, False
    elif reaction_str == "gray_scott":
        reaction_fn = gray_scott
        params = [config['feed'], config['kill']]
        initial_condition = ic_localized_seed(N)
        T_default, early_stop_default = 2000, False
    elif reaction_str == "brusselator":
        reaction_fn = brusselator
        a, b = config['a'], config['b']
        params = [a, b]
        initial_condition = ic_homogeneous_plus_noise(N, u_ss=a, v_ss=b / a)
        T_default, early_stop_default = 200, False
    elif reaction_str == "schnakenberg":
        reaction_fn = schnakenberg
        a, b = config['a'], config['b']
        u_ss, v_ss = a + b, b / (a + b) ** 2
        params = [a, b]
        initial_condition = ic_homogeneous_plus_noise(N, u_ss=u_ss, v_ss=v_ss)
        T_default, early_stop_default = 200, False
    else:
        raise ValueError(f"Unknown reaction function specified: {reaction_str}")

    T = config.get("T", T_default)
    early_stop = config.get("early_stop", early_stop_default)

    # 4. Create save name dynamically based on the input JSON filename
    json_filename = os.path.basename(config_path)
    base_name = json_filename.replace('.json', '')
    save_name = os.path.join(save_path, base_name)

    print(f"Starting simulation for: {save_name} (T={T}, early_stop={early_stop})", flush=True)

    # 5. Simulate and animate
    u_array, x_array, t_array = simulate_reaction_cpu(
        reaction_fn, params, diff_coeffs,
        initial_condition=initial_condition, T=T, early_stop=early_stop)
    animate_u_array(u_array, t_array, name=f'{save_name}.gif', titles=("u", "v"))

    # 6. Reformat to training data and save
    training_data = format_u_array_to_training_data(u_array, x_array, t_array)
    torch.save({'training_data': training_data}, f'{save_name}.pt')

    print(f"Successfully saved data to {save_name}.pt", flush=True)