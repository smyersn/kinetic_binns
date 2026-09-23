import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

import time
from numba import njit, prange
from modules.utils.imports import *
from modules.simulation.animation import animate_u_array
from modules.simulation.reaction_library import REACTION_REGISTRY
from modules.utils.format_data import format_u_array_to_training_data

N_GRID = 200  # grid points per side -- shared between simulate_reaction_cpu
              # and the IC builders dispatched from REACTION_REGISTRY below

def simulate_uvmlp(training_data, model):
    # --- Initial Conditions ---
    dimensions = model.model.dimensions
    species = model.model.species
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    xt = training_data[:, :dimensions+1]
    t_array = torch.unique(xt[:, dimensions])
    x_array = torch.unique(xt[:, 0])

    grid_x, grid_y = torch.meshgrid(x_array, x_array, indexing='xy')
    spatial_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # --- Storage ---
    u_array = torch.zeros((len(t_array), len(x_array), len(x_array), species))

    # --- Simulation loop ---
    for n, t in enumerate(t_array):
        # We create a tensor of shape (40000, 1) filled with the scalar t
        t_column = torch.full((spatial_coords.shape[0], 1), t)

        # [x, y] + [t] -> [x, y, t]
        input_tensor = torch.cat([spatial_coords, t_column], dim=1).to(device)

        # Calculate surface
        with torch.no_grad():
            uv = model.model(input_tensor).view(len(x_array), len(x_array), species)

        u_array[n] = uv

    return u_array.cpu(), x_array.cpu(), t_array.cpu()


def simulate_feql(training_data, model, dt_cap=1e-4, cfl_safety=0.4):
    """
    Integrate the learned PDE (explicit Euler, periodic BCs) and save a
    snapshot at each of the training data's frame times.
 
    dt_cap: upper bound on the timestep from the reaction timescale. The
        step actually used is min(dt_cap, cfl_safety * dx^2 / (4 max D)),
        matching simulate_reaction_cpu -- so large learned diffusion
        coefficients cannot silently destabilize the integration, and small
        ones are not paid for with unnecessary steps.
 
    N-species general: each species' reaction term comes from its own output
    column of model.model.reaction, rather than the old [+R, -R] trick that
    hardcoded two mass-conserving species.
    """
    # --- Initial conditions ---
    dimensions = model.model.dimensions
    species = model.model.species
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    xt = training_data[:, :dimensions + 1]
    t_array = torch.unique(xt[:, dimensions])
    x_array = torch.unique(xt[:, 0])
    points = len(x_array)
 
    ic = training_data[training_data[:, dimensions] == 0]
    grid_shape = [points] * dimensions
    species_ics = [ic[:, dimensions + 1 + s] for s in range(species)]
    uv_grid = torch.stack(species_ics, dim=0).reshape(1, species, *grid_shape).float().to(device)
 
    # --- Diffusion coefficients (one per species) ---
    if model.model.diff_coeffs:
        D_list = list(model.model.diff_coeffs)
    else:
        with torch.no_grad():
            D_list = [float(d) for d in model.model.diffusion_fitter()]
    D_tensor = torch.tensor(D_list, device=device).view(1, species, 1, 1)
 
    # --- Timestep: whichever of the reaction and diffusion limits is tighter ---
    nx, ny = points, points
    L = float(xt[:, 0].max().item())
    dx = L / nx
    max_D = max(D_list)
    cfl_limit = dx ** 2 / (4 * max_D) if max_D > 0 else float('inf')
    dt = min(dt_cap, cfl_safety * cfl_limit)
 
    # --- Save schedule: the training data's own frame times ---
    # The old code saved every int(0.5 / dt) steps, which assumed frames are
    # 0.5 apart. Datasets store a fixed number of frames spread over T, so
    # for a long run (Gray-Scott, T=8000) that filled every storage slot in
    # the first fraction of the run and discarded the rest.
    save_steps = [int(round(float(t) / dt)) for t in t_array]
    nits = save_steps[-1]
    limiter = 'reaction dt_cap' if dt_cap <= cfl_safety * cfl_limit else 'diffusion CFL'
    print(f"Simulating to t={float(t_array[-1]):.4g} with dt={dt:.3e} "
          f"({limiter} is binding; CFL limit={cfl_limit:.3e} for max D={max_D:.4g}) "
          f"-> {nits:,} steps, {len(save_steps)} frames", flush=True)
 
    # --- Laplacian as a grouped convolution (one independent channel per species) ---
    laplace_kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32, device=device)
    conv = nn.Conv2d(in_channels=species, out_channels=species, kernel_size=3,
                     padding=1, groups=species, padding_mode='circular', bias=False)
    conv.weight.data = laplace_kernel.unsqueeze(0).unsqueeze(0).repeat(species, 1, 1, 1)
    conv.weight.requires_grad = False
    conv = conv.to(device)
 
    reaction = model.model.reaction.eval()
    u_array = torch.zeros((len(t_array), nx, ny, species), device=device)
 
    @torch.compile
    def physics_step(state):
        lap = conv(state) / (dx ** 2)
 
        # (1, species, H, W) -> (n_points, species) for the reaction, and back.
        # The rows keep the H-major flatten order, so transpose-then-reshape
        # recovers each channel's grid without reordering the data.
        flat = state.permute(0, 2, 3, 1).contiguous().view(-1, species)
        r_grid = reaction(flat).permute(1, 0).reshape(1, species, nx, ny)
 
        return state + dt * (D_tensor * lap + r_grid)
 
    print_interval = max(1, nits // 10)
    last_time = time.time()
    next_frame = 0
 
    with torch.no_grad():
        for step in range(nits + 1):
            # while, not if: with a coarse dt two frame times can land on the
            # same step, and every frame must still get written.
            while next_frame < len(save_steps) and save_steps[next_frame] == step:
                u_array[next_frame] = uv_grid.squeeze(0).permute(1, 2, 0)
                next_frame += 1
 
            if step == nits:
                break
            uv_grid = physics_step(uv_grid)
 
            if step == 0:
                print("Progress: 0%", flush=True)
            elif step % print_interval == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                print(f"Progress: {(step / nits) * 100:.0f}% | "
                      f"Segment took: {int(elapsed // 60)}m {int(elapsed % 60)}s", flush=True)
 
    if not torch.isfinite(u_array).all():
        raise FloatingPointError(
            f"Learned-PDE simulation diverged (dt={dt:.3e}, CFL limit={cfl_limit:.3e}, "
            f"D={D_list}). The learned reaction may be stiffer than the true one; "
            f"lower dt_cap.")
 
    return u_array.cpu(), x_array.cpu(), t_array.cpu()

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


def simulate_reaction_cpu(reaction, params, diff_coeffs, initial_condition,
                           T=50, early_stop=True, min_steady_state_check_frac=0.1,
                           num_frames=101, dt_cap=0.0001):
    """
    reaction: callable (uv, params) -> (Fu, Fv). Every entry in
        REACTION_REGISTRY['...']['fn'] already matches this -- no wrapping
        needed at the call site regardless of whether the underlying
        system is mass-conserving or not.
    initial_condition: (u0, v0) tuple of NxN arrays. REACTION_REGISTRY['...']
        ['ic_builder'](N, params_dict) produces this.
    min_steady_state_check_frac: fraction of T to wait before the early-stop
        check activates. Turing-unstable systems start almost exactly at
        their homogeneous fixed point, so frame-to-frame diff can sit below
        ss_tolerance before the instability has visibly grown -- checking
        from t=0 would falsely declare "steady state" before any pattern
        forms. Ignored when early_stop=False.
    num_frames: number of snapshots saved, evenly spaced across [0, T],
        regardless of T. This keeps training data size (and downstream
        BINN training time) constant across reactions with very different
        natural timescales -- gray_scott's T=8000 and hill_poly's T=50
        both produce 101 frames, not 16001 vs 101.
    dt_cap: upper bound on the timestep from the REACTION timescale.
        The actual dt used is min(dt_cap, CFL limit from diffusion), so
        neither stiff kinetics nor large diffusion coefficients can
        silently destabilize the integration. Comes from
        REACTION_SPECS[name]['dt_cap'].
    """
    species = 2
    N = N_GRID
    L = 10                            # domain length
    ss_tolerance = 0.025

    dx = L / N
    du, dv = diff_coeffs

    # Explicit-Euler diffusion is only stable for dt < dx^2 / (4*D), while
    # the reaction kinetics impose their own separate ceiling (dt_cap).
    # The old code hardcoded dt=0.0001 for both, which was simultaneously
    # too loose for high diffusion (Brusselator Dv=8 and Schnakenberg
    # Dv=10 diverged to NaN within two saved frames) and far too strict
    # for the low-diffusion coefficients now in use (~20x more steps than
    # needed). Take the min of both limits instead.
    cfl_safety = 0.4
    max_D = max(du, dv)
    cfl_limit = dx**2 / (4 * max_D) if max_D > 0 else float('inf')
    dt = min(dt_cap, cfl_safety * cfl_limit)

    nits = int(T / dt)
    limiter = 'reaction dt_cap' if dt_cap <= cfl_safety * cfl_limit else 'diffusion CFL'
    print(f"Using dt={dt:.3e} ({limiter} is binding; CFL limit={cfl_limit:.3e} "
          f"for max D={max_D}, dt_cap={dt_cap:.3e}) -> {nits:,} steps", flush=True)

    inv_dx2 = 1.0 / (dx**2)

    u, v = initial_condition
    u, v = u.copy(), v.copy()

    # Preallocated once, reused every step -- avoids reallocating on each
    # of up to hundreds of thousands of Euler steps.
    lap_u = np.zeros((N, N), dtype=np.float64)
    lap_v = np.zeros((N, N), dtype=np.float64)

    # Save a snapshot every `save_interval` steps, chosen so exactly
    # num_frames snapshots span the full run.
    save_interval = max(1, nits // (num_frames - 1))

    u_list, v_list, t_list = [], [], []
    min_steady_state_check_time = min_steady_state_check_frac * T

    for t in range(nits + 1):
        if t % save_interval == 0 or t == nits:
            u_list.append(u.copy())
            v_list.append(v.copy())
            t_list.append(t * dt)

            if early_stop and len(t_list) > 1 and t_list[-1] > min_steady_state_check_time:
                u_diff = np.max(np.abs(u_list[-1] - u_list[-2]))
                v_diff = np.max(np.abs(v_list[-1] - v_list[-2]))
                diff = max(u_diff, v_diff)
                if diff < ss_tolerance:
                    print(f'Steady state at t = {t_list[-1]:.2f}', flush=True)
                    break

        if t == nits:
            break

        fast_laplacian(u, lap_u)
        fast_laplacian(v, lap_v)

        uv = np.column_stack((u.flatten(), v.flatten()))
        Fu, Fv = reaction(uv, params)

        if hasattr(Fu, 'detach'):
            Fu = Fu.detach().cpu().numpy()
        if hasattr(Fv, 'detach'):
            Fv = Fv.detach().cpu().numpy()

        Fu = Fu.reshape(N, N)
        Fv = Fv.reshape(N, N)

        u = u + dt * (du * lap_u * inv_dx2 + Fu)
        v = v + dt * (dv * lap_v * inv_dx2 + Fv)

        if not np.isfinite(u).all() or not np.isfinite(v).all():
            # Should not happen now that dt respects the CFL limit, but
            # fail loudly rather than silently saving a file of NaNs --
            # that is exactly how the first Brusselator/Schnakenberg
            # sweep produced 101 frames of garbage without any warning.
            raise FloatingPointError(
                f"Simulation diverged to NaN/inf at t={t * dt:.4f} (step {t}/{nits}). "
                f"dt={dt:.3e}, CFL limit={cfl_limit:.3e}, D=({du}, {dv}).")

        if t % max(1, nits // 10) == 0 and t > 0:
            print(f"Progress: {(t / nits) * 100:.0f}%", flush=True)

    u_array = np.stack([np.stack([uf, vf], axis=-1) for uf, vf in zip(u_list, v_list)]).astype(np.float32)
    x_array = np.linspace(0, L, N, dtype=np.float32)
    t_array = np.array(t_list, dtype=np.float32)

    return torch.tensor(u_array), torch.tensor(x_array), torch.tensor(t_array)


if __name__ == '__main__':
    import json

    config_path = str(sys.argv[1])
    with open(config_path, "r") as f:
        config = json.load(f)

    save_path = config["save_path"]
    diff_coeffs = (float(config["du"]), float(config["dv"]))
    reaction_str = config["reaction"]

    if reaction_str not in REACTION_REGISTRY:
        raise ValueError(f"Unknown reaction function specified: {reaction_str}. "
                          f"Available: {sorted(REACTION_REGISTRY)}")

    spec = REACTION_REGISTRY[reaction_str]
    params_dict = {k: config[k] for k in spec["param_keys"]}
    params = [params_dict[k] for k in spec["param_keys"]]

    initial_condition = spec["ic_builder"](N_GRID, params_dict)
    T = config.get("T", spec["T"])
    early_stop = config.get("early_stop", spec["early_stop"])
    num_frames = config.get("num_frames", 101)  # keep dataset size consistent across reactions
                                                 # regardless of how large T needs to be
    dt_cap = config.get("dt_cap", spec["dt_cap"])

    json_filename = os.path.basename(config_path)
    base_name = json_filename.replace('.json', '')
    save_name = os.path.join(save_path, base_name)

    print(f"Starting simulation for: {save_name} (T={T}, early_stop={early_stop})", flush=True)

    u_array, x_array, t_array = simulate_reaction_cpu(
        spec["fn"], params, diff_coeffs,
        initial_condition=initial_condition, T=T, early_stop=early_stop,
        num_frames=num_frames, dt_cap=dt_cap)
    animate_u_array(u_array, t_array, name=f'{save_name}.gif', titles=("u", "v"))

    training_data = format_u_array_to_training_data(u_array, x_array, t_array)
    torch.save({'training_data': training_data}, f'{save_name}.pt')

    print(f"Successfully saved data to {save_name}.pt", flush=True)