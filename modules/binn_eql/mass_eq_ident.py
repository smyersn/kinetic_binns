"""
Test 0 — Ground-truth identifiability check for Du, Dv via the mass equation.

Does NOT touch the network at all. Takes the raw simulator grid (u_array-equivalent,
built straight from training_data), finite-differences it in time and space, and
asks: does

    w_t = (u+v)_t = Du * lap(u) + Dv * lap(v)

even have a well-conditioned solution for (Du, Dv) once you restrict to t >= cutoff?

If this fails on the ground-truth grid, no amount of network/loss tuning will fix
it — the mass equation is structurally underdetermined for that dataset at that
cutoff, most likely because:
  (a) the transient has died out by the cutoff (|w_t| ~ 0 everywhere kept), or
  (b) lap_u and lap_v are nearly collinear over the kept points (Du/Dv ratio only
      constrains a direction, not a point, when the two Laplacians point the same way)

Usage: edit `training_data_paths` / `cutoffs` below and run.
    python test0_mass_equation_identifiability.py
"""

import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# training_data_paths = [
#     "/hpc/home/nsmyers1/projects/kinetic_binns/data/wave_pinning/diff_coeffs_du_sweep/du_0.01_dv_1.0_a_1.0_b_1.0_k_0.01.pt",
#     "/hpc/home/nsmyers1/projects/kinetic_binns/data/wave_pinning/diff_coeffs_du_sweep/du_0.025_dv_0.1_a_1.0_b_1.0_k_0.01.pt",
#     "/hpc/home/nsmyers1/projects/kinetic_binns/data/wave_pinning/diff_coeffs_du_sweep/du_0.05_dv_0.1_a_1.0_b_1.0_k_0.01.pt",
# ]

training_data_paths = [
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/wave_pinning/diff_coeffs_du_sweep/du_0.01_dv_1.0_a_1.0_b_1.0_k_0.01.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/wave_pinning/diff_coeffs_du_sweep/du_0.025_dv_0.1_a_1.0_b_1.0_k_0.01.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/wave_pinning/diff_coeffs_du_sweep/du_0.05_dv_0.1_a_1.0_b_1.0_k_0.01.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/wave_pinning/diff_coeffs_du_sweep/du_0.05_dv_1.0_a_1.0_b_1.0_k_0.01.pt",
    "/hpc/home/nsmyers1/projects/kinetic_binns/data/wave_pinning/diff_coeffs_du_sweep/du_0.1_dv_1.0_a_1.0_b_1.0_k_0.01.pt"
]

SPECIES = 2                    # u, v
CUTOFFS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]   # sweep to see where identifiability kicks in
OUTPUT_DIR = Path("./test0_diagnostics")
OUTPUT_DIR.mkdir(exist_ok=True)

# Real per-dataset Laplacian relative error at t>=2, measured from
# train_surface_only.py's surface_fit_summary.json (see analyze_step1_lap_err.py).
# Replaces the flat 10% assumption below. Keyed by filename stem.
MEASURED_LAP_REL_ERR = {
    # "du_0.01_dv_1.0_a_1.0_b_1.0_k_0.01": 0.083,
    # "du_0.025_dv_0.1_a_1.0_b_1.0_k_0.01": 0.068,
    # "du_0.05_dv_0.1_a_1.0_b_1.0_k_0.01": 0.204,
}


# ---------------------------------------------------------------------------
# Grid reconstruction + FD
# ---------------------------------------------------------------------------

def parse_true_D(path):
    m = re.search(r"du_([\d.]+)_dv_([\d.]+)", str(path))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def load_grid(path, species=SPECIES):
    """
    Loads raw training_data (columns: x*dimensions, t, species concentrations)
    and reshapes into a dense (nt, nx[, ny]) grid per species, assuming the raw
    simulator output is a complete regular grid (true before noise_and_interpolate
    is ever applied — we deliberately load the raw .pt, not a noised/interpolated
    copy).
    """
    raw = torch.load(path)["training_data"]
    data = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)

    n_cols = data.shape[1]
    dimensions = n_cols - 1 - species
    if dimensions < 1:
        raise ValueError(f"Can't infer spatial dimensions from {n_cols} columns "
                          f"with species={species}")

    coords = data[:, :dimensions]
    t_col = data[:, dimensions]
    conc = data[:, dimensions + 1: dimensions + 1 + species]

    if dimensions == 1:
        x_vals = np.unique(coords[:, 0])
        t_vals = np.unique(t_col)
        nx, nt = len(x_vals), len(t_vals)
        if nx * nt != data.shape[0]:
            raise ValueError(
                f"{path}: grid not complete/regular (nx*nt={nx*nt} != "
                f"n_rows={data.shape[0]}). Are you accidentally loading a "
                f"noised/interpolated copy instead of the raw simulator grid?")
        order = np.lexsort((coords[:, 0], t_col))  # primary key = t_col
        sorted_conc = conc[order]
        fields = [sorted_conc[:, i].reshape(nt, nx) for i in range(species)]
        coord_grids = (x_vals,)

    elif dimensions == 2:
        x_vals = np.unique(coords[:, 0])
        y_vals = np.unique(coords[:, 1])
        t_vals = np.unique(t_col)
        nx, ny, nt = len(x_vals), len(y_vals), len(t_vals)
        if nx * ny * nt != data.shape[0]:
            raise ValueError(
                f"{path}: grid not complete/regular (nx*ny*nt={nx*ny*nt} != "
                f"n_rows={data.shape[0]}).")
        order = np.lexsort((coords[:, 1], coords[:, 0], t_col))
        sorted_conc = conc[order]
        fields = [sorted_conc[:, i].reshape(nt, nx, ny) for i in range(species)]
        coord_grids = (x_vals, y_vals)

    else:
        raise NotImplementedError(f"dimensions={dimensions} not handled")

    return fields, t_vals, coord_grids, dimensions


def compute_dt_and_laplacian(field, t_vals, coord_grids, dimensions):
    """field_t via central FD in time; laplacian via central FD in space."""
    field_t = np.gradient(field, t_vals, axis=0)

    if dimensions == 1:
        x_vals = coord_grids[0]
        d1 = np.gradient(field, x_vals, axis=1)
        lap = np.gradient(d1, x_vals, axis=1)
    else:
        x_vals, y_vals = coord_grids
        d1x = np.gradient(field, x_vals, axis=1)
        lap_x = np.gradient(d1x, x_vals, axis=1)
        d1y = np.gradient(field, y_vals, axis=2)
        lap_y = np.gradient(d1y, y_vals, axis=2)
        lap = lap_x + lap_y

    return field_t, lap


# ---------------------------------------------------------------------------
# Core analysis per dataset / cutoff
# ---------------------------------------------------------------------------

def analyze(path, cutoffs):
    print("=" * 100)
    print(path)
    true_Du, true_Dv = parse_true_D(path)
    print(f"True Du={true_Du}, Dv={true_Dv}, ratio Du/Dv={true_Du/true_Dv:.4f}"
          if true_Du else "Could not parse true Du/Dv from filename")

    fields, t_vals, coord_grids, dimensions = load_grid(path)
    u, v = fields
    u_t, lap_u = compute_dt_and_laplacian(u, t_vals, coord_grids, dimensions)
    v_t, lap_v = compute_dt_and_laplacian(v, t_vals, coord_grids, dimensions)
    w_t = u_t + v_t

    # --- |w_t| magnitude over time, to see when/if the transient has died ---
    spatial_axes = tuple(range(1, w_t.ndim))
    w_t_mag_by_t = np.mean(np.abs(w_t), axis=spatial_axes)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_vals, w_t_mag_by_t, marker="o", ms=3)
    ax.set_xlabel("t")
    ax.set_ylabel("mean |w_t| over space")
    ax.set_title(f"Transient strength\n{Path(path).name}")
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{Path(path).stem}_wt_magnitude.png", dpi=120)
    plt.close(fig)

    print(f"\n  mean|w_t| by timestep (first 10 saved frames): "
          f"{np.array2string(w_t_mag_by_t[:10], precision=4)}")
    print(f"  mean|w_t| at final frame: {w_t_mag_by_t[-1]:.4e}")

    results = []
    for cutoff in cutoffs:
        t_mask = t_vals >= cutoff
        if t_mask.sum() < 2:
            print(f"  cutoff={cutoff}: not enough timesteps kept, skipping")
            continue

        lu = lap_u[t_mask]
        lv = lap_v[t_mask]
        wt = w_t[t_mask]

        lu_flat = lu.reshape(-1)
        lv_flat = lv.reshape(-1)
        wt_flat = wt.reshape(-1)

        # weighted least squares: w_t = Du*lap_u + Dv*lap_v, weights = |w_t|
        weights = np.abs(wt_flat)
        weights = weights / (weights.mean() + 1e-12)
        sw = np.sqrt(weights)

        A = np.stack([lu_flat, lv_flat], axis=1)
        Aw = A * sw[:, None]
        bw = wt_flat * sw

        sol, _, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
        Du_hat, Dv_hat = sol
        cond_number = (sv[0] / sv[-1]) if sv[-1] > 0 else np.inf

        corr = np.corrcoef(lu_flat, lv_flat)[0, 1]

        results.append(dict(
            cutoff=cutoff, Du_hat=Du_hat, Dv_hat=Dv_hat,
            cond_number=cond_number, lap_corr=corr,
            mean_abs_wt=np.mean(np.abs(wt_flat)),
            n_points=len(wt_flat),
        ))

    # --- SNR check: how big is the true signal vs. the surface's ACTUAL
    # measured Laplacian noise floor (falls back to an assumed 10% if this
    # dataset isn't in MEASURED_LAP_REL_ERR yet).
    ref_cutoff = 2.0
    t_mask = t_vals >= ref_cutoff
    lu_ref = lap_u[t_mask]
    lv_ref = lap_v[t_mask]
    wt_ref = w_t[t_mask]

    assumed_rel_err = 0.10  # fallback only
    dataset_key = Path(path).stem
    measured_rel_err = MEASURED_LAP_REL_ERR.get(dataset_key)
    rel_err_used = measured_rel_err if measured_rel_err is not None else assumed_rel_err
    source_label = "MEASURED" if measured_rel_err is not None else "ASSUMED (fallback)"

    du_true_val = true_Du if true_Du is not None else 1.0
    dv_true_val = true_Dv if true_Dv is not None else 1.0

    mean_abs_lap_u = np.mean(np.abs(lu_ref))
    mean_abs_lap_v = np.mean(np.abs(lv_ref))
    lap_magnitude_term = du_true_val * mean_abs_lap_u + dv_true_val * mean_abs_lap_v

    noise_floor = rel_err_used * lap_magnitude_term
    signal = np.mean(np.abs(wt_ref))

    print(f"\n  Absolute FD magnitudes @ cutoff={ref_cutoff}: "
          f"mean|lap_u|={mean_abs_lap_u:.4e}, mean|lap_v|={mean_abs_lap_v:.4e}  "
          f"(Du*mean|lap_u| + Dv*mean|lap_v| = {lap_magnitude_term:.4e})")
    print(f"  SNR check @ cutoff={ref_cutoff} [{source_label} rel_err={rel_err_used:.3f}]: "
          f"mean|w_t| (signal) = {signal:.4e}, noise floor = {noise_floor:.4e}, "
          f"signal/noise = {signal/noise_floor:.3f}")

    print(f"\n  {'cutoff':>7} {'Du_hat':>10} {'Dv_hat':>10} {'ratio':>8} "
          f"{'cond#':>10} {'lap_corr':>9} {'mean|w_t|':>11}")
    for r in results:
        ratio = r['Du_hat'] / r['Dv_hat'] if r['Dv_hat'] != 0 else float('nan')
        print(f"  {r['cutoff']:>7.1f} {r['Du_hat']:>10.4e} {r['Dv_hat']:>10.4e} "
              f"{ratio:>8.4f} {r['cond_number']:>10.2e} {r['lap_corr']:>9.4f} "
              f"{r['mean_abs_wt']:>11.4e}")

    if true_Du is not None:
        print(f"\n  (true Du={true_Du:.4e}, Dv={true_Dv:.4e}, "
              f"ratio={true_Du/true_Dv:.4f})")

    # scatter of lap_u vs lap_v at the largest cutoff, to visualize collinearity
    if results:
        cutoff_for_plot = results[-1]['cutoff']
        t_mask = t_vals >= cutoff_for_plot
        lu_flat = lap_u[t_mask].reshape(-1)
        lv_flat = lap_v[t_mask].reshape(-1)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(lu_flat, lv_flat, s=2, alpha=0.3)
        ax.set_xlabel("lap_u")
        ax.set_ylabel("lap_v")
        ax.set_title(f"lap_u vs lap_v, t>={cutoff_for_plot}\n{Path(path).name}\n"
                     f"corr={results[-1]['lap_corr']:.4f}")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{Path(path).stem}_lap_collinearity.png", dpi=120)
        plt.close(fig)

    return results


if __name__ == "__main__":
    all_results = {}
    for path in training_data_paths:
        try:
            all_results[path] = analyze(path, CUTOFFS)
        except Exception as e:
            print(f"FAILED on {path}: {e}")
    print("\nDiagnostics (plots) written to:", OUTPUT_DIR.resolve())