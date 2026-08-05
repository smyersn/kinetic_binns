import torch

def compute_curvature_scale(lap_labels, quantile=0.99, eps=1e-6):
    """
    Per-species normalization scale for curvature loss, analogous to
    BINN's mean_scale for GLS. Uses a high quantile (not max) to be
    robust to the rare highest-curvature outlier points, matching the
    robust-scaling philosophy already used for max_scale/pde_scale.
    Returns shape (1, species).
    """
    abs_labels = lap_labels.abs()
    species = lap_labels.shape[1]
    scales = []
    for i in range(species):
        scales.append(torch.quantile(abs_labels[:, i], quantile))
    return torch.stack(scales).view(1, -1).clamp(min=eps)

def build_curvature_labels(train_data, u_array, x_array, t_array, dimensions, species):
    """
    Builds a per-row FD Laplacian label tensor aligned with train_data,
    by looking up each row's own (x1, ..., x_dimensions, t) coordinates
    against x_array/t_array. This sidesteps needing to know train_data's
    internal row ordering (whichever way format_u_array_to_training_data
    flattened things).

    ASSUMPTION: u_array has shape (T, H, W, species) where H indexes the
    same physical axis as train_data's first spatial column and W indexes
    the second spatial column. This holds for a square domain built from
    a single x_array reused on both axes, which is your setup. If labels
    look wrong (see sanity check below), swap idx_h/idx_w.

    Returns: (num_points, species) tensor, same device as train_data,
    row-aligned with train_data.
    """
    device = train_data.device
    dx = float(x_array[1] - x_array[0])

    # --- FD Laplacian on the full grid, periodic BCs (matches simulator) ---
    up, um = torch.roll(u_array, -1, dims=1), torch.roll(u_array, 1, dims=1)
    ul, ur = torch.roll(u_array, -1, dims=2), torch.roll(u_array, 1, dims=2)
    lap_fd = (up + um + ul + ur - 4 * u_array) / dx**2   # (T, H, W, species)
    lap_fd = lap_fd.to(device)

    x_array_d = x_array.to(device)
    t_array_d = t_array.to(device)

    if dimensions != 2:
        raise NotImplementedError("build_curvature_labels currently supports dimensions=2 only")

    # train_data columns: [x1, x2, t, u, v]
    coords = train_data[:, :dimensions + 1]

    idx_h = torch.searchsorted(x_array_d, coords[:, 0].contiguous()).clamp(0, len(x_array_d) - 1)
    idx_w = torch.searchsorted(x_array_d, coords[:, 1].contiguous()).clamp(0, len(x_array_d) - 1)
    idx_t = torch.searchsorted(t_array_d, coords[:, dimensions].contiguous()).clamp(0, len(t_array_d) - 1)

    labels = lap_fd[idx_t, idx_h, idx_w, :]   # (num_points, species)
    return labels.detach()


def sanity_check_labels(train_data, labels, u_array, x_array, t_array, dimensions, n_samples=5):
    """
    Spot-checks a handful of rows: recompute the FD Laplacian directly from
    scratch for a random row's exact grid location and compare to the label
    produced by build_curvature_labels. Run this once after building labels
    to confirm the H/W axis assumption above is correct.
    """
    dx = float(x_array[1] - x_array[0])
    N = len(x_array)
    idx = torch.randint(0, train_data.shape[0], (n_samples,))

    for i in idx.tolist():
        x1, x2, t = train_data[i, 0].item(), train_data[i, 1].item(), train_data[i, dimensions].item()
        h = int(torch.searchsorted(x_array, torch.tensor([x1])).clamp(0, N - 1))
        w = int(torch.searchsorted(x_array, torch.tensor([x2])).clamp(0, N - 1))
        tt = int(torch.searchsorted(t_array, torch.tensor([t])).clamp(0, len(t_array) - 1))

        hp, hm = (h + 1) % N, (h - 1) % N
        wp, wm = (w + 1) % N, (w - 1) % N
        direct_lap = (u_array[tt, hp, w, 0] + u_array[tt, hm, w, 0] +
                      u_array[tt, h, wp, 0] + u_array[tt, h, wm, 0] -
                      4 * u_array[tt, h, w, 0]) / dx**2

        print(f"row {i}: label={labels[i,0].item():.6e}  direct={direct_lap.item():.6e}  "
              f"match={'OK' if abs(labels[i,0].item() - direct_lap.item()) < 1e-6 else 'MISMATCH -> check H/W axis order'}")