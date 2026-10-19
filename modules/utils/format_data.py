import numpy as np
import torch

# Rows are [x, y, t, u_1, ..., u_S]: two spatial columns, then time, then one
# column per species (S = 2 gives the original [x, y, t, u, v]).
N_COORD_COLUMNS = 3


def format_u_array_to_training_data(u_array, x_array, t_array):
    # 1. Unpack expecting Channel-Last (Time, Y, X, Species)
    T, H, W, C = u_array.shape  # e.g., T=101, H=200, W=200, C=2 or 3

    # Build spatial grids (x and y from positions)
    yy, xx = torch.meshgrid(x_array, x_array, indexing='ij')

    # Expand spatial grids across time
    xx = xx.unsqueeze(0).expand(T, H, W)
    yy = yy.unsqueeze(0).expand(T, H, W)

    # Expand time vector across spatial grid
    tt = t_array.view(T, 1, 1).expand(T, H, W)

    # 2. Flatten coordinates and every species' concentrations (LAST dimension)
    columns = [xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)]
    columns += [u_array[:, :, :, s].reshape(-1) for s in range(C)]

    # Stack into shape (N, 3 + species).
    training_data = torch.stack(columns, dim=1)

    return training_data


def format_training_data_to_u_array(training_data):
    # 1. Extract the individual columns
    x_raw = training_data[:, 0]
    y_raw = training_data[:, 1]
    t_raw = training_data[:, 2]
    species = training_data.shape[1] - N_COORD_COLUMNS

    # 2. Extract unique coordinates to determine the grid dimensions
    # return_inverse=True gives us the exact grid index for every single row
    x_array, x_indices = torch.unique(x_raw, sorted=True, return_inverse=True)
    y_array, y_indices = torch.unique(y_raw, sorted=True, return_inverse=True)
    t_array, t_indices = torch.unique(t_raw, sorted=True, return_inverse=True)

    # Automatically infer the resolution (handles downsampled/reduced data naturally)
    T = len(t_array)
    H = len(y_array)
    W = len(x_array)

    # 3. Initialize the empty target tensor (Time, Y, X, Species)
    u_array = torch.zeros((T, H, W, species), dtype=training_data.dtype, device=training_data.device)

    # 4. Map the flattened data back into the 4D grid simultaneously
    # This routes every concentration directly to its (t, y, x) address
    for s in range(species):
        u_array[t_indices, y_indices, x_indices, s] = training_data[:, N_COORD_COLUMNS + s]

    return u_array, x_array, t_array