import numpy as np
import torch

def format_u_array_to_training_data(u_array, x_array, t_array):
    # 1. Unpack expecting Channel-Last (Time, Y, X, Species)
    T, H, W, C = u_array.shape  # e.g., T=101, H=200, W=200, C=2

    # Build spatial grids (x and y from positions)
    yy, xx = torch.meshgrid(x_array, x_array, indexing='ij')

    # Expand spatial grids across time
    xx = xx.unsqueeze(0).expand(T, H, W)
    yy = yy.unsqueeze(0).expand(T, H, W)

    # Expand time vector across spatial grid
    tt = t_array.view(T, 1, 1).expand(T, H, W)

    # 2. Extract concentrations from the LAST dimension
    u = u_array[:, :, :, 0]  # (T, H, W)
    v = u_array[:, :, :, 1]  # (T, H, W)

    # Flatten all to 1D vectors
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)
    t_flat = tt.reshape(-1)
    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)

    # Stack into shape (N, 5). 
    training_data = torch.stack([x_flat, y_flat, t_flat, u_flat, v_flat], dim=1)
    
    return training_data

def format_training_data_to_u_array(training_data):
    # 1. Extract the individual columns
    x_raw = training_data[:, 0]
    y_raw = training_data[:, 1]
    t_raw = training_data[:, 2]
    u_raw = training_data[:, 3]
    v_raw = training_data[:, 4]

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
    u_array = torch.zeros((T, H, W, 2), dtype=training_data.dtype, device=training_data.device)

    # 4. Map the flattened data back into the 4D grid simultaneously
    # This routes every u and v value directly to its (t, y, x) address
    u_array[t_indices, y_indices, x_indices, 0] = u_raw
    u_array[t_indices, y_indices, x_indices, 1] = v_raw

    return u_array, x_array, t_array