import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
from modules.loaders.visualize_training_data import *

def format_data_general(dimensions, species, file=None, x_array=None, 
                        t_array=None, u_array=None, v_array=None):
    if file is not None:
        # load data from npz file
        npz = np.load(file)
        
        x_array = npz[npz.files[0]]
        t_array = npz[npz.files[-1]]
        outputs = [npz[npz.files[i]] for i in range(1, len(npz.files)-1)]
    else:
        outputs = [u_array, v_array]
        
    # create array of all points in n-dimensional space
    points = np.array(list(itertools.product(x_array, repeat=dimensions)))
    
    # create array to store formatted training data
    rows = len(t_array) * len(points)
    columns = dimensions + species + 1

    training_data = np.zeros((rows, columns))
    
    # load spatiotemporal coords into training data array
    i = 0

    for t in t_array:
        for point in points:
            inputs = np.hstack((point, t))
            training_data[i, : dimensions + 1] = inputs
            i += 1

    # load species concentrations into training data array
    i = dimensions + 1

    for array in outputs:
        training_data[:, i] = array.flatten()
        i += 1

    return training_data

def format_data_torch(u_array, x_array, t_array):
    T, C, H, W = u_array.shape  # T=201, C=2, H=W=200

    # Build spatial grids (x and y from positions)
    yy, xx = torch.meshgrid(x_array, x_array, indexing='ij')  # shape: (200, 200)

    # Expand spatial grids across time
    xx = xx.unsqueeze(0).expand(T, H, W)  # (T, H, W)
    yy = yy.unsqueeze(0).expand(T, H, W)  # (T, H, W)

    # Expand time vector across spatial grid
    tt = t_array.view(T, 1, 1).expand(T, H, W)  # (T, H, W)

    # Extract concentrations
    u = u_array[:, 0, :, :]  # (T, H, W)
    v = u_array[:, 1, :, :]  # (T, H, W)

    # Flatten all to 1D vectors of length N = T * H * W
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)
    t_flat = tt.reshape(-1)
    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)

    # Stack into shape (N, 5) → [x, y, t, u, v]
    training_data = torch.stack([x_flat, y_flat, t_flat, u_flat, v_flat], dim=1)
    
    return training_data