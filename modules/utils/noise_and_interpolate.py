# import numpy as np
# import itertools
# from scipy.interpolate import RegularGridInterpolator

# def noise_and_interpolate(training_data, num_points, epsilon, dimensions, species):
    
#     # get times and points from training data
#     times = np.unique(training_data[:, dimensions])
#     points = np.unique(training_data[:, 0])
    
#     if num_points == 0 or num_points > len(points):
#         num_points = len(points)
    
#     # define new points for interpolation
#     x_interp = np.linspace(np.min(points), np.max(points), num_points)
#     x_interp_pairs = np.array(list(itertools.product(x_interp, repeat=dimensions)))

#     # create array to store all interpolated data
#     training_data_interp = np.zeros((0, dimensions + species + 1))
   
#     # iterate over frames in training data
#     for t in np.unique(times):
#         frame = training_data[training_data[:, dimensions] == t]
        
#         # create array to store interpolated data for each frame
#         training_data_temp = np.hstack((x_interp_pairs, 
#                                         np.expand_dims(np.repeat(t, len(x_interp_pairs)), axis=1)))
                                       
#         for specie in range(species):
#             # reshape training data to meshgrid
#             values = np.reshape(frame[:, dimensions + specie + 1], 
#                                 (len(points),) * dimensions)
            
#             # define interpolator function and interpolate at new points
#             interp_fn = RegularGridInterpolator((points,)*dimensions, values, method='linear')
#             interpolated_values = interp_fn(x_interp_pairs)
            
#             # append frame data to all data
#             training_data_temp = np.hstack((training_data_temp,
#                                            np.expand_dims(interpolated_values, axis=1)))

#         training_data_interp = np.vstack((training_data_interp, training_data_temp))
            
            
#     # add noise to outputs and set negative values to zero
#     noise = np.random.normal(0, epsilon, training_data_interp[:, dimensions+1:].shape)
#     training_data_noise = training_data_interp.copy()
#     training_data_noise[:, dimensions+1:] = np.maximum(training_data_noise[:, dimensions+1:] + noise, 0)
    
#     return training_data_noise

import torch
import itertools
import torch.nn.functional as F

def noise_and_interpolate(training_data, num_points, epsilon, dimensions, species, multiplicative_noise=False):
    device = training_data.device
    dtype = training_data.dtype

    # Extract time and space coordinates
    times = torch.unique(training_data[:, dimensions])
    points = torch.unique(training_data[:, 0])

    if num_points == 0 or num_points > len(points):
        num_points = len(points)

    x_interp = torch.linspace(points.min(), points.max(), num_points, device=device, dtype=dtype)
    x_interp_pairs = torch.tensor(list(itertools.product(x_interp.tolist(), repeat=dimensions)),
                                  device=device, dtype=dtype)

    # Prepare output container
    training_data_interp = []

    for t in times:
        frame = training_data[training_data[:, dimensions] == t]
        training_data_temp = torch.cat([
            x_interp_pairs,
            t.repeat(len(x_interp_pairs)).unsqueeze(1)
        ], dim=1)

        for specie in range(species):
            # Extract grid values
            values = frame[:, dimensions + specie + 1]

            # Reshape for interpolation
            grid_shape = [len(points)] * dimensions
            values = values.reshape(*grid_shape)

            # Prepare values for grid_sample (N, C, H, W)
            if dimensions == 1:
                values = values.view(1, 1, 1, -1)  # (N=1, C=1, H=1, W=width)
                grid = x_interp_pairs.view(1, -1, 1, dimensions)
                grid = 2 * (grid - points.min()) / (points.max() - points.min()) - 1
            elif dimensions == 2:
                values = values.view(1, 1, *grid_shape)  # (N=1, C=1, H, W)
                grid = x_interp_pairs.view(1, -1, 1, dimensions)  # (1, N, 1, 2)
                grid = 2 * (grid - points.min()) / (points.max() - points.min()) - 1
                grid = grid.flip(-1)  # grid_sample uses (x, y), so reverse
            else:
                raise NotImplementedError("Only 1D or 2D interpolation is supported with PyTorch.")

            # Perform grid sampling
            interpolated = F.grid_sample(values, grid, mode='bilinear', align_corners=True)
            interpolated_values = interpolated.view(-1).unsqueeze(1)

            training_data_temp = torch.cat([training_data_temp, interpolated_values], dim=1)

        training_data_interp.append(training_data_temp)

    training_data_interp = torch.cat(training_data_interp, dim=0)

    if multiplicative_noise:
        # Multiplicative noise
        signal = training_data_interp[:, dimensions+1:]
        relative_noise = 1 + torch.randn_like(signal) * epsilon
        training_data_interp[:, dimensions+1:] = torch.clamp(signal * relative_noise, min=0)
        
    else: 
        # Additive Gaussian noise and clip at zero
        noise = torch.randn_like(training_data_interp[:, dimensions+1:]) * epsilon
        training_data_interp[:, dimensions+1:] = torch.clamp(training_data_interp[:, dimensions+1:] + noise, min=0)

    return training_data_interp