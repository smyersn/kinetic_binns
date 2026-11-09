import torch
import itertools
import torch.nn.functional as F

NOISE_MODELS = ('multiplicative', 'additive_std', 'additive')


def noise_and_interpolate(training_data, num_points, epsilon, dimensions, species,
                          multiplicative_noise=False, noise_model=None, seed=None):
    """
    Bilinear interpolation of every frame to num_points per side, then noise.

    noise_model (None: 'multiplicative' if multiplicative_noise else 'additive'):
        'multiplicative'  u * (1 + epsilon * N(0, 1)), clipped at 0: noise
                          proportional to the value (the BINN literature's
                          proportional-error model, which the GLS loss assumes)
        'additive_std'    u + epsilon * std(u_s) * N(0, 1), NOT clipped: the
                          SINDy / PDE-FIND convention, epsilon = noise as a
                          fraction of each species' standard deviation over the
                          whole (interpolated, noise-free) dataset. Use this to
                          compare against PDE-FIND, weak SINDy, DeepMoD.
        'additive'        u + epsilon * N(0, 1), clipped at 0 (absolute size;
                          kept for old configs)

    seed: if given, the noise comes from a torch.Generator seeded with it, so
        the same (data, points, epsilon, noise_model, seed) always gives the
        same noisy data -- the BINN and every baseline can be trained on
        IDENTICAL noise. On CPU this draws the same numbers as
        torch.manual_seed(seed) followed by the unseeded call.
        None: the global random state (the original behaviour).
    """
    if noise_model is None:
        noise_model = 'multiplicative' if multiplicative_noise else 'additive'
    if noise_model not in NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {NOISE_MODELS}, got {noise_model!r}")
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

    signal = training_data_interp[:, dimensions+1:]
    if seed is None:
        standard_normal = torch.randn_like(signal)
    else:
        generator = torch.Generator(device=signal.device).manual_seed(int(seed))
        standard_normal = torch.randn(signal.shape, generator=generator,
                                      device=signal.device, dtype=signal.dtype)

    if noise_model == 'multiplicative':
        # Multiplicative noise
        relative_noise = 1 + standard_normal * epsilon
        training_data_interp[:, dimensions+1:] = torch.clamp(signal * relative_noise, min=0)

    elif noise_model == 'additive_std':
        # Additive Gaussian noise scaled per species to its standard deviation
        # over the whole noise-free dataset; no clipping, as in the SINDy papers.
        scale = signal.std(dim=0, keepdim=True)
        training_data_interp[:, dimensions+1:] = signal + standard_normal * epsilon * scale

    else:
        # Additive Gaussian noise and clip at zero
        noise = standard_normal * epsilon
        training_data_interp[:, dimensions+1:] = torch.clamp(signal + noise, min=0)

    return training_data_interp