import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DEFAULT_SPECIES_NAMES = ("u", "v", "w")


def _species_titles(titles, species, fmt="{}"):
    """
    One title per species. Missing ones (e.g. the 2-species default
    titles=("u", "v") on a 3-species array) are filled with u, v, w, ...
    and then s3, s4, ... beyond that.
    """
    titles = list(titles or [])
    for s in range(len(titles), species):
        name = DEFAULT_SPECIES_NAMES[s] if s < len(DEFAULT_SPECIES_NAMES) else f"s{s}"
        titles.append(fmt.format(name))
    return titles[:species]


def _to_numpy(array):
    if hasattr(array, 'detach'):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def animate_u_array(u_array, t_array, name=None, titles=("u", "v")):
    """
    Plots every species side by side, one panel each.
    Assumes input shape: (Time, Height, Width, Species).
    """
    u_array = _to_numpy(u_array)
    species = u_array.shape[-1]
    titles = _species_titles(titles, species)

    # 1. Setup Figure: 1 row, one column per species
    fig, axes = plt.subplots(1, species, figsize=(6 * species, 5))
    axes = np.atleast_1d(axes)

    # 2. Initialize Plots, each with its own colour limits over the whole run
    # (fixed in time, so the colours mean the same thing in every frame)
    images = []
    for s, ax in enumerate(axes):
        lo, hi = u_array[..., s].min(), u_array[..., s].max()
        im = ax.imshow(u_array[0, :, :, s], cmap='viridis', vmin=lo, vmax=hi)
        ax.set_title(titles[s])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        images.append(im)

    # Shared Main Title (for Time)
    main_title = fig.suptitle(f'T = {t_array[0]:.1f}', fontsize=16)

    # 3. Define Update Function
    def animate(frame):
        for s, im in enumerate(images):
            im.set_array(u_array[frame, :, :, s])
        main_title.set_text(f'T = {t_array[frame]:.2f}')
        return (*images, main_title)

    # 4. Create Animation
    anim = animation.FuncAnimation(fig, animate, frames=len(t_array), interval=200)

    # Save if name provided
    if name:
        writergif = animation.PillowWriter(fps=10)
        anim.save(name, writer=writergif)

    return anim


def animate_residuals(pred_array, true_array, times, name=None,
                      titles=("Pred u - True u", "Pred v - True v")):
    """
    Plots the difference (Pred - True) side-by-side, one panel per species.
    Assumes inputs are (Time, Height, Width, Species); channel-first
    (Time, Species, Height, Width) arrays are transposed automatically.
    Red: Model > Truth (Overestimation).
    Blue: Model < Truth (Underestimation).
    """
    # 1. Shape Handling: Ensure (Time, Height, Width, Channel)
    pred_array = _to_numpy(pred_array)
    true_array = _to_numpy(true_array)

    # Channel-first if the second axis is short and the last is a grid axis.
    def channel_last(a):
        if a.ndim == 4 and a.shape[1] <= 4 and a.shape[-1] > 4:
            return np.transpose(a, (0, 2, 3, 1))
        return a

    pred_array = channel_last(pred_array)
    true_array = channel_last(true_array)

    # 2. Compute Residuals (Predicted - Ground Truth)
    residuals = pred_array - true_array
    species = residuals.shape[-1]
    titles = _species_titles(titles, species, fmt="Pred {0} - True {0}")

    # 3. Setup Figure, with a symmetric scale per species so 0 is always
    # white/centered (and never zero-width if the match is perfect)
    fig, axes = plt.subplots(1, species, figsize=(6 * species, 5))
    axes = np.atleast_1d(axes)

    images = []
    for s, ax in enumerate(axes):
        max_err = max(np.max(np.abs(residuals[..., s])), 1e-6)
        im = ax.imshow(residuals[0, :, :, s], cmap='seismic', vmin=-max_err, vmax=max_err)
        ax.set_title(f"{titles[s]}\n(Range: +/- {max_err:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        images.append(im)

    # Shared Main Title
    main_title = fig.suptitle(f'T = {times[0]:.2f}', fontsize=16)

    # 4. Define Update Function
    def animate(frame):
        for s, im in enumerate(images):
            im.set_array(residuals[frame, :, :, s])
        main_title.set_text(f'Residuals at T = {times[frame]:.2f}')
        return (*images, main_title)

    # 5. Create Animation
    anim = animation.FuncAnimation(fig, animate, frames=len(times), interval=200)

    # Save if name provided
    if name:
        writergif = animation.PillowWriter(fps=10)
        anim.save(name, writer=writergif)

    return anim