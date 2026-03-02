import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_u_array(u_array, t_array, name=None, titles=("u", "v")):
    """
    Plots two arrays side-by-side.
    Assumes input shape: (Time, Channel, Height, Width)
    """    
    # 1. Setup Figure: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2. Determine Shared Color Limits
    # This ensures the colors mean the same thing in both plots
    umin, umax = u_array[:, :, :, 0].min(), u_array[:, :, :, 0].max()
    vmin, vmax = u_array[:, :, :, 1].min(), u_array[:, :, :, 1].max()

    # 3. Initialize Plots
    # Plot 1
    im1 = axes[0].imshow(u_array[0, :, :, 0], cmap='viridis', vmin=umin, vmax=umax)
    axes[0].set_title(titles[0])
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot 2
    im2 = axes[1].imshow(u_array[0, :, :, 1], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1])
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Shared Main Title (for Time)
    main_title = fig.suptitle(f'T = {t_array[0]:.1f}', fontsize=16)

    # 4. Define Update Function
    def animate(frame):
        # Update data for both plots
        im1.set_array(u_array[frame, :, :, 0])
        im2.set_array(u_array[frame, :, :, 1])
        
        # Update time text
        main_title.set_text(f'T = {t_array[frame]:.2f}')
        
        return im1, im2, main_title

    # 5. Create Animation
    anim = animation.FuncAnimation(fig, animate, frames=len(t_array), interval=200)
    
    # Save if name provided
    if name:
        writergif = animation.PillowWriter(fps=10)
        anim.save(name, writer=writergif)

    return anim

def animate_residuals(pred_array, true_array, times, name=None, 
                      titles=("Pred u - True u", "Pred v - True v")):
    """
    Plots the difference (Pred - True) side-by-side for u and v.
    Assumes inputs are (Time, Height, Width, 2).
    Red: Model > Truth (Overestimation).
    Blue: Model < Truth (Underestimation).
    """
    # 1. Shape Handling: Ensure (Time, Height, Width, Channel)
    # Check if inputs are PyTorch Tensors (CPU or GPU) and convert
    if hasattr(pred_array, 'detach'): 
        pred_array = pred_array.detach().cpu().numpy()
    
    if hasattr(true_array, 'detach'): 
        true_array = true_array.detach().cpu().numpy()

    if pred_array.ndim == 4 and pred_array.shape[1] == 2: 
        pred_array = np.transpose(pred_array, (0, 2, 3, 1))
        
    if true_array.ndim == 4 and true_array.shape[1] == 2:
        true_array = np.transpose(true_array, (0, 2, 3, 1))
        
    # 2. Compute Residuals (Predicted - Ground Truth)
    residuals = pred_array - true_array
    
    # 3. Determine Scale (Symmetric for diverging colormap)
    # We define the range as [-max_error, +max_error] so 0 is always white/centered
    # Use slicing [..., 0] for u channel and [..., 1] for v channel
    max_err_u = np.max(np.abs(residuals[..., 0]))
    max_err_v = np.max(np.abs(residuals[..., 1]))
    
    # Avoid div by zero if perfect match
    max_err_u = max(max_err_u, 1e-6)
    max_err_v = max(max_err_v, 1e-6)

    # 4. Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Residual U (Channel 0)
    im1 = axes[0].imshow(residuals[0, :, :, 0], cmap='seismic', vmin=-max_err_u, vmax=max_err_u)
    axes[0].set_title(f"{titles[0]}\n(Range: +/- {max_err_u:.3f})")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot 2: Residual V (Channel 1)
    im2 = axes[1].imshow(residuals[0, :, :, 1], cmap='seismic', vmin=-max_err_v, vmax=max_err_v)
    axes[1].set_title(f"{titles[1]}\n(Range: +/- {max_err_v:.3f})")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Shared Main Title
    main_title = fig.suptitle(f'T = {times[0]:.2f}', fontsize=16)

    # 5. Define Update Function
    def animate(frame):
        im1.set_array(residuals[frame, :, :, 0])
        im2.set_array(residuals[frame, :, :, 1])
        main_title.set_text(f'Residuals at T = {times[frame]:.2f}')
        return im1, im2, main_title

    # 6. Create Animation
    anim = animation.FuncAnimation(fig, animate, frames=len(times), interval=200)
    
    # Save if name provided
    if name:
        writergif = animation.PillowWriter(fps=10)
        anim.save(name, writer=writergif)
        
    return anim