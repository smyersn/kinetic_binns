import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit, prange
import sys
import os

# --- 0. Parse Arguments ---
a = float(sys.argv[1])
b = float(sys.argv[2])
k = float(sys.argv[3])
Du, Dv = 0.01, 1

# --- 1. Parameters ---
L, N = 10.0, 200
dx = L / N
t_final = 5000.0

# Time step criteria for stability: dt < dx^2 / (4 * max(Du, Dv))
# 0.05^2 / 4 = 0.000625. We use 0.0005 to be safe.
dt = 0.0001
n_steps = int(t_final / dt)
total_frames = 500
save_every = n_steps // total_frames

# --- 2. JIT-Compiled Solver Kernel ---
# The 'parallel=True' flag allows Numba to use multiple CPU cores
@njit(parallel=True)
def update_step(u, v, du, dv, dx, Du, Dv, a, k, b):
    rows, cols = u.shape
    inv_dx2 = 1.0 / (dx**2)
    
    # prange tells Numba this loop can be run in parallel
    for i in prange(rows):
        for j in range(cols):
            # Periodic Boundary Indices
            ip = (i + 1) % rows
            im = (i - 1) % rows
            jp = (j + 1) % cols
            jm = (j - 1) % cols
            
            # 5-point Laplacian Stencil
            lap_u = (u[ip, j] + u[im, j] + u[i, jp] + u[i, jm] - 4*u[i, j]) * inv_dx2
            lap_v = (v[ip, j] + v[im, j] + v[i, jp] + v[i, jm] - 4*v[i, j]) * inv_dx2
            
            # Reaction Term (Equation 5)
            # reaction = (a * u[i, j]**2 / (1.0 + k * u[i, j]**2)) * v[i, j] - b * u[i, j]
            reaction = a * v[i, j] * u[i, j]**2 - (b * u[i, j]**2 / (1.0 + k * u[i, j]**2))
            
            # Calculate rates of change
            du[i, j] = Du * lap_u + reaction
            dv[i, j] = Dv * lap_v - reaction
            
# --- 3. Initial Conditions (NumPy translation) ---
u0_val, v0_val = 1.0, 1.0246

# u: uniform random noise between [0.5 * u0_val, 1.5 * u0_val)
# np.random.rand gives [0, 1)
u = (np.random.rand(N, N) + 0.5) * u0_val

# v: homogeneous field
v = np.ones((N, N)) * v0_val

# Buffers for history
history_u = []
history_v = []
time_points = []

# --- 4. Main Simulation Loop ---
print(f"Starting Numba Simulation for {n_steps} steps...", flush=True)
# Pre-allocate temporary arrays outside the loop for speed
du_arr = np.zeros((N, N))
dv_arr = np.zeros((N, N))

for step in range(n_steps):
    # Run the compiled JIT function
    update_step(u, v, du_arr, dv_arr, dx, Du, Dv, a, k, b)
    
    # Euler integration step
    u += dt * du_arr
    v += dt * dv_arr
    
    # Progress tracking
    if step % (n_steps // 10) == 0:
        print(f"Progress: {int(step/n_steps*100)}% complete (t={step*dt:.2f}s)", flush=True)
        
    # Saving history for animation
    if step % save_every == 0:
        history_u.append(u.copy().astype(np.float32))
        history_v.append(v.copy().astype(np.float32))
        time_points.append(step * dt)

print("Simulation complete.", flush=True)

# --- 5. Two-Subplot Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Initial plots
# Using different colormaps (viridis vs magma) helps distinguish u and v visually
im1 = ax1.imshow(history_u[0], extent=[0, L, 0, L], cmap='viridis', origin='lower', vmin=np.min(history_u), vmax=np.max(history_u))
im2 = ax2.imshow(history_v[0], extent=[0, L, 0, L], cmap='viridis', origin='lower', vmin=np.min(history_v), vmax=np.max(history_v))

ax1.set_title("Active GTPase (u)")
ax2.set_title("Inactive GTPase (v)")
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

fig.suptitle(f"Time: {time_points[0]:.2f}s", fontsize=14)

def animate(i):
    # Update image data
    im1.set_data(history_u[i])
    im2.set_data(history_v[i])
    
    fig.suptitle(f"Time: {time_points[i]:.2f}s", fontsize=14)
    # Return changed artists for blitting
    return im1, im2,

ani = FuncAnimation(fig, animate, frames=len(history_u), blit=True)

# Use PillowWriter for GIF generation
writer = PillowWriter(fps=10)
gif_path = f"gifs/a_{a}_b_{b}_k_{k}_long.gif"
ani.save(gif_path, writer=writer)
plt.close()