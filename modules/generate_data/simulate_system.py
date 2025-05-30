import sys, importlib, os
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{file_dir}/../../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from modules.loaders.visualize_training_data import animate_data
from modules.utils.numpy_torch_conversion import *
from modules.loaders.format_data import format_data_general

# Define 2-dimensional Laplacian
def laplace(M, dx, dim):
    if dim == 1:
        grid = (-2 * M + np.roll(M, 1) + np.roll(M, -1)) / dx**2
    
    elif dim == 2:
            grid = (-4 * M + np.roll(M, 1, axis=0) + np.roll(M, -1, axis=0) + 
                    np.roll(M, 1, axis=1) + np.roll(M, -1, axis=1)) / dx**2

    return grid

def calc_squared_wavenumbers(L, N, Du, Dv):
    kx = (2*np.pi/L) * 1j * np.hstack((np.arange(0, N//2), np.array([0]), np.arange(-N//2+1, 0)))
    ky = kx.copy()

    k2x = kx**2
    k2y = ky**2

    kxx, kyy = np.meshgrid(k2x, k2y)

    ksqu = Du * (kxx + kyy)
    ksqv = Dv * (kxx + kyy)
    
    return ksqu, ksqv

# Define reactions
def wave_pinning(uv, params=(1, 1, 0.01)):
    u, v = uv[:, 0], uv[:, 1]
    a, b, k = params
    
    F = (a * u**2 * v) / (1 + k * u**2) - b * u
    return F

def turing_type(uv, params=(1, 1)):
    u, v = uv[:, 0], uv[:, 1]
    a, b = params
    
    F = a * u**2 * v - b * u
    return F

# Define update functions for simulation
def update_laplace(u, v, reaction, Du, Dv, dt, dx, points, dim, params=None):
    if params:
        F = reaction(np.column_stack((u.ravel(), v.ravel())), params)
        
    else:
        F = to_numpy(reaction(to_torch(np.column_stack((u.ravel(), v.ravel())))))
        
    F = np.reshape(F, (points,)*dim)
        
    Lu = laplace(u, dx, dim)
    Lv = laplace(v, dx, dim)
    
    # print((Du * Lu * dt)[:10])
    # print((F * dt)[:10])
    # print((Dv * Lv * dt)[:10])
    # print((-F * dt)[:10])
    
    u = u + (Du * Lu + F) * dt
    v = v + (Dv * Lv - F) * dt
    
    return u, v

def update_fourier(reaction, u0, v0, params, dt, ksqu, ksqv, points, nn=None):
    if nn:
        F = to_numpy(nn.reaction(to_torch(
            np.column_stack((u0.ravel(), v0.ravel())))[:, None]))
        F = np.reshape(F, (points,)*2)            

    else:
        F = reaction(u0, v0, params)
        
    u1r = u0 + F * dt
    v1r = v0 - F * dt
    
    u1r_hat = np.fft.fft2(u1r)
    v1r_hat = np.fft.fft2(v1r)
    
    u1 = np.real(np.fft.ifft2(u1r_hat / (1 - dt * ksqu)))
    v1 = np.real(np.fft.ifft2(v1r_hat / (1 - dt * ksqv)))
    
    return u1, v1  

def generate_initial_conditions(u0, v0, N, dim, spikes=0, custom=False, 
                                random=False):
    if spikes != 0 and dim == 1:
        u = np.full(N, u0) - np.cos(2 * spikes * np.pi * np.arange(N) / N) * 0.1 
        v = np.full(N, v0)
        
    elif custom and dim == 1:
        # Generate custom initial conditions from file 
        initial_data_path = '../../data/custom_initial_state.csv'
        initial_data = np.loadtxt(initial_data_path, delimiter=',')

        # Get coordinates to interpolate from 
        xp = np.linspace(0, 10, 500)
        up = initial_data[0]
        vp = initial_data[1]

        # Interpolate
        xi = np.linspace(0, 10, N)
        u = np.interp(xi, xp, up)
        v = np.interp(xi, xp, vp)
        
    elif random and dim == 1: 
        # Generate initial conditions with random noise
        u = u0 * (np.random.rand(N) * 2)
        v = v0 * np.ones(N)
        
    elif dim > 1:
        u = (np.random.rand(*(N,) * dim) + 0.5) * u0
        v = np.ones((N,) * dim) * v0

    return u, v
    
def simulate(u0, v0, L, N, T, dim, reaction, params=None, Du=0.01, Dv=1, 
             save_name=None, early_stop=True):
    # Define system parameters
    dt = 0.0001
    dx = L / N
    ss_tolerance = 0.005 if dim == 1 else 0.05
                
    nits = int(T / dt)
    half_sec_nits = 0.5 / dt

    x_array = np.linspace(0, L, N)
                   
    # Set up storage (only records at half second intervals)
    rows = int(nits / half_sec_nits) + 1

    u_array = np.zeros(((rows,) + (N,)*dim))
    v_array = np.zeros(((rows,) + (N,)*dim))
    t_array = np.zeros(rows)
    
    # Set initial conditions
    u, v = u0, v0
    
    # Solve
    for t in range(nits):
        
        if t % (nits // 10) == 0:
            print(f"Progress: {(t / nits) * 100}%", flush=True)
            
        if t % half_sec_nits == 0:
            time_step = int(t / half_sec_nits)
            u_array[time_step, Ellipsis] = u
            v_array[time_step, Ellipsis] = v
            t_array[time_step] = t * dt
            
            if early_stop == True:
                current_save = u_array[time_step]
                last_save = u_array[time_step-1]
                
                # Early stop if change over last 0.5 seconds below ss_tolerance
                if t > 0 and np.max(np.abs(current_save - last_save)) < ss_tolerance:
                    print(f'Steady state at t = {t * dt}', flush=True)
                    break
                
        u, v = update_laplace(u, v, reaction, Du, Dv, dt, dx, N, dim, params)

    # Remove zeros from arrays due to reaching steady state
    u_array = u_array[~np.all(u_array == 0, axis=1)]
    v_array = v_array[~np.all(v_array == 0, axis=1)]
    t_array = np.trim_zeros(t_array, 'b')    
    
    if save_name:
        np.savez(save_name, x_array, u_array, v_array, t_array)
                
    else:
        return x_array, u_array, v_array, t_array
    
if __name__ == '__main__':   
    # Define initial conditions
    u0 = 1
    v0 = 1.0246
    
    # Define parameters
    N = 200
    L = 10
    T = 100
    dim = 2
    
    # Define reaction
    reaction = turing_type
    params = [1, 1]
    Du, Dv = 0.01, 1
    
    # Calculate initial conditions for grid
    save_name = '/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/turing_type'
    
    u0, v0 = generate_initial_conditions(u0, v0, N, dim, random=True)
    
    # Simulate
    simulate(u0, v0, L, N, T, dim, reaction, params, Du, Dv, save_name=save_name)
    
    sim_formatted = format_data_general(dim, 2, file=f'{save_name}.npz')

    animate_data(sim_formatted, dim, 2, name=save_name)