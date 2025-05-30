import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Define function to plot steady state
def plot_steady_state(x_array, u_array, v_array):
    plt.figure()
    plt.plot(x_array, u_array[-1], color='orange', label='Cdc42T')
    plt.plot(x_array, v_array[-1], color='blue', label='Cdc42D')
    plt.legend()
    plt.ylim([0, 12])
    
# Define function to plot animation
def plot_animation(x_array, u_array, v_array, t_array, save=False, name='simulation'):
    # Create list of included frame numbers
    frames = range(0, len(t_array))
    
    # Define function to update data
    def animate(i):
        t_line.set_ydata(u_array[i, :])
        d_line.set_ydata(v_array[i, :])
        ax.set_title(f'T = {i/2}')

    # Plot animation
    fig, ax = plt.subplots()
    ax.set(ylim=(0, 12))
    t_line = ax.plot(x_array, u_array[0, :], color='orange', label='Cdc42T')[0]
    d_line = ax.plot(x_array, v_array[0, :], color='blue', label='Cdc42D')[0]

    anim = animation.FuncAnimation(fig, animate, frames=frames, repeat=True)
    
    if save:
        writergif = animation.PillowWriter(fps=5)
        anim.save(f'{name}.gif', writer=writergif)

    plt.draw()
    plt.show()
    
def animate_data(training_data, dimensions, species, name=None):
       
    frames = np.unique(training_data[:, dimensions])
    positions = np.unique(training_data[:, 0])
    
    species_arrays = []
    
    for i in range(species):
        specie = training_data[:, dimensions+i+1]
        specie_array = np.reshape(specie, ((len(frames),) + (len(positions),) * dimensions))
        species_arrays.append(specie_array)

    if dimensions == 1:
        # Instantiate figure and lines
        fig, ax = plt.subplots()
        ax.set(xlim=(np.min(positions), np.max(positions)), ylim=(0, 12))
        
        lines = [ax.plot([], [])[0] for _ in range(species)]

        # Define initialization function
        def init():
            for line in lines:
                line.set_data([], [])
            return lines
        
        # Define update function
        def animate(frame):
            for i, line in enumerate(ax.get_lines()):
                line.set_data(positions, species_arrays[i][frame, :])
                ax.set_title(f'T = {frames[frame]}')
        
    if dimensions == 2:
        # Instantiate figure
        fig, ax = plt.subplots()
        u_plot = ax.imshow(species_arrays[0][0, :, :], cmap='viridis')
        u_plot.set_clim(vmin=species_arrays[0][:, :, :].min(),
                        vmax=species_arrays[0][:, :, :].max())
        cbar = plt.colorbar(u_plot, ax=ax)
        
        # Define initialization function
        def init():
            return None

        # Define update function
        def animate(frame):
            u_plot.set_array(species_arrays[0][frame, :, :])
            ax.set_title(f'T = {frames[frame]}')
        
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=range(len(frames)), init_func=init, repeat=True)
    
    if name:
        writergif = animation.PillowWriter(fps=5)
        anim.save(f'{name}.gif', writer=writergif)

    return anim