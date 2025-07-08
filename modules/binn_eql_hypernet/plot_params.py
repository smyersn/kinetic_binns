import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_params(all_ws, all_sigmas, save_path=None):
    ws = np.array(all_ws)    # shape (E, P)
    sigmas  = np.array(all_sigmas)     # shape (E, P)
    epochs = np.arange(0, ws.shape[0])
    
    labels = ['u', 'v', 'u*u', 'v*v', 'u*v', 'inchill(u)', 'inchill(v)',
              'inchill(u)*v', 'inchill(v)*u', 'dechill(u)', 'dechill(v)',
              'dechill(u)*v', 'dechill(v)*u']

    plt.figure(figsize=(10, 6))

    for p in range(ws.shape[1]):
        m = ws[:, p]
        s = sigmas[:, p]
        label = labels[p]

        # Plot the mean and grab its color
        line, = plt.plot(epochs, m, label=label)
        color = line.get_color()

        # Plot mean ± std in the same color
        plt.plot(epochs, m + s, linestyle=':', alpha=0.7, color=color)
        plt.plot(epochs, m - s, linestyle=':', alpha=0.7, color=color)

    plt.xlabel('Epoch (1000)')
    plt.ylabel('Value')
    plt.title('Parameter Evolution')
    plt.xticks(epochs)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)

    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}/parameter_evolution.png', dpi=300)
    plt.show()
