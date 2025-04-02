import numpy as np
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