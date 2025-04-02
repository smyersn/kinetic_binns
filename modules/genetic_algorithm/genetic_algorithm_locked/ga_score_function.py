import sys
sys.path.append('../../')
import numpy as np
from modules.genetic_algorithm_locked.individual_to_surface import (
    individual_to_surface)


def score_function(individual, u_mesh, v_mesh, F_mlp, F_true, nonzero_coef,
                   error, surface_to_fit):
    
    individual = np.array(individual)
    individual_surface = individual_to_surface(individual, u_mesh, v_mesh)
    if surface_to_fit == 'F_true':
        F = F_true
    elif surface_to_fit == 'F_mlp':
        F = F_mlp
    
    if error == 'mse':
        error_loss = np.nanmean(np.square(np.subtract(individual_surface, F)))
    elif error == 'sse':
        error_loss = np.nansum(np.square(np.subtract(individual_surface, F)))

    return error_loss,

def score_function_helper(individual, u_mesh, v_mesh, F_mlp, F_true,
                          nonzero_coef, error, surface_to_fit):
    return score_function(individual, u_mesh, v_mesh, F_mlp, F_true,
                          nonzero_coef, error, surface_to_fit)