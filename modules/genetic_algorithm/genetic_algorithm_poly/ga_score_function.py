import sys
sys.path.append('../../')
import numpy as np
from modules.genetic_algorithm_poly.individual_to_surface import (
    individual_to_surface)


def score_function(individual, u_mesh, v_mesh, poly_terms, F_mlp, 
                   F_true, nonzero_coef, error, surface_to_fit):
    
    individual = np.array(individual)
    individual_surface = individual_to_surface(individual, u_mesh, v_mesh,
                                               poly_terms)
    if surface_to_fit == 'F_true':
        F = F_true
    elif surface_to_fit == 'F_mlp':
        F = F_mlp
    
    if error == 'mse':
        error_loss = np.nanmean(np.square(np.subtract(individual_surface, F)))
    elif error == 'sse':
        error_loss = np.nansum(np.square(np.subtract(individual_surface, F)))

    # Calculate loss from number of non-zero coefficients
    num_nonzero_params = np.count_nonzero(individual[:len(poly_terms)])
    nonzero_loss = nonzero_coef * num_nonzero_params
    return error_loss + nonzero_loss,

def score_function_helper(individual, u_mesh, v_mesh, poly_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit):
    return score_function(individual, u_mesh, v_mesh, poly_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit)