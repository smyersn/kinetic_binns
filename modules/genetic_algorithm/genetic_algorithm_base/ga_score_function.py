import sys
sys.path.append('../../')
import numpy as np
from modules.genetic_algorithm.individual_to_surface import (
    individual_to_surface)


def score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, F_mlp, 
                   F_true, nonzero_coef, error, surface_to_fit):
    
    individual = np.array(individual)
    individual_surface = individual_to_surface(individual, u_mesh, v_mesh,
                                               poly_terms, hill_terms)
    if surface_to_fit == 'F_true':
        F = F_true
    elif surface_to_fit == 'F_mlp':
        F = F_mlp
    
    if error == 'mse':
        error_loss = np.nanmean(np.square(np.subtract(individual_surface, F)))
    elif error == 'sse':
        error_loss = np.nansum(np.square(np.subtract(individual_surface, F)))

    # Calculate loss from number of non-zero coefficients
    num_nonzero_params = np.count_nonzero(individual[:len(poly_terms) 
                                                     + 2 * len(hill_terms)])
    nonzero_loss = nonzero_coef * num_nonzero_params
    return error_loss + nonzero_loss,

def score_function_helper(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit):
    return score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit)

#
#
#
   
def aic_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, F_mlp, 
                   F_true, nonzero_coef, error, surface_to_fit):
    
    individual = np.array(individual)
    individual_surface = individual_to_surface(individual, u_mesh, v_mesh,
                                               poly_terms, hill_terms)
    if surface_to_fit == 'F_true':
        F = F_true
    elif surface_to_fit == 'F_mlp':
        F = F_mlp
    
    RSS = np.nansum(np.square(np.subtract(individual_surface, F)))

    # Calculate loss from number of non-zero coefficients
    num_poly_params = np.count_nonzero(individual[:len(poly_terms)])
                                                     
    num_hill_params = 3 * np.count_nonzero(individual[len(poly_terms):len(poly_terms) 
                                                     + 2 * len(hill_terms)])
    
    k = num_poly_params + num_hill_params
    
    flat_mesh = np.column_stack((u_mesh.flatten(order='F'), v_mesh.flatten(order='F')))
    
    n = np.sum(np.all(~np.isnan(flat_mesh), axis=1))
    
    aic = 2 * k + n * np.log(RSS / n)
    
    return aic,

def aic_score_function_helper(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit):
    return aic_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit)

#
#
#
    
def bic_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, F_mlp, 
                   F_true, nonzero_coef, error, surface_to_fit):
    
    individual = np.array(individual)
    individual_surface = individual_to_surface(individual, u_mesh, v_mesh,
                                               poly_terms, hill_terms)
    if surface_to_fit == 'F_true':
        F = F_true
    elif surface_to_fit == 'F_mlp':
        F = F_mlp
    
    RSS = np.nansum(np.square(np.subtract(individual_surface, F)))

    # Calculate loss from number of non-zero coefficients
    num_poly_params = np.count_nonzero(individual[:len(poly_terms)])
                                                     
    num_hill_params = 3 * np.count_nonzero(individual[len(poly_terms):len(poly_terms) 
                                                     + 2 * len(hill_terms)])
    
    k = num_poly_params + num_hill_params
    
    flat_mesh = np.column_stack((u_mesh.flatten(order='F'), v_mesh.flatten(order='F')))
    
    n = np.sum(np.all(~np.isnan(flat_mesh), axis=1))
    
    bic = np.log(n) * k + n * np.log(RSS / n)
    
    return bic,

def bic_score_function_helper(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit):
    return bic_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit)
    
#
#
#

def num_terms_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, F_mlp, 
                   F_true, nonzero_coef, error, surface_to_fit):
    
    individual = np.array(individual)
    individual_surface = individual_to_surface(individual, u_mesh, v_mesh,
                                               poly_terms, hill_terms)
    if surface_to_fit == 'F_true':
        F = F_true
    elif surface_to_fit == 'F_mlp':
        F = F_mlp
        
    RSS = np.nansum(np.square(np.subtract(individual_surface, F)))
    
    # Calculate loss from number of non-zero coefficients
    num_terms = np.count_nonzero(individual[:len(poly_terms) 
                                                     + 2 * len(hill_terms)])
    
    term_error = np.abs(2 - num_terms) * 10000
    
    return RSS + term_error,

def num_terms_score_function_helper(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit):
    return num_terms_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit)

#
#
# 

def exact_terms_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, F_mlp, 
                   F_true, nonzero_coef, error, surface_to_fit):
    
    individual = np.array(individual)
    individual_surface = individual_to_surface(individual, u_mesh, v_mesh,
                                               poly_terms, hill_terms)
    if surface_to_fit == 'F_true':
        F = F_true
    elif surface_to_fit == 'F_mlp':
        F = F_mlp
    
    RSS = np.nansum(np.square(np.subtract(individual_surface, F)))

    # Calculate loss from number of non-zero coefficients
    num_poly_terms = np.count_nonzero(individual[:len(poly_terms)])
                                                     
    num_hill_terms = np.count_nonzero(individual[len(poly_terms):len(poly_terms) 
                                                     + 2 * len(hill_terms)])
    
    poly_term_error = np.abs(1 - num_poly_terms) * 10000
    
    hill_term_error = np.abs(1 - num_hill_terms) * 10000
    
    return RSS + poly_term_error + hill_term_error,

def exact_terms_score_function_helper(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit):
    return exact_terms_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit)
    
#
#
#

def bic_num_terms_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, F_mlp, 
                   F_true, nonzero_coef, error, surface_to_fit):
    
    individual = np.array(individual)
    individual_surface = individual_to_surface(individual, u_mesh, v_mesh,
                                               poly_terms, hill_terms)
    if surface_to_fit == 'F_true':
        F = F_true
    elif surface_to_fit == 'F_mlp':
        F = F_mlp
        
    RSS = np.nansum(np.square(np.subtract(individual_surface, F)))

    # Calculate BIC
    num_poly_params = np.count_nonzero(individual[:len(poly_terms)])
                                                     
    num_hill_params = 3 * np.count_nonzero(individual[len(poly_terms):len(poly_terms) 
                                                     + 2 * len(hill_terms)])
    
    k = num_poly_params + num_hill_params
    
    flat_mesh = np.column_stack((u_mesh.flatten(order='F'), v_mesh.flatten(order='F')))
    
    n = np.sum(np.all(~np.isnan(flat_mesh), axis=1))
    
    bic = np.log(n) * k + n * np.log(RSS / n)
    
    # Calculate loss from number number of terms
    num_terms = np.count_nonzero(individual[:len(poly_terms) 
                                                     + 2 * len(hill_terms)])
    
    term_error = np.abs(2 - num_terms) * 10000
    
    return bic + term_error,

def bic_num_terms_score_function_helper(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit):
    return bic_num_terms_score_function(individual, u_mesh, v_mesh, poly_terms, hill_terms, 
                          F_mlp, F_true, nonzero_coef, error, surface_to_fit)
