import numpy as np

def individual_to_surface(individual, u_mesh, v_mesh, poly_terms):
    #unpack individual
    poly_params = individual[:len(poly_terms)]
    
    # Create vector to store surfaceprediction for learned function
    mesh = np.column_stack((np.ravel(u_mesh), np.ravel(v_mesh)))
    total_surface = np.zeros(len(mesh))

    # Contribution to surface from polynomial terms
    for term, param in zip(poly_terms, poly_params):
        term_surface = np.repeat(param, len(mesh))
        for idx in term:
            term_surface = (term_surface * mesh[:, idx])
        total_surface += term_surface
	    
    # Reshape to match mesh dimensions    
    total_surface = np.reshape(total_surface, (101, 101))

    return total_surface