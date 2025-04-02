import numpy as np

def individual_to_surface(individual, u_mesh, v_mesh, poly_terms, hill_terms):
    #unpack individual
    poly_params = individual[:len(poly_terms)]

    hill_params = individual[len(poly_terms):len(poly_terms) + 2 * len(hill_terms)]
    hill_params_increasing, hill_params_decreasing = np.split(hill_params, 2)

    hill_ks = individual[len(poly_terms) + 2 * len(hill_terms):len(poly_terms) + 4 * len(hill_terms)]
    hill_ks_increasing, hill_ks_decreasing = np.split(hill_ks, 2)
 
    hill_ns = individual[len(poly_terms) + 4 * len(hill_terms):]
    hill_ns_increasing, hill_ns_decreasing = np.split(hill_ns, 2)
    
    # Create vector to store surfaceprediction for learned function
    mesh = np.column_stack((np.ravel(u_mesh), np.ravel(v_mesh)))
    total_surface = np.zeros(len(mesh))

    # Contribution to surface from polynomial terms
    for term, param in zip(poly_terms, poly_params):
        term_surface = np.repeat(param, len(mesh))
        for idx in term:
            term_surface *= mesh[:, idx]
        total_surface += term_surface
	
    # Contriubtion to surface from increasing Hill function terms
    for term, param, k, n in zip(hill_terms, hill_params_increasing, hill_ks_increasing, hill_ns_increasing):
        term_surface = np.repeat(param, len(mesh)) 
        if len(term) == 1:
            term_surface *= (mesh[:, term[0]]**n / (1 + k * mesh[:, term[0]]**n))
        else:
            term_surface *= ((mesh[:, term[0]] * mesh[:, term[1]]**n) / (1 + k * mesh[:, term[1]]**n))
        total_surface += term_surface
 
    # Contriubtion to surface from decreasing Hill function terms
    for term, param, k, n in zip(hill_terms, hill_params_decreasing, hill_ks_decreasing, hill_ns_decreasing): 
        term_surface = np.repeat(param, len(mesh))
        if len(term) == 1:
            term_surface *= 1 - (mesh[:, term[0]]**n / (1 + k * mesh[:, term[0]]**n))
        else:
            term_surface *= (mesh[:, term[0]] * (1 - (mesh[:, term[1]]**n) / (1 + k * mesh[:, term[1]]**n)))
        total_surface += term_surface
    
    # Reshape to match mesh dimensions    
    total_surface = np.reshape(total_surface, (101, 101))

    return total_surface