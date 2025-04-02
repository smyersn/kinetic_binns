import numpy as np
from itertools import combinations_with_replacement, product

# Define functions for generating candidate terms
def calculate_poly_terms(n_species, degree):
    # Get indices for all polynomial terms up to degree for n_species (order insensitive)
    all_combos = []
    for degree in range(1, degree+1):
        combos = combinations_with_replacement(range(n_species), degree)
        for combo in combos:
            all_combos.append(combo)
    return all_combos

def calculate_hill_terms(n_species, degree):
    # Get indices for all possible Hill functions for n_species (order sensitive)
    all_combos = []
    for degree in range(1, degree+1):
        combos = product(range(n_species), repeat=degree)
        for combo in combos:
            all_combos.append(combo)
            
    # Remove uu and vv repeat combos
    for combo in all_combos:
        if len(combo) > 1 and combo[0] == combo[1]:
            all_combos.remove(combo)
    return all_combos

def custom_individual(ind_class, poly_terms, num_nonzero_terms, 
                      coef_bounds):
    
    lower = coef_bounds[0]
    upper = coef_bounds[1]

    # individual initialized by randomly selecting num_nonzero_terms to sample
    # the coef bounds    
    coefs = np.zeros(len(poly_terms))

    return ind_class(coefs)

def custom_mutation_function(individual, indpb, number_of_params, poly_terms, 
                            coef_bounds):
    lower = coef_bounds[0]
    upper = coef_bounds[1]
    
    lows = np.repeat(lower, len(individual))
    ups = np.repeat(upper, len(individual))
    
    
    # mutate
    for i, low, up in zip(range(number_of_params), lows, ups):
        draw = np.random.random()
        if draw < indpb:
            individual[i] = np.random.uniform(low, up)
        # zero mutation excludes n and k parameters        
        if draw > 1 - indpb:
            individual[i] = 0
        
    return individual,
