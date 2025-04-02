import time as timeski
import os
import numpy as np

def write_terms(individual, poly_terms, hill_terms):
    
    terms = []
    
    # drop fitness, keep params
    individual = individual[1:]

    # Break up learned parameters by type. poly_params are just one param for 
    # each term. hill_params are coefs, k, and n for each term
    poly_params = individual[:len(poly_terms)]  
    
    hill_params = individual[len(poly_terms):len(poly_terms) + 2 * len(hill_terms)]
    hill_params_increasing, hill_params_decreasing = np.split(hill_params, 2)

    hill_ks = individual[len(poly_terms) + 2 * len(hill_terms):len(poly_terms) + 4 * len(hill_terms)]
    hill_ks_increasing, hill_ks_decreasing = np.split(hill_ks, 2)
 
    hill_ns = individual[len(poly_terms) + 4 * len(hill_terms):]
    hill_ns_increasing, hill_ns_decreasing = np.split(hill_ns, 2)

    species = ['u', 'v']
 
    for term, param in zip(poly_terms, poly_params):
        if param != 0:
            string = f'{param:.2f}'
            for ind in term:
                string += f' * {species[ind]}'
            terms.append(string)
    
    for term, param, k, n in zip(hill_terms, hill_params_increasing, hill_ks_increasing, hill_ns_increasing):
        if param != 0:
            string = f'{param:.2f}'
            if len(term) == 1:
                string += f' * {species[term[0]]}^{int(n)} / (1 + {k:.2f} * {species[term[0]]}^{int(n)})'
            else:     
                string += f' * {species[term[0]]} * {species[term[1]]}^{int(n)} / (1 + {k:.2f} * {species[term[1]]}^{int(n)})'
            terms.append(string)
    
    for term, param, k, n in zip(hill_terms, hill_params_decreasing, hill_ks_decreasing, hill_ns_decreasing):
        if param != 0:
            string = f'{param:.2f}'
            if len(term) == 1:
                string += f' * (1 - {species[term[0]]}^{int(n)} / (1 + {k:.2f} * {species[term[0]]}^{int(n)}))'
            else:     
                string += f' * {species[term[0]]} * (1 - {species[term[1]]}^{int(n)} / (1 + {k:.2f} * {species[term[1]]}^{int(n)}))'
            terms.append(string)
    
    return terms
