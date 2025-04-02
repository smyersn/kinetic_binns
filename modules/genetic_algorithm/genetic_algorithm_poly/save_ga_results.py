import time as timeski
import os
import numpy as np

def write_terms(individual, poly_terms):
    
    terms = []
    
    # drop fitness, keep params
    individual = individual[1:]

    # Break up learned parameters by type. poly_params are just one param for 
    # each term. hill_params are coefs, k, and n for each term
    poly_params = individual[:len(poly_terms)]  
    
    species = ['u', 'v']
 
    for term, param in zip(poly_terms, poly_params):
        if param != 0:
            string = f'{param:.2f}'
            for ind in term:
                string += f' * {species[ind]}'
            terms.append(string)
        
    return terms
