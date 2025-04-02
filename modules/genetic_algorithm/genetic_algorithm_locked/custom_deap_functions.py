import numpy as np
from itertools import combinations_with_replacement, product

def custom_individual(ind_class, coef_bounds):
    
    lower = coef_bounds[0]
    upper = coef_bounds[1]

    # individual initialized by randomly selecting num_nonzero_terms to sample
    # the coef bounds    
    coefs = np.zeros(3)
    
    a = np.random.uniform(lower, upper)
    b = np.random.uniform(lower, upper)
    k = np.random.uniform(0, upper)
    n = np.random.randint(1, 5)
    all = np.hstack((a, b, k, n))

    return ind_class(all)

def custom_mutation_function(individual, indpb, coef_bounds):
    lower = coef_bounds[0]
    upper = coef_bounds[1]
    
    # bound coefficient mutations from -1000 to 1000
    lows = np.repeat(lower, 4)
    ups = np.repeat(upper, 4)
    
    # k and n can't be negative
    lows[2:] = 0
    
    # n can't be less than one
    lows[3:] = 1
    
    # n can't be greater than four
    ups[3:] = 5
    
    # mutate
    for i, low, up in zip(range(4), lows, ups):
        draw = np.random.random()
        if draw < indpb:
            if i <= 2:
                individual[i] = np.random.uniform(low, up)
            if i == 3:
                individual[i] = np.random.randint(low, up)
        
    return individual,
