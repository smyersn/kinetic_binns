import time as timeski
import os
import numpy as np

def write_terms(individual):
    
    terms = []
    
    # drop fitness, keep params
    individual = individual[1:]
    
    a, b, k, n = individual
    
    terms.append(f'a = {a}')
    terms.append(f'b = {b}')
    terms.append(f'k = {k}')
    terms.append(f'n = {n}')

    
    return terms
