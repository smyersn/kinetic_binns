import numpy as np

def individual_to_surface(individual, u_mesh, v_mesh):
 
    # Create vector to store surfaceprediction for learned function
    mesh = np.column_stack((np.ravel(u_mesh), np.ravel(v_mesh)))
    total_surface = np.zeros(len(mesh))
    
    a, b, k, n = individual

    total_surface += (a * mesh[:, 0]**n * mesh[:, 1]) / (1 + k * mesh[:, 0]**n) + b * mesh[:, 0]
	
    # Reshape to match mesh dimensions    
    total_surface = np.reshape(total_surface, (101, 101))

    return total_surface