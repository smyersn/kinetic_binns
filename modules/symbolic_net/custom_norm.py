import torch

def custom_norm(w, a=0.05):
    # Compute the absolute values of w
    abs_w = torch.abs(w)
    
    # Initialize an empty tensor to store results
    norm = torch.zeros_like(w)
    
    # Case 1: |w| >= a
    mask1 = abs_w >= a
    norm[mask1] = abs_w[mask1].sqrt()  # |w|^(1/2)
    
    # Case 2: |w| < a
    mask2 = abs_w < a
    w2 = w[mask2]
    
    term = (-w2**4 / (8 * a**3) + 3 * w2**2 / (4 * a) + 3 * a / 8)
    norm[mask2] = term
    
    return norm.sum()