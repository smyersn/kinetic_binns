def hill_poly(uv, params):
    u, v = uv[:, 0], uv[:, 1]
    a, b, k, n, *_ = params
    
    F = (a * u**n * v) / (1 + k * u**n) - b * u
    return F

def poly_poly(uv, params):
    u, v = uv[:, 0], uv[:, 1]
    a, b, *_ = params
    
    F = a * u**2 * v - b * u
    return F

def hill_hill(uv, params):
    u, v = uv[:, 0], uv[:, 1]
    a, b, k1, n1, k2, n2, *_ = params
    
    F = (a * u**n1 * v) / (1 + k1 * u**n1) -  (b * u**n2) / (1 + k2 * u**n2)
    return F

def poly_hill(uv, params):
    u, v = uv[:, 0], uv[:, 1]
    a, b, k, n, *_ = params
    
    F = a * u**2 * v - (b * u**n) / (1 + k * u**n)
    return F