def wave_pinning(uv, params=(1, 1, 0.01)):
    u, v = uv[:, 0], uv[:, 1]
    a, b, k = params
    
    F = (a * u**2 * v) / (1 + k * u**2) - b * u
    return F

def turing_type(uv, params=(1, 1)):
    u, v = uv[:, 0], uv[:, 1]
    a, b = params
    
    F = a * u**2 * v - b * u
    return F

def custom_equation(uv, params=(1, 1, 0.1)):
    u, v = uv[:, 0], uv[:, 1]
    a, b, k = params
    
    F = a * u**2 * v - (b * u**2 / (1 + k * u**2))
    return F
