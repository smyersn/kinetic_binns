from modules.utils.imports import *
from modules.generate_data.simulate_system import wave_pinning
from modules.symbolic_net.write_terms import write_terms
from modules.symbolic_net.visualize_surface import visualize_surface
from modules.symbolic_net.individual import individual

def analyze_model(model, dir_name, training_data, device):
    # Get all uv vals seen during training
    uv_training_data = training_data[:, -2:]
    
    # Create triangle mesh from min and max uv vals seen in training data
    u_triangle_mesh, v_triangle_mesh = lltriangle(uv_training_data[:, 0], 
                                                  uv_training_data[:, 1])
    # Create 1d arrays from meshes
    u_triangle, v_triangle = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)
    
    # Create separate variables for arrays containing and not containing nans
    uv_nans = np.stack((u_triangle, v_triangle), axis=1)
    mask = ~np.isnan(uv_nans).any(axis=1)
    uv = uv_nans[mask]

    # Generate true surface
    params = 1, 1, 0.01
    F_true = wave_pinning(u_triangle, v_triangle, params).reshape(501, 501)
    
    # Generate learned surface
    ind = individual(model.model.reaction.params, model.model.species, model.model.degree)
    F_mlp_unformatted = ind.predict_f(torch.tensor(uv_nans).to(device))
    F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)
    
    # Visualize
    visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                    F_true, F_mlp, 'f_mlp_surface_refined')

    # Refine and print equation       
    fn = f'{dir_name}/equation.txt'
    file = open(fn, 'w')

    file.write(f'Original Equation:\n')
    terms = ind.write_terms()
    for term in terms:
        file.write(f'{term}\n')

    file.write(f'\nNo insignificant terms:\n')
    ind.fix_insignificant_terms(torch.tensor(uv).to(device))
    terms = ind.write_terms()
    for term in terms:
        file.write(f'{term}\n')

    file.write(f'\nNo cheating Hill functions:\n')
    ind.fix_cheating_hill_functions(torch.tensor(uv).to(device))
    terms = ind.write_terms()
    for term in terms:
        file.write(f'{term}\n')

    file.close()