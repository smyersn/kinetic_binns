from modules.utils.imports import *
from modules.generate_data.simulate_system import wave_pinning
from modules.symbolic_net.visualize_surface import visualize_surface
from modules.binn_eql_normalized.simulate_surface import simulate_surface

def analyze_model(model, dir_name, training_data, device):
    # Create triangle mesh from min and max uv vals seen in training data
    u_triangle_mesh, v_triangle_mesh = lltriangle(training_data[:, -2:], 
                                                  training_data[:, -1:])
    # Create 1d arrays from meshes
    u_triangle, v_triangle = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)
    
    # Create separate variables for arrays containing and not containing nans
    uv_nans = np.stack((u_triangle, v_triangle), axis=1)
    mask = ~np.isnan(uv_nans).any(axis=1)
    uv = torch.from_numpy(uv_nans[mask]).to(device)
    
    # Generate true surface
    params = 1, 1, 0.01
    F_true = wave_pinning(uv_nans, params).reshape(501, 501)
    
    # Generate learned surface
    F_mlp_unformatted = model.model.reaction(torch.tensor(uv_nans).float().to(device))
    F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)
    
    # Visualize
    visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                    F_true, F_mlp, 'f_mlp_surface_unrefined')

    # Refine and print equation       
    fn = f'{dir_name}/equation.txt'
    file = open(fn, 'w')

    file.write(f'Original Equation:\n')
    for term in model.model.generate_equation():
        file.write(f'{term}\n')

    file.write(f'\nNo insignificant terms:\n')
    model.model.remove_insignificant_terms(uv)
    for term in model.model.generate_equation():
        file.write(f'{term}\n')

    file.write(f'\nNo cheating Hill functions:\n')
    model.model.fix_cheating_hill_functions(uv)
    for term in model.model.generate_equation():
        file.write(f'{term}\n')

    file.close()
    
    # Generate learned surface
    F_mlp_unformatted = model.model.reaction(torch.tensor(uv_nans).float().to(device))
    F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)
        
    # Visualize
    visualize_surface(dir_name, u_triangle_mesh, v_triangle_mesh,
                    F_true, F_mlp, 'f_mlp_surface_refined')
    
    # Simulate
    reaction = model.model.reaction.eql_layer
    
    if not model.model.diff_coeffs:
        D = model.model.diffusion_fitter()
        Du = D[0].detach().numpy()
        Dv = D[1].detach().numpy()
        
    else:
        Du, Dv = 0.01, 1

    simulate_surface(training_data, 2, 2, reaction, Du, Dv, dir_name)