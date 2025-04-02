import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from pathlib import Path
from modules.utils.imports import *
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns import BINN
from modules.loaders.format_data import format_data
from modules.generate_data.simulate_system import reaction

from modules.genetic_algorithm_poly.plot_ga_results import (plot_best_ind_f_true, 
    plot_best_ind_f_mlp, plot_fitness_evolution)
from modules.genetic_algorithm_poly.custom_deap_functions import (
    calculate_poly_terms, calculate_hill_terms)
from modules.genetic_algorithm_poly.save_ga_results import write_terms

# load params from configuration file
config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

number_of_runs = config['number_of_runs']
number_of_generations = config['number_of_generations']
number_of_individuals = config['number_of_individuals']
number_of_species = config['number_of_species']
degree = config['degree']
mutation_rate = config['mutation_rate']
crossover_rate = config['crossover_rate']
nonzero_coef = config['nonzero_coef']
error = config['error']
surface_to_fit = config['surface_to_fit']

# calculate number of parameters
poly_terms = calculate_poly_terms(number_of_species, degree)
number_of_params = len(poly_terms)

# save individuals from all runs to best_individuals array and sort
best_individuals = np.zeros((number_of_runs, number_of_params+1))
fitness_evolution = np.zeros((number_of_runs, number_of_generations+1))

for i in range(number_of_runs):
    best_individuals[i] = np.load(f'results/runs/best_ind_run_{i}.npz')['arr_0']
    fitness_evolution[i] = np.load(f'results/runs/best_ind_run_{i}.npz')['arr_1']
    

best_individuals_sorted = sorted(best_individuals, key=lambda x: x[0])

# write best individuals to output file
fn = 'results/best_equations.txt'

file = open(fn, 'w')
for ind in range(len(best_individuals_sorted)):
    terms = write_terms(best_individuals_sorted[ind], poly_terms)
    file.write(f'Individual {ind + 1} (Fitness: {best_individuals_sorted[ind][0]}):\n')
    for term in terms:
        file.write(f'{term}\n')
    file.write('\n\n')
file.close()

# Instantiate BINN
device = torch.device(get_lowest_gpu(pick_from=[0,1,2,3]))
binn = BINN().to(device)
parameters = binn.parameters()
model = model_wrapper(
    model=binn,
    optimizer=None,
    loss=None)
model.load(f'{repo_start}weights/best_fit_weights', device=device)

# Load and format training data
xt, u, v, shape_u, shape_v = format_data(f'{repo_start}data/spikes_data.npz', plot=False)
u_triangle_mesh, v_triangle_mesh = lltriangle(u, v)

# Calculate true reaction surface
true_params = [1, 1, 0.01]
F_true = reaction(u_triangle_mesh, v_triangle_mesh, true_params)[0]

# Calculate MLP reaction surface
uv = np.column_stack((np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)))
F_mlp_stacked = to_numpy(binn.reaction(to_torch(uv)[:, None]))
F_mlp = np.reshape(F_mlp_stacked, (101, 101))

# plot best individual
if surface_to_fit == 'F_true':
    plot_best_ind_f_true(best_individuals_sorted[0][1:], u_triangle_mesh,
                        v_triangle_mesh, F_true, poly_terms,
                        'results')
elif surface_to_fit == 'F_mlp':
    plot_best_ind_f_mlp(best_individuals_sorted[0][1:], u_triangle_mesh,
                        v_triangle_mesh, F_true, F_mlp, poly_terms,
                        'results')

plot_fitness_evolution(number_of_generations, fitness_evolution, 
                       'results', 1000)