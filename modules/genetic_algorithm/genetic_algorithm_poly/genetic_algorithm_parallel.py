import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns import BINN
from modules.loaders.format_data import format_data
from modules.generate_data.simulate_system import reaction

from modules.genetic_algorithm_poly.save_ga_results import write_terms
from modules.genetic_algorithm_poly.custom_deap_functions import (
    calculate_poly_terms, calculate_hill_terms, custom_individual,
    custom_mutation_function)
from modules.genetic_algorithm_poly.ga_score_function import (score_function, 
    score_function_helper)

from deap import base, creator, tools, algorithms

###############################################################################
#PARAMETERS
###############################################################################
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
num_nonzero_terms=config['num_nonzero_terms']
coef_bounds=config['coef_bounds']
nonzero_coef = config['nonzero_coef']
error = config['error']
surface_to_fit = config['surface_to_fit']

# Generate candidate terms for 
poly_terms = calculate_poly_terms(number_of_species, degree)
# Caclulate total parameters (3 per Hill and 2 types of Hill: incr. + decr.)
number_of_params = len(poly_terms)

###############################################################################
#LOAD/FORMAT DATA AND GENERATE OUTPUT FILE
###############################################################################
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

 
###############################################################################
#RUN GENETIC ALGORITHM
###############################################################################
    
# TYPE
# Create minimizing fitness class w/ single objective:
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
# Create individual class:
creator.create('Individual', list, fitness=creator.FitnessMin)

# TOOLBOX
toolbox = base.Toolbox()
toolbox.register('individual', custom_individual, creator.Individual,
                poly_terms, num_nonzero_terms, coef_bounds)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# GENETIC OPERATORS:
# score function
toolbox.register('evaluate', score_function_helper, u_mesh=u_triangle_mesh,
                    v_mesh=v_triangle_mesh, poly_terms=poly_terms, 
                    F_mlp=F_mlp, F_true=F_true,
                    nonzero_coef=nonzero_coef, error=error,
                    surface_to_fit=surface_to_fit)
# crossover function
toolbox.register('mate', tools.cxTwoPoint)
# mutation function
toolbox.register('mutate', custom_mutation_function, indpb=0.2, 
                number_of_params=number_of_params, poly_terms=poly_terms, 
                coef_bounds=coef_bounds)
# selection function
toolbox.register('select', tools.selTournament, tournsize=3)

# EVOLUTION!
pop = toolbox.population(n=number_of_individuals)
hof = tools.HallOfFame(1)
stats = tools.Statistics(key = lambda ind: np.hstack((ind.fitness.values, ind)))
stats.register('all', np.copy)

# Using built in eaSimple algo
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_rate,
                                    mutpb=mutation_rate, 
                                    ngen=number_of_generations,
                                    stats=stats, halloffame=hof,
                                    verbose=False)
    
# Determine best fitness for each generation
best_fit_per_gen = np.zeros(number_of_generations + 1)

for gen in range(number_of_generations + 1):
    gen_scores = logbook[gen]['all'][:, 0]
    best_fit_per_gen[gen] = np.min(gen_scores)

# Determine and save best fitness individual of all generations    
best_gen_of_run_idx = np.argmin(best_fit_per_gen)
best_ind_of_gen_idx = np.argmin(logbook[best_gen_of_run_idx]['all'][:, 0])
best_ind_of_run = logbook[best_gen_of_run_idx]['all'][best_ind_of_gen_idx]
np.savez(f'results/runs/best_ind_run_{sys.argv[2]}', best_ind_of_run, best_fit_per_gen)