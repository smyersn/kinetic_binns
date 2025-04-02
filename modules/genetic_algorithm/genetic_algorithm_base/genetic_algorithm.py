import sys
sys.path.append('../../')

from modules.utils.imports import *
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns import BINN
from modules.loaders.format_data import format_data
from modules.generate_data.simulate_system import reaction

from save_ga_results import write_terms
from custom_deap_functions import (calculate_poly_terms, calculate_hill_terms,
                                   custom_individual, custom_mutation_function)
from plot_ga_results import plot_best_ind_f_true, plot_fitness_evolution
from ga_score_function import score_function, score_function_helper

from deap import base, creator, tools, algorithms

###############################################################################
#PARAMETERS
###############################################################################
number_of_runs = 100
number_of_generations = 10000
number_of_individuals = 100
number_of_species = 2
degree = 2
mutation_rate = 0.3
crossover_rate = 0.5
nonzero_coef = 0.001
filename = 'output'

# Generate candidate terms for 
poly_terms = calculate_poly_terms(number_of_species, degree)
hill_terms = calculate_hill_terms(number_of_species, degree)
# Caclulate total parameters (3 per Hill and 2 types of Hill: incr. + decr.)
number_of_params = len(poly_terms) + 3 * 2 * len(hill_terms) 

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
model.load('model_weights_best_val_model', device=device)

# Load and format training data
path = '/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/'
file_name = 'spikes_data.npz'
xt, u, v, shape_u, shape_v = format_data(path+file_name, plot=False)
u_triangle_mesh, v_triangle_mesh = lltriangle(u, v)

# Calculate true reaction surface
true_params = [1, 1, 0.01]
F_true = reaction(u_triangle_mesh, v_triangle_mesh, true_params)[0]


dir_to_use = os.getcwd() + '/' + filename
# Check if dir exists and make
if not os.path.isdir(dir_to_use):
	os.makedirs(dir_to_use)
	# and make README file:
	fn = dir_to_use + '/' + 'output.txt'
	file = open(fn, 'w')

	# Write pertinent info at top
	file.write('OUTPUT\n\n')
	file.write('Filename: ' + filename + '\n')
	file.write('Directory: ' + dir_to_use + '\n')
	file.write('Generations: ' + str(number_of_generations) + '\n')
	file.write('Individuals: ' + str(number_of_individuals) + '\n')
	file.write('Mutation rate: ' + str(mutation_rate) + '\n')
	file.write('Crossover rate: ' + str(crossover_rate) + '\n\n')
	file.close()
 
###############################################################################
#RUN GENETIC ALGORITHM
###############################################################################

# create arrays to store average from each run, best from each run, and 
best_ind_of_all_runs = np.zeros((number_of_runs, number_of_params + 1))
best_fit_of_all_gens = np.zeros((number_of_runs, number_of_generations + 1))

for run in range(number_of_runs):
    
    # TYPE
    # Create minimizing fitness class w/ single objective:
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    # Create individual class:
    creator.create('Individual', list, fitness=creator.FitnessMin)

    # TOOLBOX
    toolbox = base.Toolbox()
    toolbox.register('individual', custom_individual, creator.Individual,
                  poly_terms, hill_terms)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # GENETIC OPERATORS:
    # score function
    toolbox.register('evaluate', score_function_helper, u_mesh=u_triangle_mesh,
                     v_mesh=v_triangle_mesh, poly_terms=poly_terms, 
                     hill_terms=hill_terms, F=F_true,
                     nonzero_coef=nonzero_coef, mse=False)
    # crossover function
    toolbox.register('mate', tools.cxTwoPoint)
    # mutation function
    toolbox.register('mutate', custom_mutation_function, indpb=0.2, 
                  number_of_params=number_of_params, poly_terms=poly_terms, 
                  hill_terms=hill_terms)
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
    
    print(f'{run} of {number_of_runs} runs completed', flush=True)
        
    # Save best individual/fitness from run and save in all-time array
    for gen in range(number_of_generations + 1):
        # get best fitness for each generation
        gen_scores = logbook[gen]['all'][:, 0]
        best_fit_of_all_gens[run, gen] = np.min(gen_scores)
        
    best_gen_of_run_idx = np.argmin(best_fit_of_all_gens[run, :])
    best_ind_of_gen_idx = np.argmin(logbook[best_gen_of_run_idx]['all'][:, 0])
    best_ind_of_run = logbook[best_gen_of_run_idx]['all'][best_ind_of_gen_idx]
 
    best_ind_of_all_runs[run] = best_ind_of_run

# Find single best individual in any run and save array
best_of_all_runs_sorted = sorted(best_ind_of_all_runs, key=lambda x: x[0])
np.savez(dir_to_use + '/best_of_all_runs_sorted', best_of_all_runs_sorted)

# Write equations to output file
file = open(fn, 'a')
for ind in range(len(best_of_all_runs_sorted)):
    terms = write_terms(best_of_all_runs_sorted[ind], poly_terms, hill_terms)
    file.write(f'Individual {ind + 1} (Fitness: {best_of_all_runs_sorted[ind][0]}):\n')
    for term in terms:
        file.write(f'{term}\n')
    file.write('\n\n')
file.close()

###############################################################################
#PLOT RESULTS
###############################################################################

plot_best_ind_f_true(best_of_all_runs_sorted[0][1:], u_triangle_mesh,
                     v_triangle_mesh, F_true, poly_terms, hill_terms,
                     dir_to_use)

plot_fitness_evolution(number_of_generations, best_fit_of_all_gens, dir_to_use,
                       1000)