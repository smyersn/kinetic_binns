import sys
sys.path.append('../../')
from modules.utils.imports import *
from modules.genetic_algorithm.individual_to_surface import (
    individual_to_surface)

def plot_best_ind_f_mlp(individual, u_mesh, v_mesh, F_true, F_mlp, poly_terms,
                         hill_terms, dir_to_use, zmax=11, ztick=2):
    
    ga_surface = individual_to_surface(individual, u_mesh, v_mesh, poly_terms,
                                       hill_terms)
    
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type':'scene'}, {'type':'scene'}, {'type':'scene'}]],
                        subplot_titles=('F(A, B)', 'F*(A, B)', 'Eq. Fit by Gen. Alg.'),
                        horizontal_spacing = 0)

    fig.layout.annotations[0].update(y=0.8)
    fig.layout.annotations[1].update(y=0.8)
    fig.layout.annotations[2].update(y=0.8)
    fig.update_annotations(font_size=24, font_color='#000000')

    
    fig.add_trace(
        go.Surface(z=F_true, x=u_mesh, y=v_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        x=1.15,
                                        len=0.5)),
        row=1, col=1)


    fig.add_trace(
        go.Surface(z=F_mlp, x=u_mesh, y=v_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        x=1.15,
                                        len=0.5)),
        row=1, col=2)
    
    fig.add_trace(
        go.Surface(z=ga_surface, x=u_mesh, y=v_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        len=0.5)),
        row=1, col=3)

    scene_dict = dict( 
        xaxis_title='[A] (uM)',
        yaxis_title='[B] (uM)',
        zaxis_title='F',
        camera=dict(eye=dict(x=1, y=-2.5, z=1)),
        xaxis = dict(
            tick0 = 0,
            dtick = 2,
            tickfont = dict(size=18)),
        yaxis = dict(
            tick0 = 0.2,
            dtick = 0.4,
            tickfont = dict(size=18)),
        zaxis = dict(
            tick0 = 0,
            dtick = ztick,
            tickfont = dict(size=18),
            range=[-1.5, zmax]),)

    
    fig.update_layout(autosize=True,
        width=1800, 
        height=800,
        font=dict(color = '#000000',
                size=20),
      
        scene=scene_dict,
                    
        scene2=scene_dict,
        
        scene3=scene_dict)

    fig.update_coloraxes(showscale=False)
    fig.show()
    fig.write_image(dir_to_use + '/eq_fitting.png')
    
def plot_best_ind_f_true(individual, u_mesh, v_mesh, F_true, poly_terms,
                         hill_terms, dir_to_use, zmax=11, ztick=2):
    
    ga_surface = individual_to_surface(individual, u_mesh, v_mesh, poly_terms,
                                       hill_terms)
        
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type':'scene'}, {'type':'scene'}]],
                        subplot_titles=('F(A, B)', 'Eq. Fit by Gen. Alg.'),
                        horizontal_spacing = 0)

    fig.layout.annotations[0].update(y=0.8)
    fig.layout.annotations[1].update(y=0.8)
    fig.update_annotations(font_size=24, font_color='#000000')

    fig.add_trace(
        go.Surface(z=F_true, x=u_mesh, y=v_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        x=1.15,
                                        len=0.5)),
        row=1, col=1)


    
    fig.add_trace(
        go.Surface(z=ga_surface, x=u_mesh, y=v_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        len=0.5)),
        row=1, col=2)

    scene_dict = dict( 
        xaxis_title='[A] (uM)',
        yaxis_title='[B] (uM)',
        zaxis_title='F',
        camera=dict(eye=dict(x=1, y=-2.5, z=1)),
        xaxis = dict(
            tick0 = 0,
            dtick = 2,
            tickfont = dict(size=18)),
        yaxis = dict(
            tick0 = 0.2,
            dtick = 0.4,
            tickfont = dict(size=18)),
        zaxis = dict(
            tick0 = 0,
            dtick = ztick,
            tickfont = dict(size=18),
            range=[-1.5, zmax]),)

    
    fig.update_layout(autosize=True,
        width=1800, 
        height=800,
        font=dict(color = '#000000',
                size=20),
      
        scene=scene_dict,
                    
        scene2=scene_dict)

    fig.update_coloraxes(showscale=False)
    fig.show()
    fig.write_image(dir_to_use + '/eq_fitting.png')

def plot_fitness_evolution(number_of_generations, best_fit_of_all_gens,
                           dir_to_use, start_idx=0):
    plt.figure()
    plt.title('Fitness Evolution for Genetic Algorithm Runs')
    plt.plot(np.arange(0, number_of_generations + 1)[start_idx:], best_fit_of_all_gens.T[start_idx:, :], label='_nolegend_')
    plt.plot(np.arange(0, number_of_generations + 1)[start_idx:], np.mean(best_fit_of_all_gens.T, axis=1)[start_idx:], linewidth = 5, c='black')
    plt.legend(['Average Fitness'])
    plt.savefig(dir_to_use + '/fitness_evolution.png')