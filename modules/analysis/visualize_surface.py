import os, sys
import numpy as np
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go

file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.triangle import lltriangle

def plot_surfaces(model_dir, u_triangle_mesh, v_triangle_mesh, F_true,
                  F_mlp, filename=None):
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type':'scene'}, {'type':'scene'}]],
                        subplot_titles=("F<sub>True</sub>(u, v)", 
                                        "F<sub>EQL</sub>(u, v)"),
                        horizontal_spacing = 0)

    fig.layout.annotations[0].update(y=0.8)
    fig.layout.annotations[1].update(y=0.8)
    
    fig.update_annotations(font_size=24, font_color='#000000')

    fig.add_trace(
        go.Surface(z=F_true, x=u_triangle_mesh, y=v_triangle_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        x=1.15,
                                        title='True',
                                        len=0.5)),
        row=1, col=1)
    
    fig.add_trace(
        go.Surface(z=F_mlp, x=u_triangle_mesh, y=v_triangle_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        title='MLP',
                                        len=0.5)),
        row=1, col=2)

    scene_dict = dict(
        aspectmode='cube', # Forces the 3D bounding box to be a perfect cube
        xaxis_title='[u] (uM)',
        yaxis_title='[v] (uM)',
        zaxis_title='F',
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
            dtick = 2,
            tickfont = dict(size=18),
            range=[-1.5, 11]),
        camera=dict(eye=dict(x=1, y=-2.5, z=1)))

    # 1. Update the overall figure layout (Remove scene and scene2 from here)
    fig.update_layout(autosize=True,
        width=1600, 
        height=800,
        font=dict(color = '#000000', size=20))
        
    # 2. Safely apply the scene dictionary to all 3D subplots
    fig.update_scenes(**scene_dict)

    fig.update_coloraxes(showscale=False)
        
    if filename:
        fig.write_image(f'{model_dir}/{filename}.png')

def compare_surfaces_over_training_domain(training_data, model, device, 
                                          reaction, params, dir_name, 
                                          filename='feql_surface'):
    # Create triangle mesh from min and max uv vals seen in training data
    u_triangle_mesh, v_triangle_mesh = lltriangle(training_data[:, -2:].cpu().detach().numpy(),
                                                training_data[:, -1:].cpu().detach().numpy())

    # Create 1d arrays from meshes
    u_triangle, v_triangle = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)
    uv = np.stack((u_triangle, v_triangle), axis=1)
        
    # Generate true surface
    F_true = reaction(uv, params).reshape(501, 501)

    # # Generate learned learned surface after initial training
    # uv_scaled = np.zeros_like(uv)
    # uv_scaled[:, 0] = uv[:, 0] / model.model.max_scale[0, 0].cpu().detach().numpy()
    # uv_scaled[:, 1] = uv[:, 1] / model.model.max_scale[0, 1].cpu().detach().numpy()
    # F_mlp_unformatted = model.model.reaction(torch.tensor(uv_scaled).float().to(device))
    F_mlp_unformatted = model.model.reaction(torch.tensor(uv).float().to(device))
    F_mlp = F_mlp_unformatted.cpu().detach().numpy().reshape(501, 501)

    # Visualize surfaces
    plot_surfaces(dir_name, u_triangle_mesh, v_triangle_mesh,
                  F_true, F_mlp, filename)