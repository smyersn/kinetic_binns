import os, sys
import numpy as np
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go

file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.triangle import lltriangle
from modules.binn_eql.build_binn_eql_net import default_species_names

def plot_surfaces(model_dir, u_triangle_mesh, v_triangle_mesh, F_true_list, F_mlp_list,
                  species_names=None, filename=None, use_latex=True):
    """
    Grid of 3D surface plots: one ROW per equation being shown, 2 COLUMNS
    (True, Learned). n_rows = len(F_true_list) -- NOT necessarily the
    number of species/concentration fields in the system: when the caller
    passes mcas-collapsed data (one shared F, not one per species),
    n_rows is 1 even though species_names has 2 entries ('u', 'v'), since
    the equation's argument list still needs both names even though only
    one row is plotted.

    F_true_list, F_mlp_list: length-n_rows lists of (501, 501) arrays.
    species_names: the full concentration field list (e.g. ['u', 'v']),
        used only for the parenthetical argument list in titles/axis
        labels -- its length is independent of n_rows and is NOT
        validated against it.

    Subplot titles follow F_i^{True/MLP}(all species names) when n_rows
    > 1 -- i is the 1-indexed equation number (row). When n_rows == 1,
    the numeric subscript is dropped entirely (just F^{True/MLP}(...))
    since there's nothing to index.

    use_latex: if True (default), titles use "$F_1^{\\text{True}}(u, v)$"
        LaTeX so the superscript stacks directly above the subscript
        (needs MathJax). HTML <sub>/<sup> tags render sequentially, not
        stacked -- that's unfixable with tag reordering, it's how HTML
        inline elements work. write_image()'s kaleido-based static export
        has an inconsistent track record rendering LaTeX across versions
        (sometimes shows literal "$...$" instead of typeset math) --
        render ONE figure and check it before trusting this for a full
        sweep. If it doesn't render, set use_latex=False for a unicode-
        subscript fallback ("F\u2081 True (u, v)") that has no MathJax
        dependency and is guaranteed to render identically everywhere,
        at the cost of not being stacked math notation.
    """
    n_rows = len(F_true_list)
    if len(F_mlp_list) != n_rows:
        raise ValueError(f"F_true_list has {n_rows} entries but F_mlp_list has {len(F_mlp_list)}.")
    if not species_names:
        # No names given at all -- best guess is one name per row, since
        # we have no other information about the true species count.
        species_names = default_species_names(n_rows)

    args = ", ".join(species_names)
    specs = [[{'type': 'scene'}, {'type': 'scene'}] for _ in range(n_rows)]
    subplot_titles = []
    subscript_digits = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
    for i in range(1, n_rows + 1):
        if n_rows == 1:
            # Nothing to index when there's only one equation -- drop the subscript.
            if use_latex:
                subplot_titles.append(fr"$F^{{\text{{True}}}}({args})$")
                subplot_titles.append(fr"$F^{{\text{{MLP}}}}({args})$")
            else:
                subplot_titles.append(f"F True ({args})")
                subplot_titles.append(f"F MLP ({args})")
        elif use_latex:
            subplot_titles.append(fr"$F_{{{i}}}^{{\text{{True}}}}({args})$")
            subplot_titles.append(fr"$F_{{{i}}}^{{\text{{MLP}}}}({args})$")
        else:
            i_sub = str(i).translate(subscript_digits)
            subplot_titles.append(f"F{i_sub} True ({args})")
            subplot_titles.append(f"F{i_sub} MLP ({args})")

    fig = make_subplots(rows=n_rows, cols=2,
                        specs=specs,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=0,
                        vertical_spacing=min(0.08, 0.6 / max(n_rows, 1)))

    fig.update_annotations(font_size=24, font_color='#000000')

    for row_idx in range(1, n_rows + 1):
        F_true = F_true_list[row_idx - 1]
        F_mlp = F_mlp_list[row_idx - 1]

        fig.add_trace(
            go.Surface(z=F_true, x=u_triangle_mesh, y=v_triangle_mesh,
                                        colorscale='mint',
                                        showscale=False),
            row=row_idx, col=1)

        fig.add_trace(
            go.Surface(z=F_mlp, x=u_triangle_mesh, y=v_triangle_mesh,
                                        colorscale='mint',
                                        showscale=False),
            row=row_idx, col=2)

    # NOTE: the old zaxis had a hardcoded range=[-1.5, 11] and fixed
    # tick0/dtick, tuned for wave_pinning's single F magnitude. Different
    # species' reaction functions can have very different magnitudes (a
    # Gray-Scott Fu and an FHN Fv are not on the same scale), so those are
    # dropped here in favor of per-subplot auto-ranging. If you want a
    # shared/fixed range again for a specific system, add it back per-row
    # via fig.update_scenes(zaxis_range=[...], row=r, col=c).
    scene_dict = dict(
        aspectmode='cube', # Forces the 3D bounding box to be a perfect cube
        xaxis_title='[u] (uM)',
        yaxis_title='[v] (uM)',
        zaxis_title='F',
        xaxis = dict(tickfont = dict(size=18)),
        yaxis = dict(tickfont = dict(size=18)),
        zaxis = dict(tickfont = dict(size=18)),
        camera=dict(eye=dict(x=1, y=-2.5, z=1)))

    # 1. Update the overall figure layout (Remove scene and scene2 from here)
    fig.update_layout(autosize=True,
        width=1600,
        height=800 * n_rows,
        font=dict(color = '#000000', size=20))

    # 2. Safely apply the scene dictionary to all 3D subplots
    fig.update_scenes(**scene_dict)

    fig.update_coloraxes(showscale=False)

    if filename:
        fig.write_image(f'{model_dir}/{filename}.png')

    return fig

def compare_surfaces_over_training_domain(training_data, model, device,
                                          reaction, params, dir_name,
                                          filename='feql_surface', species_names=None,
                                          use_latex=True):
    """
    reaction: (uv, params) -> tuple of length n_species (one array per
        species equation), matching every function in reaction_library.py.
        The old convention -- a single scalar F, implicitly F_u=+F,
        F_v=-F -- no longer applies to anything in the registry (that was
        only wave_pinning/turing_type/custom_equation, which have been
        removed); every reaction function returns its full tuple natively
        now, so no wrapping is needed here regardless of which reaction
        this is.
    species_names: optional override; defaults to model.model.species_names.
    """
    # Create triangle mesh from min and max uv vals seen in training data
    u_triangle_mesh, v_triangle_mesh = lltriangle(training_data[:, -2:].cpu().detach().numpy(),
                                                training_data[:, -1:].cpu().detach().numpy())

    # Create 1d arrays from meshes
    u_triangle, v_triangle = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)
    uv = np.stack((u_triangle, v_triangle), axis=1)

    # Generate true surfaces -- one per species
    F_true_tuple = reaction(uv, params)
    if not isinstance(F_true_tuple, (tuple, list)):
        raise TypeError(
            "reaction(uv, params) must return a tuple/list (F_species0, F_species1, ...). "
            "Got a single array/scalar -- this looks like a pre-stage-1 reaction function; "
            "every entry in reaction_library.py's REACTION_REGISTRY already returns the "
            "full tuple, so double-check you're passing REACTION_REGISTRY[name]['fn'] "
            "and not something wrapped for the old single-F convention.")

    F_true_list = []
    for F in F_true_tuple:
        F_arr = F.detach().cpu().numpy() if hasattr(F, 'detach') else np.asarray(F)
        F_true_list.append(F_arr.reshape(501, 501))
    n_species = len(F_true_list)

    # Generate learned surfaces -- model.model.reaction now returns
    # (n_points, n_species) instead of the old (n_points, 1).
    F_mlp_unformatted = model.model.reaction(torch.tensor(uv).float().to(device))
    F_mlp_np = F_mlp_unformatted.cpu().detach().numpy()

    if F_mlp_np.ndim != 2 or F_mlp_np.shape[1] != n_species:
        raise ValueError(
            f"Learned reaction module outputs shape {F_mlp_np.shape} (expected "
            f"(n_points, {n_species})) -- the true reaction function returned "
            f"{n_species} species but the model's reaction module doesn't match. "
            f"Check that `species` in the BINN config equals the true system's "
            f"species count.")

    F_mlp_list = [F_mlp_np[:, s].reshape(501, 501) for s in range(n_species)]

    if species_names is None:
        species_names = getattr(model.model, 'species_names', None)

    # When mcas=True, model.model.n_equations is 1 -- there's only ONE
    # independent equation (F1 = -F2 architecturally), so only show it,
    # not both the equation and its redundant sign-flipped mirror.
    # model.model.reaction still had to compute both columns above (the
    # physics needs u_t and v_t regardless), this only affects the plot.
    n_equations = getattr(model.model, 'n_equations', n_species)
    F_true_list = F_true_list[:n_equations]
    F_mlp_list = F_mlp_list[:n_equations]

    # Visualize surfaces
    plot_surfaces(dir_name, u_triangle_mesh, v_triangle_mesh,
                  F_true_list, F_mlp_list, species_names, filename, use_latex=use_latex)