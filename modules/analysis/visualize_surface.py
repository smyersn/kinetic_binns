import os, sys
import numpy as np
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go

file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.triangle import lltriangle
from modules.binn_eql.model.binn import default_species_names

def _shared_range(arrays, pad=0.04):
    """
    (data min, data max, padded axis range) across several surfaces, ignoring
    NaNs (the triangle mesh leaves the region outside the data as NaN).
    Returns (None, None, None) if nothing is finite.
    """
    finite = [a for a in arrays if np.any(np.isfinite(a))]
    if not finite:
        return None, None, None
 
    lo = float(min(np.nanmin(a) for a in finite))
    hi = float(max(np.nanmax(a) for a in finite))
    if hi - lo < 1e-12:                       # flat surface: invent a window round it
        span = max(abs(hi), 1e-6)
        return lo, hi, [lo - span, hi + span]
    margin = (hi - lo) * pad
    return lo, hi, [lo - margin, hi + margin]
 
 
def plot_surfaces(model_dir, u_triangle_mesh, v_triangle_mesh, F_true_list, F_mlp_list,
                  species_names=None, filename=None, use_latex=True,
                  title_font_size=40, axis_title_size=22, tick_size=16,
                  width=1500, height_per_row=640, camera_eye=(0.85, -2.05, 0.75),
                  share_z=True, scale=2):
    """
    Grid of 3D surface plots: one ROW per equation, 2 COLUMNS (True, Learned).
 
    n_rows = len(F_true_list) -- NOT necessarily the number of species: with
    mcas-collapsed data (one shared F) n_rows is 1 even though species_names
    has two entries, since the equation's argument list still needs both.
 
    F_true_list, F_mlp_list: length-n_rows lists of (501, 501) arrays.
    species_names: full concentration field list (e.g. ['u', 'v']), used only
        for the parenthetical argument list in titles; its length is
        independent of n_rows and is NOT validated against it.
 
    share_z: give both columns of a row one z-axis range and one color range,
        taken from both surfaces. Without it each subplot auto-ranges, so a
        learned F an order of magnitude too small looks like a shape
        difference rather than an amplitude error. x and y are shared too:
        both columns are evaluated on the same mesh.
 
    camera_eye: viewer position. The distance from the origin sets the zoom;
        ~2.3 fills the subplot with the cube, while the old (1, -2.5, 1)
        (distance 2.9) left a wide empty border.
 
    use_latex: titles as "$F_1^{\\text{True}}(u, v)$" so the superscript
        stacks above the subscript (needs MathJax). HTML <sub>/<sup> render
        sequentially instead, which no tag reordering fixes. kaleido's static
        export has an inconsistent record with LaTeX across versions
        (sometimes emitting literal "$...$"), so render ONE figure and check
        it before trusting a full sweep; use_latex=False falls back to
        unicode subscripts ("F\u2081 True (u, v)"), which render everywhere.
 
    scale: write_image resolution multiplier; 2 keeps the larger titles crisp.
    """
    n_rows = len(F_true_list)
    if len(F_mlp_list) != n_rows:
        raise ValueError(f"F_true_list has {n_rows} entries but F_mlp_list has {len(F_mlp_list)}.")
    if not species_names:
        # No names at all -- best guess is one per row, since we have no other
        # information about the true species count.
        species_names = default_species_names(n_rows)
 
    args = ", ".join(species_names)
    subplot_titles = []
    subscript_digits = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
    for i in range(1, n_rows + 1):
        if n_rows == 1:                        # nothing to index with one equation
            index_latex, index_plain = "", ""
        else:
            index_latex, index_plain = f"_{{{i}}}", str(i).translate(subscript_digits)
        if use_latex:
            subplot_titles.append(fr"$F{index_latex}^{{\text{{True}}}}({args})$")
            subplot_titles.append(fr"$F{index_latex}^{{\text{{MLP}}}}({args})$")
        else:
            subplot_titles.append(f"F{index_plain} True ({args})")
            subplot_titles.append(f"F{index_plain} MLP ({args})")
 
    fig = make_subplots(
        rows=n_rows, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}] for _ in range(n_rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.04 if n_rows > 1 else 0.0)
 
    # x and y come from the same mesh in every panel, so range them once.
    _, _, x_range = _shared_range([u_triangle_mesh], pad=0.0)
    _, _, y_range = _shared_range([v_triangle_mesh], pad=0.0)
 
    for row in range(1, n_rows + 1):
        F_true, F_mlp = F_true_list[row - 1], F_mlp_list[row - 1]
        if share_z:
            cmin, cmax, z_range = _shared_range([F_true, F_mlp])
        else:
            cmin = cmax = z_range = None
 
        for col, F in ((1, F_true), (2, F_mlp)):
            fig.add_trace(
                go.Surface(z=F, x=u_triangle_mesh, y=v_triangle_mesh,
                           colorscale='mint', showscale=False,
                           cmin=cmin, cmax=cmax),   # one color mapping per row
                row=row, col=col)
 
        # Per-row z range. Set here rather than in the shared scene dict below,
        # which applies to every subplot at once.
        fig.update_scenes(zaxis=dict(range=z_range), row=row, col=1)
        fig.update_scenes(zaxis=dict(range=z_range), row=row, col=2)
 
    fig.update_scenes(
        aspectmode='cube',
        xaxis=dict(title='[u] (uM)', range=x_range, tickfont=dict(size=tick_size)),
        yaxis=dict(title='[v] (uM)', range=y_range, tickfont=dict(size=tick_size)),
        zaxis=dict(title='F', tickfont=dict(size=tick_size)),
        camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])))
 
    # Pull each title down onto its subplot: make_subplots places them at the
    # top of the subplot's domain, which the camera zoom leaves empty.
    for annotation in fig.layout.annotations:
        annotation.font.size = title_font_size
        annotation.yanchor = 'bottom'
        annotation.y = annotation.y - 0.03 / max(n_rows, 1)
 
    fig.update_layout(
        autosize=False,
        width=width,
        height=height_per_row * n_rows,
        margin=dict(l=0, r=0, t=int(1.6 * title_font_size), b=0),
        font=dict(color='#000000', size=axis_title_size))
    fig.update_coloraxes(showscale=False)
 
    if filename:
        fig.write_image(f'{model_dir}/{filename}.png', scale=scale)
 
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