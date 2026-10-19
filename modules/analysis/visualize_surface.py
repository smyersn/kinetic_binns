import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
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
                  species_names=None, filename=None, use_latex=False,
                  title_font_size=48, latex_title_size=r"\huge", axis_title_size=22, tick_size=16,
                  width=1500, height_per_row=640, camera_eye=(0.85, -2.05, 0.75),
                  share_z=True, scale=2, axis_names=None, arg_names=None):
    """
    Grid of 3D surface plots: one ROW per equation, 2 COLUMNS (True, EQL).
 
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
 
    use_latex: False (default) writes the titles in plotly's own HTML markup,
        <i>F</i><sub>1</sub><sup>EQL</sup>(<i>u</i>, <i>v</i>), in a serif
        font: it renders the same way every time, because nothing has to be
        loaded. True uses LaTeX instead, which kaleido typesets with MathJax
        loaded at render time (usually from the internet); when that load
        fails or is late -- intermittently on compute nodes, especially with
        several repeats rendering at once -- the raw "$...$" string is
        printed instead. Old notes on the LaTeX option:
        titles as "$F_1^{\\text{True}}(u, v)$" so the superscript
        stacks above the subscript (needs MathJax). HTML <sub>/<sup> render
        sequentially instead, which no tag reordering fixes. kaleido's static
        export has an inconsistent record with LaTeX across versions
        (sometimes emitting literal "$...$"), so render ONE figure and check
        it before trusting a full sweep; use_latex=False falls back to
        unicode subscripts ("F\u2081 True (u, v)"), which render everywhere.
 
    scale: write_image resolution multiplier; 2 keeps the larger titles crisp.

    title_font_size: subplot title size for the plain-text titles
        (use_latex=False).
    latex_title_size: TeX size command put inside every LaTeX title
        (\\large < \\Large < \\LARGE < \\huge < \\Huge; "" for MathJax's default).
        MathJax typesets LaTeX titles itself and does not reliably follow
        the annotation's font size, so title_font_size alone may not change
        them; the size command does.

    axis_names: names of the two concentrations on the x and y axes
        (default: the first two species names). Used for axis titles.
    arg_names: argument list shown in the titles, e.g. ['u', 'v=0.43', 'w']
        when v is held fixed (default: species_names).
    """
    n_rows = len(F_true_list)
    if len(F_mlp_list) != n_rows:
        raise ValueError(f"F_true_list has {n_rows} entries but F_mlp_list has {len(F_mlp_list)}.")
    if not species_names:
        # No names at all -- best guess is one per row, since we have no other
        # information about the true species count.
        species_names = default_species_names(n_rows)
 
    args_list = list(arg_names) if arg_names else list(species_names)
    args = ", ".join(args_list)
    args_latex = ", ".join(a.replace("=", "{=}") for a in args_list)
    size_cmd = f"{latex_title_size} " if latex_title_size else ""
    if axis_names is None:
        axis_names = list(species_names[:2]) if len(species_names) >= 2 else ['u', 'v']
    subplot_titles = []
    subscript_digits = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
    for i in range(1, n_rows + 1):
        if n_rows == 1:                        # nothing to index with one equation
            index_latex, index_plain = "", ""
        else:
            index_latex, index_plain = f"_{{{i}}}", str(i).translate(subscript_digits)
        if use_latex:
            subplot_titles.append(fr"${size_cmd}F{index_latex}^{{\text{{True}}}}({args_latex})$")
            subplot_titles.append(fr"${size_cmd}F{index_latex}^{{\text{{EQL}}}}({args_latex})$")
        else:
            # Plotly HTML: italic symbols, subscript index, superscript label.
            index_html = f"<sub>{i}</sub>" if n_rows > 1 else ""
            args_html = ", ".join(
                f"<i>{a.split('=')[0]}</i>=" + a.split('=', 1)[1] if '=' in a else f"<i>{a}</i>"
                for a in args_list)
            subplot_titles.append(f"<i>F</i>{index_html}<sup>True</sup>({args_html})")
            subplot_titles.append(f"<i>F</i>{index_html}<sup>EQL</sup>({args_html})")
 
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
        xaxis=dict(title=f'[{axis_names[0]}] (uM)', range=x_range, tickfont=dict(size=tick_size)),
        yaxis=dict(title=f'[{axis_names[1]}] (uM)', range=y_range, tickfont=dict(size=tick_size)),
        zaxis=dict(title='F', tickfont=dict(size=tick_size)),
        camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])))
 
    # Pull each title down onto its subplot: make_subplots places them at the
    # top of the subplot's domain, which the camera zoom leaves empty.
    for annotation in fig.layout.annotations:
        annotation.font.size = title_font_size
        if not use_latex:
            annotation.font.family = "Times New Roman, Times, serif"   # match the LaTeX look
        annotation.yanchor = 'bottom'
        annotation.y = annotation.y - 0.03 / max(n_rows, 1)
 
    fig.update_layout(
        autosize=False,
        width=width,
        height=height_per_row * n_rows,
        margin=dict(l=0, r=0, t=int(1.8 * title_font_size), b=0),
        font=dict(color='#000000', size=axis_title_size))
    fig.update_coloraxes(showscale=False)
 
    if filename:
        fig.write_image(f'{model_dir}/{filename}.png', scale=scale)
 
    return fig

def _as_numpy(F):
    return F.detach().cpu().numpy() if hasattr(F, 'detach') else np.asarray(F)


def _true_and_learned(reaction, params, model, device, states):
    """
    True and learned reaction terms at the given concentration states,
    (n_points, n_species) each. reaction is REACTION_REGISTRY[name]['fn'].
    """
    F_true_tuple = reaction(states, params)
    if not isinstance(F_true_tuple, (tuple, list)):
        raise TypeError(
            "reaction(uv, params) must return a tuple/list (F_species0, F_species1, ...). "
            "Got a single array/scalar -- this looks like a pre-stage-1 reaction function; "
            "every entry in reaction_library.py's REACTION_REGISTRY already returns the "
            "full tuple, so double-check you're passing REACTION_REGISTRY[name]['fn'] "
            "and not something wrapped for the old single-F convention.")
    F_true = np.stack([_as_numpy(F).reshape(-1) for F in F_true_tuple], axis=1)

    with torch.no_grad():
        F_learned = model.model.reaction(torch.tensor(states).float().to(device))
    F_learned = F_learned.cpu().numpy()

    if F_learned.ndim != 2 or F_learned.shape[1] != F_true.shape[1]:
        raise ValueError(
            f"Learned reaction module outputs shape {F_learned.shape} (expected "
            f"(n_points, {F_true.shape[1]})) -- the true reaction function returned "
            f"{F_true.shape[1]} species but the model's reaction module doesn't match. "
            f"Check that `species` in the BINN config equals the true system's "
            f"species count.")
    return F_true, F_learned


def _blur(a, sigma):
    """Separable Gaussian blur of a 2D array (sigma in cells), zero outside."""
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, a)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, a)


def _occupied_mask(a, b, a_grid, b_grid, bins=120, sigma=2.0, min_points=3.0):
    """
    True where the data's (a, b) states are dense enough to say something
    about the reaction there; False in the parts of the plotting plane the data
    never visit, where true and learned reactions can differ without the data
    constraining them.

    The data are histogrammed, the counts blurred with a Gaussian of `sigma`
    bins, the blurred density interpolated (bilinearly) onto the fine plotting
    mesh, and thresholded at roughly `min_points` points within a couple of
    bins. Interpolating the smooth density before thresholding gives a smooth
    boundary at the mesh's resolution, instead of a staircase of histogram
    cells, and min_points > 1 keeps lone noisy points from adding islands.
    """
    edges_a = np.linspace(a_grid.min(), a_grid.max(), bins + 1)
    edges_b = np.linspace(b_grid.min(), b_grid.max(), bins + 1)
    counts, _, _ = np.histogram2d(a, b, bins=[edges_a, edges_b])
    density = _blur(counts, sigma)
    threshold = min_points / (2 * np.pi * sigma ** 2)    # a blurred cluster of min_points points

    # Bilinear interpolation of density (defined at bin centres) onto the mesh.
    fa = np.clip((a_grid - edges_a[0]) / (edges_a[1] - edges_a[0]) - 0.5, 0, bins - 1)
    fb = np.clip((b_grid - edges_b[0]) / (edges_b[1] - edges_b[0]) - 0.5, 0, bins - 1)
    ia, ib = np.minimum(fa.astype(int), bins - 2), np.minimum(fb.astype(int), bins - 2)
    ta, tb = fa - ia, fb - ib
    smooth = ((1 - ta) * (1 - tb) * density[ia, ib] + ta * (1 - tb) * density[ia + 1, ib]
              + (1 - ta) * tb * density[ia, ib + 1] + ta * tb * density[ia + 1, ib + 1])
    return smooth > threshold


def plot_parity(model_dir, F_true, F_learned, equation_labels, filename=None,
                max_points=20000, title_size=26, label_size=18, seed=0):
    """
    Learned vs true reaction term at the data's own concentration states, one
    panel per equation. Unlike a surface, this uses every species at once, so
    it is the full comparison for three or more species.

    Each panel reports the least-squares slope through the origin (1 = right
    amplitude; below 1 = shrunk) and the correlation r (1 = right shape).
    """
    rng = np.random.default_rng(seed)
    if len(F_true) > max_points:
        keep = rng.choice(len(F_true), max_points, replace=False)
        F_true, F_learned = F_true[keep], F_learned[keep]

    n = len(equation_labels)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), squeeze=False)
    for i, (ax, label) in enumerate(zip(axes[0], equation_labels)):
        t, l = F_true[:, i], F_learned[:, i]
        slope = float(np.dot(t, l) / max(np.dot(t, t), 1e-30))
        r = float(np.corrcoef(t, l)[0, 1]) if np.std(t) > 0 and np.std(l) > 0 else float('nan')
        ax.scatter(t, l, s=2, alpha=0.25, color='#1f6f8b', rasterized=True)
        lo, hi = float(min(t.min(), l.min())), float(max(t.max(), l.max()))
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=1.5, linestyle='--')
        ax.set_xlabel(f'{label} true', fontsize=label_size)
        ax.set_ylabel(f'{label} EQL', fontsize=label_size)
        ax.set_title(f'{label}\nslope {slope:.3f}, r {r:.3f}', fontsize=title_size)
        ax.tick_params(labelsize=label_size - 4)
    fig.tight_layout()
    if filename:
        fig.savefig(f'{model_dir}/{filename}.png', dpi=120)
    plt.close(fig)
    return fig


def compare_surfaces_over_training_domain(training_data, model, device,
                                          reaction, params, dir_name,
                                          filename='feql_surface', species_names=None,
                                          use_latex=False, fixed_value='median',
                                          parity=True):
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

    Two species: surfaces over the (u, v) triangle mesh, as before.

    Three or more species: F depends on more concentrations than a surface
    can show, so the surfaces are drawn over the TWO species that vary most
    in the data (largest std / mean), with every other species held at its
    median ('median', or 'mean' via fixed_value). In the three-species
    polarity models that holds the fast cytosolic species fixed, which
    barely varies anyway. Mesh points whose (a, b) the data never visit are
    left blank. The titles show the fixed value, e.g. F(u, v=0.43, w).

    parity: also write <filename>_parity.png, learned vs true F at the
    data's own states (all species varying), one panel per equation.
    """
    if species_names is None:
        species_names = getattr(model.model, 'species_names', None)
    conc = training_data[:, -model.model.species:].cpu().detach().numpy().astype(np.float64)
    n_species = conc.shape[1]
    if not species_names or len(species_names) != n_species:
        species_names = default_species_names(n_species)

    # When mcas=True, model.model.n_equations is 1 -- there's only ONE
    # independent equation (F1 = -F2 architecturally), so only show it,
    # not both the equation and its redundant sign-flipped mirror.
    # model.model.reaction still has to compute every column (the physics
    # needs every species' rate), this only affects the plots.
    n_equations = getattr(model.model, 'n_equations', n_species)

    if n_species == 2:
        # Create triangle mesh from min and max uv vals seen in training data
        u_mesh, v_mesh = lltriangle(training_data[:, -2:].cpu().detach().numpy(),
                                    training_data[:, -1:].cpu().detach().numpy())
        states = np.stack((np.ravel(u_mesh), np.ravel(v_mesh)), axis=1)
        axis_names, arg_names = list(species_names), list(species_names)
    else:
        spread = conc.std(axis=0) / np.maximum(np.abs(conc.mean(axis=0)), 1e-12)
        a, b = sorted(np.argsort(spread)[-2:])            # the two that vary most, in order
        fixed = [s for s in range(n_species) if s not in (a, b)]
        centre = np.median(conc, axis=0) if fixed_value == 'median' else conc.mean(axis=0)

        a_points = np.linspace(conc[:, a].min(), conc[:, a].max(), 501)
        b_points = np.linspace(conc[:, b].min(), conc[:, b].max(), 501)
        u_mesh, v_mesh = np.meshgrid(a_points, b_points)
        mask = _occupied_mask(conc[:, a], conc[:, b], u_mesh, v_mesh)
        u_mesh = np.where(mask, u_mesh, np.nan)
        v_mesh = np.where(mask, v_mesh, np.nan)

        states = np.tile(centre, (u_mesh.size, 1))
        states[:, a] = np.nan_to_num(np.ravel(u_mesh))   # masked points: evaluated, then blanked
        states[:, b] = np.nan_to_num(np.ravel(v_mesh))
        axis_names = [species_names[a], species_names[b]]
        arg_names = [species_names[s] if s in (a, b) else f"{species_names[s]}={centre[s]:.3g}"
                     for s in range(n_species)]
        print(f"Surface plot: F over ({species_names[a]}, {species_names[b]}), "
              + ", ".join(f"{species_names[s]} fixed at {centre[s]:.4g}" for s in fixed)
              + f" ({fixed_value}); {100 * mask.mean():.0f}% of the plane visited by the data",
              flush=True)

    F_true, F_learned = _true_and_learned(reaction, params, model, device, states)
    blank = np.isnan(np.ravel(u_mesh))
    F_true[blank], F_learned[blank] = np.nan, np.nan
    F_true_list = [F_true[:, s].reshape(u_mesh.shape) for s in range(n_equations)]
    F_mlp_list = [F_learned[:, s].reshape(u_mesh.shape) for s in range(n_equations)]

    # Visualize surfaces
    plot_surfaces(dir_name, u_mesh, v_mesh, F_true_list, F_mlp_list, species_names,
                  filename, use_latex=use_latex, axis_names=axis_names, arg_names=arg_names)

    if parity:
        # Subsample first: at native resolution there are millions of rows.
        rng = np.random.default_rng(0)
        sample = conc[rng.choice(len(conc), min(len(conc), 20000), replace=False)]
        P_true, P_learned = _true_and_learned(reaction, params, model, device, sample)
        if n_equations == 1 and getattr(model.model, 'mcas', False):
            labels = [f"F({', '.join(species_names)})"]
        else:
            labels = [f"d{name}/dt reaction" for name in species_names[:n_equations]]
        plot_parity(dir_name, P_true[:, :n_equations], P_learned[:, :n_equations], labels,
                    filename=f'{filename}_parity' if filename else None)