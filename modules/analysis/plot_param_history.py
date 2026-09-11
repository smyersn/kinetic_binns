import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict
import os

from modules.binn_eql.build_binn_eql_net import default_species_names

def plot_param_history(binn_model,
                       param_history,
                       save_path = None,
                       max_terms=None,
                       figsize=(14, 10),
                       alpha_dup=0.25,
                       alpha_sum=0.9,
                       cmap_name='Set1'):
    """
    Fixed 2x2 plotting: left column polynomials, right column hills.
    - increasing and decreasing hill terms are handled separately (no mixing).
    - legend order follows the colormap native order (for discrete colormaps like Set1).
    - Creates a supplementary figure for diffusion coefficient convergence.

    Stage-1 BINN change: extract_params() now returns one set of
    coefficients PER SPECIES (each species has its own independently-
    gated reaction equation), so param_history['raw_w_unscaled'] and
    ['effective_unscaled'] are (epochs, species, features) instead of the
    old (epochs, features). This produces one 2x2 figure PER SPECIES
    equation -- saved as '{save_path stem}_{species_name}{ext}' -- instead
    of a single combined figure.
    """
    # --- load arrays
    if 'raw_w_unscaled' in param_history:
        raw_list = param_history['raw_w_unscaled']
    else:
        raise KeyError("param_history must contain 'raw_w_unscaled'")

    if 'effective_unscaled' not in param_history:
        raise KeyError("param_history must contain 'effective_unscaled'")

    # Each element of raw_list is itself a list of `species` arrays (one
    # per equation) -- stack twice: inner over species, outer over epochs.
    raw_arr = np.stack([np.stack([np.asarray(sp) for sp in epoch]) for epoch in raw_list], axis=0)
    eff_arr = np.stack([np.stack([np.asarray(sp) for sp in epoch])
                         for epoch in param_history['effective_unscaled']], axis=0)
    epoch_arr = np.array(param_history['epoch'])

    if raw_arr.ndim != 3:
        raise ValueError(
            f"Expected raw_w_unscaled shaped (epochs, species, features); got shape {raw_arr.shape}. "
            "If this param_history came from a pre-stage-1 BINN (single flat equation), "
            "wrap each epoch's array in a length-1 list before calling this function.")

    E, S, M = raw_arr.shape
    if eff_arr.shape != raw_arr.shape:
        raise ValueError(f"raw and effective shapes mismatch: {raw_arr.shape} vs {eff_arr.shape}")

    # Two DIFFERENT things, easy to conflate: basis_names is the full
    # concentration field list (length = binn_model.species), used only
    # for term formulas like "u*v^2" -- term power-tuples index up to
    # `species`, not S, so this must never be truncated to S. row_names
    # is the per-ROW plot label (length = S = n_equations): when
    # binn_model.mcas is True, S is 1 and there's no single species this
    # equation "belongs" to, so it's labeled 'F' rather than mislabeling
    # it as e.g. "u" (which would misleadingly suggest this is du/dt
    # specifically, when it's really the shared term added to species 0
    # and subtracted from species 1).
    species = getattr(binn_model, 'species', S)
    basis_names = list(getattr(binn_model, 'species_names', []))
    if len(basis_names) != species:
        basis_names = default_species_names(species)

    is_mcas = getattr(binn_model, 'mcas', False)
    if is_mcas and S == 1:
        row_names = ['F']
    elif len(basis_names) == S:
        row_names = basis_names
    else:
        row_names = default_species_names(S)

    # --- reconstruct base-term groups from model
    # These describe the SHARED feature basis (e.g. "u*v^2") and are the
    # same regardless of which species' equation is being plotted -- only
    # the coefficient values (raw_arr_s / eff_arr_s below) differ per species.
    poly_terms, hill_terms = binn_model.generate_terms()   # lists for a single duplicate
    dup = int(getattr(binn_model, 'duplicates', 1))

    n_poly_single = len(poly_terms)
    n_hill_single = len(hill_terms)
    n_poly_total = n_poly_single * dup
    n_hill_total = 2 * n_hill_single * dup
    if n_poly_total + n_hill_total != M:
        # fallback
        n_poly_total = binn_model.reaction.eql_layer.num_poly_features
        n_hill_total = binn_model.reaction.eql_layer.num_hill_features
        if n_poly_total + n_hill_total != M:
            raise RuntimeError("Feature count mismatch. Check generate_terms()/duplicates vs M.")

    # --- polynomial groups (base term -> list of indices across duplicates)
    poly_base_names = []
    for t in poly_terms:
        term_parts = []
        for i, power in enumerate(t):
            if power == 1:
                term_parts.append(f"{basis_names[i]}")
            elif power > 1:
                term_parts.append(f"{basis_names[i]}^{power}")

        # If the term is a constant (all powers are 0), call it '1'
        if not term_parts:
            poly_base_names.append("1")
        else:
            poly_base_names.append("*".join(term_parts))

    groups_poly = {}
    for d in range(dup):
        for i, name in enumerate(poly_base_names):
            idx = d * n_poly_single + i
            groups_poly.setdefault(name, []).append(idx)

    # --- hill groups: create separate dicts for inc and dec (NEVER combine)
    hill_inc_names = []
    hill_dec_names = []
    for t in hill_terms:
        if len(t) == 1:
            hill_inc_names.append(f"H_inc({basis_names[t[0]]})")
            hill_dec_names.append(f"H_dec({basis_names[t[0]]})")
        else:
            hill_inc_names.append(f"H_inc({basis_names[t[0]]})*{basis_names[t[1]]}")
            hill_dec_names.append(f"H_dec({basis_names[t[0]]})*{basis_names[t[1]]}")

    groups_hill_inc = defaultdict(list)
    groups_hill_dec = defaultdict(list)

    current_idx = n_poly_total  # Start after all polynomials

    for d in range(dup):
        # 1. Assign Increasing indices for this duplicate
        for i, name in enumerate(hill_inc_names):
            groups_hill_inc[name].append(current_idx)
            current_idx += 1

        # 2. Assign Decreasing indices for this duplicate
        for i, name in enumerate(hill_dec_names):
            groups_hill_dec[name].append(current_idx)
            current_idx += 1

    # --- selection of top terms (optional)
    def pick_top(groups, arr2d, K):
        keys = list(groups.keys())
        if K is None or K >= len(keys):
            return keys
        mags = {k: np.abs(arr2d[-1, groups[k]].sum()) for k in keys}
        return sorted(keys, key=lambda k: mags[k], reverse=True)[:K]

    def plot_panel(ax, names_to_plot, groups_dict, data_arr2d, title, color_map):
        proxies = []
        for name in names_to_plot:
            idxs = groups_dict[name]
            color = color_map[name]
            # plot duplicates faint
            for idx in idxs:
                ax.plot(epoch_arr, data_arr2d[:, idx], color=color, alpha=alpha_dup, linewidth=1)
            # plot sum (thicker)
            summed = data_arr2d[:, idxs].sum(axis=1)
            ax.plot(epoch_arr, summed, color=color, alpha=alpha_sum, linewidth=2)
            proxies.append(mlines.Line2D([], [], color=color, linewidth=2, alpha=alpha_sum, label=name))
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('rate')
        if proxies:
            ax.legend(handles=proxies, fontsize='small', ncol=1, loc='upper right')

    # ==========================================
    # --- PLOTTING FIGURE 1 (per species): Reaction Terms
    # ==========================================
    figs = []
    # eq_label: what to call this row in panel/figure titles. For a normal
    # (non-mcas) row this is just "d{species}/dt". For the single mcas row,
    # "dF/dt" would misleadingly imply F is a concentration whose time
    # derivative is being tracked -- it's the shared reaction term itself,
    # so label it "F(u, v)" instead, matching generate_equation()'s
    # convention.
    for s_idx, sp_name in enumerate(row_names):
        raw_arr_s = raw_arr[:, s_idx, :]
        eff_arr_s = eff_arr[:, s_idx, :]

        eq_label = f"F({', '.join(basis_names)})" if is_mcas else f"d{sp_name}/dt"

        poly_plot_names = pick_top(groups_poly, eff_arr_s, max_terms)
        hill_inc_plot_names = pick_top(groups_hill_inc, eff_arr_s, max_terms)
        hill_dec_plot_names = pick_top(groups_hill_dec, eff_arr_s, max_terms)

        # Combine order for colors & legend: first polys, then hill_inc, then hill_dec
        combined_names = list(poly_plot_names) + list(hill_inc_plot_names) + list(hill_dec_plot_names)
        N = max(len(combined_names), 1)

        # --- sample colormap in native order if discrete, otherwise sample evenly
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, 'colors'):
            palette = list(cmap.colors)
        else:
            palette = [cmap(i / float(max(N - 1, 1))) for i in range(N)]
        if len(palette) < N:
            palette = [palette[i % len(palette)] for i in range(N)]
        else:
            palette = palette[:N]
        color_map = {name: palette[i] for i, name in enumerate(combined_names)}

        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
        ax_raw_poly, ax_raw_hill = axes[0,0], axes[0,1]
        ax_eff_poly, ax_eff_hill = axes[1,0], axes[1,1]

        # raw polynomials (left top)
        plot_panel(ax_raw_poly, poly_plot_names, groups_poly, raw_arr_s,
                   f'Raw polynomial weights ({eq_label})', color_map)
        # raw hills: show inc then dec separate in same axes
        plot_panel(ax_raw_hill, hill_inc_plot_names + hill_dec_plot_names,
                   {**groups_hill_inc, **groups_hill_dec}, raw_arr_s,
                   f'Raw hill rates ({eq_label}, inc & dec)', color_map)
        # effective polynomials
        plot_panel(ax_eff_poly, poly_plot_names, groups_poly, eff_arr_s,
                   f'Effective polynomial rates ({eq_label}, gated)', color_map)
        # effective hills
        plot_panel(ax_eff_hill, hill_inc_plot_names + hill_dec_plot_names,
                   {**groups_hill_inc, **groups_hill_dec}, eff_arr_s,
                   f'Effective hill rates ({eq_label}, gated)', color_map)

        fig.suptitle(f"Parameter history: {eq_label}")
        fig.tight_layout()
        figs.append(fig)

        if save_path:
            base, ext = os.path.splitext(save_path)
            fig.savefig(f"{base}_{sp_name}{ext}")

    # ==========================================
    # --- PLOTTING FIGURE 2: Diffusion Coeffs (one line per species, N-general)
    # ==========================================
    fig_diff = None
    axes_diff = None

    # Only plot diffusion convergence for LEARNED coefficients. diff_coeffs
    # being truthy means they were passed in fixed (BINN's convention:
    # falsy/None -> learned via diffusion_fitter). Checking this directly
    # rather than only relying on param_history staying empty for fixed
    # coefficients, in case an older model_wrapper still records them.
    diff_coeffs_are_fixed = bool(getattr(binn_model, 'diff_coeffs', None))

    if (not diff_coeffs_are_fixed
            and 'diffusion_coeffs' in param_history
            and len(param_history['diffusion_coeffs']) > 0):
        diff_arr = np.array(param_history['diffusion_coeffs'])

        if diff_arr.ndim == 2 and diff_arr.shape[1] >= 1:
            fig_diff, axes_diff = plt.subplots(figsize=(10, 5))
            colors = plt.get_cmap('tab10').colors
            # Diffusion coefficients are always per-SPECIES (physics-level:
            # one Du, one Dv, ...), independent of mcas/n_equations -- use
            # basis_names (full concentration list), never row_names.
            diff_species_names = basis_names if len(basis_names) == diff_arr.shape[1] \
                else default_species_names(diff_arr.shape[1])

            for i in range(diff_arr.shape[1]):
                axes_diff.plot(epoch_arr, diff_arr[:, i], label=f'Learned $D_{{{diff_species_names[i]}}}$',
                               color=colors[i % len(colors)], linewidth=2.5)

            axes_diff.set_title('Diffusion Coefficient Convergence')
            axes_diff.set_xlabel('Epoch')
            axes_diff.set_ylabel('Coefficient Magnitude')
            axes_diff.grid(True, linestyle=':', alpha=0.6)
            axes_diff.legend(loc='best')

            fig_diff.tight_layout()

    # ==========================================
    # --- Save or Return logic
    # ==========================================
    if save_path:
        # Per-species figures already saved above in the loop.
        if fig_diff is not None:
            base, ext = os.path.splitext(save_path)
            diff_save_path = f"{base}_diffusion{ext}"
            fig_diff.savefig(diff_save_path)
    else:
        # Return everything so it can be rendered directly in a notebook cell.
        return figs, fig_diff, axes_diff