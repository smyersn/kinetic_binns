import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict
import os

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
    """
    # --- load arrays
    if 'raw_w_unscaled' in param_history:
        raw_list = param_history['raw_w_unscaled']
    else:
        raise KeyError("param_history must contain 'raw_w_unscaled'")

    if 'effective_unscaled' not in param_history:
        raise KeyError("param_history must contain 'effective_unscaled'")

    raw_arr = np.stack([np.asarray(a) for a in raw_list], axis=0)   # (E, M)
    eff_arr = np.stack([np.asarray(a) for a in param_history['effective_unscaled']], axis=0)
    epoch_arr = np.array(param_history['epoch'])
    E, M = raw_arr.shape
    if eff_arr.shape != raw_arr.shape:
        raise ValueError(f"raw and effective shapes mismatch: {raw_arr.shape} vs {eff_arr.shape}")

    # --- reconstruct base-term groups from model
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

    species = ['u', 'v']  # adapt if needed

    # --- polynomial groups (base term -> list of indices across duplicates)
    poly_base_names = []
    for t in poly_terms:
        if len(t) == 1:
            poly_base_names.append(f"{species[t[0]]}")
        else:
            poly_base_names.append("*".join([species[i] for i in t]))
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
            hill_inc_names.append(f"H_inc({species[t[0]]})")
            hill_dec_names.append(f"H_dec({species[t[0]]})")
        else:
            hill_inc_names.append(f"H_inc({species[t[0]]})*{species[t[1]]}")
            hill_dec_names.append(f"H_dec({species[t[0]]})*{species[t[1]]}")

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
    def pick_top(groups, arr, K):
        keys = list(groups.keys())
        if K is None or K >= len(keys):
            return keys
        mags = {k: np.abs(arr[-1, groups[k]].sum()) for k in keys}
        return sorted(keys, key=lambda k: mags[k], reverse=True)[:K]

    poly_plot_names = pick_top(groups_poly, eff_arr, max_terms)
    hill_inc_plot_names = pick_top(groups_hill_inc, eff_arr, max_terms)
    hill_dec_plot_names = pick_top(groups_hill_dec, eff_arr, max_terms)

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

    # ==========================================
    # --- PLOTTING FIGURE 1: Reaction Terms
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    ax_raw_poly, ax_raw_hill = axes[0,0], axes[0,1]
    ax_eff_poly, ax_eff_hill = axes[1,0], axes[1,1]

    def plot_panel(ax, names_to_plot, groups_dict, data_arr, title):
        proxies = []
        for name in names_to_plot:
            idxs = groups_dict[name]
            color = color_map[name]
            # plot duplicates faint
            for idx in idxs:
                ax.plot(epoch_arr, data_arr[:, idx], color=color, alpha=alpha_dup, linewidth=1)
            # plot sum (thicker)
            summed = data_arr[:, idxs].sum(axis=1)
            ax.plot(epoch_arr, summed, color=color, alpha=alpha_sum, linewidth=2)
            proxies.append(mlines.Line2D([], [], color=color, linewidth=2, alpha=alpha_sum, label=name))
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('rate')
        if proxies:
            ax.legend(handles=proxies, fontsize='small', ncol=1, loc='upper right')

    # raw polynomials (left top)
    plot_panel(ax_raw_poly, poly_plot_names, groups_poly, raw_arr, 'Raw polynomial weights')
    # raw hills: show inc then dec separate in same axes
    plot_panel(ax_raw_hill, hill_inc_plot_names + hill_dec_plot_names,
               {**groups_hill_inc, **groups_hill_dec}, raw_arr, 'Raw hill rates (inc & dec)')
    # effective polynomials
    plot_panel(ax_eff_poly, poly_plot_names, groups_poly, eff_arr, 'Effective polynomial rates (gated)')
    # effective hills
    plot_panel(ax_eff_hill, hill_inc_plot_names + hill_dec_plot_names,
               {**groups_hill_inc, **groups_hill_dec}, eff_arr, 'Effective hill rates (gated)')

    fig.tight_layout()
    
    # ==========================================
    # --- PLOTTING FIGURE 2: Diffusion Coeffs
    # ==========================================
    fig_diff = None
    axes_diff = None
    
    if 'diffusion_coeffs' in param_history and len(param_history['diffusion_coeffs']) > 0:
        diff_arr = np.array(param_history['diffusion_coeffs'])
        
        # Ensure we actually have 2 coefficients to plot (d_u and d_v)
        if diff_arr.ndim == 2 and diff_arr.shape[1] >= 2:
            fig_diff, axes_diff = plt.subplots(figsize=(10, 5))
            
            axes_diff.plot(epoch_arr, diff_arr[:, 0], label='Learned $d_u$', color='#1f77b4', linewidth=2.5)
            axes_diff.plot(epoch_arr, diff_arr[:, 1], label='Learned $d_v$', color='#ff7f0e', linewidth=2.5)
            
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
        # Save the primary 2x2 parameter history plot
        fig.savefig(save_path)
        
        # Save the supplementary diffusion plot 
        if fig_diff is not None:
            base, ext = os.path.splitext(save_path)
            diff_save_path = f"{base}_diffusion{ext}"
            fig_diff.savefig(diff_save_path)
            
    else:
        # Return both figures so they can be rendered directly in a Jupyter Notebook cell
        return fig, axes, fig_diff, axes_diff