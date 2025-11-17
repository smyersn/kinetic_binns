import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict
# epoch_arr = np.array(param_history['epoch'])

def plot_param_history(binn_model,
                       param_history,
                       save_path = None,
                       max_terms=None,
                       figsize=(14, 5),
                       alpha_dup=0.25,
                       alpha_sum=0.9,
                       cmap_name='Set1'):
    """
    Fixed 2x2 plotting: left column polynomials, right column hills.
    - increasing and decreasing hill terms are handled separately (no mixing).
    - legend order follows the colormap native order (for discrete colormaps like Set1).
    """
    # --- load arrays
    if 'raw_w_unscaled' in param_history:
        raw_list = param_history['raw_w_unscaled']
    else:
        raise KeyError("param_history must contain 'raw_w_unscaled'")

    raw_arr = np.stack([np.asarray(a) for a in raw_list], axis=0)   # (E, M)
    epoch_arr = np.array(param_history['epoch'])
    E, M = raw_arr.shape

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
    base = n_poly_total
    for d in range(dup):
        start = base + d * 2 * n_hill_single
        inc_idxs = [start + i for i in range(n_hill_single)]
        dec_idxs = [start + n_hill_single + i for i in range(n_hill_single)]
        # assign inc indices
        for i, name in enumerate(hill_inc_names):
            groups_hill_inc[name].append(inc_idxs[i])
        # assign dec indices
        for i, name in enumerate(hill_dec_names):
            groups_hill_dec[name].append(dec_idxs[i])

    # --- selection of top terms (optional)
    def pick_top(groups, arr, K):
        keys = list(groups.keys())
        if K is None or K >= len(keys):
            return keys
        mags = {k: np.abs(arr[-1, groups[k]].sum()) for k in keys}
        return sorted(keys, key=lambda k: mags[k], reverse=True)[:K]

    poly_plot_names = pick_top(groups_poly, raw_arr, max_terms)
    hill_inc_plot_names = pick_top(groups_hill_inc, raw_arr, max_terms)
    hill_dec_plot_names = pick_top(groups_hill_dec, raw_arr, max_terms)

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

    # --- plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    ax_raw_poly, ax_raw_hill = axes[0], axes[1]

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
    # raw hills: show inc then dec separate in same axes (order of legend follows combined_names)
    plot_panel(ax_raw_hill, hill_inc_plot_names + hill_dec_plot_names,
               {**groups_hill_inc, **groups_hill_dec}, raw_arr, 'Raw hill rates (inc & dec)')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        return fig, axes