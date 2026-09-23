"""
Per-frame statistics of the training data.

Both quantities below are built from the spatial standard deviation of each
species in each output frame. The standard deviation, not the mean, is used
because under mass conservation the mean barely moves while a pattern forms.

    unresolved_t_cutoff    first time at which the data resolves its own dynamics
    frame_activity_weights per-frame GLS weights, large where the pattern is changing
"""
import torch


def frame_stats(train_data, dimensions, species):
    """
    Spatial standard deviation of each species in each frame,
    (n_frames, species), and the frame times. Rows may be in any order.
    """
    times, frame = torch.unique(train_data[:, dimensions], return_inverse=True)
    conc = train_data[:, -species:]
    n_frames = len(times)

    counts = torch.zeros(n_frames, device=conc.device)
    counts.index_add_(0, frame, torch.ones(len(frame), device=conc.device))
    sums = torch.zeros(n_frames, species, device=conc.device).index_add_(0, frame, conc)
    sq_sums = torch.zeros(n_frames, species, device=conc.device).index_add_(0, frame, conc ** 2)

    mean = sums / counts[:, None]
    var = (sq_sums / counts[:, None] - mean ** 2).clamp_min(0.0)
    return var.sqrt(), times


def _leading_outlier_frames(std, factor):
    """
    Number of consecutive frame-to-frame changes, counted from the first
    frame, that exceed `factor` times the median change. None if the median
    change is zero (a static run).
    """
    delta = (std[1:] - std[:-1]).abs().max(dim=1).values
    typical = delta.median()
    if typical <= 0:
        return None

    flagged = delta > factor * typical
    k = 0
    while k < len(flagged) and flagged[k]:
        k += 1
    return k


def unresolved_t_cutoff(train_data, dimensions, species, factor=5.0):
    """
    Earliest time from which u_t is resolvable by the output sampling.

    An initial condition that relaxes faster than the sampling interval
    shows up as outlier frame-to-frame changes at the start of the run.
    There, neither finite differences nor a smooth surface can represent
    u_t, and fitting the PDE residual to it drives D and F down together.
    Only a contiguous run from the first frame is trimmed; a large change
    mid-run is a real event (a front arriving, a bifurcation).

    Returns 0.0 when nothing is flagged.
    """
    std, times = frame_stats(train_data, dimensions, species)
    if len(times) < 4:
        return 0.0
    k = _leading_outlier_frames(std, factor)
    return times[k].item() if k else 0.0


def frame_activity_weights(train_data, dimensions, species, max_weight=50.0, factor=5.0):
    """
    Per-frame GLS weights in [1, max_weight], proportional to how fast the
    pattern is changing, so fitting capacity goes where the dynamics are.

    Returns (weights, frame_times).
    """
    std, times = frame_stats(train_data, dimensions, species)
    std = std.cpu()
    n_frames = len(times)
    if n_frames < 4 or max_weight <= 1.0:
        return torch.ones(n_frames), times

    # Central difference of the spatial std: large while the pattern grows,
    # ~0 at steady state.
    activity = torch.zeros(n_frames)
    activity[1:-1] = (std[2:] - std[:-2]).abs().max(dim=1).values / 2
    activity[0] = (std[1] - std[0]).abs().max()
    activity[-1] = (std[-1] - std[-2]).abs().max()

    # Zero the unresolved initial transient (see unresolved_t_cutoff). Frame
    # k is zeroed as well: its central difference still straddles the
    # unresolved frame and would otherwise put the peak weight on the IC.
    k = _leading_outlier_frames(std, factor)
    if k is not None:
        activity[:k + 1] = 0.0

    peak = activity.max()
    if peak <= 0:
        return torch.ones(n_frames), times
    return 1.0 + (max_weight - 1.0) * (activity / peak), times