"""
Check what a trained model's reaction actually computes.

    python check_reaction.py <run_dir> [--no-prune] [--set SPECIES:TERM=VALUE ...]

Compares, on concentrations drawn from the training data:

    forward   binn.reaction(u), the function simulate_feql integrates
    printed   the equation in equation.txt, rebuilt from the reported coefficients
    true      the ground-truth reaction from REACTION_REGISTRY

and checks the L0 gates directly: the gate value used in the forward pass (eval
mode) against the value the equation printout uses (gate.get_gates()). Then
compares the Jacobians of forward and true at the initial condition's mean
state, which decide between oscillation (Hopf) and spots (Turing).

If forward disagrees with printed, the equation files and --set are describing
a different function from the one being simulated.

Keep next to train_binn_eql.py and resimulate_feql.py.
"""
import argparse

import torch

import resimulate_feql as rs
import train_binn_eql as train
from modules.binn_eql.equations.extract import extract_params, gate_values
from modules.binn_eql.equations.prune import fine_tune_eql
from modules.simulation.reaction_library import REACTION_REGISTRY


def printed_reaction(binn, x):
    """The polynomial part of the equation as printed, evaluated at x: (n, species)."""
    eql = binn.reaction.eql_layer
    params = extract_params(binn, full=True)
    rows = []
    for p in params:
        coeffs = torch.as_tensor(p['poly_coeffs_unscaled'], dtype=x.dtype, device=x.device)
        f = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        for powers, c in zip(p['poly_terms'] * binn.duplicates, coeffs):
            term = torch.ones_like(f)
            for i, k in enumerate(powers):
                term = term * x[:, i] ** k
            f = f + c * term
        rows.append(f)
    # Map reported equations back onto species (mcas mirrors are -1 x their row).
    cols = []
    for role, row in eql.species_role:
        cols.append(rows[row] if role == 'free' else -rows[row])
    return torch.stack(cols, dim=1)


def jacobian_summary(f, point):
    J = torch.autograd.functional.jacobian(lambda y: f(y.unsqueeze(0))[0], point)
    eig = torch.linalg.eigvals(J)
    return J, torch.trace(J).item(), torch.linalg.det(J).item(), eig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('run_dir')
    parser.add_argument('--no-prune', action='store_true')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='SPECIES:TERM=VALUE')
    args = parser.parse_args()

    from pathlib import Path
    run_dir = Path(args.run_dir)
    cfg = train.RunConfig.load(run_dir / 'config.json')
    raw_data, training_data = train.load_training_data(cfg)
    binn = rs.load_model(run_dir, cfg, training_data)
    if not args.no_prune:
        fine_tune_eql(binn, threshold=train.PRUNE_THRESHOLD, epsilon=train.HILL_MERGE_EPSILON)
    for species, term, value in (rs.parse_override(o) for o in args.overrides):
        rs.set_coefficient(binn, species, term, value)
    binn.eval()
    eql = binn.reaction.eql_layer

    print("\nEquation as printed:")
    for line in binn.equations_as_strings():
        print(f"  {line}")

    # --- 1. Gates: forward-pass value vs the value the printout uses ----------
    print("\n1. L0 gates (eval mode): value used by the forward pass vs by the printout")
    with torch.no_grad():
        for row, gate in enumerate(eql.l0_gates):
            used = gate().reshape(-1)
            reported = gate_values(gate).reshape(-1)
            open_cols = (eql.fc.weight[row].abs() > 1e-12).nonzero().reshape(-1)
            worst = (used - reported).abs()[open_cols].max().item() if len(open_cols) else 0.0
            print(f"   row {row}: max |forward gate - printed gate| over active terms = {worst:.3e}")
            for c in open_cols.tolist():
                print(f"      column {c:3d}: forward {used[c].item():.6f}   printed {reported[c].item():.6f}")

    # --- 2. Function values ----------------------------------------------------
    x = training_data[:, -binn.species:]
    x = x[torch.randperm(len(x), device=x.device)[:20000]].to(rs.DEVICE)
    true_fn = REACTION_REGISTRY[cfg.reaction]['fn']
    with torch.no_grad():
        f_forward = binn.reaction(x)
        f_printed = printed_reaction(binn, x)
        f_true = torch.stack(true_fn(x, cfg.params), dim=1)

    print("\n2. Reaction values on 20,000 training-data concentrations (max over points)")
    if eql.hill is not None:
        print("   (library has Hill terms; 'printed' covers polynomial terms only)")
    for s, name in enumerate(binn.species_names):
        scale = f_true[:, s].abs().max().item()
        d_fp = (f_forward[:, s] - f_printed[:, s]).abs().max().item()
        d_ft = (f_forward[:, s] - f_true[:, s]).abs().max().item()
        d_pt = (f_printed[:, s] - f_true[:, s]).abs().max().item()
        print(f"   {name}: |true| up to {scale:.3f}")
        print(f"      forward vs printed {d_fp:.3e}  ({100 * d_fp / scale:.2f}% of scale)")
        print(f"      forward vs true    {d_ft:.3e}  ({100 * d_ft / scale:.2f}%)")
        print(f"      printed vs true    {d_pt:.3e}  ({100 * d_pt / scale:.2f}%)")

    # --- 3. Jacobians at the initial condition's mean state --------------------
    ic = raw_data[raw_data[:, cfg.dimensions] == raw_data[:, cfg.dimensions].min()]
    point = ic[:, -binn.species:].mean(dim=0).to(rs.DEVICE)
    print(f"\n3. Jacobian at the initial condition's mean state {point.tolist()}")
    for label, f in (("forward", lambda y: binn.reaction(y)),
                     ("true", lambda y: torch.stack(true_fn(y, cfg.params), dim=1))):
        J, tr, det, eig = jacobian_summary(f, point)
        print(f"   {label:8} trace {tr:+.4f}  det {det:+.4f}  "
              f"eigenvalues {[f'{e.real:+.4f}{e.imag:+.4f}i' for e in eig.tolist()]}")
        print(f"            J = {[[round(v, 4) for v in r] for r in J.tolist()]}")
    print("\n   A positive trace with a complex pair means oscillation grows; compare the"
          "\n   two traces. Forward and printed should agree to rounding; if section 1 or 2"
          "\n   shows otherwise, the simulated reaction is not the printed one.")


if __name__ == '__main__':
    main()