"""
Phase 2: Complete Experiment Runner
====================================

Runs all Phase 2 experiments:
  1. Gradient verification
  2. Adjoint+NN on Problem 2 (baseline comparison)
  3. High-frequency challenge: MLP vs Fourier vs SIREN

Usage (from repo root):
    python scripts/run_phase2.py              # Run all experiments
    python scripts/run_phase2.py --verify     # Gradient verification only
    python scripts/run_phase2.py --problem2   # Problem 2 only
    python scripts/run_phase2.py --highosci   # High-freq experiments only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import argparse
import numpy as np


def run_gradient_verification():
    """Step 2.2: Verify adjoint gradients."""
    from adjoint_nn import verify_gradient

    print("\n" + "=" * 70)
    print("  STEP 2.2: GRADIENT VERIFICATION")
    print("=" * 70)

    for prob_name in ['problem2', 'high-osci']:
        err = verify_gradient(prob_name, n_checks=8, eps=1e-5)
        print()
    return err


def run_problem2_comparison():
    """Step 2.4: Compare pure adjoint vs adjoint+NN on Problem 2."""
    from adjoint_nn import train_adjoint_nn, plot_comparison
    from pde_adjoint_solver import adjoint_optimize
    from problems import get_problem

    print("\n" + "=" * 70)
    print("  STEP 2.4: PROBLEM 2 — PURE ADJOINT vs ADJOINT+NN")
    print("=" * 70)

    # --- Pure Adjoint baseline ---
    print("\n--- Pure Adjoint (baseline) ---")
    prob = get_problem('problem2')
    f_adj, losses_adj = adjoint_optimize(
        f_init=np.zeros((prob.nt, prob.nx)),
        u_obs=prob.u_obs, u0=prob.u0,
        bc_left=prob.bc_left, bc_right=prob.bc_right,
        alpha=prob.alpha, dx=prob.dx, dt=prob.dt,
        nx=prob.nx, nt=prob.nt,
        lr=5.0, max_iter=3000, scheme='implicit',
        log_every=500,
    )
    print(f"  Pure Adjoint final loss: {losses_adj[-1]:.6e}")

    # --- Adjoint + NN (MLP) ---
    print("\n--- Adjoint + NN (MLP) ---")
    res_nn = train_adjoint_nn(
        problem_name='problem2',
        arch='mlp',
        lr=1e-3,
        max_iter=3000,
        log_every=500,
    )
    print(f"  Adjoint+NN final loss: {res_nn['losses'][-1]:.6e}")

    plot_comparison(res_nn, save_path='phase2_problem2_mlp.png')

    # --- Summary ---
    print(f"\n  {'Method':<25} {'Final Loss':>12}")
    print(f"  {'-'*25} {'-'*12}")
    print(f"  {'Pure Adjoint':<25} {losses_adj[-1]:>12.6e}")
    print(f"  {'Adjoint+NN (MLP)':<25} {res_nn['losses'][-1]:>12.6e}")

    return losses_adj, res_nn


def run_highosci_comparison():
    """Step 2.5: High-frequency challenge with different architectures."""
    from adjoint_nn import train_adjoint_nn, plot_comparison
    from pde_adjoint_solver import adjoint_optimize
    from problems import get_problem

    print("\n" + "=" * 70)
    print("  STEP 2.5: HIGH-FREQUENCY CHALLENGE (ω=15)")
    print("=" * 70)

    results = {}

    # --- Pure Adjoint baseline ---
    print("\n--- Pure Adjoint (baseline) ---")
    prob = get_problem('high-osci')
    f_adj, losses_adj = adjoint_optimize(
        f_init=np.zeros((prob.nt, prob.nx)),
        u_obs=prob.u_obs, u0=prob.u0,
        bc_left=prob.bc_left, bc_right=prob.bc_right,
        alpha=prob.alpha, dx=prob.dx, dt=prob.dt,
        nx=prob.nx, nt=prob.nt,
        lr=5.0, max_iter=5000, scheme='implicit',
        log_every=1000,
    )
    results['Pure Adjoint'] = losses_adj[-1]
    print(f"  Final loss: {losses_adj[-1]:.6e}")

    # --- Adjoint + MLP ---
    print("\n--- Adjoint + MLP ---")
    res_mlp = train_adjoint_nn(
        problem_name='high-osci',
        arch='mlp',
        model_kwargs={'hidden_dims': (256, 256, 256)},
        lr=1e-3,
        max_iter=5000,
        log_every=1000,
    )
    results['Adjoint+MLP'] = res_mlp['losses'][-1]
    plot_comparison(res_mlp, save_path='phase2_highosci_mlp.png')

    # --- Adjoint + Fourier MLP ---
    print("\n--- Adjoint + Fourier MLP ---")
    res_fourier = train_adjoint_nn(
        problem_name='high-osci',
        arch='fourier',
        model_kwargs={
            'hidden_dims': (256, 256, 256),
            'num_frequencies': 128,
            'frequency_scale': 15.0,  # Match omega
        },
        lr=1e-3,
        max_iter=5000,
        log_every=1000,
    )
    results['Adjoint+Fourier'] = res_fourier['losses'][-1]
    plot_comparison(res_fourier, save_path='phase2_highosci_fourier.png')

    # --- Adjoint + SIREN ---
    print("\n--- Adjoint + SIREN ---")
    res_siren = train_adjoint_nn(
        problem_name='high-osci',
        arch='siren',
        model_kwargs={
            'hidden_dims': (256, 256, 256),
            'omega_0': 30.0,
        },
        lr=1e-4,  # SIREN needs smaller LR
        max_iter=5000,
        log_every=1000,
    )
    results['Adjoint+SIREN'] = res_siren['losses'][-1]
    plot_comparison(res_siren, save_path='phase2_highosci_siren.png')

    # --- Summary Table ---
    print(f"\n{'='*50}")
    print(f"  HIGH-FREQUENCY COMPARISON (ω=15)")
    print(f"{'='*50}")
    print(f"  {'Method':<25} {'Final Loss':>12}")
    print(f"  {'-'*25} {'-'*12}")
    for method, loss in results.items():
        print(f"  {method:<25} {loss:>12.6e}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Gradient verification only')
    parser.add_argument('--problem2', action='store_true', help='Problem 2 comparison only')
    parser.add_argument('--highosci', action='store_true', help='High-freq experiments only')
    args = parser.parse_args()

    run_all = not (args.verify or args.problem2 or args.highosci)

    if args.verify or run_all:
        run_gradient_verification()

    if args.problem2 or run_all:
        run_problem2_comparison()

    if args.highosci or run_all:
        run_highosci_comparison()

    print("\n  Done. Check generated PNG files for plots.")
