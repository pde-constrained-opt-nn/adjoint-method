"""
Phase 1 - Step 1.3: Baseline Reproduction
==========================================

Run all problems through the modular solver and verify results
match the original notebooks.

Usage (from repo root):
    python scripts/baseline_verification.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pde_adjoint_solver import forward_solve, adjoint_solve, adjoint_optimize
from problems import get_problem

# ===================================================================
# Configuration: which problems to run and with what settings
# ===================================================================

CONFIGS = [
    {
        'name': 'problem1',
        'label': 'Problem 1 (simple sine)',
        'lr': 1.0,
        'max_iter': 5000,
        'constraint': None,
        'expected_loss': 4.6e-4,  # from original notebook
    },
    {
        'name': 'problem3',
        'label': 'Problem 3 (separable)',
        'lr': 50.0,
        'max_iter': 500,
        'constraint': None,
        'expected_loss': 1.1e-5,
    },
    {
        'name': 'problem2',
        'label': 'Problem 2 (implicit, constrained)',
        'lr': 5.0,
        'max_iter': 5000, #5000 Original 
        'constraint': 'non_negative',
        'expected_loss': 1.68e-3,
    },
    {
        'name': 'high-osci',
        'label': 'High-Osci ω=15 (implicit)',
        'lr': 5.0,
        'max_iter': 10000,
        'constraint': None,
        'expected_loss': 5.99e-2,
    },
]


def run_baseline(config):
    """Run one problem and return results."""
    prob = get_problem(config['name'])
    print(f"\n{'='*60}")
    print(f"  {config['label']}")
    print(f"{'='*60}")
    print(prob.summary())

    f_init = np.zeros((prob.nt, prob.nx))

    f_opt, losses = adjoint_optimize(
        f_init=f_init,
        u_obs=prob.u_obs,
        u0=prob.u0,
        bc_left=prob.bc_left,
        bc_right=prob.bc_right,
        alpha=prob.alpha,
        dx=prob.dx,
        dt=prob.dt,
        nx=prob.nx,
        nt=prob.nt,
        lr=config['lr'],
        max_iter=config['max_iter'],
        scheme=prob.scheme,
        constraint=config['constraint'],
        log_every=config['max_iter'] // 5,
    )

    final_loss = losses[-1]
    expected = config['expected_loss']
    ratio = final_loss / expected
    status = "✓ MATCH" if 0.5 < ratio < 2.0 else "✗ MISMATCH"

    print(f"\n  Final loss:    {final_loss:.6e}")
    print(f"  Expected:      {expected:.6e}")
    print(f"  Ratio:         {ratio:.2f}")
    print(f"  Status:        {status}")

    return {
        'prob': prob,
        'f_opt': f_opt,
        'losses': losses,
        'final_loss': final_loss,
        'status': status,
        'config': config,
    }


def plot_results(results, savepath='baseline_results.png'):
    """Generate comparison plots for all problems."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, res in enumerate(results):
        prob = res['prob']
        f_opt = res['f_opt']
        losses = res['losses']
        cfg = res['config']

        # True f
        ax = axes[i, 0]
        im = ax.imshow(prob.f_true, aspect='auto',
                       extent=[0, prob.L, 0, prob.T_final],
                       origin='lower', cmap='viridis')
        ax.set_title(f"{cfg['label']}\nTrue f(x,t)")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        fig.colorbar(im, ax=ax)

        # Recovered f
        ax = axes[i, 1]
        im = ax.imshow(f_opt, aspect='auto',
                       extent=[0, prob.L, 0, prob.T_final],
                       origin='lower', cmap='viridis')
        ax.set_title(f"Recovered f(x,t)\n{res['status']}")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        fig.colorbar(im, ax=ax)

        # Loss curve
        ax = axes[i, 2]
        ax.plot(losses, linewidth=1.5)
        ax.set_title(f"Loss: {res['final_loss']:.4e}")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches='tight')
    print(f"\nPlot saved to: {savepath}")


# ===================================================================
# Main
# ===================================================================

if __name__ == '__main__':
    results = []

    for cfg in CONFIGS:
        res = run_baseline(cfg)
        results.append(res)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Problem':<35} {'Final Loss':>12} {'Expected':>12} {'Status':>10}")
    print(f"{'-'*35} {'-'*12} {'-'*12} {'-'*10}")
    for res in results:
        cfg = res['config']
        print(f"{cfg['label']:<35} {res['final_loss']:>12.4e} "
              f"{cfg['expected_loss']:>12.4e} {res['status']:>10}")

    plot_results(results,
                 savepath='/sessions/gallant-nifty-ramanujan/mnt/PDE/'
                          '1_Adjoint_Method/baseline_results.png')
