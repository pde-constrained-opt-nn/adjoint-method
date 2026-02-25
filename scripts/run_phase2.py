"""
Phase 2 Experiment Runner (Reproducible + Reporting)
====================================================

Runs gradient checks, pure-adjoint baselines, and Adjoint+NN experiments,
then exports both machine-readable and human-readable summaries.

Usage (from 1_Adjoint_Method/):
    python scripts/run_phase2.py
    python scripts/run_phase2.py --problem4
    python scripts/run_phase2.py --highosci --seed 42 --output_dir results/phase2
"""

import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

from problems import get_problem
from pde_adjoint_solver import adjoint_optimize


def _import_adjoint_nn():
    """Lazy import JAX-dependent module with a clearer error message."""
    try:
        from adjoint_nn import train_adjoint_nn, verify_gradient, plot_comparison
    except Exception as exc:
        raise RuntimeError(
            "Failed to import adjoint_nn (JAX/Flax/Optax environment issue).\n"
            "Please pin compatible versions before running NN experiments.\n"
            "Example: jax[cpu]==0.4.28, flax==0.8.5, optax==0.2.2."
        ) from exc
    return train_adjoint_nn, verify_gradient, plot_comparison


def _count_loss_spikes(losses, rel_jump=0.12, abs_jump=1e-4):
    """Count abrupt upward jumps in loss to quantify instability."""
    arr = np.asarray(losses, dtype=np.float64)
    if arr.size < 2:
        return 0
    prev = arr[:-1]
    curr = arr[1:]
    spikes = (curr > prev * (1.0 + rel_jump)) & ((curr - prev) > abs_jump)
    return int(np.sum(spikes))


def _run_pure_adjoint(problem_name, lr, max_iter, log_every, constraint=None):
    """Run pure adjoint baseline and collect timing metrics."""
    prob = get_problem(problem_name)
    start = time.perf_counter()
    _, losses = adjoint_optimize(
        f_init=np.zeros((prob.nt, prob.nx)),
        u_obs=prob.u_obs,
        u0=prob.u0,
        bc_left=prob.bc_left,
        bc_right=prob.bc_right,
        alpha=prob.alpha,
        dx=prob.dx,
        dt=prob.dt,
        nx=prob.nx,
        nt=prob.nt,
        lr=lr,
        max_iter=max_iter,
        scheme=prob.scheme,
        constraint=constraint,
        log_every=log_every,
    )
    wall = time.perf_counter() - start
    return {
        'problem': problem_name,
        'scheme': prob.scheme,
        'grid': {'nx': prob.nx, 'nt': prob.nt, 'dx': prob.dx, 'dt': prob.dt},
        'final_loss': float(losses[-1]),
        'best_loss': float(np.min(losses)),
        'wall_time_sec': float(wall),
        'sec_per_1000_iter': float(wall / max(1, max_iter + 1) * 1000.0),
        'spike_count': _count_loss_spikes(losses),
        'losses': [float(v) for v in losses],
    }


def _run_problem4_alias_baseline(max_iter):
    """problem4 is an alias of high-osci; run pure adjoint baseline on alias."""
    p4 = get_problem('problem4')
    ho = get_problem('high-osci')
    alias_ok = bool(
        p4.nx == ho.nx
        and p4.nt == ho.nt
        and np.isclose(p4.omega, ho.omega)
        and np.allclose(p4.f_true, ho.f_true)
    )
    res = _run_pure_adjoint('problem4', lr=5.0, max_iter=max_iter,
                            log_every=max(200, max_iter // 5))
    res['alias_to'] = 'high-osci'
    res['alias_check_passed'] = alias_ok
    return res


def run_gradient_verification():
    """Verify discrete-adjoint gradients on problem2 and high-osci."""
    _, verify_gradient, _ = _import_adjoint_nn()

    print("\n" + "=" * 72)
    print("  STEP 1: GRADIENT VERIFICATION")
    print("=" * 72)

    errors = {}
    for prob_name in ['problem2', 'high-osci']:
        err = verify_gradient(prob_name, n_checks=8, eps=1e-5)
        errors[prob_name] = float(err)
        print()
    return errors


def run_problem2_comparison(seed=42, max_iter=3000):
    """Compare pure adjoint baseline with Adjoint+NN (MLP) on problem2."""
    train_adjoint_nn, _, plot_comparison = _import_adjoint_nn()

    print("\n" + "=" * 72)
    print("  STEP 2: PROBLEM2 — PURE ADJOINT vs ADJOINT+NN")
    print("=" * 72)

    print("\n--- Pure Adjoint (problem2) ---")
    pure = _run_pure_adjoint(
        'problem2',
        lr=5.0,
        max_iter=max_iter,
        log_every=max(200, max_iter // 6),
    )
    print(f"  Final loss: {pure['final_loss']:.6e}")
    print(f"  Time: {pure['wall_time_sec']:.2f}s | sec/1k: {pure['sec_per_1000_iter']:.2f}")

    print("\n--- Adjoint + NN (MLP, problem2) ---")
    nn_res = train_adjoint_nn(
        problem_name='problem2',
        arch='mlp',
        model_kwargs={'hidden_dims': (128, 128, 128)},
        lr=1e-3,
        max_iter=max_iter,
        seed=seed,
        log_every=max(200, max_iter // 6),
        lr_schedule='exponential',
        grad_clip=1.0,
    )
    nn_out = {
        'arch': 'mlp',
        'final_loss': float(nn_res['losses'][-1]),
        'best_loss': float(nn_res['best_loss']),
        'wall_time_sec': float(nn_res['wall_time_sec']),
        'sec_per_1000_iter': float(nn_res['sec_per_1000_iter']),
        'spike_count': _count_loss_spikes(nn_res['losses']),
        'losses': [float(v) for v in nn_res['losses']],
    }
    print(f"  Final loss: {nn_out['final_loss']:.6e}")
    print(f"  Best loss:  {nn_out['best_loss']:.6e}")
    print(f"  Time: {nn_out['wall_time_sec']:.2f}s | sec/1k: {nn_out['sec_per_1000_iter']:.2f}")

    plot_path = PROJECT_ROOT / 'results' / 'phase2_problem2_mlp.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(nn_res, save_path=str(plot_path))

    return {
        'pure_adjoint': pure,
        'adjoint_nn_mlp': nn_out,
    }


def run_highosci_comparison(seed=42, max_iter=5000):
    """
    High-frequency challenge (problem4 alias high-osci):
    compare MLP / Fourier(fixed basis) / SIREN with stability diagnostics.
    """
    train_adjoint_nn, _, plot_comparison = _import_adjoint_nn()

    print("\n" + "=" * 72)
    print("  STEP 3: HIGH-OSCI (problem4 alias) — MLP vs FOURIER vs SIREN")
    print("=" * 72)

    print("\n--- Pure Adjoint baseline (high-osci) ---")
    pure = _run_pure_adjoint(
        'high-osci',
        lr=5.0,
        max_iter=max_iter,
        log_every=max(250, max_iter // 5),
    )
    print(f"  Final loss: {pure['final_loss']:.6e}")
    print(f"  Time: {pure['wall_time_sec']:.2f}s | sec/1k: {pure['sec_per_1000_iter']:.2f}")

    nn_cfgs = [
        {
            'label': 'Adjoint+MLP',
            'arch': 'mlp',
            'model_kwargs': {'hidden_dims': (256, 256, 256)},
            'lr': 1e-3,
            'lr_schedule': 'exponential',
            'grad_clip': 1.0,
            'warmup_ratio': 0.05,
        },
        {
            'label': 'Adjoint+Fourier',
            'arch': 'fourier_fixed',
            'model_kwargs': {
                'hidden_dims': (256, 256, 256),
                'num_frequencies': 128,
                'frequency_scale': 15.0,
            },
            'lr': 1e-3,
            'lr_schedule': 'cosine',
            'grad_clip': 1.0,
            'warmup_ratio': 0.10,
        },
        {
            'label': 'Adjoint+SIREN',
            'arch': 'siren',
            'model_kwargs': {
                'hidden_dims': (256, 256, 256),
                'omega_0': 20.0,
            },
            'lr': 5e-5,
            'lr_schedule': 'cosine',
            'grad_clip': 0.5,
            'warmup_ratio': 0.10,
        },
    ]

    outputs = {}
    for cfg in nn_cfgs:
        print(f"\n--- {cfg['label']} ({cfg['arch']}) ---")
        res = train_adjoint_nn(
            problem_name='high-osci',
            arch=cfg['arch'],
            model_kwargs=cfg['model_kwargs'],
            lr=cfg['lr'],
            max_iter=max_iter,
            seed=seed,
            log_every=max(250, max_iter // 5),
            lr_schedule=cfg['lr_schedule'],
            warmup_ratio=cfg['warmup_ratio'],
            grad_clip=cfg['grad_clip'],
        )
        out = {
            'arch': cfg['arch'],
            'final_loss': float(res['losses'][-1]),
            'best_loss': float(res['best_loss']),
            'wall_time_sec': float(res['wall_time_sec']),
            'sec_per_1000_iter': float(res['sec_per_1000_iter']),
            'spike_count': _count_loss_spikes(res['losses']),
            'losses': [float(v) for v in res['losses']],
            'train_cfg': {
                'lr': cfg['lr'],
                'lr_schedule': cfg['lr_schedule'],
                'grad_clip': cfg['grad_clip'],
                'warmup_ratio': cfg['warmup_ratio'],
                'seed': seed,
            },
        }
        outputs[cfg['label']] = out
        print(f"  Final loss: {out['final_loss']:.6e}")
        print(f"  Best loss:  {out['best_loss']:.6e}")
        print(f"  Spikes:     {out['spike_count']}")
        print(f"  Time: {out['wall_time_sec']:.2f}s | sec/1k: {out['sec_per_1000_iter']:.2f}")

        plot_path = PROJECT_ROOT / 'results' / f"phase2_highosci_{cfg['arch']}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(res, save_path=str(plot_path))

    print("\n" + "-" * 72)
    print(f"{'Method':<20} {'Final':>12} {'Best':>12} {'Spikes':>8} {'sec/1k':>10}")
    print("-" * 72)
    print(f"{'Pure Adjoint':<20} {pure['final_loss']:>12.4e} {pure['best_loss']:>12.4e}"
          f" {pure['spike_count']:>8d} {pure['sec_per_1000_iter']:>10.2f}")
    for name, out in outputs.items():
        print(f"{name:<20} {out['final_loss']:>12.4e} {out['best_loss']:>12.4e}"
              f" {out['spike_count']:>8d} {out['sec_per_1000_iter']:>10.2f}")

    return {
        'pure_adjoint': pure,
        'adjoint_nn': outputs,
    }


def _make_summary_md(metrics):
    """Generate concise review markdown from experiment metrics."""
    lines = []
    lines.append("# Phase2 Summary")
    lines.append("")
    lines.append(f"- Generated: {metrics['timestamp']}")
    lines.append(f"- Seed: {metrics['seed']}")
    lines.append("")

    alias = metrics['results'].get('problem4')
    if alias:
        lines.append("## Problem4 Alias Check")
        lines.append(f"- `problem4 -> high-osci`: "
                     f"{'PASS' if alias['alias_check_passed'] else 'FAIL'}")
        lines.append(f"- Pure adjoint final loss on `problem4`: {alias['final_loss']:.6e}")
        lines.append("")

    grad = metrics['results'].get('gradient_verification')
    if grad:
        lines.append("## Gradient Verification")
        lines.append("| Problem | Max Relative Error |")
        lines.append("|---|---:|")
        for k, v in grad.items():
            lines.append(f"| {k} | {v:.3e} |")
        lines.append("")

    p2 = metrics['results'].get('problem2')
    if p2:
        lines.append("## Problem2")
        lines.append("| Method | Final Loss | Best Loss | sec/1k iter |")
        lines.append("|---|---:|---:|---:|")
        pure = p2['pure_adjoint']
        nnm = p2['adjoint_nn_mlp']
        lines.append(f"| Pure Adjoint | {pure['final_loss']:.3e} | "
                     f"{pure['best_loss']:.3e} | {pure['sec_per_1000_iter']:.2f} |")
        lines.append(f"| Adjoint+NN (MLP) | {nnm['final_loss']:.3e} | "
                     f"{nnm['best_loss']:.3e} | {nnm['sec_per_1000_iter']:.2f} |")
        lines.append("")

    ho = metrics['results'].get('highosci')
    if ho:
        lines.append("## High-Osci / Problem4")
        lines.append("| Method | Final Loss | Best Loss | Spikes | sec/1k iter |")
        lines.append("|---|---:|---:|---:|---:|")
        pure = ho['pure_adjoint']
        lines.append(f"| Pure Adjoint | {pure['final_loss']:.3e} | "
                     f"{pure['best_loss']:.3e} | {pure['spike_count']} | "
                     f"{pure['sec_per_1000_iter']:.2f} |")
        for name, out in ho['adjoint_nn'].items():
            lines.append(f"| {name} ({out['arch']}) | {out['final_loss']:.3e} | "
                         f"{out['best_loss']:.3e} | {out['spike_count']} | "
                         f"{out['sec_per_1000_iter']:.2f} |")
        lines.append("")
        lines.append("- Note: `Adjoint+Fourier` here uses `arch=fourier_fixed` "
                     "for deterministic frequency coverage on ω=15.")
        lines.append("")

    return "\n".join(lines) + "\n"


def _export_reports(metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'metrics.json'
    summary_path = output_dir / 'summary.md'
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    summary_path.write_text(_make_summary_md(metrics), encoding='utf-8')
    print("\n" + "=" * 72)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print("=" * 72)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase2 experiment runner')
    parser.add_argument('--verify', action='store_true',
                        help='Run gradient verification')
    parser.add_argument('--problem2', action='store_true',
                        help='Run problem2 pure-adjoint vs adjoint+NN')
    parser.add_argument('--highosci', action='store_true',
                        help='Run high-osci (problem4 alias) architecture comparison')
    parser.add_argument('--problem4', action='store_true',
                        help='Run pure adjoint baseline on problem4 alias')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--p2_iters', type=int, default=3000, help='Iterations for problem2')
    parser.add_argument('--ho_iters', type=int, default=5000, help='Iterations for high-osci/problem4')
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'phase2'),
                        help='Directory to write metrics.json and summary.md')
    args = parser.parse_args()

    run_all = not (args.verify or args.problem2 or args.highosci or args.problem4)

    metrics = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'seed': args.seed,
        'results': {},
    }
    run_errors = []

    if args.verify or run_all:
        try:
            metrics['results']['gradient_verification'] = run_gradient_verification()
        except RuntimeError as exc:
            run_errors.append(f"gradient_verification: {exc}")

    if args.problem2 or run_all:
        try:
            metrics['results']['problem2'] = run_problem2_comparison(
                seed=args.seed,
                max_iter=args.p2_iters,
            )
        except RuntimeError as exc:
            run_errors.append(f"problem2: {exc}")

    if args.highosci or run_all:
        try:
            metrics['results']['highosci'] = run_highosci_comparison(
                seed=args.seed,
                max_iter=args.ho_iters,
            )
        except RuntimeError as exc:
            run_errors.append(f"highosci: {exc}")

    if args.problem4 or run_all:
        print("\n" + "=" * 72)
        print("  STEP 4: PROBLEM4 (ALIAS TO HIGH-OSCI) — PURE ADJOINT")
        print("=" * 72)
        p4 = _run_problem4_alias_baseline(max_iter=args.ho_iters)
        metrics['results']['problem4'] = p4
        print(f"  Alias check: {'PASS' if p4['alias_check_passed'] else 'FAIL'}")
        print(f"  Final loss:  {p4['final_loss']:.6e}")
        print(f"  Time: {p4['wall_time_sec']:.2f}s | sec/1k: {p4['sec_per_1000_iter']:.2f}")

    _export_reports(metrics, Path(args.output_dir))
    if run_errors:
        print("\nEncountered errors:")
        for err in run_errors:
            print(f"- {err}")
        raise SystemExit(1)
