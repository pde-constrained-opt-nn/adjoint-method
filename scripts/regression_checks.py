"""
Lightweight regression checks for key fixes.

Usage:
    python scripts/regression_checks.py
"""

from pathlib import Path
import sys
import importlib

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.problems import get_problem
from src import pde_adjoint_solver as pde


def check_problem4_alias():
    p4 = get_problem('problem4')
    ho = get_problem('high-osci')
    assert p4.nx == ho.nx and p4.nt == ho.nt
    assert np.isclose(p4.omega, ho.omega)
    assert np.allclose(p4.f_true, ho.f_true)
    return "problem4 alias maps to high-osci"


def check_scheme_time_index_updates():
    # Monkeypatch PDE calls so gradient is known and time-index updates can be asserted.
    forward_orig = pde.forward_solve
    adjoint_orig = pde.adjoint_solve

    def fake_forward(f, *_args, **_kwargs):
        return np.ones_like(f)

    def fake_adjoint(residual, *_args, **_kwargs):
        return np.ones_like(residual)

    pde.forward_solve = fake_forward
    pde.adjoint_solve = fake_adjoint

    try:
        nx, nt = 8, 6
        f_init = np.zeros((nt, nx))
        u_obs = np.zeros((nt, nx))
        u0 = np.zeros(nx)
        bc_left = np.zeros(nt)
        bc_right = np.zeros(nt)

        f_imp, _ = pde.adjoint_optimize(
            f_init=f_init,
            u_obs=u_obs,
            u0=u0,
            bc_left=bc_left,
            bc_right=bc_right,
            alpha=1.0,
            dx=1.0,
            dt=0.1,
            nx=nx,
            nt=nt,
            lr=0.5,
            max_iter=0,
            scheme='implicit',
            verbose=False,
        )
        assert np.allclose(f_imp[0, 1:-1], 0.0), "implicit must keep t=0 unchanged"
        assert np.allclose(f_imp[1:, 1:-1], -0.5), "implicit must update t=1..end"

        f_exp, _ = pde.adjoint_optimize(
            f_init=f_init,
            u_obs=u_obs,
            u0=u0,
            bc_left=bc_left,
            bc_right=bc_right,
            alpha=1.0,
            dx=1.0,
            dt=0.1,
            nx=nx,
            nt=nt,
            lr=0.5,
            max_iter=0,
            scheme='explicit',
            verbose=False,
        )
        assert np.allclose(f_exp[-1, 1:-1], 0.0), "explicit must keep last time layer unchanged"
        assert np.allclose(f_exp[:-1, 1:-1], -0.5), "explicit must update t=0..end-1"
    finally:
        pde.forward_solve = forward_orig
        pde.adjoint_solve = adjoint_orig

    return "scheme-aware active time indices are correct"


def check_baseline_savepath_relative():
    script_path = PROJECT_ROOT / 'scripts' / 'baseline_verification.py'
    text = script_path.read_text(encoding='utf-8')
    expected = "Path(__file__).resolve().parent.parent / 'results' / 'baseline_results.png'"
    assert expected in text
    assert '/Users/' not in text and 'C:\\' not in text
    return "baseline save path is project-relative"


def check_lazy_import_behavior():
    # Re-import src package and verify adjoint_nn is not imported eagerly.
    to_remove = [k for k in sys.modules if k == 'src' or k.startswith('src.')]
    for k in to_remove:
        del sys.modules[k]

    mod = importlib.import_module('src')
    assert 'src.adjoint_nn' not in sys.modules
    _ = mod.get_problem('problem1')
    assert 'src.adjoint_nn' not in sys.modules
    return "src lazy-import keeps numpy-only path lightweight"


def main():
    checks = [
        check_problem4_alias,
        check_scheme_time_index_updates,
        check_baseline_savepath_relative,
        check_lazy_import_behavior,
    ]
    failures = []
    print("=" * 72)
    print("Regression checks")
    print("=" * 72)
    for fn in checks:
        name = fn.__name__
        try:
            msg = fn()
            print(f"[PASS] {name}: {msg}")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}")

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"- {name}: {err}")
        raise SystemExit(1)

    print("\nAll regression checks passed.")


if __name__ == '__main__':
    main()
