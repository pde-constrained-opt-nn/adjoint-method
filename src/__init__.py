"""
Adjoint Method for PDE-Constrained Optimization
================================================

Core modules:
  pde_adjoint_solver — Forward & adjoint PDE solvers (numpy only)
  problems           — Problem registry (numpy only)
  adjoint_nn         — Adjoint+NN hybrid (requires JAX/Flax/Optax)

Light imports (numpy-only modules) are available immediately.
JAX-dependent symbols are loaded lazily on first access.
"""

# Light imports: numpy-only, always available
from .pde_adjoint_solver import forward_solve, adjoint_solve, adjoint_optimize
from .problems import get_problem, PROBLEMS


def __getattr__(name):
    """Lazy import for JAX-dependent symbols (adjoint_nn)."""
    _nn_symbols = {
        'discrete_adjoint_gradient',
        'make_pde_solver_vjp',
        'create_model',
        'nn_to_grid',
        'train_adjoint_nn',
        'verify_gradient',
        'plot_comparison',
    }
    if name in _nn_symbols:
        from . import adjoint_nn
        return getattr(adjoint_nn, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
