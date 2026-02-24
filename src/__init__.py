"""
Adjoint Method for PDE-Constrained Optimization
================================================

Core modules:
  solver   — Forward & adjoint PDE solvers (numpy)
  problems — Problem registry (Problem 1–3, High-Osci)
  nn       — Adjoint+NN hybrid method (JAX/Flax/Optax)
"""

from .pde_adjoint_solver import forward_solve, adjoint_solve, adjoint_optimize
from .problems import get_problem, PROBLEMS
from .adjoint_nn import (
    discrete_adjoint_gradient,
    make_pde_solver_vjp,
    create_model,
    nn_to_grid,
    train_adjoint_nn,
    verify_gradient,
    plot_comparison,
)
