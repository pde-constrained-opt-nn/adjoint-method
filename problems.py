"""
Problem Definitions for 1D Heat Equation Inverse Problems
==========================================================

Each problem provides:
  - u_exact(X, T)  : analytical solution on meshgrid
  - f_exact(X, T)  : analytical source term (derived from u_t - alpha * u_xx)
  - domain config   : spatial/temporal bounds, diffusion coefficient

All functions take meshgrid arrays (X, T) and return arrays of same shape.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Problem Registry
# ---------------------------------------------------------------------------

PROBLEMS = {}


def register(name):
    """Decorator to register a problem class."""
    def wrapper(cls):
        PROBLEMS[name] = cls
        return cls
    return wrapper


def get_problem(name, **overrides):
    """
    Instantiate a registered problem by name.

    Parameters
    ----------
    name : str
        Problem identifier (e.g., 'problem1', 'high-osci').
    **overrides
        Override default grid/domain parameters (nx, nt, etc.).

    Returns
    -------
    problem : ProblemBase instance
    """
    if name not in PROBLEMS:
        available = ', '.join(sorted(PROBLEMS.keys()))
        raise ValueError(f"Unknown problem {name!r}. Available: {available}")
    return PROBLEMS[name](**overrides)


# ---------------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------------

class ProblemBase:
    """Base class for 1D heat equation test problems."""

    def __init__(self, nx=50, nt=5000, L=1.0, T=0.5, alpha=1.0,
                 scheme='explicit', **kwargs):
        self.nx = nx
        self.nt = nt
        self.L = L
        self.T_final = T
        self.alpha = alpha
        self.scheme = scheme

        self.dx = L / (nx - 1)
        self.dt = T / (nt - 1)
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T, nt)
        self.X, self.T_mesh = np.meshgrid(self.x, self.t)

        # Compute ground truth
        self.u_true = self.u_exact(self.X, self.T_mesh)
        self.f_true = self.f_exact(self.X, self.T_mesh)
        self.u_obs = self.u_true.copy()

        # Extract IC / BCs from analytical solution
        self.u0 = self.u_true[0, :]          # u(x, 0)
        self.bc_left = self.u_true[:, 0]     # u(0, t)
        self.bc_right = self.u_true[:, -1]   # u(L, t)

    def u_exact(self, X, T):
        raise NotImplementedError

    def f_exact(self, X, T):
        raise NotImplementedError

    @property
    def cfl(self):
        return self.alpha * self.dt / (self.dx ** 2)

    def summary(self):
        return (
            f"{self.__class__.__name__}\n"
            f"  Domain: x ∈ [0, {self.L}], t ∈ [0, {self.T_final}]\n"
            f"  Grid:   {self.nx} × {self.nt}  (dx={self.dx:.4e}, dt={self.dt:.4e})\n"
            f"  Scheme: {self.scheme}  |  CFL = {self.cfl:.4f}\n"
            f"  Alpha:  {self.alpha}"
        )


# ---------------------------------------------------------------------------
# Concrete Problems
# ---------------------------------------------------------------------------

@register('problem1')
class Problem1(ProblemBase):
    """
    Simple additive solution.
      u(x,t) = sin(2πx) + sin(πt)
      f(x,t) = π cos(πt) + 4π² sin(2πx)
    """

    def __init__(self, nx=50, nt=5000, **kw):
        super().__init__(nx=nx, nt=nt, T=0.5, scheme='explicit', **kw)

    def u_exact(self, X, T):
        return np.sin(2 * np.pi * X) + np.sin(np.pi * T)

    def f_exact(self, X, T):
        return (np.pi * np.cos(np.pi * T)
                + 4 * np.pi ** 2 * np.sin(2 * np.pi * X))


@register('problem2')
class Problem2(ProblemBase):
    """
    Superimposed sine waves (implicit scheme recommended).
      u(x,t) = [sin(πx) - 0.5 sin(2πx)] sin(πt)
      f derived analytically.
    """

    def __init__(self, nx=150, nt=50, **kw):
        super().__init__(nx=nx, nt=nt, T=0.5, scheme='implicit', **kw)

    def u_exact(self, X, T):
        return (np.sin(np.pi * X) - 0.5 * np.sin(2 * np.pi * X)) * np.sin(np.pi * T)

    def f_exact(self, X, T):
        spatial = np.sin(np.pi * X) - 0.5 * np.sin(2 * np.pi * X)
        term1 = np.pi * np.cos(np.pi * T) * spatial
        term2 = np.pi ** 2 * np.sin(np.pi * T) * (
            np.sin(np.pi * X) - 2 * np.sin(2 * np.pi * X)
        )
        return term1 + term2


@register('problem3')
class Problem3(ProblemBase):
    """
    Space-time separable standing wave.
      u(x,t) = sin(2πx) cos(πt)
      f derived analytically.
    Fastest convergence due to separability.
    """

    def __init__(self, nx=50, nt=10000, **kw):
        super().__init__(nx=nx, nt=nt, T=1.0, scheme='explicit', **kw)

    def u_exact(self, X, T):
        return np.sin(2 * np.pi * X) * np.cos(np.pi * T)

    def f_exact(self, X, T):
        return (-np.pi * np.sin(np.pi * T) * np.sin(2 * np.pi * X)
                + 4 * np.pi ** 2 * np.cos(np.pi * T) * np.sin(2 * np.pi * X))


@register('high-osci')
class HighOscillatory(ProblemBase):
    """
    High-frequency oscillatory problem (ω = 15).
      u(x,t) = sin(ω π x) cos(ω π t)
    Tests the limits of the adjoint method.
    """

    def __init__(self, nx=200, nt=50, omega=15, **kw):
        self.omega = omega
        super().__init__(nx=nx, nt=nt, T=0.5, scheme='implicit', **kw)

    def u_exact(self, X, T):
        w = self.omega
        return np.sin(w * np.pi * X) * np.cos(w * np.pi * T)

    def f_exact(self, X, T):
        w = self.omega
        return (-w * np.pi * np.sin(w * np.pi * T) * np.sin(w * np.pi * X)
                + w ** 2 * np.pi ** 2 * np.cos(w * np.pi * T) * np.sin(w * np.pi * X))
