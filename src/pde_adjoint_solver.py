"""
Modular 1D Heat Equation Solver with Adjoint Method
=====================================================

Decoupled forward/adjoint solvers for PDE-constrained optimization.
No reference to any specific problem (u_true) inside the solver.

PDE:  u_t - alpha * u_xx = f   on [x0, x1] x [0, T]
      u(x, 0)  = IC(x)         initial condition
      u(x0, t) = bc_left(t)    left Dirichlet BC
      u(x1, t) = bc_right(t)   right Dirichlet BC

Adjoint: -p_t - alpha * p_xx = residual   (backward in time)
         p(x, T) = 0                      terminal condition
         p(x0, t) = p(x1, t) = 0          homogeneous Dirichlet
"""

import numpy as np


# ---------------------------------------------------------------------------
# Forward Solver
# ---------------------------------------------------------------------------

def forward_solve(f, u0, bc_left, bc_right, alpha, dx, dt, nx, nt,
                  scheme='implicit'):
    """
    Solve the 1D heat equation forward in time.

    Parameters
    ----------
    f : ndarray, shape (nt, nx)
        Source term on the full space-time grid.
    u0 : ndarray, shape (nx,)
        Initial condition u(x, 0).
    bc_left : ndarray, shape (nt,)
        Left Dirichlet boundary u(x0, t).
    bc_right : ndarray, shape (nt,)
        Right Dirichlet boundary u(x1, t).
    alpha : float
        Diffusion coefficient.
    dx, dt : float
        Grid spacings.
    nx, nt : int
        Number of grid points in space and time.
    scheme : str
        'implicit' (Backward Euler, unconditionally stable) or
        'explicit' (Forward Euler, requires CFL <= 0.5).

    Returns
    -------
    u : ndarray, shape (nt, nx)
        Solution on the full space-time grid.
    """
    u = np.zeros((nt, nx))

    # Initial condition
    u[0, :] = u0

    # Boundary conditions at t=0 (override corners for consistency)
    u[0, 0] = bc_left[0]
    u[0, -1] = bc_right[0]

    lam = alpha * dt / (dx ** 2)

    if scheme == 'implicit':
        N_int = nx - 2
        main = (1.0 + 2.0 * lam) * np.ones(N_int)
        off = -lam * np.ones(N_int - 1)
        A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

        for n in range(nt - 1):
            u[n + 1, 0] = bc_left[n + 1]
            u[n + 1, -1] = bc_right[n + 1]

            rhs = u[n, 1:-1].copy() + dt * f[n + 1, 1:-1]
            rhs[0] += lam * u[n + 1, 0]
            rhs[-1] += lam * u[n + 1, -1]

            u[n + 1, 1:-1] = np.linalg.solve(A, rhs)

    elif scheme == 'explicit':
        cfl = lam
        if cfl > 0.5:
            raise ValueError(
                f"Explicit scheme unstable: CFL = {cfl:.4f} > 0.5. "
                f"Reduce dt or increase nx."
            )
        for n in range(nt - 1):
            u[n + 1, 0] = bc_left[n + 1]
            u[n + 1, -1] = bc_right[n + 1]

            u_xx = (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]) / (dx ** 2)
            u[n + 1, 1:-1] = u[n, 1:-1] + dt * (alpha * u_xx + f[n, 1:-1])
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}. Use 'implicit' or 'explicit'.")

    return u


# ---------------------------------------------------------------------------
# Adjoint Solver
# ---------------------------------------------------------------------------

def adjoint_solve(residual, alpha, dx, dt, nx, nt, scheme='implicit'):
    """
    Solve the adjoint equation backward in time.

    The adjoint PDE is:
        -p_t - alpha * p_xx = residual(x, t)
    with terminal condition p(x, T) = 0 and homogeneous Dirichlet BCs.

    Parameters
    ----------
    residual : ndarray, shape (nt, nx)
        Source term for adjoint equation, typically (u_pred - u_obs).
    alpha : float
        Diffusion coefficient.
    dx, dt : float
        Grid spacings.
    nx, nt : int
        Number of grid points in space and time.
    scheme : str
        'implicit' or 'explicit'.

    Returns
    -------
    p : ndarray, shape (nt, nx)
        Adjoint variable on the full space-time grid.
    """
    p = np.zeros((nt, nx))
    # Terminal condition p(T) = 0 is satisfied by initialization.

    lam = alpha * dt / (dx ** 2)

    if scheme == 'implicit':
        N_int = nx - 2
        main = (1.0 + 2.0 * lam) * np.ones(N_int)
        off = -lam * np.ones(N_int - 1)
        A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

        for n in range(nt - 2, -1, -1):
            rhs = p[n + 1, 1:-1] + dt * residual[n + 1, 1:-1]
            p[n, 1:-1] = np.linalg.solve(A, rhs)

    elif scheme == 'explicit':
        cfl = lam
        if cfl > 0.5:
            raise ValueError(
                f"Explicit adjoint unstable: CFL = {cfl:.4f} > 0.5."
            )
        for n in range(nt - 2, -1, -1):
            p_xx = (p[n + 1, 2:] - 2 * p[n + 1, 1:-1] + p[n + 1, :-2]) / (dx ** 2)
            p[n, 1:-1] = (p[n + 1, 1:-1]
                          + dt * (alpha * p_xx + residual[n + 1, 1:-1]))
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    return p


# ---------------------------------------------------------------------------
# Optimization Loop
# ---------------------------------------------------------------------------

def adjoint_optimize(f_init, u_obs, u0, bc_left, bc_right,
                     alpha, dx, dt, nx, nt,
                     lr=1.0, max_iter=1000, scheme='implicit',
                     lambda_reg=0.0, constraint=None, constraint_args=None,
                     log_every=100, verbose=True):
    """
    Gradient descent with adjoint-based gradients.

    Parameters
    ----------
    f_init : ndarray, shape (nt, nx)
        Initial guess for the source term.
    u_obs : ndarray, shape (nt, nx)
        Observed/target solution.
    u0 : ndarray, shape (nx,)
        Initial condition.
    bc_left, bc_right : ndarray, shape (nt,)
        Dirichlet boundary conditions.
    alpha, dx, dt : float
        Physical and grid parameters.
    nx, nt : int
        Grid dimensions.
    lr : float
        Learning rate (step size).
    max_iter : int
        Number of gradient descent iterations.
    scheme : str
        'implicit' or 'explicit'.
    lambda_reg : float
        Tikhonov regularization weight.
    constraint : str or None
        'non_negative' for f >= 0,
        'box' for 0 <= f <= M (requires constraint_args={'M': value}),
        None for unconstrained.
    constraint_args : dict or None
        Additional arguments for constraint (e.g., {'M': 10.0}).
    log_every : int
        Print loss every this many iterations.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    f_opt : ndarray, shape (nt, nx)
        Optimized source term.
    loss_history : list of float
        Loss at each iteration.
    """
    f = f_init.copy()
    loss_history = []

    solver_kw = dict(alpha=alpha, dx=dx, dt=dt, nx=nx, nt=nt, scheme=scheme)

    for it in range(max_iter + 1):
        # Forward solve
        u_pred = forward_solve(f, u0, bc_left, bc_right, **solver_kw)

        # Loss
        misfit = u_pred - u_obs
        data_term = 0.5 * np.sum(misfit ** 2) * dx * dt
        reg_term = 0.5 * lambda_reg * np.sum(f ** 2) * dx * dt
        loss = data_term + reg_term
        loss_history.append(loss)

        # Adjoint solve → gradient
        p = adjoint_solve(misfit, alpha, dx, dt, nx, nt, scheme=scheme)

        # Gradient: ∇_f J = lambda_reg * f + p
        # (adjoint convention: -p_t - α p_xx = +(u - u_obs), so ∇_f J = +p)
        grad = lambda_reg * f + p

        # Gradient descent on active grid points:
        # - Spatial: only interior points (boundaries are Dirichlet, f at x=0/L unused)
        # - Temporal: depends on scheme
        #   implicit uses f[n+1] for n=0..nt-2, so f[1..nt-1] are active
        #   explicit uses f[n]   for n=0..nt-2, so f[0..nt-2] are active
        if scheme == 'implicit':
            f[1:, 1:-1] -= lr * grad[1:, 1:-1]
        else:  # explicit
            f[:-1, 1:-1] -= lr * grad[:-1, 1:-1]

        # Projection / constraint
        if constraint == 'non_negative':
            f = np.maximum(f, 0.0)
        elif constraint == 'box':
            M = constraint_args.get('M', 1.0) if constraint_args else 1.0
            f = np.clip(f, 0.0, M)

        if verbose and it % log_every == 0:
            print(f"  Iter {it:5d}/{max_iter}: loss = {loss:.6e}")

    return f, loss_history
