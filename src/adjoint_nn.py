"""
Phase 2: Adjoint + NN Hybrid Method (JAX/Flax/Optax)
=====================================================

Core idea:
  - Parameterize f(x,t) = NN(x, t; θ)  using Flax
  - Forward PDE solve:  numpy-based solver (black box)
  - Backward pass:      adjoint equation gives ∇_f J,
                         then chain rule via JAX autodiff gives ∇_θ J

The key component is jax.custom_vjp which wraps the numpy PDE solver
so that JAX sees it as differentiable, but the backward pass uses
the adjoint equation instead of autodiff through the solver.

Usage:
    python adjoint_nn.py --problem problem2 --max_iter 3000
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial

# Enable 64-bit precision (critical for PDE accuracy)
jax.config.update('jax_enable_x64', True)

try:
    from .pde_adjoint_solver import forward_solve, adjoint_solve
    from .problems import get_problem
except ImportError:
    from pde_adjoint_solver import forward_solve, adjoint_solve
    from problems import get_problem


# ===================================================================
# 1. JAX Custom VJP Wrapper for PDE Solver
# ===================================================================

def discrete_adjoint_gradient(misfit, alpha, dx, dt, nx, nt, scheme='implicit'):
    """
    Compute exact discrete gradient ∂J/∂f via discrete adjoint method.

    Uses "discretize-then-optimize" approach so the gradient exactly matches
    finite differences of the discrete loss function.

    For loss  J = 0.5 * sum(misfit^2) * dx * dt
    where misfit = forward_solve(f) - u_obs.

    Parameters
    ----------
    misfit : ndarray, shape (nt, nx)
        u_pred - u_obs from the forward solve.
    alpha, dx, dt : float
        PDE and grid parameters.
    nx, nt : int
        Grid dimensions.
    scheme : str
        'implicit' (Backward Euler) or 'explicit' (Forward Euler).

    Returns
    -------
    grad_f : ndarray, shape (nt, nx)
        Exact discrete gradient ∂J/∂f_{n,i}.
    """
    grad_f = np.zeros((nt, nx))
    lam = alpha * dt / (dx ** 2)

    if scheme == 'implicit':
        # Forward:  A * u_{n+1} = u_n + dt * f_{n+1} + bc   (n = 0..nt-2)
        #   => f_0 never used,  f_1..f_{nt-1} drive the solve.
        #
        # Discrete adjoint (from Lagrangian stationarity ∂L/∂u_n = 0):
        #   A * mu_{nt-1} = -misfit_{nt-1} * dx * dt            (terminal)
        #   A * mu_n      =  mu_{n+1} - misfit_n * dx * dt      (n = nt-2 .. 1)
        #
        # Gradient:  ∂J/∂f_m = -dt * mu_m   (m = 1..nt-1)
        #            ∂J/∂f_0 = 0

        N_int = nx - 2
        main = (1.0 + 2.0 * lam) * np.ones(N_int)
        off = -lam * np.ones(N_int - 1)
        A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

        mu = np.zeros((nt, nx))

        # Terminal
        mu[nt - 1, 1:-1] = np.linalg.solve(
            A, -misfit[nt - 1, 1:-1] * dx * dt)

        # Backward sweep
        for n in range(nt - 2, 0, -1):
            rhs = mu[n + 1, 1:-1] - misfit[n, 1:-1] * dx * dt
            mu[n, 1:-1] = np.linalg.solve(A, rhs)

        # Gradient
        grad_f[1:, 1:-1] = -dt * mu[1:, 1:-1]
        # grad_f[0, :] = 0   (f_0 never enters the forward solve)

    elif scheme == 'explicit':
        # Forward:  u_{n+1} = B * u_n + dt * f_n + bc   (n = 0..nt-2)
        #   where B = I + dt * alpha * Laplacian_h
        #   => f_0..f_{nt-2} drive the solve,  f_{nt-1} never used.
        #
        # Discrete adjoint:
        #   mu_{nt-2} = -misfit_{nt-1} * dx * dt               (terminal)
        #   mu_{n-1}  =  B * mu_n - misfit_n * dx * dt         (n = nt-2 .. 1)
        #
        # Gradient:  ∂J/∂f_n = -dt * mu_n   (n = 0..nt-2)
        #            ∂J/∂f_{nt-1} = 0

        N_int = nx - 2
        main_B = (1.0 - 2.0 * lam) * np.ones(N_int)
        off_B = lam * np.ones(N_int - 1)
        B = np.diag(main_B) + np.diag(off_B, 1) + np.diag(off_B, -1)

        mu = np.zeros((nt, nx))

        # Terminal
        mu[nt - 2, 1:-1] = -misfit[nt - 1, 1:-1] * dx * dt

        # Backward sweep (explicit: matrix-vector multiply, no solve)
        for n in range(nt - 2, 0, -1):
            mu[n - 1, 1:-1] = B @ mu[n, 1:-1] - misfit[n, 1:-1] * dx * dt

        # Gradient
        grad_f[:-1, 1:-1] = -dt * mu[:-1, 1:-1]
        # grad_f[nt-1, :] = 0   (f_{nt-1} never enters the forward solve)

    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    return grad_f


def make_pde_solver_vjp(u0, bc_left, bc_right, u_obs,
                        alpha, dx, dt, nx, nt, scheme='implicit'):
    """
    Create a JAX-differentiable PDE solver using custom_vjp.

    The forward pass calls the numpy forward solver via jax.pure_callback
    (so it works inside jit / value_and_grad tracing).
    The backward pass uses the DISCRETE adjoint to compute exact ∂J/∂f.

    Parameters
    ----------
    u0, bc_left, bc_right : ndarray
        IC and BCs (fixed, not optimized).
    u_obs : ndarray, shape (nt, nx)
        Observed/target solution.
    alpha, dx, dt, nx, nt, scheme : PDE parameters.

    Returns
    -------
    pde_loss : callable
        JAX-differentiable function: f_grid (jnp array) -> scalar loss.
        Supports jax.grad(pde_loss)(f_grid).
    """
    # Pre-convert to numpy (these are constants, never traced)
    u0_np = np.asarray(u0)
    bc_left_np = np.asarray(bc_left)
    bc_right_np = np.asarray(bc_right)
    u_obs_np = np.asarray(u_obs)

    # --- Numpy callbacks (called via jax.pure_callback) ---

    def _forward_and_misfit(f_arr):
        """Numpy forward solve → (loss, misfit)."""
        f_np = np.asarray(f_arr)
        u_pred = forward_solve(f_np, u0_np, bc_left_np, bc_right_np,
                               alpha, dx, dt, nx, nt, scheme)
        misfit = (u_pred - u_obs_np).astype(np.float64)
        loss = np.float64(0.5 * np.sum(misfit ** 2) * dx * dt)
        return loss, misfit

    def _adjoint_grad(misfit_arr):
        """Discrete adjoint → ∂J/∂f."""
        misfit_np = np.asarray(misfit_arr)
        return discrete_adjoint_gradient(
            misfit_np, alpha, dx, dt, nx, nt, scheme).astype(np.float64)

    # Shape/dtype specs for pure_callback
    _loss_spec = jax.ShapeDtypeStruct((), jnp.float64)
    _misfit_spec = jax.ShapeDtypeStruct((nt, nx), jnp.float64)
    _grad_spec = jax.ShapeDtypeStruct((nt, nx), jnp.float64)

    # --- custom_vjp definition ---

    @jax.custom_vjp
    def pde_loss(f_grid):
        """Compute 0.5 * ||u(f) - u_obs||^2 * dx * dt."""
        loss, _ = jax.pure_callback(
            _forward_and_misfit, (_loss_spec, _misfit_spec), f_grid)
        return loss

    def pde_loss_fwd(f_grid):
        """Forward pass: compute loss and save misfit for backward."""
        loss, misfit = jax.pure_callback(
            _forward_and_misfit, (_loss_spec, _misfit_spec), f_grid)
        return loss, (misfit,)

    def pde_loss_bwd(res, g):
        """Backward pass: exact discrete adjoint for ∂J/∂f."""
        (misfit,) = res
        grad_f = jax.pure_callback(_adjoint_grad, _grad_spec, misfit)
        return (grad_f * g,)

    pde_loss.defvjp(pde_loss_fwd, pde_loss_bwd)
    return pde_loss


# ===================================================================
# 2. Flax Neural Network Models
# ===================================================================

class SourceMLP(nn.Module):
    """
    Standard MLP for source term estimation.
    Input: (batch, 2) for [x, t]  or  (batch, 3) for [x, y, t]
    Output: (batch,) scalar source values.
    """
    hidden_dims: tuple = (128, 128, 128)
    activation: str = 'tanh'

    @nn.compact
    def __call__(self, coords):
        act_fn = {'tanh': nn.tanh, 'relu': nn.relu, 'sigmoid': nn.sigmoid}[self.activation]
        x = coords
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = act_fn(x)
        return nn.Dense(1)(x).squeeze(-1)


class FourierMLP(nn.Module):
    """
    MLP with Fourier feature embedding (Tancik et al., 2020).
    Maps inputs through sin/cos at multiple frequencies before MLP.
    Helps overcome spectral bias for high-frequency problems.
    """
    hidden_dims: tuple = (256, 256, 256)
    num_frequencies: int = 128
    frequency_scale: float = 10.0
    activation: str = 'tanh'

    @nn.compact
    def __call__(self, coords):
        # Learnable frequency matrix
        B = self.param('fourier_B',
                       nn.initializers.normal(stddev=self.frequency_scale),
                       (coords.shape[-1], self.num_frequencies))
        proj = coords @ B  # (batch, num_frequencies)
        x = jnp.concatenate([jnp.sin(2 * jnp.pi * proj),
                             jnp.cos(2 * jnp.pi * proj)], axis=-1)

        act_fn = {'tanh': nn.tanh, 'relu': nn.relu, 'sigmoid': nn.sigmoid}[self.activation]
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = act_fn(x)
        return nn.Dense(1)(x).squeeze(-1)


class SIREN(nn.Module):
    """
    SIREN network (Sitzmann et al., 2020).
    Uses sin activation with specific initialization.
    Naturally captures high-frequency signals.
    """
    hidden_dims: tuple = (256, 256, 256)
    omega_0: float = 30.0

    @nn.compact
    def __call__(self, coords):
        # First layer with omega_0 scaling
        x = nn.Dense(self.hidden_dims[0],
                     kernel_init=nn.initializers.uniform(scale=1.0 / coords.shape[-1]))(coords)
        x = jnp.sin(self.omega_0 * x)

        # Hidden layers with sqrt(6 / n) init
        for dim in self.hidden_dims[1:]:
            x = nn.Dense(dim,
                         kernel_init=nn.initializers.uniform(
                             scale=jnp.sqrt(6.0 / dim) / self.omega_0))(x)
            x = jnp.sin(self.omega_0 * x)

        return nn.Dense(1)(x).squeeze(-1)


def create_model(arch='mlp', **kwargs):
    """Factory function for model creation."""
    models = {
        'mlp': SourceMLP,
        'fourier': FourierMLP,
        'siren': SIREN,
    }
    if arch not in models:
        raise ValueError(f"Unknown arch {arch!r}. Available: {list(models.keys())}")
    return models[arch](**kwargs)


# ===================================================================
# 3. NN → Grid Evaluation
# ===================================================================

def nn_to_grid(model, params, x_grid, t_grid):
    """
    Evaluate NN on the full space-time grid.

    Parameters
    ----------
    model : Flax Module
    params : PyTree
    x_grid : ndarray, shape (nx,)
    t_grid : ndarray, shape (nt,)

    Returns
    -------
    f_grid : jnp.ndarray, shape (nt, nx)
    """
    # Normalize to [-1, 1]
    x_norm = 2.0 * x_grid / x_grid[-1] - 1.0
    t_norm = 2.0 * t_grid / t_grid[-1] - 1.0

    # Create meshgrid coordinates
    X, T = jnp.meshgrid(x_norm, t_norm)  # both (nt, nx)
    coords = jnp.stack([X.ravel(), T.ravel()], axis=-1)  # (nt*nx, 2)

    # Evaluate NN
    f_flat = model.apply(params, coords)  # (nt*nx,)
    return f_flat.reshape(len(t_grid), len(x_grid))  # (nt, nx)


# ===================================================================
# 4. Training Loop
# ===================================================================

def train_adjoint_nn(problem_name, arch='mlp', model_kwargs=None,
                     lr=1e-3, max_iter=3000, lr_decay=0.95,
                     lambda_reg=0.0, seed=42, log_every=100,
                     verbose=True):
    """
    Train NN source estimator using adjoint-based gradients.

    Parameters
    ----------
    problem_name : str
        Problem identifier (e.g., 'problem2', 'high-osci').
    arch : str
        NN architecture: 'mlp', 'fourier', or 'siren'.
    model_kwargs : dict or None
        Extra kwargs for model constructor.
    lr : float
        Initial learning rate.
    max_iter : int
        Number of optimization steps.
    lr_decay : float
        Exponential decay rate for learning rate.
    lambda_reg : float
        Tikhonov regularization weight on NN output.
    seed : int
        Random seed for initialization.
    log_every : int
        Print interval.
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'params'       : final NN parameters
        'losses'       : loss history
        'f_opt'        : optimized f on grid, shape (nt, nx)
        'problem'      : Problem object
        'model'        : Flax model
    """
    # --- Setup problem ---
    prob = get_problem(problem_name)
    if verbose:
        print(prob.summary())
        print(f"  Arch: {arch}, LR: {lr}, Iters: {max_iter}")
        print()

    # --- Setup model ---
    kw = model_kwargs or {}
    model = create_model(arch, **kw)
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.zeros((1, 2))  # (batch=1, [x, t])
    params = model.init(key, dummy_input)

    # --- Setup PDE solver with custom_vjp ---
    pde_loss_fn = make_pde_solver_vjp(
        u0=prob.u0, bc_left=prob.bc_left, bc_right=prob.bc_right,
        u_obs=prob.u_obs,
        alpha=prob.alpha, dx=prob.dx, dt=prob.dt,
        nx=prob.nx, nt=prob.nt, scheme=prob.scheme,
    )

    # Coordinate grids (as jnp arrays)
    x_grid = jnp.array(prob.x)
    t_grid = jnp.array(prob.t)

    # --- Full loss: PDE data term + regularization ---
    def total_loss(params):
        f_grid = nn_to_grid(model, params, x_grid, t_grid)
        data_loss = pde_loss_fn(f_grid)
        reg_loss = lambda_reg * jnp.mean(f_grid ** 2) if lambda_reg > 0 else 0.0
        return data_loss + reg_loss

    # --- Optimizer ---
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=max_iter,
        decay_rate=lr_decay,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(params)

    # --- Training loop ---
    loss_history = []

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(total_loss)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for it in range(max_iter + 1):
        params, opt_state, loss = train_step(params, opt_state)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if verbose and it % log_every == 0:
            print(f"  Iter {it:5d}/{max_iter}: loss = {loss_val:.6e}")

    # --- Final evaluation ---
    f_opt = np.asarray(nn_to_grid(model, params, x_grid, t_grid))

    return {
        'params': params,
        'losses': loss_history,
        'f_opt': f_opt,
        'problem': prob,
        'model': model,
    }


# ===================================================================
# 5. Gradient Verification
# ===================================================================

def verify_gradient(problem_name='problem2', n_checks=5, eps=1e-5):
    """
    Verify adjoint gradient against finite differences.

    Uses a coarse grid for speed. Picks random grid points
    and compares adjoint gradient to central finite differences.

    Returns relative error (should be < 1e-4).
    """
    # Use coarse grid for speed
    prob = get_problem(problem_name, nx=20, nt=10)
    print(f"Gradient verification on {problem_name} (coarse: {prob.nx}×{prob.nt})")

    pde_loss_fn = make_pde_solver_vjp(
        u0=prob.u0, bc_left=prob.bc_left, bc_right=prob.bc_right,
        u_obs=prob.u_obs,
        alpha=prob.alpha, dx=prob.dx, dt=prob.dt,
        nx=prob.nx, nt=prob.nt, scheme=prob.scheme,
    )

    # Random test point
    key = jax.random.PRNGKey(0)
    f_test = jax.random.normal(key, (prob.nt, prob.nx)) * 0.1

    # Adjoint gradient
    adjoint_grad = jax.grad(pde_loss_fn)(f_test)

    # Finite difference gradient (spot check random indices)
    key2 = jax.random.PRNGKey(1)
    indices = jax.random.randint(key2, (n_checks, 2),
                                  minval=jnp.array([1, 1]),
                                  maxval=jnp.array([prob.nt - 1, prob.nx - 1]))

    max_rel_error = 0.0
    print(f"  {'Index':<12} {'Adjoint':>12} {'FD':>12} {'Rel Err':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for k in range(n_checks):
        i, j = int(indices[k, 0]), int(indices[k, 1])

        f_plus = f_test.at[i, j].set(f_test[i, j] + eps)
        f_minus = f_test.at[i, j].set(f_test[i, j] - eps)
        fd_val = float((pde_loss_fn(f_plus) - pde_loss_fn(f_minus)) / (2 * eps))
        adj_val = float(adjoint_grad[i, j])

        rel_err = abs(adj_val - fd_val) / (abs(fd_val) + 1e-12)
        max_rel_error = max(max_rel_error, rel_err)
        print(f"  ({i:2d},{j:2d})     {adj_val:>12.6e} {fd_val:>12.6e} {rel_err:>12.6e}")

    status = "PASS" if max_rel_error < 1e-3 else "FAIL"
    print(f"\n  Max relative error: {max_rel_error:.6e}  [{status}]")
    return max_rel_error


# ===================================================================
# 6. Visualization
# ===================================================================

def plot_comparison(results, save_path=None):
    """Plot true vs recovered f and loss curve.

    In notebook usage (save_path=None), keep the current interactive backend.
    In script usage (save_path provided), save and close the figure.
    """
    import matplotlib.pyplot as plt

    prob = results['problem']
    f_opt = results['f_opt']
    losses = results['losses']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # True f
    im = axes[0].imshow(prob.f_true, aspect='auto',
                        extent=[0, prob.L, 0, prob.T_final],
                        origin='lower', cmap='viridis')
    axes[0].set_title('True f(x,t)')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
    fig.colorbar(im, ax=axes[0])

    # Recovered f
    im = axes[1].imshow(f_opt, aspect='auto',
                        extent=[0, prob.L, 0, prob.T_final],
                        origin='lower', cmap='viridis')
    axes[1].set_title('Recovered f(x,t) [Adjoint+NN]')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
    fig.colorbar(im, ax=axes[1])

    # Error
    error = f_opt - prob.f_true
    im = axes[2].imshow(error, aspect='auto',
                        extent=[0, prob.L, 0, prob.T_final],
                        origin='lower', cmap='RdBu_r')
    axes[2].set_title(f'Error (L2={np.sqrt(np.sum(error**2) * prob.dx * prob.dt):.4e})')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('t')
    fig.colorbar(im, ax=axes[2])

    # Loss
    axes[3].plot(losses, lw=1.5)
    axes[3].set_title(f'Loss: {losses[-1]:.4e}')
    axes[3].set_xlabel('Iteration'); axes[3].set_ylabel('Loss')
    axes[3].set_yscale('log'); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        plt.close()
    else:
        plt.show()


# ===================================================================
# 7. CLI
# ===================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Adjoint + NN PDE Optimization')
    parser.add_argument('--problem', default='problem2', help='Problem name')
    parser.add_argument('--arch', default='mlp', choices=['mlp', 'fourier', 'siren'])
    parser.add_argument('--max_iter', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--verify_grad', action='store_true',
                        help='Run gradient verification before training')
    parser.add_argument('--save_plot', default=None, help='Save path for plot')
    args = parser.parse_args()

    # --- Gradient verification ---
    if args.verify_grad:
        print("=" * 60)
        print("  GRADIENT VERIFICATION")
        print("=" * 60)
        verify_gradient(args.problem)
        print()

    # --- Training ---
    print("=" * 60)
    print(f"  ADJOINT + NN TRAINING ({args.arch.upper()})")
    print("=" * 60)
    results = train_adjoint_nn(
        problem_name=args.problem,
        arch=args.arch,
        lr=args.lr,
        max_iter=args.max_iter,
        lambda_reg=args.lambda_reg,
    )

    print(f"\n  Final loss: {results['losses'][-1]:.6e}")

    # --- Plot ---
    save_path = args.save_plot or f"adjoint_nn_{args.problem}_{args.arch}.png"
    plot_comparison(results, save_path=save_path)
