# Alignment Failure Postmortem (2026-02-26)

## Symptoms
1. After alignment, NN performance on `high-osci` degraded significantly:
- Previous (better): `Adjoint+Fourier` reached around `~1e-2` (or lower).
- Current (worse): `Adjoint+Fourier` ended around `~5e-2`, with visible spikes during training.

2. VP smoke could not run initially due to missing dependencies and version conflicts.

## Primary Failure Points
1. **Training recipe drift** (main cause)
- The high-frequency run used `fourier + exponential`.
- The previously better recipe was `fourier_fixed + cosine(warmup)`.
- This reduced stability and led to worse local minima on high-frequency tasks.

2. **Environment coupling distorted reproducibility**
- Installing VP dependencies upgraded `jax/numpy/scipy`, diverging from the main experiment environment.
- Terminal environment and notebook kernel environment were not the same, causing false assumptions about reproducibility.

3. **VP optional dependencies were not preinstalled**
- Missing `matplotlib` and `vp_solver` caused VP smoke failures.
- After fixing these, VP ran, but the environment drift increased.

4. **Convention changes and recipe changes were mixed together**
- Alignment convention (`interior/exclude_t0`) introduces small numeric shifts by itself.
- The large performance drop was primarily caused by recipe/environment drift, not the convention alone.

## Confirmed Conclusions
1. `problem4` is still an alias of `high-osci`; no new PDE was introduced.
2. The observed degradation was mainly due to experiment setup changes, not a failure of the underlying method.

## Required Preconditions for Next Alignment Attempt
1. **Strict two-environment isolation**
- `pde-adjoint`: only for `problem2/high-osci` main experiments (pinned versions).
- `pde-vp`: only for VP smoke and Mark VP-related workflows.

2. **Freeze the high-frequency baseline recipe**
- Default for high-frequency: `arch=fourier_fixed`, `lr_schedule=cosine`, `warmup_ratio=0.10`, `seed=42`.
- Report both `final_loss` and `best_loss` (to avoid over-interpreting noisy tail points).

3. **Pass core acceptance before extending scope**
- First pass: no regression on `problem2` and restore `high-osci` to historical scale.
- Then integrate VP and cross-repo merged reporting.

4. **Print environment fingerprint at notebook startup**
- Log `sys.executable` and versions of `jax/flax/optax/numpy/scipy` at the start of each run.

## Suggested Recovery Path
1. Return to pre-alignment code and main environment, and verify baseline recovery.
2. Re-align incrementally with minimal changes:
- first problem aliases + metrics schema,
- then convention switching,
- finally VP integration.
