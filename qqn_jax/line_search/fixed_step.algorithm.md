# Fixed-Step Line Search

## Overview

`fixed_step_search` is the simplest possible line-search strategy in the
`qqn_jax` line-search family. Rather than iteratively probing along the
search direction to satisfy a sufficient-decrease condition (as in
backtracking, Wolfe, or bisection strategies), it evaluates the 1-D
objective `φ(t)` at a **single, constant step size** and reports the
result.

This is useful as:

- A **baseline** for comparing more sophisticated line searches.
- A **fast path** when the caller already knows a good step length.
- A building block for algorithms (e.g. fixed-schedule SGD-style updates)
  where step selection is handled elsewhere.

## Signature

```python
fixed_step_search(
    eval_at,          # Callable: t -> (params, value, grad, slope)
    params,           # current parameters
    value,            # current objective value φ(0)
    grad,             # current gradient
    slope0,           # directional derivative at t=0 (unused)
    *,
    step_size=1.0,    # nominal step length
    temperature=0.0,  # Metropolis acceptance temperature
    cooling=0.95,     # accepted for interface parity (unused here)
    seed=0,           # PRNG seed for stochastic acceptance
    max_probes=32,    # size of probe record buffers
    record_probes=True,
    max_step=1.0,     # upper clip on step length
) -> LineSearchResult
```

## Algorithm

The procedure is intentionally trivial:

1. **Clip the step.** Compute
   `α = min(step_size, max_step)`, cast to the working dtype. This
   guarantees the reported step never exceeds the caller's trust bound.

2. **Single evaluation.** Call `eval_at(α)` once to obtain the candidate
   `(new_params, new_val, new_g, slope)`. Because the search path and any
   trust region are already folded into `eval_at` by the solver, this
   routine is entirely **path-agnostic** — it never manipulates the raw
   search direction itself.

3. **Convergence / acceptance gate (`done`).**
   - When `temperature == 0`, the step is *unconditionally* accepted
     (`done = True`). The fixed step is taken as-is.
   - When `temperature > 0`, a **Metropolis meta-rule** is applied via
     `_metropolis_accept(Δφ, T, key)` where `Δφ = new_val - value`.
     `done` becomes true if the move is a genuine descent
     (`new_val < value`) **OR** the uphill move is stochastically
     accepted. This allows the fixed-step search to participate in
     simulated-annealing-style optimisation, occasionally accepting
     worse points to escape local minima.

4. **Probe recording.** A probe buffer of size `max_probes`
   (or `1` if `record_probes` is `False`) is allocated with
   `_empty_probes`, and the single evaluation is stored at index `0` with
   `_record_probe`. This keeps the output shape-compatible with the other
   line-search strategies, which may record many probes.

5. **Return.** A `LineSearchResult` is produced with `num_evals = 1`.

## Notes on Parameters

- **`slope0`** is accepted for interface uniformity but immediately
  discarded (`del slope0`); a fixed step does not use curvature/slope
  information.
- **`cooling`** is accepted for signature parity with annealing-capable
  searches but is not applied here — there is only one evaluation, so
  there is no schedule to cool. Temperature decay, if desired, must be
  handled by the caller across outer iterations.
- **`seed`** seeds `jax.random.PRNGKey`. Note the returned key from
  `_metropolis_accept` is discarded (`_key`); callers requiring
  independent randomness across steps should vary `seed`.

## JAX / Functional Considerations

- The routine is **pure and traceable**: control flow uses `jnp.where`
  rather than Python `if`, so it is safe under `jit`, `vmap`, and `grad`
  transformations.
- All scalar constants are cast to `value.dtype` to avoid dtype-promotion
  surprises inside traced code.
- Buffer sizes (`max_probes`, `record_probes`) are treated as static and
  must not be traced values.

## Complexity

- **Objective evaluations:** exactly `1`.
- **Memory:** `O(max_probes)` for the probe buffers (mostly unused, but
  kept for output-shape consistency across strategies).

## Discussion

Fixed-step search trades robustness for speed and simplicity. It offers no
guarantee of sufficient decrease and can diverge if `step_size` is too
large for the local geometry — responsibility for choosing a safe step
rests entirely with the caller (or an outer trust-region mechanism baked
into `eval_at`). Its chief virtues are its single evaluation cost, its
fully differentiable/vectorisable implementation, and its optional
Metropolis gate, which lets it double as a primitive stochastic-acceptance
step within annealing-flavoured optimisers.