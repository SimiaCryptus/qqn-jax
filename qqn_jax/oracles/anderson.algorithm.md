# Anderson (Type-II) Acceleration Oracle

## Overview

The `AndersonOracle` implements **Anderson acceleration** (Type-II variant),
a multi-secant fixed-point acceleration scheme. It is presented here as the
"variational ideal" that limited-memory quasi-Newton methods (such as
L-BFGS) approximate. Rather than forming or approximating a Hessian, the
oracle solves a small `(m × m)` least-squares problem over recent gradient
differences to construct a descent direction that captures multi-step
curvature.

## Mathematical Formulation

At each step, the oracle maintains sliding windows of the most recent
gradients (residuals) and iterates. From these it forms first-difference
matrices `ΔG` and `ΔX`, then computes a mixing coefficient vector `θ` by
solving a regularized least-squares problem:

```
min_θ  ‖ ∇f − ΔG θ ‖²  +  reg · ‖θ‖²
```

The resulting search direction is:

```
direction = −β · ( ∇f − ΔG θ )  −  ΔX θ
```

where:

- `∇f` is the current gradient.
- `ΔG` are first-differences of the stored gradient window.
- `ΔX` are first-differences of the stored iterate window.
- `θ` are the multi-secant mixing coefficients.
- `β` is the **coupling constant** (mixing parameter).

### Behavior by window size

- **`window = 1`**: The update reduces to a single secant step.
- **Deep window (`window > 1`)**: The update captures multi-step curvature
  that a single secant cannot represent.

No Hessian is ever explicitly formed; the only linear solve is over an
`(m × m)` system, where `m` is the effective window size.

## Parameters

| Parameter | Type    | Default | Description                                                        |
|-----------|---------|---------|--------------------------------------------------------------------|
| `window`  | `int`   | `5`     | Number of past gradients/iterates retained in the sliding windows. |
| `reg`     | `float` | `1e-8`  | Tikhonov regularization scaled by the mean Gram trace.             |
| `beta`    | `float` | `1.0`   | Coupling constant (mixing parameter) of the Anderson scheme.       |

### The coupling constant `β`

The mixing parameter rescales the accelerated residual toward the gradient's
natural magnitude:

- `β = 1` recovers the pure Type-II Anderson update.
- `β > 1` lets the deep-residual descent stretch, trading trajectory-AUC
  advantage for a leading *iteration* count.

## State

The oracle carries an `AndersonState`, a `NamedTuple` with:

| Field         | Shape    | Description                                        |
|---------------|----------|----------------------------------------------------|
| `g_history`   | `(m, n)` | Window of recent gradients (residuals).            |
| `x_history`   | `(m, n)` | Window of recent iterates.                         |
| `step_count`  | scalar   | Number of valid columns currently stored.          |

## Algorithm Details

### Initialization (`init`)

Given initial `params` of dimension `n`, the state is initialized with
zero-filled `g_history` and `x_history` windows of shape `(window, n)` and
a `step_count` of `0`.

### Direction Computation (`direction`)

1. Build the difference matrices `ΔG` and `ΔX`. The first column uses the
   difference between the current gradient/iterate and the oldest stored
   entry; the remaining columns are consecutive first-differences of the
   stored windows.
2. Form the Gram matrix `gram = ΔGᵀ ΔG`.
3. Compute a regularization scale from the mean trace of the Gram matrix
   (falling back to `1.0` if the trace is non-positive) and build the
   regularized system `A = gram + reg · scale · I`.
4. Compute the right-hand side `b = ΔGᵀ ∇f`.
5. Apply an **active mask** based on `step_count` to zero out contributions
   from unfilled window slots, keeping the system numerically well-posed by
   substituting identity rows/columns for inactive entries and adding a
   small `1e-12` diagonal jitter.
6. Solve `A θ = b`, zeroing inactive components of `θ`.
7. Form the residual `∇f − ΔG θ` and the direction
   `d = −β · residual − ΔX θ`.
8. **Safeguard**: If `d` is non-finite or no history has been accumulated
   (`step_count == 0`), fall back to steepest descent `d = −∇f`.

### State Update (`update`)

The update supports two modes depending on `publish(info)`:

- **Single-point update** (`points is None`): Roll the windows and insert
  the newest `params`/`grad`, incrementing `step_count` up to `window`.
- **Batched update**: Iterate over the published sequence of points using
  `jax.lax.scan`. Each element carries a `valid` flag; only valid points
  roll into the windows and increment the count. This allows history to be
  reconstructed from an externally published point history.

## Numerical Safeguards

- **Trace-scaled regularization** adapts the Tikhonov term to the problem's
  scale, avoiding over- or under-regularization.
- **Active masking** ensures that unfilled window slots do not corrupt the
  least-squares solve during the warmup phase.
- **Diagonal jitter** (`1e-12`) guarantees the system remains invertible.
- **Finite-direction fallback** protects against NaNs/Infs and the cold
  start, defaulting to gradient descent.

## Discussion

Anderson acceleration is closely related to multi-secant quasi-Newton
updates. The Type-II variant minimizes the residual norm directly, which
connects it to the Good Broyden / L-BFGS family. By exposing the full
least-squares solve, this oracle serves as a reference "ideal" — it uses the
same information (gradient and iterate differences) that L-BFGS compresses
into its two-loop recursion, but performs the exact `(m × m)` solve rather
than the memory-limited approximation.

The tunable coupling constant `β` provides a knob to trade convergence
metrics: `β = 1` yields the theoretically pure update, while `β > 1` can
accelerate iteration-count convergence at the potential cost of stability
along the optimization trajectory. This makes the oracle well-suited for
benchmarking and studying the gap between idealized multi-secant methods and
practical limited-memory approximations.