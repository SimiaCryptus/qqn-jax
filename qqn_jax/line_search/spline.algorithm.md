# Cubic Hermite Spline Line Search

    ## Overview

    `spline_search` is a one-dimensional line search that models the scalar
    reduction

    ```
    φ(t)  = f(x + d(t))
    φ'(t) = ⟨∇f(x + d(t)), d'(t)⟩
    ```

    as a **piecewise cubic Hermite spline** and solves that model for a *true
    minimum*. It is a close relative of the cubic Hermite **spline path**
    (`qqn_jax/paths/spline.py`), but its objective is different: instead of
    reusing probes to enrich a path model, it accumulates control points to
    locate an actual stationary minimizer of `φ(t)` along the pre-baked path
    exposed by `eval_at`.

    Like every line search here it is **path-agnostic** — all geometry lives
    in the `eval_at` callback, which returns `(params, value, grad, slope)`
    where `slope` is the directional derivative `φ'(t)`.

    ## Control Points

    Each evaluation yields a control point `(t_i, φ_i, φ'_i)`. Because the
    measured slope is the genuine directional derivative, the Hermite
    interpolant is exact to cubic order and **no gradient-orientation
    reflection heuristic is applied** (in contrast to the spline *path*, which
    stores absolute slopes and reflects tangents by secant sign).

    The model is seeded with two control points:

    * the origin `(0, φ(0), φ'(0))` — no evaluation needed, and
    * an initial probe at `init_step`.

    It then accumulates one control point per refinement round.

    ## Cubic Hermite Segment

    On a segment between `(t0, f0, m0)` and `(t1, f1, m1)` with `h = t1 - t0`
    and `s = (t - t0)/h`:

    ```
    f(s) = h00(s)·f0 + h10(s)·h·m0 + h01(s)·f1 + h11(s)·h·m1
    ```

    Differentiating gives `f'(s) = A·s² + B·s + C` with

    ```
    A =  6·f0 + 3·h·m0 - 6·f1 + 3·h·m1
    B = -6·f0 - 4·h·m0 + 6·f1 - 2·h·m1
    C =         h·m0
    ```

    `_segment_candidates` solves this in closed form and retains a root only
    when it is

    1. **real** (`disc = B² - 4AC ≥ 0`, with a linear `-C/B` fallback when
       `|A|` is tiny),
    2. **in range** (`s ∈ [0, 1]`), and
    3. a **genuine minimum** — the second-derivative test `f''(s) = 2A·s + B >
       0`.

    This last filter is what makes the search hunt for minima rather than
    arbitrary stationary points.

    ## Proposal

    `_propose` sorts the (padded) control-point buffer by `t`, forms
    consecutive segments whose endpoints are both valid and non-degenerate,
    `vmap`s `_segment_candidates` over them, and returns the minimizing
    candidate with the **lowest predicted value** across all segments (plus a
    `found` flag).

    ## Refinement Loop

    Implemented with `jax.lax.scan` over a fixed `max_iter` rounds; each round
    costs exactly one evaluation:

    1. **Propose** the lowest-value minimizing stationary point.
    2. **Fallback**: if the model yields no interior minimum
       (`found = False`), bisect between the current lowest-value control
       point and the opposite span endpoint, guaranteeing progress.
    3. **Clamp** the candidate to `[0, max_step]`.
    4. **Measure** it and record a probe.
    5. **Append** the measurement as a new control point.
    6. **Track best**, updating on strict improvement (and optionally via a
       cooling Metropolis acceptance when `temperature > 0`).

    Because each measurement is appended, every subsequent proposal is drawn
    from a strictly richer spline that tightens around the minimizer.

    ## Acceptance

    The returned step is `done` when any of the following hold:

    * **Armijo**: `best_v ≤ φ(0) + c1·best_a·φ'(0)`,
    * **Improvement**: `best_v < φ(0)`,
    * **Stochastic**: a Metropolis acceptance fired during refinement.

    If nothing improved, `step_size` collapses to `0` so the caller never
    moves to a worse point.

    ## JAX-Friendliness

    Fixed-capacity control-point buffers (`2 + max_iter` slots), branch-free
    masking (`jnp.where`), and a static-length `lax.scan` keep the whole search
    `jit`- and `vmap`-compatible. The number of evaluations is exactly
    `1 + max_iter`.

    ## Public API

    | Symbol                | Purpose                                        |
    |-----------------------|------------------------------------------------|
    | `spline_search`       | the line-search entry point                    |
    | `hermite_basis`       | cubic Hermite basis at `s`                     |
    | `segment_eval`        | interpolated fitness of one segment            |
    | `segment_candidates`  | closed-form *minimizing* stationary points     |