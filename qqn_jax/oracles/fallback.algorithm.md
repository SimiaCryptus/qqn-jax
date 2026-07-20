# Fallback Oracle

## Overview

The `Fallback` oracle composes a sequence of search-direction *oracles*
into a single oracle that prefers the direction produced by the
**earliest** oracle in the list whose direction is considered *valid*.
If none of the oracles produce a valid direction, the oracle falls back
to the negative gradient (steepest descent), which is always a valid
descent direction for a smooth objective.

The composition is written so that all decision logic is expressed with
`jnp.where` (functionally, `lax.select`) rather than Python `if`
statements on traced values. This keeps the function fully compatible
with JAX transformations such as `jit`, `grad`, and `vmap`.

## Interface

```python
def Fallback(oracles: Sequence[Oracle]) -> Oracle
```

- **Input**: an ordered sequence of `Oracle` objects.
- **Output**: a single `Oracle` with the standard
  `init` / `direction` / `update` triple.

An `Oracle` is a named triple of pure functions:

- `init(params) -> state`
- `direction(params, grad, state) -> (direction, new_state)`
- `update(state, info) -> new_state`

## Algorithm

### `init`

The combined state is the tuple of the sub-oracles' initial states:

```
state = (o0.init(params), o1.init(params), ...)
```

### `direction`

For each oracle `o_i` with its state `s_i`, in order:

1. Compute the candidate direction and next state:
   `d_i, ns_i = o_i.direction(params, grad, s_i)`.
2. Evaluate the candidate's **validity** using three tests:
   - **finite**: every component of `d_i` is finite
     (`jnp.all(jnp.isfinite(d))`), rejecting `NaN`/`inf`.
   - **nonzero**: the direction has positive squared norm
     (`⟨d, d⟩ > 0`), rejecting the zero vector.
   - **descent**: the direction is a descent direction,
     i.e. `⟨grad, d⟩ < 0`.
   - `valid = finite & nonzero & descent`.
3. Fold the candidate into the running selection:
   - For the first oracle, the running choice is simply `d_0` with
     validity `valid_0`.
   - For subsequent oracles, keep the previously chosen direction when
     it was already valid, otherwise take the current candidate:
     ```
     chosen        = where(chosen_valid, chosen, d_i)
     chosen_valid  = chosen_valid | valid_i
     ```
   This implements a "first valid wins" priority: once a valid
   direction has been selected, later candidates cannot override it.

After iterating all oracles, if no candidate was valid the result is
replaced by the negative gradient:

```
chosen = where(chosen_valid, chosen, -grad)
```

The function returns `(chosen, new_states)`.

### `update`

Propagates `update` to every sub-oracle, threading the shared `info`:

```
new_state = (o_i.update(s_i, info) for each i)
```

## Design Notes / Discussion

- **Branch-free selection.** All choices are made with `jnp.where`, so
  every oracle's `direction` is *always* executed and its cost is always
  paid. This is the standard trade-off for making control flow traceable
  under JAX: correctness and composability at the price of doing some
  redundant work.

- **Priority semantics.** The fold gives strict ordering priority:
  oracle 0 wins if it is valid, else oracle 1, and so on. Place the most
  preferred (e.g. quasi-Newton) oracle first and cheaper/safer oracles
  later.

- **Guaranteed descent.** Because `-grad` is used as the ultimate
  fallback and it always satisfies `⟨grad, -grad⟩ = -‖grad‖² ≤ 0`, the
  returned direction is a descent direction whenever the gradient is
  non-zero. At a stationary point (`grad = 0`) all candidate directions
  collapse to zero and the fallback returns the zero vector, which is
  the correct "no move" behaviour.

- **L-BFGS interaction.** An L-BFGS oracle with an empty history returns
  `-H∇f = -∇f`, which passes all three validity tests and is therefore
  selected normally. A degenerate update that produces `NaN`/`inf`
  triggers the fallback chain, protecting the optimizer from divergence.

## Known Issue

The `new_states` list is declared but never appended to; the per-oracle
`next_state` values (`ns`) computed inside the loop are discarded, so
`direction` currently returns an **empty tuple** as its new state. Any
oracle that relies on state being threaded through `direction` (rather
than through `update`) will not behave correctly. A fix would append
each `ns` to `new_states`:

```python
d, ns = o.direction(params, grad, s)
new_states.append(ns)
```