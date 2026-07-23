# Debugging JAX Code in a Python Debugger

JAX's tracing/JIT model makes standard `pdb`/`breakpoint()` workflows behave
differently than in normal Python code. This guide covers practical
techniques for debugging JAX code (including the patterns used in this
codebase, e.g. `qqn_jax/line_search/util.py`) with a Python debugger.

## Why JAX debugging is different

Under `jax.jit`, `jax.vmap`, `jax.grad`, and `lax.scan`/`lax.cond`/`lax.while_loop`,
your Python function is **traced once** with abstract tracer values, not
concrete arrays. Consequences:

- `print(x)` inside jitted code prints a `Tracer` repr, not real values.
- `if x > 0:` on a traced value raises `ConcreteizationTypeError` — this is
  why JAX code uses `jnp.where`, `lax.cond`, `lax.select` instead of Python
  `if`/branching (see `_metropolis_accept`, `_record_probe` above, which are
  written this way specifically to remain JIT/vmap-safe).
- A `pdb` breakpoint set inside a jitted function will only ever show you
  the *tracing* pass — usually once, with abstract shapes/dtypes instead of
  real numbers — not each actual invocation.
- Stepping with `n`/`s` in `pdb` inside jitted code often does nothing
  useful because the underlying execution is compiled XLA, not the Python
  bytecode you're stepping through.

The fix is almost always: **disable JIT (and other transforms) while
debugging**, then re-enable once you understand the bug.

## Quick recipe: disable JIT

The single most useful trick is `jax.disable_jit()`:

```python
import jax

with jax.disable_jit():
    result = my_jitted_fn(x, y)
```

With JIT disabled, every operation runs eagerly in Python, so:

- `print(x)` shows real array values.
- Standard `pdb`/`breakpoint()` calls work exactly like normal Python.
- Every loop iteration of `lax.scan`/`lax.while_loop`/`lax.cond` actually
  executes as Python control flow, so you can single-step through it.

You can also set this globally for a whole debugging session:

```bash
JAX_DISABLE_JIT=1 python -m pdb my_script.py
```

or in code:

```python
jax.config.update("jax_disable_jit", True)
```

Remember to remove/revert this before benchmarking or shipping — it disables
all compilation and will be much slower.

## Dropping into `pdb`/`breakpoint()`

### Outside of JIT (eager code, tests, notebooks)

Just use `breakpoint()` as normal:

```python
def _metropolis_accept(delta_e, temp, key, dtype):
    breakpoint()  # works fine when this function is called un-jitted
    ...
```

### Inside JIT-compiled code

Plain `breakpoint()` will not work as expected under `jit`. Use
`jax.debug.breakpoint()` instead — it inserts a debugger callback into the
compiled program and works under `jit`, `vmap`, and `pmap`:

```python
import jax

@jax.jit
def f(x):
    y = x * 2
    jax.debug.breakpoint()
    return y + 1
```

When execution hits this point, you get an interactive `pdb`-like prompt
(`jax.debug.breakpoint` REPL) where you can inspect the *concrete* runtime
values of local variables, even though the function is compiled. Type `c`
to continue, or use the debugger commands it documents on entry.

Notes:

- Under `vmap`, the debugger will show batched values; you can inspect a
  specific batch element with the usual indexing.
- `jax.debug.breakpoint()` respects `jax.disable_jit()` as well — combining
  the two lets you drop into a debugger and step through host-side Python
  around it.

## Printing values inside JIT: `jax.debug.print`

For quick inspection without a full debugger session, prefer
`jax.debug.print` over Python's `print` inside jitted/traced code — it
defers printing until the value is actually computed at runtime:

```python
import jax

@jax.jit
def f(x):
    jax.debug.print("x = {x}", x=x)
    return x + 1
```

This is often faster to iterate with than a full breakpoint when you just
need to confirm shapes/values at a point in a `scan` body or `cond` branch.

## Debugging functions with tracer-unsafe control flow

Code in this repo is deliberately written to avoid Python-level
branching on traced values (see `_record_probe`, which uses
`jnp.where`/`.at[idx].set(...)` instead of `if`). When debugging such
functions:

1. First reproduce the issue **without** `jit`/`vmap` by calling the
   function directly on plain `jnp` arrays. Since none of these helpers
   require `jit` to run, you can call them eagerly and use normal
   `breakpoint()`/`pdb` to step through line by line.

   ```python
   from qqn_jax.line_search.util import _record_probe, _empty_probes
   import jax.numpy as jnp

   probe_params, probe_grads, probe_valid, probe_values, probe_alphas = \
       _empty_probes(jnp.zeros(4), max_probes=8)

   breakpoint()
   out = _record_probe(
       probe_params, probe_grads, probe_valid, probe_values, probe_alphas,
       slot=jnp.array(2), p=jnp.ones(4), g=jnp.ones(4),
       v=jnp.array(1.0), a=jnp.array(0.5), max_probes=8,
   )
   ```

2. If the bug only appears once wrapped in `jit`/`scan`, re-enable JIT but
   wrap the call in `jax.disable_jit()` first to confirm the eager
   semantics are correct, then narrow down what changes under compilation
   (usually: shape/dtype mismatches, or reliance on Python control flow
   that silently does the wrong thing when traced — e.g. an `if` on a
   traced boolean that got "constant-folded" using tracer-time shape
   info rather than runtime values).

3. Use `jax.make_jaxpr(fn)(*args)` to inspect the traced program structure
   without running a debugger at all — useful for confirming whether a
   branch is actually present (`lax.cond`) vs. was resolved away at trace
   time (Python `if`).

## Debugging NaNs / Infs

JAX has a dedicated NaN-checker that's much more useful than manual
breakpoints for tracking down where a `NaN` first appears:

```bash
JAX_DEBUG_NANS=True python my_script.py
```

or

```python
jax.config.update("jax_debug_nans", True)
```

This raises immediately (with a traceback pointing at the offending op)
the first time a `NaN` is produced, rather than letting it silently
propagate through the rest of the computation (e.g. through
`_metropolis_accept`'s `jnp.exp(-delta_e / safe_t)`, where a bad `temp`
or huge `delta_e` could otherwise produce silent `inf`/`nan` results).
Note this option forces synchronous dispatch and disables some
optimizations, so use it only while debugging.

## Debugging `jax.random` key issues

A common source of confusing bugs is reusing a PRNG key or forgetting to
thread the split key forward (as done correctly in `_metropolis_accept`,
which returns the new `key` for the caller to use next). If results look
suspiciously correlated across calls or across `vmap` batches:

- Add a `jax.debug.print` for `key` at entry/exit to confirm it changes.
- Check for accidental key reuse by grepping for `jax.random.split` calls
  that discard one half of the returned tuple.

## Debugging under `vmap`

`jax.debug.print` and `jax.debug.breakpoint()` both work under `vmap`,
but printed/inspected values will be batched arrays. To inspect a single
batch element, temporarily replace `vmap` with a Python loop over the
batch dimension calling the un-vmapped function, and debug that directly
with plain `pdb`:

```python
for i in range(batch_size):
    breakpoint()
    out_i = fn(x[i], y[i])
```

This sidesteps `vmap`'s tracing entirely and is often the fastest way to
isolate a per-example bug.

## Recommended workflow summary

1. Reproduce the failure **outside** `jit`/`vmap` if at all possible — use
   plain `breakpoint()`/`pdb`.
2. If it only reproduces under `jit`, wrap the call in
   `jax.disable_jit()` and try again.
3. If it only reproduces *with* JIT enabled, switch to
   `jax.debug.breakpoint()` / `jax.debug.print` inserted at the suspect
   location, and/or enable `JAX_DEBUG_NANS`.
4. Use `jax.make_jaxpr` to confirm the compiled program structure matches
   your expectations (branches present, shapes as expected).
5. Remove all debug flags/prints before committing.