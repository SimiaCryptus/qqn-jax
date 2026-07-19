from typing import NamedTuple, Any

from jax import numpy as jnp


class LineSearchResult(NamedTuple):
    """Result of a line search.

    Attributes:
        step_size: chosen step size ``α``.
        new_value: function value at ``params + α·d``.
        new_grad: gradient at ``params + α·d``.
        new_params: the updated parameters.
        done: whether the search satisfied its conditions.
         probe_params: fixed-size ``(max_probes, n)`` buffer of evaluated
             points along the path (for feeding oracle curvature memory).
         probe_grads: fixed-size ``(max_probes, n)`` buffer of probe gradients.
         probe_valid: fixed-size ``(max_probes,)`` boolean mask of filled slots.
    """

    step_size: jnp.ndarray
    new_value: jnp.ndarray
    new_grad: jnp.ndarray
    new_params: jnp.ndarray
    done: jnp.ndarray
    probe_params: Any = None
    probe_grads: Any = None
    probe_valid: Any = None

    probe_values: Any = None

    probe_alphas: Any = None

    num_evals: Any = None
