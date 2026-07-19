from typing import NamedTuple, Callable, Any, Tuple


class Oracle(NamedTuple):
    """Pure, swappable oracle interface.

    Attributes:
        init: ``params -> oracle_state`` (use ``()`` when stateless).
        direction: ``(params, grad, state) -> (direction, new_state)``.
            ``direction`` is the ``t = 1`` endpoint ``-H∇f``.
        update: ``(state, info) -> state`` (no-op for stateless oracles).
    """

    init: Callable[[Any], Any]
    direction: Callable[[Any, Any, Any], Tuple[Any, Any]]
    update: Callable[[Any, Any], Any]


class OracleInfo(NamedTuple):
    """Information passed to ``Oracle.update`` after a step is accepted.

    Attributes:
        params: iterate ``x`` before the step.
        new_params: accepted iterate ``x_new``.
        grad: gradient ``∇f(x)`` before the step.
        new_grad: gradient ``∇f(x_new)`` after the step.
        t: chosen interpolation parameter.
        step_size: accepted step size ``α``.
         probe_params: optional ``(k, n)`` buffer of line-search probe points.
         probe_grads: optional ``(k, n)`` buffer of probe gradients.
         probe_valid: optional ``(k,)`` boolean mask of filled probe slots.
    """

    params: Any = None
    new_params: Any = None
    grad: Any = None
    new_grad: Any = None
    t: Any = None
    step_size: Any = None
    probe_params: Any = None
    probe_grads: Any = None
    probe_valid: Any = None
    probe_alphas: Any = None
