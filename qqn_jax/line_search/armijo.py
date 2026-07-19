from typing import Callable

from qqn_jax import backtracking_search
from qqn_jax.line_search.strategy import LineSearchResult


def armijo_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-2,
    shrink: float = 0.5,
    max_iter: int = 30,
    temperature: float = 0.0,
    cooling: float = 0.95,
    seed: int = 0,
    region=None,
    region_state=None,
    max_probes: int = 32,
    record_probes: bool = True,
    max_step: float = 1.0,
) -> LineSearchResult:
    """Alias for :func:`backtracking_search`.
    Provided so users can refer to the Armijo backtracking search by its
    classical name as well.
    """
    return backtracking_search(
        value_and_grad_fn,
        params,
        direction,
        value,
        grad,
        *args,
        init_step=init_step,
        c1=c1,
        shrink=shrink,
        max_iter=max_iter,
        temperature=temperature,
        cooling=cooling,
        seed=seed,
        region=region,
        region_state=region_state,
        max_probes=max_probes,
        record_probes=record_probes,
        max_step=max_step,
    )
