from qqn_jax.line_search.backtracking import backtracking_search
from qqn_jax.line_search.armijo_wolfe import armijo_wolfe_search
from qqn_jax.line_search.bisection import bisection_search
from qqn_jax.line_search.fixed_step import fixed_step_search
from qqn_jax.line_search.hager_zhang import hager_zhang_search
from qqn_jax.line_search.null_search import null_search
from qqn_jax.line_search.strong_wolfe import strong_wolfe_search


LINE_SEARCHES = {
    "strong_wolfe": strong_wolfe_search,
    "backtracking": backtracking_search,
    "armijo_wolfe": armijo_wolfe_search,
    "hager_zhang": hager_zhang_search,
    "fixed": fixed_step_search,
    "null": null_search,
    "bisection": bisection_search,
}

__all__ = [
    "strong_wolfe_search",
    "backtracking_search",
    "armijo_wolfe_search",
    "hager_zhang_search",
    "fixed_step_search",
    "null_search",
    "bisection_search",
    "LINE_SEARCHES",
]
