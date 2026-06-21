# QQN (Quasi-Quadratic-Newton) Algorithm Technical Documentation

## Overview

The QQN (Quasi-Quadratic-Newton) algorithm is a novel optimization method that combines the robustness of steepest descent with the efficiency of L-BFGS through a unique quadratic interpolation scheme. This implementation provides a sophisticated approach to unconstrained optimization that adaptively blends gradient descent and quasi-Newton directions.

## Conceptual Role: A Combiner for Gradient + Oracle + Search

At its core, QQN is best understood as a **combiner** that unifies three fundamental components of numerical optimization:

1. **Gradient** (steepest descent): The raw signal from `-∇f(x)`, providing a reliable, locally valid descent direction.
2. **Oracle** (L-BFGS quasi-Newton): A learned approximation of curvature (`-H∇f(x)`), acting as a black-box oracle that encodes second-order information from historical gradient differences.
3. **Search** (line search strategy): The mechanism that traverses the quadratic path `d(t)` and selects a step that guarantees sufficient descent.

These three components are not merely combined additively — the quadratic path construction means the **search strategy is the glue** that makes the gradient and oracle work together coherently.
Without a robust line search, the interpolation between directions has no principled stopping criterion and the algorithm loses its convergence guarantees entirely.

## Algorithm Description

### Core Concept

QQN operates by constructing a quadratic path between two search directions:

1. **Steepest descent direction**: `-∇f(x)` (negative gradient)
2. **L-BFGS direction**: `-H∇f(x)` (quasi-Newton direction with approximate inverse Hessian H)

The algorithm searches along a parametric curve defined by:

```
d(t) = t(1-t)(-∇f) + t²(-H∇f)
```

where `t ∈ [0, 1]` is the interpolation parameter.

### Key Properties

- **t = 0**: Pure steepest descent direction
- **t = 1**: Pure L-BFGS direction

This formulation ensures:

- The direction is always a descent direction (for small enough steps)
- Smooth transition between conservative (gradient) and aggressive (quasi-Newton) steps
- Adaptive behavior based on problem characteristics

### The Line Search Strategy: The Critical Component

**The line search is not an implementation detail — it is a first-class algorithmic component** and the mechanism by which QQN's theoretical properties are realized in practice.

The line search operates over the quadratic path `d(t)` and must:

- **Select `t`**: Determine the interpolation parameter, effectively choosing how much to trust the gradient vs. the oracle at each iteration.
- **Select step size `α`**: Scale the chosen direction `d(t)` to satisfy sufficient decrease conditions (e.g., Armijo/Wolfe conditions).
- **Enforce descent**: Guarantee that `f(x + α·d(t)) < f(x)`, which is the foundation of global convergence.
- **Exploit curvature**: A strong Wolfe condition on the line search ensures the curvature information fed back into the L-BFGS oracle remains accurate and well-conditioned.

The interplay between `t` and `α` is subtle: a poor line search can cause the algorithm to degenerate into neither effective gradient descent nor effective quasi-Newton steps, losing the benefits of both. A well-implemented line search, by contrast, allows QQN to **automatically discover the right blend** of gradient and oracle directions without manual tuning.

> **Key insight**: The quadratic path `d(t)` defines a one-dimensional search space over direction blends. The line search is what navigates this space. The quality of the overall optimization is therefore directly bounded by the quality of the line search.

## Advantages

1. **Adaptive Behavior**: Automatically balances between conservative and aggressive steps
2. **Robustness**: Multiple fallback mechanisms ensure progress
3. **Efficiency**: L-BFGS acceleration when appropriate
4. **Smooth Transitions**: Quadratic interpolation avoids abrupt direction changes
5. **Modular Design**: The gradient, oracle (L-BFGS), and search components are conceptually separable, making the algorithm extensible — alternative oracles or search strategies can be substituted independently.

## Limitations

1. **Memory Requirements**: Stores L-BFGS history (O(m×n) where m is history size, n is parameter dimension)
2. **Computational Overhead**: Quadratic path evaluation adds complexity
3. **Parameter Tuning**: Performance sensitive to configuration settings
4. **Line Search Sensitivity**: The algorithm's effectiveness is highly sensitive to the line search implementation. An inexact or poorly tuned line search undermines both convergence speed and the quality of L-BFGS curvature updates.

## Theoretical Guarantees

Under standard assumptions (smooth, bounded gradients):

- **Global Convergence**: Guaranteed due to steepest descent fallback
- **Superlinear Convergence**: Near optimum when L-BFGS direction dominates
- **Descent Property**: Every step decreases function value (enforced)

> **Note on guarantees**: All three theoretical guarantees above are contingent on the line search satisfying sufficient decrease conditions. The steepest descent fallback provides global convergence only if the line search can always find a valid step along the gradient direction. The descent property is enforced *by* the line search, not independently of it.

## References

The QQN algorithm combines ideas from:

- L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
- Trust region methods (quadratic models)
- Adaptive step size selection
- Gradient descent with momentum

- Wolfe condition line searches (critical for curvature update validity)
- Backtracking line search with Armijo conditions (fallback robustness)