# Brief — Outline

## Goal
A short, results-forward demonstration that one simple change to L-BFGS —
a permissive line search along a quadratic path between the gradient and
the quasi-Newton direction — beats both existing methods and every more
elaborate QQN variant. Deliberately minimal; defers all generalization to
the Theory part.

## 1. The Direction Dilemma
- First-order methods (GD, momentum): cheap, robust, but slow on
  ill-conditioned valleys.
- Quasi-Newton (L-BFGS): fast (superlinear) but fragile — `-H∇f` only
  descends when `H` is positive-definite.
- The classical compromise: pick one direction, then line-search along it.
- Its flaw: committing to a single direction wastes evaluations when the
  oracle direction is poor.

## 2. The Minimal Enhancement
- Refuse the binary choice; blend instead.
- The quadratic path:
  `d(t) = t(1-t)(-∇f) + t²(-H∇f)`, `t ∈ [0, 1]`.
- Three properties: `d(0)=0`, `d'(0)=-∇f`, `d(1)=-H∇f`.
- A permissive line search walks `t` directly — no direction re-scaling,
  the curve *is* the search space.

## 3. Why It Works (one paragraph)
- Gradient tangent at the origin → guaranteed descent → global
  convergence.
- Oracle endpoint at `t=1` → superlinear behavior near the optimum.
- The blend is discovered automatically; no manual tuning.

## 4. Headline Result
- On a 4-layer MLP (Fashion-MNIST, 335k params), the simple variant beats
  standalone L-BFGS (2.64× iteration speedup, cheaper per iteration),
  SGD, and Adam.
- It also beats the elaborate QQN variants (splines, regions, exotic
  oracles): simplest configuration is the strongest.
- Speedup widens as the loss target tightens.

## 5. Forward Pointer
- The rest of the paper generalizes and justifies this single move:
  the four-axis factoring (Theory), the fairness harness (Methodology),
  and the full empirical story (Results).