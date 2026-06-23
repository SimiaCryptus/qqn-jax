---
specifies: genesis.md
related:
  - algorithm.md
  - ../README.md
  - https://blog.simiacrypt.us/posts/symmetric_textures/
  - https://blog.simiacrypt.us/posts/optimization_research/
  - https://raw.githubusercontent.com/SimiaCryptus/qqn-optimizer/refs/heads/master/README.md
  - https://raw.githubusercontent.com/SimiaCryptus/MindsEye/refs/heads/master/src/main/java/com/simiacryptus/mindseye/opt/orient/QQN.java
  - https://math.cognotik.com/companion/geometric-entropy/index.html
  - https://raw.githubusercontent.com/SimiaCryptus/Fun_With_Math/refs/heads/main/js/optimizer-qqn.js
---

# The Genesis of QQN

This document traces the origin and evolution of the **Quadratic Quasi-Newton
(QQN)** algorithm — from a hands-on optimization experiment inside a homegrown
deep-learning framework, through a formal paper and reference implementations,
to the present JAX/Optax library. It is a narrative companion to the technical
reference in [`algorithm.md`](theory/algorithm.md) and the project overview in
[`../README.md`](../README.md).

## 1. A Research Project, and a Framework Built to Learn (~2016)

QQN did not begin as an algorithm — it began as a *question about how
optimizers actually work*. Around 2016 I was deep into neural-network image
research, and rather than treat optimization as a black box, I wanted to learn
it the way I learn most things: by building it myself and experimenting.

So I wrote a complete deep-learning framework in Java — **MindsEye**. It was an
open-source hobby project that grew over roughly five years, and it became the
laboratory in which every idea here was first tried. Building the framework
from the differentiable layers up meant I controlled the entire stack: the
forward/backward passes, the training loop, and — crucially — the *orientation
and line-search machinery* that decides where each step goes.

This "learn by implementing" stance is the through-line of QQN's whole history.
Every subsequent implementation — Rust, TensorFlow.js, and now JAX/Optax — is
another pass at understanding the same core idea in a new setting.

## 2. Encountering L-BFGS — and Being Puzzled by Backtracking

While implementing quasi-Newton methods in MindsEye, I worked through **L-BFGS**
(Limited-memory Broyden–Fletcher–Goldfarb–Shanno). L-BFGS does something
appealing: instead of merely predicting a *direction* (like steepest descent),
it predicts a full *optimum point* — both a direction and a distance — by
accumulating curvature information from the history of gradient differences.

But one part bothered me: the **backtracking line search**. L-BFGS proposes a
confident step to a predicted minimum, and then, when that step misbehaves (as
it often does in regions of high nonlinearity), the standard remedy is to
simply *shrink* the step — backtrack along the same straight ray under the
Armijo/Wolfe conditions until a tolerable point is found.

That felt unsatisfying. A few observations crystallized the discomfort:

* **2nd-order estimates are fragile.** Curvature estimates assume a smooth,
  deterministic function. Re-sampling data invalidates the gradient-ray
  history, and numeric resolution issues can corrupt the approximation.
* **Bad steps are common.** In nonlinear regions the predicted step can
  *increase* the loss, which is exactly why the line search exists as a
  safety net.
* **Small steps waste the prediction.** Once you backtrack to a very small
  step size, you've spent real compute building a second-order model, only to
  end up moving in a direction you *know* is suboptimal — it sits at an angle
  to the true steepest-descent vector you already had in hand.

The straight-line backtrack throws away a structural fact: at small step sizes
we *know* the best local direction (the negative gradient), and at the full
step we have a *curvature-aware* prediction (the L-BFGS point). Why search a
straight line between an arbitrary origin and the L-BFGS point, when those two
endpoints encode genuinely different information?

## 3. Writing QQN Interpolation

The fix was to replace the straight-line search path with a **quadratic path**.
Instead of a line from the current point toward the L-BFGS prediction, I
defined a curve that:

1. **starts at the current point** (`d(0) = 0`),
2. **departs along steepest descent** (`d'(0) = -∇f`), and
3. **arrives at the L-BFGS prediction** (`d(1) = -H∇f`).

These three constraints yield the canonical QQN path:

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

The behavior is exactly what the puzzle demanded: for small `t` the path *is*
gradient descent (so a good small step always exists), and at `t = 1` it is
exactly the quasi-Newton optimum. The line search no longer backtracks along a
suboptimal ray — it **traverses a curve** that smoothly blends the reliable
first-order signal with the aggressive second-order prediction. A step size of
`1` recovers the pure L-BFGS point; a tiny step recovers pure gradient descent;
everything in between is a principled, curvature-aware compromise discovered by
the search itself.

The original MindsEye implementation
(`com.simiacryptus.mindseye.opt.orient.QQN`) captured this directly: it asked
L-BFGS for its direction, formed the negative gradient as the steepest-descent
direction, scaled the gradient to match the L-BFGS magnitude, and returned a
`LineSearchCursor` whose `position(t)` evaluated
`scaledGradient·(t − t²) + lbfgs·t²` — the same quadratic blend. When the two
directions were already nearly aligned, it degraded gracefully back to the
plain L-BFGS cursor.

## 4. The Focal Problem: Symmetric Deep Texture Synthesis

QQN was not invented in the abstract — it was forged against a concrete, hard
optimization problem: **deep texture synthesis** in the style of DeepDream and
neural style transfer, *with added symmetry operators*.

The MindsEye image work optimized a canvas so that a pretrained vision network
"saw" a desired texture — but with extra **opticals** placed in front of the
network that enforced strict geometric symmetry: kaleidoscopic rotations,
color-permutation groups, tilings of the plane, spherical (polyhedral)
symmetry, and even hyperbolic (Poincaré-disk) tilings. Conceptually, the
optimizer was painting a canvas *as seen through a kaleidoscope*, and only
certain symmetry configurations would even converge.

These problems are highly nonlinear, multi-scale, and unforgiving — exactly the
regime where naive L-BFGS backtracking struggles and where the stability of the
QQN path paid off. The multiresolution rendering pipeline (solve at low
resolution, upscale, repeat) made robust, well-behaved steps especially
valuable, since each scale had to converge cleanly before handing off to the
next.

## 5. Formalization: A Paper and a Rust Reference Implementation

Having validated QQN as a practical tool, I later **formalized** it. This
produced:

* An **academic paper** —
  *"Quadratic-Quasi-Newton Optimization: Combining Gradient and Quasi-Newton
  Directions Through Quadratic Interpolation"* — giving the full mathematical
  derivation, the descent/convergence analysis, and a comprehensive empirical
  study.
* A **Rust reference implementation**, the
  [`qqn-optimizer`](https://github.com/SimiaCryptus/qqn-optimizer) crate, built
  around a rigorous benchmarking framework.

The Rust project framed QQN's core innovation cleanly: rather than *choosing*
between gradient and quasi-Newton directions or solving an expensive
subproblem, QQN constructs the smooth path `d(t) = t(1-t)(-∇f) + t²·d_LBFGS`
and performs univariate optimization along it. The accompanying benchmark suite
(62 problems across convex, non-convex, multimodal, and ML categories; many
optimizer variants; statistically rigorous comparisons) put numbers behind the
intuition — including the headline observations that QQN variants won a majority
of problems and that QQN's **guaranteed descent** holds *regardless of L-BFGS
direction quality*, precisely because the path begins tangent to `-∇f`.

This phase emphasized the **line search as a first-class algorithmic
component**: Backtracking, Strong Wolfe, Golden Section, Bisection, and
Moré–Thuente variants were all explored as ways to walk the quadratic path.

## 6. A TensorFlow.js Lab — and the Basin-of-Attraction Property

To make the algorithm tangible (and to probe a property I'd noticed during the
image work), I built a **TensorFlow.js** implementation
([`optimizer-qqn.js`](https://raw.githubusercontent.com/SimiaCryptus/Fun_With_Math/refs/heads/main/js/optimizer-qqn.js))
for an interactive **Geometric Entropy** lab. There, points on a manifold
(sphere, torus, cube, saddle, or arbitrary STL mesh) are arranged to extremize
the Shannon entropy of their pairwise-distance distribution — a continuous
analogue of the Erdős distinct-distance problem.

This browser version is a faithful, compact QQN: it maintains an L-BFGS history
via the two-loop recursion, builds the quadratic path
`step(t) = t(1-t)·d_sd + t²·d_lbfgs`, and selects `t ∈ [0, 1]` with a golden-
section search over the loss.

The entropy objective is deliberately **degenerate**: a vast, high-dimensional
set of point configurations all achieve the same `H ≈ ln N` optimum. Because
the optimum is so flat, *which* configuration an optimizer lands on is
determined by the **path it takes through configuration space** — an effect the
lab calls "optimizer fingerprinting." Adam, QQN, and L-BFGS each produce
visibly different extremal arrangements.

This made visible the very property that made QQN valuable in image synthesis:
**adherence to a local basin of attraction**. The quadratic path's
steepest-descent tangent keeps the optimizer faithful to the local structure it
starts in, curving toward curvature-aware steps without abandoning the basin —
smoother, more symmetric, curvature-aligned results, rather than the noisy,
isotropic packings of less path-coherent methods.

## 7. The Present: A JAX/Optax Implementation

The current project, **`qqn-jax`**, is the latest pass at the same idea — now in
a modern, composable, hardware-accelerated setting. It re-expresses QQN as a
pure-functional JAX/Optax optimizer:

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

where `t = 0` is pure gradient descent and `t = 1` is the pure oracle
(L-BFGS by default) direction, and a robust line search selects the
interpolation parameter `t` and step size `α` together.

What's new in this incarnation is **explicit modularity**. The lessons of the
earlier implementations are factored into four orthogonal, independently
swappable axes (see [`algorithm.md`](theory/algorithm.md)):

* **Gradient** — the steepest-descent signal anchoring global convergence.
* **Oracle** — the `t = 1` endpoint, swappable beyond L-BFGS (Momentum,
  Shampoo, and combinators like `Fallback`/`Blend`).
* **Search** — the line search that walks the path (Armijo/backtracking,
  Strong Wolfe, Hager–Zhang, fixed), optionally augmented by the
  information-reusing cubic-Hermite **spline** refinement.
* **Region** — optional projective constraints (box, orthant, trust-region,
  and combinators) that let the search navigate a *feasible* path.

Because it is built on JAX's functional model, the whole solver composes with
`jit`, `vmap`, `pmap`, and `grad` — the same algorithm I first hand-wrote in
Java, now differentiable end-to-end and vectorizable across batched starting
points.

## Lineage at a Glance

| Era           | Setting                          | Implementation                                                                                                         | What it added                                                              |
|---------------|----------------------------------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| ~2016         | Java DL framework (MindsEye)     | `QQN.java` orientation strategy                                                                                        | The original quadratic-path idea, born from the L-BFGS backtracking puzzle |
| (image work)  | Symmetric deep texture synthesis | MindsEye opticals + multiresolution solvers                                                                            | The hard focal problem that validated QQN's stability                      |
| Formalization | Paper + Rust                     | [`qqn-optimizer`](https://github.com/SimiaCryptus/qqn-optimizer)                                                       | Rigorous derivation, convergence analysis, benchmark suite                 |
| Interactive   | TensorFlow.js                    | [`optimizer-qqn.js`](https://raw.githubusercontent.com/SimiaCryptus/Fun_With_Math/refs/heads/main/js/optimizer-qqn.js) | Demonstrated basin-of-attraction adherence ("optimizer fingerprinting")    |
| Present       | JAX/Optax                        | `qqn-jax`                                                                                                              | Modular oracle/search/region axes; full JAX transform composability        |

## See Also

* [`algorithm.md`](theory/algorithm.md) — the complete technical reference for the
  QQN path, line search, oracles, and regions.
* [`../README.md`](../README.md) — installation, quick start, and configuration.
* [Optimization research blog post](https://blog.simiacrypt.us/posts/optimization_research/)
  — the original introduction of QQN (and Recursive Subspace Optimization).
* [Symmetric textures blog post](https://blog.simiacrypt.us/posts/symmetric_textures/)
  — the focal image-synthesis problem QQN was forged against.
* [Geometric Entropy lab](https://math.cognotik.com/companion/geometric-entropy/index.html)
  — the interactive demonstration of the basin-of-attraction property.