Results discussion notes

Adam has fastest start - JIT compilation time? L-BFGS and QQN both show an extra 2.6s on the first iteration (need to implement time-from-first-iteration)
LBFGS implementation only provides iterations - we don't have the eval count (need to implement an adaptive estimation workaround)
Anomolous "Fix" strategy performance (open question for study)


need to show:
Reference optimizers vs best-of-breed
Adam comparisons - Temperature and oracle
Line search comparison
Sine & Rolling-Sine case study - L-BFGS shows superios performance, compare and show PSD-region effect 



















Professional Intro Sequence: (drafted)

* Basic Optimizer Theory - Assume known, but may not see the component categories yet; provide orientation
* QQN Innovations
  * Quadratic Path
  * Strategy Pattern
  * Geometrically Principled
  * Provable Convergence
* Cross-Cutting Enhancements
  * Temperature = Allow controlled exploration
  * Projective Regions - Constrain the search
* Core Enhancements
  * Spline - Reuse all measured gradients
  * Partitioned Oracles - We can partition the wieghts, e.g. by layer, and consider each oracle a partitioned regime to make some strategies faster
  * Recursive Descent - If we instead paralellelize the QQN pathbuilding, we remap each step to a low-dimensional regime which can be solved with an inner multidimensional continuous optimization strategy, e.g. basic l-bfgs-based QQN


