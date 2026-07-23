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


