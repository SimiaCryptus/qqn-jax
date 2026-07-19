Outline for new QQN paper with a focus on being a canonical design document for pluggable optimizers

* Introduction
  * Goals:
    * Discuss optimization methods, explain how they relate
    * Introduce a new framework for pluggable optimizers
  * New components:
    * Quadratic and spline path methods for continuous optimization
    * Strategy framework that unifies various optimization methods

* Objective Gradient Function - Needed input
* Oracle Strategy
  * Strategies
    * Gradient - uses gradient to find next point, configured rate. Simple, needed fallback for initial iterations
    * Momentum - Integrates past gradient evaluations to smooth out the path and avoid oscillations
    * Adam - Adaptive moment estimation, combines momentum and RMSProp ideas
    * L-BFGS - Quasi-newton estimate given history of gradient point measurements
  * Foreach Strategy:
    * Explain method, pros/cons
    * Discuss hyperparameters
  * Orthogonal concern: Provide line search points to the oracle integrator? This is optional
* Path Strategy
  * Linear - discards gradient
  * Quadratic - simplest that follows gradient
  * Spline - extends quadratic and utilizes all gradient evaluations
* Line Search Strategy
  * Temperature - Orthogonal to the inner strategy, used for early acceptance and exploration
  * Strategies
    * Backtracking
    * Armijo-Wolfe
  * Foreach Strategy:
      * Explain method, pros/cons
      * Discuss hyperparameters
* Regions
  * Orthant
  * Trust Region
  * No Decrease
  * Value Bounds
* Test method
  * Problems
    * Analytical: Rosenbrock, Rastrigin, Ackley, etc.
    * MNIST, FashionMNIST, etc: Simple multi-layer with various activation functions
  * Measures
    * Iterations (bad - not representative)
    * Time (honest)
    * Evaluations (stable)
  * Analysis