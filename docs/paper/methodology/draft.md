# Methodology

## 1. Overview

We conduct a controlled empirical study comparing the Quadratic-Quasi-Newton
(QQN) optimizer against three widely used baselines — stochastic gradient
descent (SGD), Adam, and L-BFGS — on non-convex neural-network training
problems. Our central methodological concern is *fairness*: we design the
experimental harness so that any measured performance difference is
attributable to the optimizers themselves rather than to incidental
differences in initialization, termination, evaluation accounting, or
timing. To that end we impose a set of explicit fairness invariants
(Section 3) that every optimizer variant must satisfy.

The benchmark objective is multi-class image classification with a
multilayer perceptron (MLP). We evaluate across MNIST and Fashion-MNIST and
across a battery of activation functions, since the geometry of the loss
landscape — and hence the relative advantage of each optimizer — depends
strongly on the nonlinearity used.

## 2. Experimental Design

### 2.1 Objective and Models

All experiments optimize a flat multilayer perceptron (`FlatMLP`) whose
parameters are stored as a single flat vector. The model exposes a
configurable number of hidden layers, hidden width, and per-layer activation
function. The training objective is the cross-entropy classification loss
with an L2 (weight-decay) regularization term of strength `l2`. Crucially,
a *single* loss closure is constructed once from the training data and
regularization strength and is shared by every optimizer:

```
loss_fn = model.make_loss(X_train, y_train, l2=config.l2)
```

Sharing one loss function guarantees that all optimizers see byte-identical
objectives, gradients, and regularization.

### 2.2 Datasets

We use MNIST and Fashion-MNIST. From each dataset we draw balanced subsets of
configurable train and test sizes, using a fixed `subset_seed` so that every
optimizer trains and is evaluated on exactly the same examples. Balanced
sampling ensures equal class representation and removes class-imbalance as a
confounding variable.

### 2.3 Optimizers Under Test

We compare four optimizer families:

- **QQN** — the proposed method, which follows a quadratic (or optionally
higher-order spline) path built from an oracle direction, refined by a
configurable line search and optionally constrained to a trust region.
- **SGD** — plain stochastic gradient descent with a fixed learning rate.
- **Adam** — the adaptive-moment optimizer with a fixed learning rate.
- **L-BFGS** — the limited-memory quasi-Newton method (Optax
implementation, using a zoom line search).

SGD, Adam, and L-BFGS serve as reference baselines and do not participate in
the QQN configuration cross-product (Section 4), since they lack QQN's
oracle / line-search / region axes.

## 3. Fairness Invariants

The driver (`experiments/driver.py`) enforces the following invariants,
which are the backbone of our methodology:

1. **Identical initialization.** Every optimizer starts from the same
parameter vector, produced by a single PRNG key:
`params0 = model.init_params(jax.random.PRNGKey(config.seed))`.
No optimizer receives a more or less favorable starting point.

2. **Shared termination.** All optimizers use one termination
specification — a target loss `f_target`, a gradient-norm tolerance
`gtol`, and a wall-clock `time_budget` — expressed once in
`ExperimentConfig.stop` and threaded identically into each runner. The
convergence test `converged(value, gnorm, f_target, gtol)` and the
milestone recorder `update_milestones(...)` are defined once and reused
by all runners, so termination semantics cannot silently diverge across
optimizers.

3. **Genuine evaluation accounting.** Rather than assuming a fixed cost per
iteration, each runner records the *actual* cumulative number of function
and gradient evaluations. QQN and the Optax runners maintain explicit
forward (`fwd_counts`) and backward (`bwd_counts`) evaluation counters.
This lets us compare optimizers on the honest currency of oracle calls,
not merely iterations — important because a single QQN or L-BFGS
iteration may internally evaluate the objective several times during its
line search.

4. **Identical loss / data / regularization.** As noted in Section 2.1, one
loss closure is shared, so the objective is provably identical.

5. **Warmup exclusion.** JIT compilation cost is excluded from timing. Each
runner performs one real update step *before* starting the wall-clock
timer, blocking until the result is materialized
(`jax.block_until_ready(...)`). Consequently the multi-second trace/
compile cost of the first `update` is not charged to iteration 1. (We
revisit residual compilation effects in Section 8.)

Each optimizer variant is additionally wrapped in its own profiling region,
so per-variant device activity can be inspected independently.

## 4. QQN Configuration Space

QQN exposes many orthogonal design choices. Rather than hand-enumerate
combinations, we generate profiles as the *Cartesian product* of a small set
of independent axes, each defined as a `{token: kwargs}` map
(`reports/searches/profiles.py`). The empty token `""` is always an axis's
default and contributes nothing to the generated profile name; every other
entry both selects keyword arguments for the QQN constructor and appends its
token to the profile name.

The axes are:

- **Oracle** — the search-direction generator (e.g. L-BFGS with a given
history size, Adam, momentum, or fallback chains such as
`Fallback([LBFGSOracle(...), AdamOracle(...)])`).
- **Line search** — backtracking, Armijo-Wolfe, strong Wolfe, Hager-Zhang,
bisection, spline, or fixed, each with its own `c1`/`c2`/`max_iter`
parameters.
- **Parametric bound (`max_t`)** — the largest value of the quadratic path
parameter the line search may explore.
- **Temperature** — an optional Metropolis-style stochastic acceptance in
the line search (for the backtracking/Armijo family).
- **Spline / path strategy** — quadratic (default), cubic-Hermite spline,
or value-only linear interpolation.
- **Trust region** — an optional constraint on the step (trust region, box,
or PSD-secant region).
- **Probe feeding** — whether line-search probe points are fed back to the
oracle.
- **Partitioning** — whether the flat parameter vector is split per-layer,
each block getting its own oracle curvature history.
- **Step-size memory** — whether the line search warm-starts from the last
accepted step size.

A profile's name is built by hyphenating its non-empty tokens after a
leading `QQN` in a fixed axis order (e.g. oracle `L80` + line search `BT`
gives `QQN-L80-BT`). Only profiles whose names appear in an `ENABLED` list
are actually built and run. Disabling a variant is as simple as commenting
out its line, keeping the search space explicit and version-controlled.

## 5. Metrics

For every run we record a `RunResult` containing the raw measured
trajectories (loss history, per-iteration wall times, and cumulative eval
counts) and, after enrichment by the driver, a set of derived quantities:

- **Loss quality** — final loss, best loss over the trajectory.
- **Accuracy** — train and test classification accuracy of the final
parameters.
- **Iteration count** — number of optimizer iterations.
- **Wall time and ms/iter** — total time and mean per-iteration time.
- **Trajectory AUC** — the area under the log-loss-vs-normalized-progress
curve, a scalar summary of how quickly loss falls over the whole run.
- **Time / iterations / evals to target** — the first point at which the
run reaches the shared target loss (and a boolean `reached` flag).
- **Per-milestone crossings** — for a ladder of loss milestones we record
the first iteration, wall time, and evaluation counts (forward, backward,
and combined) at which each milestone is crossed.

The milestone ladder is central: it lets us credit an optimizer that reaches
intermediate loss levels quickly even if it is eventually overtaken.

## 6. Pareto Analysis

Because no single scalar fully characterizes an optimizer, we report Pareto
frontiers over competing objectives. Two constructions are used:

- **Loss vs. wall-time** — the set of variants not dominated on both final
loss and total wall time.
- **Per-milestone (time, evals)** — for each loss milestone, the min-min
Pareto frontier over the two honest cross-optimizer cost axes: wall-clock
time and total forward+backward evaluations to *first* reach that loss
level. Evaluating domination independently at each milestone credits fast
early winners (e.g. Adam) that later plateau.

A point is dominated when another is at least as good on both axes and
strictly better on at least one; points missing a value on either axis are
excluded from the frontier.

## 7. Reproducibility Infrastructure

### 7.1 Configuration

All experiment knobs live in a single `ExperimentConfig` dataclass: dataset,
class count, train/test sizes, hidden topology, activation, regularization,
seeds, iteration cap, and the full termination profile (target loss,
gradient tolerance, time budget, milestones, and target profile).
`ExperimentConfig.from_env` layers environment-variable overrides over the
benchmark's narrative defaults, so a run's configuration can be reproduced
exactly from its recorded environment.

### 7.2 Driver Pipeline

The driver executes a deterministic pipeline: load the dataset subset,
build the model and shared loss, construct the initial parameters, assemble
the enabled runners via `build_runners(ctx, enabled=...)`, run each variant
inside its profiling region, enrich its `RunResult`, and finally emit
tables, axis analysis, JSON artifacts, and plots.

### 7.3 Artifacts and Viewer

Each run and experiment is serialized to a JSON artifact following a fixed
schema. A manifest generator (`generate-results-manifest.js`) scans the
results directory and produces a compact `manifest.json` index carrying
summary metadata (final/best loss, iterations, accuracy, wall time,
time/iters/evals to target, etc.) without the full trajectories. A
browser-based viewer (`results/index.html`) reads the manifest and lets us
interactively select, filter, and overlay runs — plotting loss against
iteration, wall time, warmup-adjusted wall time, or evaluation count on
linear or logarithmic axes.

### 7.4 Batch Runner and Activation Sweeps

A Node driver (`run_reports.js`) orchestrates batches of runs as named
*variants*. Each variant fixes a report script and a set of environment
overrides. We define an activation sweep that generates one variant per
activation function (relu, sigmoid, sine, gaussian, triangle, logabs, tanh,
gelu, swish, softplus, sawtooth, abs, identity, rolling_sin, rolling_atan2)
under a common set of sweep parameters, so the entire battery of
nonlinearities is exercised under identical conditions. Each run's stdout
and stderr are tee'd to a timestamped, self-describing log recording the
exact command line and environment.

## 8. Threats to Validity

Several harness artifacts warrant discussion:

- **Residual compilation cost.** Although we exclude the first JIT trace/
compile from timing via a warmup step, both L-BFGS and QQN still exhibit
an extra ~2.6 s on their first *timed* iteration attributable to JAX
compilation. We report this explicitly rather than smoothing it away.

- **L-BFGS eval accounting.** The Optax L-BFGS implementation does not
directly expose its per-iteration line-search evaluation count. We
therefore estimate it: when the optimizer state exposes a
`num_linesearch_steps` field we read it directly and fold it into a
running average; when it is unavailable we fall back to that running
average (rather than a fixed constant) so the eval count adapts to the
problem's observed behavior. This estimation is a known limitation of the
cross-optimizer eval comparison for L-BFGS.

- **Anomalous line-search behavior.** We observe that certain line-search
strategies (notably the `Fix` fixed-step strategy) can perform
unexpectedly well or poorly depending on the activation and topology. We
flag these as open questions for further study rather than tuning them
away, in keeping with our fairness-first methodology.