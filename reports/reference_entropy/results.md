---
documents: results/*.log
related:
  - entropy_gated_benchmark.py
  - run_reports.cjs
  - ../../qqn_jax/regions/entropy_gated.py
---

# EG-PTGP sweep — results (2026-09-16)

Second sweep of the entropy-gated per-sample trust region (`qqn_jax/regions/entropy_gated.py`, spec in
`docs/entropy_gated_regions.md`).
Every log in `results/` was produced by `run_reports.cjs` with the default
variant set (`eg_baseline`, `eg_h_mem_{0p3,0p4,0p6}`, the `tau` axis and the
`policy` axis). The `kappa`, `zeta`, `num_regions`, `max_constraints` and
`warmup` axes, and the extreme `h_mem` values (0.1, 0.2, 0.35, 0.7, 1.0) are
defined in `run_reports.cjs` but were **not** run in this batch.

## Protocol

Shared by every cell (`BASE_PARAMS` in `run_reports.cjs`):

| knob      | value                                                                                                                |
|-----------|----------------------------------------------------------------------------------------------------------------------|
| data      | Fashion-MNIST, 5 000 train / 2 000 test, balanced, `DATA_SEED=0`                                                     |
| model     | `FlatMLP` 784→128→128→10, tanh, 118 282 parameters                                                                   |
| optimizer | QQN, `Fallback[LBFGS(50), Adam(1e-3)]`, `armijo_wolfe{c1=1e-9,c2=0.7,max_iter=10}`, `remember_step_size=False`       |
| budget    | `MAXITER=300`, `TIME_BUDGET=0` (disabled), `FREEZE_PATIENCE=25`, `FREEZE_TOL=1e-9`                                   |
| seeds     | `SEEDS=0,1,2,3,4` (init only; data split pinned)                                                                     |
| region    | `MAX_CONSTRAINTS=256`, `GRAD_CHUNK=64`, `WARMUP=5`, `dead_zone_ratio=0.05`, `dead_zone_floor=True`, `REGION_DEBUG=1` |
| L2        | `1e-4`                                                                                                               |

Pivot operating point (`EG_BASELINE`): `H_MEM=0.5`, `TAU=0.1`, `KAPPA=0.1`,
`ZETA=0.0`, `POLICY=mean`, `NUM_REGIONS=8`.

Each run evaluates the four cells of §7.4:

| cell        | objective                         | region               |
|-------------|-----------------------------------|----------------------|
| ERM         | plain CE                          | none                 |
| Gate-Obj    | `make_gated_loss` (α-weighted CE) | none                 |
| Region-Only | plain CE                          | `EntropyGatedRegion` |
| EG-PTGP     | `make_gated_loss`                 | `EntropyGatedRegion` |

All cells ran the full 300 iterations (`stop=maxiter`); no cell was
truncated by wall clock or by the freeze detector, so every averaged metric
is iteration-matched. Reported values are mean±std over the 5 seeds.
"ECE" is the mean over the last 20 logged iterations; "matched ERM" is ERM's
ECE linearly interpolated to the cell's final train accuracy; sECE is
signed calibration error (mean confidence − accuracy, + = overconfident);
"cross" is cross-seed test-set disagreement (the H1 quantity).

Hypothesis thresholds (from `entropy_gated_benchmark.py`): H1 passes if
EG-PTGP cross-seed churn is ≤ −30 % relative to ERM; H2 passes if EG-PTGP ECE
< matched-accuracy ERM ECE; H8 passes if any cell shows the dead zone (`t=0 ∧ Δf≈0`) or the inert gate (`|M|/N→1, H→0`).

## Baseline (`eg_baseline`, `results/…_eg_baseline_20260916_140751.log`)

| cell        | train acc     | test acc      | \|M\|/N | H     | ECE           | sECE    | matched ERM | cross  | time/run |
|-------------|---------------|---------------|---------|-------|---------------|---------|-------------|--------|----------|
| ERM         | 1.0000        | 0.8387±0.0019 | 1.000   | 0.011 | 0.1195        | +0.1189 | 0.1325      | 0.0663 | 7.5 s    |
| Gate-Obj    | 0.8770±0.0410 | 0.8207±0.0190 | 0.721   | 0.513 | 0.0309±0.0163 | −0.0143 | 0.0438      | 0.0978 | 10.3 s   |
| Region-Only | 1.0000        | 0.8405±0.0032 | 1.000   | 0.011 | 0.1195        | +0.1188 | 0.1325      | 0.0678 | 14.0 s   |
| EG-PTGP     | 0.8868±0.0291 | 0.8301±0.0112 | 0.735   | 0.496 | 0.0302±0.0135 | −0.0186 | 0.0486      | 0.0771 | 27.2 s   |

Region diagnostics (`ProjectionRecorder`, means over projections):

| cell        | binding | \|s*\|/\|step\| | cos    | #λ>0 | #valid | λmax   | guard | resid |
|-------------|---------|-----------------|--------|------|--------|--------|-------|-------|
| Region-Only | 0.466   | 0.9997          | 0.9998 | 0.47 | 0.98   | 3.1e-4 | 0     | 7e-9  |
| EG-PTGP     | 0.125   | 0.9989          | 0.9993 | 0.13 | 1.00   | 7.3e-5 | 0     | 4e-10 |

Verdicts: **H1 FAIL** (+16.3 %), **H2 PASS** (−37.9 %, underconfident), **H8 PASS** (inert gate in every ERM /
Region-Only seed; no `t=0 ∧ Δf≈0`
freeze anywhere).

## H_mem sweep (`POLICY=mean`, `TAU=0.1`) — EG-PTGP row

| H_mem | train acc     | test acc      | \|M\|/N | ECE           | sECE    | matched ERM | rel.    | cross (vs ERM 0.0663) | binding | H1   | H2   |
|-------|---------------|---------------|---------|---------------|---------|-------------|---------|-----------------------|---------|------|------|
| 0.3   | 0.9630±0.0381 | 0.8315±0.0024 | 0.883   | 0.0785±0.0407 | +0.0779 | 0.1010      | −22.3 % | 0.0862 (+29.9 %)      | 0.040   | FAIL | PASS |
| 0.4   | 0.8898±0.0657 | 0.8177±0.0142 | 0.716   | 0.0511±0.0452 | +0.0158 | 0.0555      | −7.9 %  | 0.1167 (+76.1 %)      | 0.064   | FAIL | PASS |
| 0.5   | 0.8868±0.0291 | 0.8301±0.0112 | 0.735   | 0.0302±0.0135 | −0.0186 | 0.0486      | −37.9 % | 0.0771 (+16.3 %)      | 0.125   | FAIL | PASS |
| 0.6   | 0.8801±0.0178 | 0.8283±0.0086 | 0.760   | 0.0395±0.0127 | −0.0299 | 0.0391      | +0.9 %  | 0.0725 (+9.4 %)       | 0.111   | FAIL | FAIL |

At `H_mem=0.3` two of five EG-PTGP seeds and two Gate-Obj seeds escaped the
hinge and trained to ≈100 % train accuracy (ECE ≈ 0.13, i.e. ERM), which is
what drives the large std. At `0.6` the gate is *underconfident* (sECE −0.03)
and no longer beats the matched-accuracy ERM baseline.

## τ sweep (`POLICY=mean`, `H_MEM=0.5`) — EG-PTGP row

| τ    | train acc     | test acc      | \|M\|/N | ECE           | sECE    | matched ERM | rel.    | cross            | H1   | H2    |
|------|---------------|---------------|---------|---------------|---------|-------------|---------|------------------|------|-------|
| 0.02 | 0.8288±0.0306 | 0.7954±0.0261 | 0.442   | 0.0527±0.0100 | −0.0445 | 0.0272      | +93.6 % | 0.1003 (+51.3 %) | FAIL | FAIL  |
| 0.05 | 0.8488±0.0143 | 0.8124±0.0081 | 0.548   | 0.0494±0.0133 | −0.0418 | 0.0278      | +77.9 % | 0.0651 (−1.7 %)  | FAIL | FAIL  |
| 0.1  | 0.8868±0.0291 | 0.8301±0.0112 | 0.735   | 0.0302±0.0135 | −0.0186 | 0.0486      | −37.9 % | 0.0771 (+16.3 %) | FAIL | PASS  |
| 0.25 | 1.0000        | 0.8380±0.0041 | 1.000   | 0.1206±0.0062 | +0.1201 | 0.1325      | −9.0 %  | 0.0744 (+12.2 %) | FAIL | PASS* |
| 0.5  | 1.0000        | 0.8401±0.0051 | 1.000   | 0.1188±0.0029 | +0.1181 | 0.1325      | −10.4 % | 0.0708 (+6.8 %)  | FAIL | PASS* |

\* trivial: for τ ≥ 0.25 the sigmoid band is so wide that α never goes to
zero, every gated cell reaches `|M|/N = 1, H ≈ 0.01` and is indistinguishable
from ERM (`inert` flag set on all seeds; region `binding` ≈ 0.01–0.02). The
−9…−10 % "pass" is the same interpolation artifact ERM itself shows (0.1195 vs 0.1325). For τ ≤ 0.05 the gate is sharp
enough to stall training
at ≈83–85 % train accuracy and the cells are strongly underconfident. τ = 0.1
is the only setting in the sweep where the hinge is active without being
inert.

## Region-policy sweep (`H_MEM=0.5`, `TAU=0.1`) — EG-PTGP row

| policy   | train acc     | test acc      | ECE               | sECE    | matched ERM | rel.    | cross               | binding | \|s*\|/\|step\| | #valid | time/run | H1   | H2   |
|----------|---------------|---------------|-------------------|---------|-------------|---------|---------------------|---------|-----------------|--------|----------|------|------|
| mean     | 0.8868±0.0291 | 0.8301±0.0112 | 0.0302±0.0135     | −0.0186 | 0.0486      | −37.9 % | 0.0771 (+16.3 %)    | 0.125   | 0.9989          | 1.00   | 27 s     | FAIL | PASS |
| class    | 0.8880±0.0410 | 0.8271±0.0082 | 0.0360±0.0090     | −0.0119 | 0.0483      | −25.5 % | 0.0870 (+31.2 %)    | 0.259   | 0.9973          | 3.91   | 41 s     | FAIL | PASS |
| pair     | 0.8917±0.0194 | 0.8317±0.0042 | **0.0205±0.0072** | −0.0066 | 0.0447      | −54.2 % | 0.0742 (+11.8 %)    | 0.481   | 0.9931          | 10.78  | 157 s    | FAIL | PASS |
| gradient | 0.8837±0.0112 | 0.8299±0.0030 | 0.0260±0.0039     | −0.0146 | 0.0389      | −33.1 % | **0.0670 (+1.1 %)** | 0.349   | 0.9946          | 4.56   | 90 s     | FAIL | PASS |
| sample   | 0.9022±0.0292 | 0.8295±0.0095 | 0.0282±0.0136     | +0.0070 | 0.0523      | −46.0 % | 0.0805 (+21.5 %)    | 0.614   | 0.9754          | 255.1  | 336 s    | FAIL | PASS |
| random   | 0.8744±0.0205 | 0.8278±0.0100 | 0.0341±0.0101     | −0.0289 | 0.0365      | −6.5 %  | 0.0681 (+2.7 %)     | 0.140   | 0.9984          | 7.98   | 41 s     | FAIL | PASS |

Region-Only for every policy: train acc 1.000, test acc 0.836–0.841, ECE
0.120–0.128, `|s*|/|step|` ≥ 0.994 — statistically indistinguishable from
ERM at 2–7× the cost. For `sample` (k=256) the Hildreth dual with
`dual_sweeps=20` has not converged: mean primal residual 2.7e-3 (Region-Only)
vs ≤ 1e-7 for k ≤ 100.

The `pair` policy has the lowest ECE and lowest seed-to-seed ECE variance and
`gradient` the lowest cross-seed churn, but neither difference vs `mean`
clears the seed noise (ECE std ≈ 0.007–0.014) and both cost 3–6× more.

## What the logs say

1. **The constraint channel does nothing measurable.** Across 15 runs
   Region-Only tracks ERM to within seed noise on accuracy, ECE and churn.
   The constraints *do* bind (13–61 % of projections in EG-PTGP, up to 91 %
   in Region-Only/`sample`) but the projected step is ≥ 97.5 % of the raw
   step in norm and cosine ≥ 0.98 — the cone is almost never violated by
   more than a rounding error, so `λ` stays ~1e-4. The dead-zone guard never
   fired (`guard = 0` everywhere).
2. **Everything attributed to EG-PTGP comes from the objective channel.**
   Gate-Obj and EG-PTGP are within noise on every metric at every operating
   point; EG-PTGP just costs 2.5–30× more.
3. **The entropy hinge behaves as early stopping.** Gated cells plateau at
   85–92 % train accuracy with `|M|/N ≈ 0.7` and mean entropy ≈ 0.5 nats,
   and their ECE (0.02–0.05) sits on ERM's own accuracy/ECE curve, which is
   why the matched-accuracy comparison lands anywhere between −54 % and
   +94 % depending on where the plateau happens to fall.
4. **The hinge stalls the line search rather than the parameters.** Gated
   cells spend most of their 300 iterations at `t = 0` with
   `‖Δθ‖ ≈ 1e-4` and `f` constant to 8 digits — the objective is flat on
   the hinge plateau, Armijo rejects every trial, and progress happens in
   rare bursts when a few samples cross `H_mem`. This is *not* the H8
   dead zone (which requires `‖Δθ‖ = 0`), so `FREEZE_PATIENCE` never
   triggered; the benchmark logs the discrepancy as the
   `state.step_size==0 while ||dtheta||>0` note.
5. **H1 is not supported at any operating point.** Cross-seed churn of
   EG-PTGP is 0.065–0.117 vs ERM 0.066; the best cells (`gradient`,
   `random`, τ=0.05) merely match ERM. Under-converged models disagree *more* across seeds, not less.
6. **H2 passes only in the τ=0.1, H_mem∈[0.3,0.5] band, and only
   because of Gate-Obj.** Outside that band the gate is either inert (τ ≥ 0.25, H_mem too low) or underconfident (τ ≤
   0.05, H_mem = 0.6).

## Reproducing / aggregating

```sh
cd reports/reference_entropy
node run_reports.cjs --list                # all variants and axes
node run_reports.cjs                       # the default set used here
node run_reports.cjs --axis kappa          # an axis not yet run
node run_reports.cjs eg_baseline --env SEEDS=0,1 --env MAXITER=100
node run_reports.cjs --summarize           # results/*.log -> results/summary.csv
```

Each log is self-describing (`# command:` header carries the full
environment) and ends with one machine-readable `[row] …` line per (variant, seed) that `--summarize` collects into
`results/summary.csv`.