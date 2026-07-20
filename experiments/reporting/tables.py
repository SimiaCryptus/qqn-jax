"""All textual reporting: summary, Pareto, leaderboards, profiles, stalls.

Moved verbatim (in spirit) from the example driver, parameterized by the
``{name: RunResult}`` results dict + an ``ExperimentConfig``.
"""

import numpy as np

from experiments.metrics.pareto import (
    milestone_pareto_frontiers,
    pareto_frontier,
)

__all__ = ["report_tables"]


def _total_evals(r):
    """Best available cumulative (fwd + bwd) evals for a run.
    Prefers the deepest-milestone hit's fwd/bwd counters; falls back to the
    combined ``evals`` field when the split isn't populated.
    """
    if not r.milestone_hits:
        return None
    best = None
    for hit in r.milestone_hits.values():
        if hit is None:
            continue
        fwd = hit[3] if len(hit) >= 4 else None
        bwd = hit[4] if len(hit) >= 5 else None
        if fwd is not None and bwd is not None:
            cand = int(fwd) + int(bwd)
        elif len(hit) >= 3 and hit[2] is not None:
            cand = int(hit[2])
        else:
            continue
        if best is None or cand > best:
            best = cand
    return best


def report_tables(results, config):
    _summary(results)
    _pareto(results)
    _milestone_pareto(results, config)
    _eval_leaderboard(results)
    _milestone_profiles(results, config)
    _trajectory(results)


def _summary(results):
    ordered = sorted(results.items(), key=lambda kv: kv[1].final_loss)
    lbfgs_ref = results["L-BFGS"].iters_to_target if "L-BFGS" in results else None
    print(
        f"{'optimizer':<12}{'final_loss':>14}{'iters':>8}"
        f"{'train_acc':>12}{'test_acc':>11}{'time(s)':>10}"
        f"{'ms/it':>8}{'->target':>10}{'t->tgt':>9}{'vs LBFGS':>10}"
        f"{'evals':>9}{'AUC':>8}"
    )
    print("-" * 130)
    for name, r in ordered:
        it_tgt = "—" if r.iters_to_target is None else f"{r.iters_to_target}"
        t_tgt = "—" if r.time_to_target is None else f"{r.time_to_target:.3f}"
        if lbfgs_ref is not None and r.iters_to_target is not None:
            spd = f"{lbfgs_ref / r.iters_to_target:.2f}x"
        else:
            spd = "—"
        ev = "—" if r.evals_to_target is None else f"{r.evals_to_target}"
        print(
            f"{name:<12}{r.final_loss:>14.6e}{r.iters:>8}"
            f"{r.train_acc:>12.4f}{r.test_acc:>11.4f}{r.wall:>10.3f}"
            f"{r.ms_per_iter:>8.2f}{it_tgt:>10}{t_tgt:>9}{spd:>10}"
            f"{ev:>9}{r.traj_auc:>8.2f}"
        )


def _pareto(results):
    print("\nPareto frontier (loss vs. time vs. total evals — non-dominated variants):")
    pareto = pareto_frontier(results)
    for name, r in sorted(pareto, key=lambda kv: kv[1].wall):
        tot = _total_evals(r)
        tot_s = "—" if tot is None else f"{tot}"
        print(
            f"  {name:<14} loss={r.final_loss:.4e}  "
            f"time={r.wall:.3f}s  total_evals~{tot_s:>8}"
        )


def _milestone_pareto(results, config):
    """Per-milestone Pareto frontiers (time & evals to *reach* each loss).
    Unlike the single final-loss Pareto frontier above, this evaluates
    domination independently at every milestone on the honest cross-optimizer
    cost axes (wall-time, total fwd+bwd evals). It surfaces methods that win
    the *race* to early loss levels even if they later stall — e.g. Adam
    typically dominates the first milestones on both time and evals.
    """
    milestones = config.milestones
    if not milestones:
        return
    frontiers = milestone_pareto_frontiers(results, milestones)
    if not frontiers:
        return
    print(
        "\nPer-milestone Pareto frontier (non-dominated on time & total "
        "fwd+bwd evals to *first* reach each loss):"
    )
    for m in milestones:
        frontier = frontiers.get(m)
        if not frontier:
            continue
        print(f"  <= {m:.1e}:")
        for name, t, tot in sorted(frontier, key=lambda x: x[1]):
            print(f"      {name:<16} time={t:.3f}s  total_evals~{tot:>8}")


def _eval_leaderboard(results):
    print("\nCost-aware leaderboard (estimated function/grad evals to target):")
    eval_ranked = [(n, r) for n, r in results.items() if r.evals_to_target is not None]
    eval_ranked.sort(key=lambda kv: kv[1].evals_to_target)
    lbfgs_evals = results["L-BFGS"].evals_to_target if "L-BFGS" in results else None
    for name, r in eval_ranked[:12]:
        spd = (
            f"{lbfgs_evals / r.evals_to_target:.2f}x"
            if lbfgs_evals is not None
            else "—"
        )
        print(
            f"  {name:<14} evals~{r.evals_to_target:>5}  "
            f"(={r.evals_per_iter:.1f}/it x {r.iters_to_target} it)  "
            f"vs_LBFGS={spd:>6}  final={r.final_loss:.4e}"
        )


def _milestone_profiles(results, config):
    milestones = config.milestones
    if not milestones:
        return
    tightest = milestones[-1]

    def _sort_key_time(kv):
        hit = kv[1].milestone_hits.get(tightest)
        return hit[1] if hit is not None else float("inf")

    # Race-to-milestone: the honest cross-optimizer comparison is *time*
    # and *total work* (fwd+bwd evals) to first reach each loss level —
    # not iterations, which are apples-to-oranges across methods.
    print(
        "\nRace to each milestone (wall-clock seconds / total fwd+bwd evals "
        "to first reach each loss):"
    )
    header = (
        "  " + f"{'optimizer':<12}" + "".join(f"{f'<={m:.1e}':>20}" for m in milestones)
    )
    print(header)
    for name, r in sorted(results.items(), key=_sort_key_time):
        cells = []
        for m in milestones:
            hit = r.milestone_hits.get(m)
            if hit is None:
                cells.append("—")
                continue
            t = hit[1]
            fwd = hit[3] if len(hit) >= 4 else None
            bwd = hit[4] if len(hit) >= 5 else None
            if fwd is not None and bwd is not None:
                tot = int(fwd) + int(bwd)
            elif len(hit) >= 3 and hit[2] is not None:
                tot = int(hit[2])
            else:
                tot = None
            tot_s = "—" if tot is None else f"{tot}"
            cells.append(f"{t:.3f}s/{tot_s}")
        print("  " + f"{name:<12}" + "".join(f"{c:>20}" for c in cells))

    print(
        "\nInter-milestone cost breakdown "
        "(Δtime[s] / Δevals between consecutive milestones):"
    )
    seg_labels = []
    prev = None
    for m in milestones:
        seg_labels.append(f"start->{m:.1e}" if prev is None else f"{prev:.1e}->{m:.1e}")
        prev = m
    header = "  " + f"{'optimizer':<12}" + "".join(f"{s:>20}" for s in seg_labels)
    print(header)
    for name, r in sorted(results.items(), key=_sort_key_time):
        cells = []
        prev_hit = (0, 0.0, 0, 0, 0)
        for m in milestones:
            hit = r.milestone_hits.get(m)
            if hit is None:
                cells.append("—")
                continue
            dt = hit[1] - prev_hit[1]
            if len(hit) >= 3 and hit[2] is not None and prev_hit[2] is not None:
                de = hit[2] - prev_hit[2]
                cells.append(f"{dt:.3f}/{de}")
            else:
                cells.append(f"{dt:.3f}/—")
            prev_hit = hit
        print("  " + f"{name:<12}" + "".join(f"{c:>20}" for c in cells))


def _trajectory(results):
    print("\nLoss trajectory (log10, sampled):")
    sample_points = 10
    for name, r in results.items():
        hist = r.history
        idxs = np.linspace(0, len(hist) - 1, sample_points).astype(int)
        vals = [f"{np.log10(max(hist[i], 1e-12)):6.2f}" for i in idxs]
        print(f"  {name:<10} " + " ".join(vals))
