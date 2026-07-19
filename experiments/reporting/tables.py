"""All textual reporting: summary, Pareto, leaderboards, profiles, stalls.

Moved verbatim (in spirit) from the example driver, parameterized by the
``{name: RunResult}`` results dict + an ``ExperimentConfig``.
"""

import numpy as np

from experiments.metrics.pareto import pareto_frontier

__all__ = ["report_tables"]


def report_tables(results, config):
    _summary(results)
    _pareto(results)
    _iter_leaderboard(results)
    _eval_leaderboard(results)
    _target_profile(results, config)
    _milestone_profiles(results, config)
    _stall_report(results, config)
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
    print("\nPareto frontier (loss vs. time — non-dominated variants):")
    pareto = pareto_frontier(results)
    for name, r in sorted(pareto, key=lambda kv: kv[1].wall):
        print(f"  {name:<12} loss={r.final_loss:.4e}  time={r.wall:.3f}s")


def _iter_leaderboard(results):
    print("\nIteration-efficiency leaderboard (target reached, fewest iters):")
    lbfgs_ref = results["L-BFGS"].iters_to_target if "L-BFGS" in results else None
    converged = [(n, r) for n, r in results.items() if r.iters_to_target is not None]
    converged.sort(key=lambda kv: (kv[1].iters_to_target, kv[1].wall))
    for name, r in converged[:12]:
        spd = f"{lbfgs_ref / r.iters_to_target:.2f}x" if lbfgs_ref is not None else "—"
        print(
            f"  {name:<14} iters={r.iters_to_target:>4}  "
            f"time={r.time_to_target:.3f}s  vs_LBFGS={spd:>6}  "
            f"final={r.final_loss:.4e}"
        )


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


def _target_profile(results, config):
    target_profile = config.target_profile
    print("\nTarget-sensitivity profile (iterations to reach each loss target):")
    header = (
        "  "
        + f"{'optimizer':<14}"
        + "".join(f"{f'<={t:.2e}':>14}" for t in target_profile)
    )
    print(header)
    tightest = target_profile[-1]

    def _tgt_key(kv):
        v = kv[1].target_iters.get(tightest)
        return v if v is not None else 10**9

    for name, r in sorted(results.items(), key=_tgt_key):
        cells = []
        for t in target_profile:
            it = r.target_iters.get(t)
            cells.append("—" if it is None else f"{it}")
        print("  " + f"{name:<14}" + "".join(f"{c:>14}" for c in cells))

    if "L-BFGS" in results:
        for ref_name in ("QQN-L50", "QQN-L80", "QQN-L120", "QQN-L160", "QQN-Lean"):
            if ref_name not in results:
                continue
            print(f"\n  vs-LBFGS speedup stability across targets ({ref_name}):")
            ref = results[ref_name].target_iters
            lbf = results["L-BFGS"].target_iters
            for t in target_profile:
                a, b = ref.get(t), lbf.get(t)
                if a and b and a > 0:
                    print(f"    <= {t:.2e}:  {b / a:.2f}x  ({ref_name}={a}, LBFGS={b})")
                else:
                    print(f"    <= {t:.2e}:  — (not both reached)")


def _milestone_profiles(results, config):
    milestones = config.milestones
    if not milestones:
        return
    tightest = milestones[-1]

    def _sort_key(kv):
        hit = kv[1].milestone_hits.get(tightest)
        return hit[0] if hit is not None else 10**9

    def _sort_key_time(kv):
        hit = kv[1].milestone_hits.get(tightest)
        return hit[1] if hit is not None else float("inf")

    print("\nConvergence-rate profile (iteration first reaching each loss):")
    header = (
        "  " + f"{'optimizer':<12}" + "".join(f"{f'<={m:.1e}':>12}" for m in milestones)
    )
    print(header)
    for name, r in sorted(results.items(), key=_sort_key):
        cells = []
        for m in milestones:
            hit = r.milestone_hits.get(m)
            cells.append("—" if hit is None else f"{hit[0]}")
        print("  " + f"{name:<12}" + "".join(f"{c:>12}" for c in cells))

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

    print("\nConvergence-rate profile (wall-clock seconds first reaching each loss):")
    header = (
        "  " + f"{'optimizer':<12}" + "".join(f"{f'<={m:.1e}':>12}" for m in milestones)
    )
    print(header)
    for name, r in sorted(results.items(), key=_sort_key_time):
        cells = []
        for m in milestones:
            hit = r.milestone_hits.get(m)
            cells.append("—" if hit is None else f"{hit[1]:.3f}")
        print("  " + f"{name:<12}" + "".join(f"{c:>12}" for c in cells))

    print(
        "\nConvergence-rate profile "
        "(estimated function/grad evals first reaching each loss):"
    )
    header = (
        "  " + f"{'optimizer':<12}" + "".join(f"{f'<={m:.1e}':>12}" for m in milestones)
    )
    print(header)

    def _sort_key_evals(kv):
        hit = kv[1].milestone_hits.get(tightest)
        if hit is None or len(hit) < 3 or hit[2] is None:
            return 10**9
        return hit[2]

    for name, r in sorted(results.items(), key=_sort_key_evals):
        cells = []
        for m in milestones:
            hit = r.milestone_hits.get(m)
            if hit is None or len(hit) < 3 or hit[2] is None:
                cells.append("—")
            else:
                cells.append(f"{hit[2]}")
        print("  " + f"{name:<12}" + "".join(f"{c:>12}" for c in cells))
    print("\nConvergence-rate profile (forward value evals first reaching each loss):")
    header = (
        "  " + f"{'optimizer':<12}" + "".join(f"{f'<={m:.1e}':>12}" for m in milestones)
    )
    print(header)

    def _sort_key_fwd(kv):
        hit = kv[1].milestone_hits.get(tightest)
        if hit is None or len(hit) < 4 or hit[3] is None:
            return 10**9
        return hit[3]

    for name, r in sorted(results.items(), key=_sort_key_fwd):
        cells = []
        for m in milestones:
            hit = r.milestone_hits.get(m)
            if hit is None or len(hit) < 4 or hit[3] is None:
                cells.append("—")
            else:
                cells.append(f"{hit[3]}")
        print("  " + f"{name:<12}" + "".join(f"{c:>12}" for c in cells))
    print(
        "\nConvergence-rate profile (backward gradient evals first reaching each loss):"
    )
    header = (
        "  " + f"{'optimizer':<12}" + "".join(f"{f'<={m:.1e}':>12}" for m in milestones)
    )
    print(header)

    def _sort_key_bwd(kv):
        hit = kv[1].milestone_hits.get(tightest)
        if hit is None or len(hit) < 5 or hit[4] is None:
            return 10**9
        return hit[4]

    for name, r in sorted(results.items(), key=_sort_key_bwd):
        cells = []
        for m in milestones:
            hit = r.milestone_hits.get(m)
            if hit is None or len(hit) < 5 or hit[4] is None:
                cells.append("—")
            else:
                cells.append(f"{hit[4]}")
        print("  " + f"{name:<12}" + "".join(f"{c:>12}" for c in cells))


def _stall_report(results, config):
    stalled = [(n, r) for n, r in results.items() if r.iters_to_target is None]
    if not stalled:
        return
    print("\nStall report (never reached the shared target):")
    stalled.sort(key=lambda kv: kv[1].final_loss)
    budget = config.time_budget
    for name, r in stalled:
        if r.wall >= budget - 0.5:
            cause = "time-budget exhausted"
        elif r.final_loss > 0.7:
            cause = "stalled (plateau)"
        else:
            cause = "slow (no target in maxiter)"
        print(
            f"  {name:<14} final={r.final_loss:.4e}  "
            f"iters={r.iters:>3}  time={r.wall:.3f}s  [{cause}]"
        )


def _trajectory(results):
    print("\nLoss trajectory (log10, sampled):")
    sample_points = 10
    for name, r in results.items():
        hist = r.history
        idxs = np.linspace(0, len(hist) - 1, sample_points).astype(int)
        vals = [f"{np.log10(max(hist[i], 1e-12)):6.2f}" for i in idxs]
        print(f"  {name:<10} " + " ".join(vals))
