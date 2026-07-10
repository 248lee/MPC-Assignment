"""
compare.py -- compare several methods on ONE environment
========================================================

Usage
-----
    python compare.py --env lqr --methods optimal sac cem tdmpc tdmpc-noprior
    python compare.py --env clqr --methods cem mppi tdmpc-noprior
    python compare.py --env terminal --methods cem tdmpc-noprior tdmpc-vf
    python compare.py --env lqr                     # all compatible methods

For each requested method it loads the cached ``results/{env}__{method}.npz``
(produced by ``evaluate.py``); if the cache is missing or stale it re-runs just
that method's sweep and caches it (unless ``--no-run``). Then it plots episode
COST = -reward vs planning horizon on a log scale (so the diverging short/long
horizons stay visible), with horizon-independent methods (optimal / sac) drawn
as horizontal reference lines.

Only the requested methods are ever (re)computed -- each method is cached in its
own file, so adding one method to a comparison never re-runs the others.
"""

from __future__ import annotations

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from registry import ENV_REGISTRY, METHOD_REGISTRY, check_compatible, compatible_methods
from runner import evaluate_method, save_result, load_result, is_fresh


DEFAULT_HORIZONS = list(range(1, 21))


def _print_premature_log(results: dict) -> None:
    """Reproduce the per-episode premature-convergence log from each result
    dict. Works identically for freshly-computed and cached results, since the
    counts are stored in the npz (see runner.evaluate_method)."""
    print("-" * 70)
    print("Premature convergence per episode (premature / total plannings):")
    for m, r in results.items():
        spec = METHOD_REGISTRY[m]
        prem = np.asarray(r.get("premature_counts", []), dtype=np.int64)
        plan = np.asarray(r.get("planning_counts", []), dtype=np.int64)
        # optimal / SAC (and any non-sampling method) never run the convergence
        # loop, so they log no plannings -> report n/a.
        if plan.size == 0 or int(plan.sum()) == 0:
            print(f"  {m:20s} ({spec.label}): n/a (no sampling-based convergence)")
            continue
        print(f"  {m:20s} ({spec.label})")
        if bool(r["horizon_dependent"]):
            hs = np.asarray(r["horizons"])
            for i in range(len(hs)):
                print(f"      H={int(hs[i]):>3}  premature {int(prem[i]):>5}/{int(plan[i]):<5}")
        else:
            print(f"      premature {int(prem[0])}/{int(plan[0])} (horizon-independent)")


def _get_result(env, method, horizons, seed, allow_run, force):
    """Return a fresh result dict for (env, method), running/caching if needed."""
    if not force:
        cached = load_result(env, method)
        if is_fresh(cached, env, method, horizons, seed):
            return cached
    if not allow_run:
        print(f"  [skip] {env}/{method}: no fresh cache and --no-run set.")
        return None
    result = evaluate_method(env, method, horizons, seed=seed)
    save_result(result)
    return result


def main():
    p = argparse.ArgumentParser(description="Compare several methods on one environment.")
    p.add_argument("--env", required=True, choices=list(ENV_REGISTRY))
    p.add_argument("--methods", nargs="+", default=None,
                   help="methods to compare (default: all compatible with the env)")
    p.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output figure path (default compare_{env}.png)")
    p.add_argument("--no-run", action="store_true",
                   help="only plot from existing caches; do not run missing/stale methods")
    p.add_argument("--force", action="store_true", help="recompute every requested method")
    args = p.parse_args()

    env = args.env
    methods = args.methods or compatible_methods(env)

    # validate / filter incompatible requests up front
    valid = []
    for m in methods:
        reason = check_compatible(env, m)
        if reason is not None:
            print(f"  [skip] {reason}")
            continue
        valid.append(m)
    if not valid:
        p.error(f"no compatible methods to compare on env {env!r}")

    print(f"Comparing on env '{env}' ({ENV_REGISTRY[env].label}): {', '.join(valid)}")
    print("-" * 70)

    results = {}
    for m in valid:
        r = _get_result(env, m, args.horizons, args.seed, allow_run=not args.no_run, force=args.force)
        if r is not None:
            results[m] = r

    if not results:
        p.error("nothing to plot (no results available).")

    # ---- plot: cost = -reward vs horizon (log scale) -------------------- #
    plt.figure(figsize=(9.5, 5.5))
    for m, r in results.items():
        spec = METHOD_REGISTRY[m]
        rewards = np.asarray(r["rewards"], dtype=np.float64)
        cost = -rewards
        if not bool(r["horizon_dependent"]):
            plt.axhline(cost[0], color=spec.color, ls="--", lw=2,
                        label=f"{spec.label} (cost={cost[0]:.2f})")
        else:
            hs = np.asarray(r["horizons"])
            plt.plot(hs, cost, marker=spec.marker or "o", color=spec.color, label=spec.label)

    plt.yscale("log")
    plt.xlabel("Planning horizon H")
    plt.ylabel("Episode cost = -reward (log scale)")
    plt.title(f"Method comparison on '{env}' ({ENV_REGISTRY[env].label}) -- lower is better")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out = args.out or f"compare_{env}.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print("-" * 70)
    print(f"Saved plot -> {out}")

    # ---- premature-convergence log (reproduced from cache too) ---------- #
    _print_premature_log(results)

    # ---- summary -------------------------------------------------------- #
    for m, r in results.items():
        rewards = np.asarray(r["rewards"], dtype=np.float64)
        if bool(r["horizon_dependent"]):
            hs = np.asarray(r["horizons"])
            if np.all(np.isnan(rewards)):
                continue
            bi = int(np.nanargmax(rewards))
            print(f"  {m:20s} best: H={int(hs[bi]):<3} reward={rewards[bi]:.4f}")
        else:
            print(f"  {m:20s} reward={rewards[0]:.4f} (horizon-independent)")


if __name__ == "__main__":
    main()
