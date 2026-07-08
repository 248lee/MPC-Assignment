"""
evaluate.py -- run ONE environment x ONE method, cache to an npz
================================================================

Usage
-----
    python evaluate.py --env lqr --method cem
    python evaluate.py --env clqr --method tdmpc-noprior --horizons 1 5 10 15 20
    python evaluate.py --env terminal --method tdmpc-vf
    python evaluate.py --list                      # show envs, methods, compat

One (env, method) pair -> one file ``results/{env}__{method}.npz`` holding the
per-horizon episode rewards (or a single scalar for horizon-independent methods
like optimal / sac). A cached result is reused unless ``--force`` is given or the
config signature (seed / horizons / init state / prior checkpoint) changed.
"""

from __future__ import annotations

import argparse

from registry import ENV_REGISTRY, METHOD_REGISTRY, check_compatible, compatible_methods
from runner import evaluate_method, save_result, load_result, is_fresh, result_path


DEFAULT_HORIZONS = list(range(1, 21))


def print_catalogue():
    print("Environments:")
    for name, spec in ENV_REGISTRY.items():
        extra = []
        if spec.has_optimal:
            extra.append("optimal")
        if spec.prior_path:
            extra.append(f"prior={spec.prior_path}")
        if spec.terminal_value:
            extra.append("terminal-value")
        tag = f"  [{', '.join(extra)}]" if extra else ""
        print(f"  {name:10s} {spec.label}{tag}")
    print("\nMethods:")
    for name, spec in METHOD_REGISTRY.items():
        req = f"  requires: {', '.join(sorted(spec.requires))}" if spec.requires else ""
        hi = "" if spec.horizon_dependent else "  (horizon-independent)"
        print(f"  {name:20s} {spec.label}{hi}{req}")
    print("\nCompatible methods per env:")
    for name in ENV_REGISTRY:
        print(f"  {name:10s} {', '.join(compatible_methods(name))}")


def main():
    p = argparse.ArgumentParser(description="Evaluate one env x one method; cache to npz.")
    p.add_argument("--env", choices=list(ENV_REGISTRY))
    p.add_argument("--method", choices=list(METHOD_REGISTRY))
    p.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                   help="planning horizons to sweep (default 1..20)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="recompute even if a fresh cache exists")
    p.add_argument("--list", action="store_true", help="list envs / methods / compatibility and exit")
    args = p.parse_args()

    if args.list:
        print_catalogue()
        return
    if not args.env or not args.method:
        p.error("--env and --method are required (or use --list)")

    reason = check_compatible(args.env, args.method)
    if reason is not None:
        p.error(reason)

    if not args.force:
        cached = load_result(args.env, args.method)
        if is_fresh(cached, args.env, args.method, args.horizons, args.seed):
            print(f"[cache] {result_path(args.env, args.method)} is up to date -- skipping "
                  f"(use --force to recompute).")
            return

    result = evaluate_method(args.env, args.method, args.horizons, seed=args.seed)
    path = save_result(result)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
