"""
Evaluation runner: one environment x one method
================================================

`evaluate_method` builds the chosen env + planner from the registries, rolls
one episode per planning horizon (or a single episode for horizon-independent
methods like optimal / SAC), and returns a plain dict of arrays ready to cache
in an npz. `run_episode` is the shared rollout loop used by every method.
"""

from __future__ import annotations

import os

import numpy as np

from registry import ENV_REGISTRY, METHOD_REGISTRY, check_compatible


# --------------------------------------------------------------------------- #
# shared rollout
# --------------------------------------------------------------------------- #
def run_episode(env, agent, init_state, T):
    """Roll out an agent (``.act(state)``, optional ``.reset()``) for T steps;
    return (total_reward, final_state)."""
    if hasattr(agent, "reset"):
        agent.reset()
    s = env.reset(state=None if init_state is None else np.asarray(init_state, np.float64))
    total = 0.0
    last = s
    for _ in range(T):
        a = agent.act(s)
        s, r, term, trunc, _ = env.step(a)
        total += r
        last = s
        if term or trunc:
            break
    return total, last


# --------------------------------------------------------------------------- #
# per-(env, method) signature for cache staleness
# --------------------------------------------------------------------------- #
def _prior_mtime(prior_path: str | None) -> float:
    return os.path.getmtime(prior_path) if prior_path and os.path.exists(prior_path) else -1.0


def config_sig(env_name: str, method_name: str, horizons, seed: int) -> np.ndarray:
    espec = ENV_REGISTRY[env_name]
    mspec = METHOD_REGISTRY[method_name]
    init = espec.default_init_state
    init = np.array([], np.float64) if init is None else np.asarray(init, np.float64)
    uses_prior = ("prior" in mspec.requires)
    parts = [
        float(seed), float(espec.eval_T), float(len(init)), *init,
        float(len(horizons)), *[float(h) for h in horizons],
        _prior_mtime(espec.prior_path) if uses_prior else 0.0,
    ]
    return np.array(parts, dtype=np.float64)


# --------------------------------------------------------------------------- #
# evaluate one (env, method) over the horizon sweep
# --------------------------------------------------------------------------- #
def evaluate_method(env_name: str, method_name: str, horizons, seed: int = 0, verbose: bool = True):
    reason = check_compatible(env_name, method_name)
    if reason is not None:
        raise ValueError(reason)

    espec = ENV_REGISTRY[env_name]
    mspec = METHOD_REGISTRY[method_name]
    horizons = [int(h) for h in horizons]
    init = espec.default_init_state
    T = espec.eval_T

    # resolve the policy prior (instance for planners, checkpoint path for sac)
    prior = None
    if method_name == "sac":
        prior = espec.prior_path
    elif "prior" in mspec.requires:
        from policy_prior_CEM import SACPrior
        prior = SACPrior(espec.prior_path)

    # resolve the terminal value function only for methods that ask for it
    terminal_value = None
    if "terminal_value" in mspec.requires and espec.terminal_value is not None:
        # build a throwaway env to construct the terminal-value callable
        terminal_value = espec.terminal_value(espec.make(seed))

    if not mspec.horizon_dependent:
        # single episode; result is horizon-independent (flat line)
        env = espec.make(seed)
        agent = mspec.build(env, horizons[0] if horizons else 1, seed, prior, terminal_value)
        reward, _ = run_episode(env, agent, init, T)
        n_prem = int(getattr(agent, "n_premature", 0))
        n_plan = int(getattr(agent, "n_plans", 0))
        if verbose:
            print(f"[{env_name}/{method_name}]  reward (T={T}): {reward:.4f}  (horizon-independent)")
        return dict(
            env=env_name, method=method_name,
            horizons=np.array(horizons, dtype=np.int64),
            rewards=np.array([reward], dtype=np.float64),
            premature_counts=np.array([n_prem], dtype=np.int64),
            planning_counts=np.array([n_plan], dtype=np.int64),
            horizon_dependent=np.array(False),
            seed=np.int64(seed), eval_T=np.int64(T),
            config_sig=config_sig(env_name, method_name, horizons, seed),
        )

    rewards = np.full(len(horizons), np.nan)
    # per-horizon (== per-episode) premature-convergence bookkeeping
    premature_counts = np.zeros(len(horizons), dtype=np.int64)
    planning_counts = np.zeros(len(horizons), dtype=np.int64)
    if verbose:
        print(f"[{env_name}/{method_name}]  sweeping H over {horizons[0]}..{horizons[-1]} (T={T})")
    for i, H in enumerate(horizons):
        env = espec.make(seed)
        agent = mspec.build(env, H, seed, prior, terminal_value)
        rewards[i], _ = run_episode(env, agent, init, T)
        premature_counts[i] = int(getattr(agent, "n_premature", 0))
        planning_counts[i] = int(getattr(agent, "n_plans", 0))
        if verbose:
            print(f"    H={H:>3}  reward={rewards[i]:12.4f}"
                  f"  premature={premature_counts[i]}/{planning_counts[i]}")

    return dict(
        env=env_name, method=method_name,
        horizons=np.array(horizons, dtype=np.int64),
        rewards=rewards,
        premature_counts=premature_counts,
        planning_counts=planning_counts,
        horizon_dependent=np.array(True),
        seed=np.int64(seed), eval_T=np.int64(T),
        config_sig=config_sig(env_name, method_name, horizons, seed),
    )


# --------------------------------------------------------------------------- #
# npz persistence
# --------------------------------------------------------------------------- #
RESULTS_DIR = "results"


def result_path(env_name: str, method_name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{env_name}__{method_name}.npz")


def save_result(result: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = result_path(result["env"], result["method"])
    np.savez(path, **result)
    return path


def load_result(env_name: str, method_name: str):
    """Load a cached result if present AND its config signature still matches
    the current setup; else return None."""
    path = result_path(env_name, method_name)
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def is_fresh(cached: dict | None, env_name: str, method_name: str, horizons, seed: int) -> bool:
    if cached is None or "config_sig" not in cached:
        return False
    # require the premature-convergence arrays too, so pre-instrumentation
    # caches are treated as stale and regenerated with the new bookkeeping.
    if "premature_counts" not in cached or "planning_counts" not in cached:
        return False
    want = config_sig(env_name, method_name, horizons, seed)
    got = cached["config_sig"]
    return got.shape == want.shape and np.allclose(got, want)
