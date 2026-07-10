"""
Environment and method registries
==================================

Central catalogue used by both ``evaluate.py`` (run one env x one method) and
``compare.py`` (compare several methods on one env). Two flat dictionaries:

  * ENV_REGISTRY[name]    -> EnvSpec   (how to build the env + its metadata)
  * METHOD_REGISTRY[name] -> MethodSpec (how to build the planner + metadata)

Design choices (per the reconstruction spec):

  * CLI-selected by name; one npz caches one (env, method) sweep.
  * Optimal-LQR and SAC are first-class *methods* (horizon-independent), not
    hard-wired baselines.
  * Every hyperparameter variant is its OWN method name (e.g. ``pp-large`` vs
    ``pp-conservative``, ``tdmpc`` vs ``tdmpc-noprior``) -- one name = one fully
    specified config.
  * torch is imported lazily inside the factories that need it, so value-free
    planners on the harder (LQR-free) envs run without the deep-RL stack.

Compatibility: a method declares ``requires`` (a subset of {"prior",
"optimal_ctrl", "terminal_value"}); ``check_compatible(env, method)`` validates
the pair against the env's advertised capabilities and returns a reason string
if unsupported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from lqr_env import LQREnv
from envs import ConstrainedLQREnv, SwitchedLinearEnv, NonQuadraticTerminalLQREnv


# --------------------------------------------------------------------------- #
# environment specs
# --------------------------------------------------------------------------- #
@dataclass
class EnvSpec:
    make: Callable[[int], object]          # make(seed) -> env instance (model = real env)
    label: str
    eval_T: int                            # episode length used for evaluation
    default_init_state: np.ndarray | None  # fixed start state (None -> random per seed)
    prior_path: str | None = None          # SAC checkpoint for policy-prior methods
    has_optimal: bool = False              # analytical optimal controller available
    terminal_value: Callable[[object], Callable] | None = None  # env -> V(states) callable


def _lqr_terminal_value(env):
    """Analytical optimal LQR terminal value V(s) = -s^T P s."""
    from optimal import LQRController
    P = LQRController(env).P
    return lambda states: -np.einsum("ni,ij,nj->n", np.atleast_2d(states), P,
                                     np.atleast_2d(states))


ENV_REGISTRY: dict[str, EnvSpec] = {
    "lqr": EnvSpec(
        make=lambda seed: LQREnv(noise_std=0.0, seed=seed),
        label="LQR",
        eval_T=200,
        default_init_state=np.array([1.0, -1.0, 0.5]),
        prior_path="sac_lqr.pt",
        has_optimal=True,
        terminal_value=_lqr_terminal_value,
    ),
    "clqr": EnvSpec(
        make=lambda seed: ConstrainedLQREnv(noise_std=0.0, seed=seed),
        label="Constrained LQR (sat + rate)",
        eval_T=200,
        default_init_state=np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0]),
        prior_path="sac_clqr.pt",          # produced by train_prior.py --env clqr
        has_optimal=False,
        terminal_value=None,
    ),
    "switched": EnvSpec(
        make=lambda seed: SwitchedLinearEnv(noise_std=0.0, seed=seed),
        label="Switched linear",
        eval_T=200,
        default_init_state=np.array([1.0, -1.0, 0.5]),
        prior_path="sac_switched.pt",      # produced by train_prior.py --env switched
        has_optimal=False,
        terminal_value=None,
    ),
    "terminal": EnvSpec(
        make=lambda seed: NonQuadraticTerminalLQREnv(noise_std=0.0, seed=seed),
        label="Linear + Huber terminal",
        eval_T=30,                          # == task_horizon
        default_init_state=np.array([1.0, -1.0, 0.5]),
        prior_path="sac_terminal.pt",      # produced by train_prior.py --env terminal
        has_optimal=False,
        # the env's own non-quadratic tail cost is the terminal value
        terminal_value=lambda env: env.terminal_value,
    ),
}


# --------------------------------------------------------------------------- #
# method specs
# --------------------------------------------------------------------------- #
@dataclass
class MethodSpec:
    # build(env, horizon, seed, prior, terminal_value) -> agent with .act(state)
    build: Callable
    label: str
    color: str
    marker: str
    horizon_dependent: bool = True
    requires: frozenset = field(default_factory=frozenset)   # {"prior","optimal_ctrl","terminal_value"}


# ---- baselines ------------------------------------------------------------ #
def _build_optimal(env, horizon, seed, prior, terminal_value):
    from optimal import LQRController
    return LQRController(env)


def _build_sac(env, horizon, seed, prior, terminal_value):
    # deterministic (mean-action) wrapper around the trained SAC actor
    import torch
    from sac_lqr import GaussianPolicy, STATE_DIM, ACTION_DIM

    class _SAC:
        def __init__(self, path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            sdim = cfg.get("state_dim", STATE_DIM)
            adim = cfg.get("action_dim", ACTION_DIM)
            self.actor = GaussianPolicy(sdim, adim, tuple(cfg["hidden"]),
                                        cfg["action_low"], cfg["action_high"])
            self.actor.load_state_dict(ckpt["actor"])
            self.actor.eval()

        def act(self, state):
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, _, mean_action = self.actor.sample(s)
            return mean_action.squeeze(0).numpy()

    return _SAC(prior)          # `prior` carries the checkpoint path for sac


# ---- planners ------------------------------------------------------------- #
# NB: ``max_iters`` is intentionally NOT passed to any iterative planner below.
# The CEM-family class defaults are ``int(1e10)`` so planning runs until it
# converges (via tol_mu / tol_sigma), rather than being capped by a value
# hard-coded here. Keep it that way -- don't reintroduce a max_iters kwarg.
# (MPPI is the exception: its fixed scalar sigma never collapses, so its class
# default keeps a finite 1000-iteration cap -- see phase2.MPPIPlanner.)
def _build_random_shooting(env, horizon, seed, prior, terminal_value):
    from phase1 import RandomShootingMPC
    return RandomShootingMPC(env, horizon=horizon, num_samples=1000, sigma=1.0, seed=seed)


def _build_cem(env, horizon, seed, prior, terminal_value):
    from phase2 import CEMPlanner
    return CEMPlanner(env, horizon=horizon, num_samples=1000, num_elites=50,
                      sigma_init=0.2, seed=seed)


def _build_mppi(env, horizon, seed, prior, terminal_value):
    from phase2 import MPPIPlanner
    return MPPIPlanner(env, horizon=horizon, num_samples=1000, temperature=20.0,
                       sigma=0.2, seed=seed)


def _build_pp_large(env, horizon, seed, prior, terminal_value):
    from policy_prior_CEM import CEMPlanner as PPCEM
    return PPCEM(env, horizon=horizon, num_samples=1000, num_elites=50,
                 prior_std_scale=5.0, prior=prior, seed=seed)


def _build_pp_conservative(env, horizon, seed, prior, terminal_value):
    from policy_prior_CEM import CEMPlanner as PPCEM
    return PPCEM(env, horizon=horizon, num_samples=1000, num_elites=50,
                 prior_std_scale=1.0 / (2.0 * horizon), prior=prior, seed=seed)


def _build_pp_random_shooting(env, horizon, seed, prior, terminal_value):
    from policy_prior_random_shooting import RandomShootingPlanner
    return RandomShootingPlanner(env, horizon=horizon, num_samples=100000, prior=prior, seed=seed)


def _build_tdmpc(env, horizon, seed, prior, terminal_value):
    from tdmpc_planning import TDMPCPlanner
    return TDMPCPlanner(env, horizon=horizon, num_samples=512, num_policy_samples=25,
                        num_elites=64, sigma_init=2.0,
                        temperature=20.0, terminal_value=terminal_value, prior=prior, seed=seed)


def _build_tdmpc_noprior(env, horizon, seed, prior, terminal_value):
    from tdmpc_planning import TDMPCPlanner
    return TDMPCPlanner(env, horizon=horizon, num_samples=512, num_policy_samples=0,
                        num_elites=64, sigma_init=2.0,
                        temperature=20.0, terminal_value=terminal_value, prior=None, seed=seed)


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "optimal": MethodSpec(_build_optimal, "Optimal LQR", "black", "",
                          horizon_dependent=False, requires=frozenset({"optimal_ctrl"})),
    "sac": MethodSpec(_build_sac, "SAC", "tab:red", "",
                      horizon_dependent=False, requires=frozenset({"prior"})),
    "random-shooting": MethodSpec(_build_random_shooting, "Random-shooting MPC", "tab:blue", "o"),
    "cem": MethodSpec(_build_cem, "CEM", "tab:green", "s"),
    "mppi": MethodSpec(_build_mppi, "MPPI", "tab:orange", "^"),
    "pp-large": MethodSpec(_build_pp_large, "Large-explore prior CEM", "tab:purple", "d",
                           requires=frozenset({"prior"})),
    "pp-conservative": MethodSpec(_build_pp_conservative, "Conservative prior CEM", "tab:olive", "P",
                                  requires=frozenset({"prior"})),
    "pp-random-shooting": MethodSpec(_build_pp_random_shooting, "Policy-prior random shooting",
                                     "tab:brown", "v", requires=frozenset({"prior"})),
    "tdmpc": MethodSpec(_build_tdmpc, "TD-MPC (with prior)", "tab:cyan", "X",
                        requires=frozenset({"prior"})),
    "tdmpc-noprior": MethodSpec(_build_tdmpc_noprior, "TD-MPC (no prior)", "tab:pink", "^"),
    # TD-MPC using the env's terminal value function (needs an env that has one)
    "tdmpc-vf": MethodSpec(_build_tdmpc_noprior, "TD-MPC (+ terminal value)", "tab:gray", "*",
                           requires=frozenset({"terminal_value"})),
}


# --------------------------------------------------------------------------- #
# compatibility
# --------------------------------------------------------------------------- #
def check_compatible(env_name: str, method_name: str) -> str | None:
    """Return None if the (env, method) pair is runnable, else a reason string."""
    if env_name not in ENV_REGISTRY:
        return f"unknown env {env_name!r} (choices: {', '.join(ENV_REGISTRY)})"
    if method_name not in METHOD_REGISTRY:
        return f"unknown method {method_name!r} (choices: {', '.join(METHOD_REGISTRY)})"
    espec, mspec = ENV_REGISTRY[env_name], METHOD_REGISTRY[method_name]
    if "prior" in mspec.requires:
        if espec.prior_path is None:
            return f"method {method_name!r} needs a policy prior, but env {env_name!r} defines none"
        if not os.path.exists(espec.prior_path):
            return (f"method {method_name!r} needs prior {espec.prior_path!r} for env "
                    f"{env_name!r} -- train it first (train_prior.py / sac_lqr.py)")
    if "optimal_ctrl" in mspec.requires and not espec.has_optimal:
        return f"method {method_name!r} needs an analytical optimum, unavailable for env {env_name!r}"
    if "terminal_value" in mspec.requires and espec.terminal_value is None:
        return f"method {method_name!r} needs a terminal value function, undefined for env {env_name!r}"
    return None


def compatible_methods(env_name: str) -> list[str]:
    """All method names runnable on ``env_name``."""
    return [m for m in METHOD_REGISTRY if check_compatible(env_name, m) is None]
