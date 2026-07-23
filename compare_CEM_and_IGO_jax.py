"""
Compare (JAX/GPU): Optimal LQR vs Random-Shooting MPC vs CEM vs IGO variants
============================================================================

JAX reconstruction of compare_CEM_and_IGO.py. Same experiment: identical env /
init state / seed, episode reward on a horizon sweep, COST = -reward on a log
scale. The planners' inner sample/rollout/refit loop now runs as one jitted GPU
kernel (see planners_jax.py); the receding-horizon loop steps the *NumPy* env so
the ground-truth dynamics (clip + optional noise) and warm-start are identical to
the reference.

Planner lines:
  * Optimal LQR feedback            -- optimal_jax.LQRController
  * Random-shooting MPC             -- planners_jax.build_random_shooting
  * CEM (dt=1, hard refit)          -- build_planner + refit_hard
  * IGO-ML (soft variance-inject)   -- build_planner + igo_variance_injection
  * IGO complete sample-based       -- build_planner + igo_weighted_mle (N,K scaled)
  * IGO SAC-penalized               -- sac_penalized_jax (needs sac_lqr_jax.pkl)

Per-timestep premature-convergence flags (fully-JAX shell probe, premature_jax)
are recorded for the CEM/IGO planners and written to
``premature_check_CEM_and_IGO_jax.csv``. NOTE: the JAX probe uses a uniform
i.i.d. shell sampler (not scipy Sobol), so these flags will NOT match the NumPy
CSV row-for-row -- this is expected (see premature_jax.py).

Each horizon recompiles the kernels once, then reuses them across all T steps.
"""

from __future__ import annotations

import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from tqdm import tqdm

from lqr_env import LQREnv
from lqr_env_jax import from_numpy_env
from optimal_jax import LQRController
from updates_jax import refit_hard, igo_variance_injection, igo_weighted_mle
from planners_jax import build_planner, build_random_shooting, nk_for_horizon
from premature_jax import make_premature_check

# ------------------------------------------------------------------------- #
# experiment configuration (mirrors compare_CEM_and_IGO.py)
# ------------------------------------------------------------------------- #
ENV_KWARGS = dict(noise_std=0.0)
INIT_STATE = np.array([1.0, -1.0, 0.5])
T = 200
NUM_SAMPLES = 1000
SIGMA = 1.0                # random-shooting proposal std
HORIZONS = list(range(2, 21))
SEED = 0

REFINE_SIGMA = 0.2         # CEM / IGO proposal std (reset each timestep)
NUM_ELITES = 250
N_ITERS = 100              # fixed iteration budget (JAX scan can't early-stop)
SACP_ITERS = 300            # SAC-penalized budget (per-step SAC forward is costly)
DT = 0.1                   # IGO step size (dt=1 -> CEM)
KAPPA = 0.8                # SAC-penalized random-shoot/SAC mix decay

SAC_CKPT = "sac_lqr_jax.pkl"     # converted/trained JAX actor
DETECT_PREMATURE = True
CSV_OUT = "premature_check_CEM_and_IGO_jax.csv"


def make_env() -> LQREnv:
    return LQREnv(seed=SEED, **ENV_KWARGS)


def rollout_mu(env, plan, H, dt, key, init_state, T, detect=None):
    """Receding-horizon rollout for a warm-started (mu, sigma) planner.

    ``plan(key, s0, mu, dt) -> (mu, sigma)``. Executes mu[0] in the NumPy env and
    shifts the plan forward. If ``detect`` (a jitted check(key, state, mu, sigma))
    is given, its per-step boolean flags are collected and returned."""
    adim = env.action_dim
    s = env.reset(state=init_state)
    mu = jnp.zeros((H, adim))
    total = 0.0
    flags = []
    for _ in range(T):
        key, sub = jax.random.split(key)
        mu, sigma = plan(sub, jnp.asarray(s), mu, dt)
        if detect is not None:
            key, dk = jax.random.split(key)
            flags.append(bool(detect(dk, jnp.asarray(s), mu, sigma)))
        s, r, term, trunc, _ = env.step(np.asarray(mu[0]))
        total += r
        mu = jnp.vstack([mu[1:], jnp.zeros((1, adim))])
        if term or trunc:
            break
    return total, flags


def rollout_shoot(env, plan, key, init_state, T):
    s = env.reset(state=init_state)
    total = 0.0
    for _ in range(T):
        key, sub = jax.random.split(key)
        a = plan(sub, jnp.asarray(s))
        s, r, term, trunc, _ = env.step(np.asarray(a))
        total += r
        if term or trunc:
            break
    return total


def run_sweep():
    env = make_env()
    params = from_numpy_env(env)
    ctrl = LQRController(params)
    s = env.reset(state=INIT_STATE)
    opt_reward = 0.0
    for _ in range(T):
        s, r, _, tr, _ = env.step(np.asarray(ctrl.act(s)))
        opt_reward += r
        if tr:
            break
    print(f"[Optimal LQR] reward (T={T}): {opt_reward:.3f} | value(s0): {ctrl.value(INIT_STATE):.3f}")

    has_sac = os.path.exists(SAC_CKPT)
    sac_actor = sac_scale = sac_bias = None
    if has_sac:
        from sac_jax import load_ckpt
        sac_actor, sac_cfg, _ = load_ckpt(SAC_CKPT)
        sac_scale, sac_bias = sac_cfg["action_scale"], sac_cfg["action_bias"]
    else:
        print(f"[IGO SAC-penalized] {SAC_CKPT} not found -> SACP column NaN.")
    if has_sac:
        from sac_penalized_jax import build_sac_penalized_planner
    print("-" * 92)

    mpc_r, cem_r, igo_r, cs_r, sacp_r = [], [], [], [], []
    premature_rows = []
    print(f"{'H':>3} | {'MPC':>11} | {'CEM':>10} | {'IGO-ML':>10} | {'IGO-CS':>10} | "
          f"{'SACP':>10} | {'premature igo/cem/cs/sacp':>26}")
    print("-" * 92)
    base_key = jax.random.PRNGKey(SEED)
    for H in tqdm(HORIZONS, desc="horizon sweep"):
        env = make_env(); params = from_numpy_env(env)
        det = make_premature_check(params, H) if DETECT_PREMATURE else None

        shoot = build_random_shooting(params, H, NUM_SAMPLES, gamma=1.0, sigma=SIGMA)
        r_mpc = rollout_shoot(env, shoot, base_key, INIT_STATE, T)

        cem = build_planner(params, H, NUM_SAMPLES, NUM_ELITES, N_ITERS, 1.0, refit_hard, REFINE_SIGMA)
        r_cem, cem_f = rollout_mu(env, cem, H, 1.0, base_key, INIT_STATE, T, det)

        igo = build_planner(params, H, NUM_SAMPLES, NUM_ELITES, N_ITERS, 1.0,
                            igo_variance_injection, REFINE_SIGMA)
        r_igo, igo_f = rollout_mu(env, igo, H, DT, base_key, INIT_STATE, T, det)

        Ncs, Kcs = nk_for_horizon(H, NUM_SAMPLES, NUM_ELITES)
        cs = build_planner(params, H, Ncs, Kcs, N_ITERS, 1.0, igo_weighted_mle, REFINE_SIGMA)
        r_cs, cs_f = rollout_mu(env, cs, H, DT, base_key, INIT_STATE, T, det)

        if has_sac:
            sacp = build_sac_penalized_planner(params, sac_actor, sac_scale, sac_bias, H,
                                               Ncs, Kcs, SACP_ITERS, 1.0, KAPPA, REFINE_SIGMA)
            r_sacp, sacp_f = rollout_mu(env, sacp, H, DT, base_key, INIT_STATE, T, det)
        else:
            r_sacp, sacp_f = np.nan, []

        for name, flags in [("igo", igo_f), ("cem", cem_f), ("igo-cs", cs_f), ("igo-sacp", sacp_f)]:
            for t, f in enumerate(flags):
                premature_rows.append((name, H, t, int(f)))

        mpc_r.append(r_mpc); cem_r.append(r_cem); igo_r.append(r_igo); cs_r.append(r_cs); sacp_r.append(r_sacp)
        n = len(igo_f)
        sp = f"{sum(sacp_f)}/{n}" if has_sac else "N/A"
        tqdm.write(f"{H:>3} | {r_mpc:>11.2f} | {r_cem:>10.3f} | {r_igo:>10.3f} | {r_cs:>10.3f} | "
                   f"{r_sacp:>10.3f} | {f'{sum(igo_f)}/{n},{sum(cem_f)}/{n},{sum(cs_f)}/{n},{sp}':>26}")

    if DETECT_PREMATURE:
        with open(CSV_OUT, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["planner", "H", "timestep", "premature"]); w.writerows(premature_rows)
        print(f"Saved {len(premature_rows)} rows to {CSV_OUT}")

    return dict(horizons=np.array(HORIZONS), opt_reward=float(opt_reward),
                mpc=np.array(mpc_r), cem=np.array(cem_r), igo=np.array(igo_r),
                cs=np.array(cs_r), sacp=np.array(sacp_r))


def make_plot(results, out, h_min):
    hs_all = list(results["horizons"])
    idx = [i for i, H in enumerate(hs_all) if H >= h_min]
    hs = [hs_all[i] for i in idx]
    opt_cost = -results["opt_reward"]
    plt.figure(figsize=(9, 5.5))
    plt.axhline(opt_cost, color="black", ls="--", lw=2, label=f"Optimal LQR (cost={opt_cost:.2f})")
    lines = [
        ("mpc", "o-", "tab:blue", "Random-shooting MPC"),
        ("igo", "s-", "tab:green", "IGO-ML"),
        ("cem", "^-", "tab:orange", "CEM (dt=1)"),
        ("cs", "d-", "tab:purple", "IGO complete sample-based"),
        ("sacp", "v-", "tab:brown", "IGO SAC-penalized"),
    ]
    for key, style, color, lab in lines:
        vals = results[key]
        if vals is None or np.all(np.isnan(vals)):
            continue
        cost = [-vals[i] for i in idx]
        plt.plot(hs, cost, style, color=color, label=lab)
    plt.yscale("log")
    plt.xlabel("Planning horizon H")
    plt.ylabel(f"Episode cost = -reward (T={T}, log scale)")
    plt.title("JAX: Optimal LQR vs Random-Shooting MPC vs CEM vs IGO (lower is better)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close()
    print(f"saved {out}")


def main():
    results = run_sweep()
    make_plot(results, "compare_CEM_and_IGO_jax_reward_vs_horizon.png", h_min=min(HORIZONS))
    make_plot(results, "compare_CEM_and_IGO_jax_more_horizon.png", h_min=5)
    print("-" * 92)
    for key, lab in [("mpc", "Random-shooting MPC"), ("cem", "CEM (dt=1)"),
                     ("igo", "IGO-ML"), ("cs", "IGO complete sample-based"),
                     ("sacp", "IGO SAC-penalized")]:
        vals = results[key]
        if vals is None or np.all(np.isnan(vals)):
            continue
        bi = int(np.nanargmax(vals))
        print(f"Best {lab:28s}: H={HORIZONS[bi]:<3} reward {vals[bi]:10.3f}")
    print(f"Optimal reward: {results['opt_reward']:.3f}")


if __name__ == "__main__":
    main()
