"""
IGO-ML dt sweep
===============

Sweeps the IGO-ML step size ``dt`` over a grid and measures episode reward vs.
planning horizon for the three IGO planners used in ``compare.py``:

  * plain IGO-ML                 (warm-started shifted plan)     -- IGO.py
  * large-explore prior IGO      (SAC seed, std * PP_STD_SCALE)  -- policy_prior_IGO.py
  * conservative prior IGO       (SAC seed, std / (2H))          -- policy_prior_IGO.py

The dt-INDEPENDENT controllers are computed once and drawn as reference lines:

  * Optimal LQR feedback            (a = -K s)
  * trained SAC policy              (reactive, no horizon)
  * random-shooting MPC             (Phase I)
  * policy-prior random shooting    (SAC seed, one-shot best-of-N)

With ``dt = 1`` every IGO planner reduces exactly to its CEM counterpart, so the
``dt = 1`` curve is the "plain CEM" reference inside each panel.

Reuses the tuned constants and planner classes from ``compare.py`` so the sweep
stays in lock-step with the main comparison. Results are cached to
``sweep_dt_results.npz`` (keyed by a config signature); re-running only re-plots
unless the configuration changed.
"""

from __future__ import annotations

import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse everything tuned in compare.py so the sweep matches the main comparison
import compare
from compare import (
    make_env, SACPolicy,
    INIT_STATE, T, NUM_SAMPLES, SIGMA, SEED,
    REFINE_SIGMA, NUM_ELITES, IGO_ITERS, PP_STD_SCALE, SAC_MODEL_FILE,
)
from optimal import LQRController, run_episode as run_optimal_episode
from phase1 import RandomShootingMPC, run_episode as run_mpc_episode
from phase2 import run_episode as run_plan_episode
from IGO import IGOPlanner
from policy_prior_IGO import IGOPlanner as PolicyPriorIGO, SACPrior
from policy_prior_random_shooting import RandomShootingPlanner as PolicyPriorRS


# --------------------------------------------------------------------------- #
# sweep configuration
# --------------------------------------------------------------------------- #
DTS = [0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]     # IGO-ML step sizes to sweep
HSWEEP = [3, 5, 8, 12, 16, 20]                  # representative horizons

RESULTS_FILE = "sweep_dt_results.npz"

CONFIG_SIG = np.array([
    T, NUM_SAMPLES, SIGMA, SEED, REFINE_SIGMA, NUM_ELITES, IGO_ITERS,
    PP_STD_SCALE, *INIT_STATE, *HSWEEP, *DTS,
], dtype=np.float64)


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def run_sweep():
    nD, nH = len(DTS), len(HSWEEP)

    # ---- dt-independent references (computed once) ---------------------- #
    env = make_env()
    ctrl = LQRController(env)
    opt_reward, _ = run_optimal_episode(env, ctrl, init_state=INIT_STATE, T=T)
    print(f"[Optimal LQR] reward (T={T}): {opt_reward:.3f}")

    sac_reward = np.nan
    if os.path.exists(SAC_MODEL_FILE):
        sac = SACPolicy(SAC_MODEL_FILE)
        sac_reward, _ = run_optimal_episode(make_env(), sac, init_state=INIT_STATE, T=T)
        print(f"[SAC]         reward (T={T}): {sac_reward:.3f}")

    prior = SACPrior(SAC_MODEL_FILE) if os.path.exists(SAC_MODEL_FILE) else None
    if prior is None:
        print(f"[warn] {SAC_MODEL_FILE} not found -> policy-prior columns will be NaN.")

    mpc_rewards = np.full(nH, np.nan)
    pprs_rewards = np.full(nH, np.nan)
    print("\n-- dt-independent references over horizons --")
    for j, H in enumerate(HSWEEP):
        env = make_env()
        mpc = RandomShootingMPC(env, horizon=H, num_samples=NUM_SAMPLES, sigma=SIGMA, seed=SEED)
        mpc_rewards[j], _ = run_mpc_episode(env, mpc, init_state=INIT_STATE, T=T)
        if prior is not None:
            env = make_env()
            pprs = PolicyPriorRS(env, horizon=H, num_samples=NUM_SAMPLES * 100, prior=prior, seed=SEED)
            pprs_rewards[j], _ = run_plan_episode(env, pprs, init_state=INIT_STATE, T=T)
        print(f"  H={H:>2} | MPC {mpc_rewards[j]:>12.3f} | PP-RandShoot {pprs_rewards[j]:>12.3f}")

    # ---- dt-dependent IGO planners ------------------------------------- #
    igo = np.full((nD, nH), np.nan)     # plain IGO
    pp = np.full((nD, nH), np.nan)      # large-explore prior IGO
    ppc = np.full((nD, nH), np.nan)     # conservative prior IGO

    for i, dt in enumerate(DTS):
        t0 = time.time()
        print(f"\n== dt = {dt:g}  ({i + 1}/{nD}) ==")
        print(f"{'H':>3} | {'plain IGO':>12} | {'PP-large':>12} | {'PP-consv':>12}")
        for j, H in enumerate(HSWEEP):
            env = make_env()
            p = IGOPlanner(
                env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
                max_iters=IGO_ITERS, sigma_init=REFINE_SIGMA, dt=dt, seed=SEED,
            )
            igo[i, j], _ = run_plan_episode(env, p, init_state=INIT_STATE, T=T)

            if prior is not None:
                env = make_env()
                pl = PolicyPriorIGO(
                    env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
                    max_iters=IGO_ITERS, prior_std_scale=PP_STD_SCALE, dt=dt,
                    prior=prior, seed=SEED,
                )
                pp[i, j], _ = run_plan_episode(env, pl, init_state=INIT_STATE, T=T)

                env = make_env()
                pc = PolicyPriorIGO(
                    env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
                    max_iters=IGO_ITERS, prior_std_scale=1.0 / (2.0 * H), dt=dt,
                    prior=prior, seed=SEED,
                )
                ppc[i, j], _ = run_plan_episode(env, pc, init_state=INIT_STATE, T=T)
            print(f"{H:>3} | {igo[i, j]:>12.3f} | {pp[i, j]:>12.3f} | {ppc[i, j]:>12.3f}")
        print(f"   dt={dt:g} done in {time.time() - t0:.1f}s")

    results = dict(
        dts=np.array(DTS), horizons=np.array(HSWEEP),
        opt_reward=np.float64(opt_reward), sac_reward=np.float64(sac_reward),
        mpc_rewards=mpc_rewards, pprs_rewards=pprs_rewards,
        igo_rewards=igo, pp_rewards=pp, ppc_rewards=ppc,
        config_sig=CONFIG_SIG,
    )
    np.savez(RESULTS_FILE, **results)
    print(f"\nSaved results to {RESULTS_FILE}")
    return results


def load_or_run():
    if os.path.exists(RESULTS_FILE):
        data = np.load(RESULTS_FILE)
        if (
            "config_sig" in data
            and data["config_sig"].shape == CONFIG_SIG.shape
            and np.allclose(data["config_sig"], CONFIG_SIG)
        ):
            print(f"Loaded cached results from {RESULTS_FILE} (config matches).")
            return {k: data[k] for k in data.files}
        print(f"Cached results in {RESULTS_FILE} are stale -> re-running sweep.")
    return run_sweep()


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _dt_colors(dts):
    cmap = plt.get_cmap("viridis")
    return [cmap(k) for k in np.linspace(0.0, 0.9, len(dts))]


def make_plots(results):
    dts = [float(d) for d in results["dts"]]
    hs = [int(h) for h in results["horizons"]]
    opt_cost = -float(results["opt_reward"])
    sac_reward = float(results["sac_reward"])
    sac_cost = -sac_reward if np.isfinite(sac_reward) else None
    mpc_cost = -results["mpc_rewards"]
    pprs = results["pprs_rewards"]
    pprs_cost = -pprs if not np.all(np.isnan(pprs)) else None

    panels = [
        ("igo_rewards", "Plain IGO-ML", mpc_cost, "Random-shooting MPC", "tab:blue"),
        ("pp_rewards", "Large-explore prior IGO (SAC)", pprs_cost,
         "Policy-prior random shooting", "tab:brown"),
        ("ppc_rewards", "Conservative prior IGO (SAC)", pprs_cost,
         "Policy-prior random shooting", "tab:brown"),
    ]
    colors = _dt_colors(dts)

    # ---- figure 1: cost vs horizon, one panel per variant, dt as colour -- #
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (key, title, ref_cost, ref_label, ref_color) in zip(axes, panels):
        rew = results[key]                     # (nD, nH)
        if np.all(np.isnan(rew)):
            ax.set_title(f"{title}\n(no data)")
            continue
        cost = -rew
        for i, dt in enumerate(dts):
            ax.plot(hs, cost[i], "o-", color=colors[i], label=f"dt={dt:g}")
        ax.axhline(opt_cost, color="black", ls="--", lw=2, label=f"Optimal ({opt_cost:.2f})")
        if sac_cost is not None:
            ax.axhline(sac_cost, color="tab:red", ls=":", lw=1.8, label=f"SAC ({sac_cost:.2f})")
        if ref_cost is not None:
            ax.plot(hs, ref_cost, "x--", color=ref_color, lw=1.5, alpha=0.8, label=ref_label)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Planning horizon H")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel(f"Episode cost = -reward  (T={T}, log scale)")
    fig.suptitle("IGO-ML dt sweep: episode cost vs. horizon  (dt=1 -> CEM)")
    fig.tight_layout()
    fig.savefig("sweep_dt_cost_vs_horizon.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: mean cost over horizons vs dt -------------------------- #
    plt.figure(figsize=(9, 5.5))
    for key, title, *_ in panels:
        rew = results[key]
        if np.all(np.isnan(rew)):
            continue
        mean_cost = np.nanmean(-rew, axis=1)   # average over horizons, per dt
        plt.plot(dts, mean_cost, "o-", label=title)
    plt.axhline(opt_cost, color="black", ls="--", lw=2, label=f"Optimal ({opt_cost:.2f})")
    plt.yscale("log")
    plt.xlabel("IGO-ML step size dt   (dt=1 -> CEM)")
    plt.ylabel(f"Mean episode cost over H={hs}  (log scale)")
    plt.title("Effect of dt on IGO-ML (lower is better)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sweep_dt_mean_cost.png", dpi=130)
    plt.close()

    print("Saved plots: sweep_dt_cost_vs_horizon.png, sweep_dt_mean_cost.png")


def print_tables(results):
    dts = list(results["dts"])
    hs = list(results["horizons"])
    for key, title in [("igo_rewards", "Plain IGO-ML"),
                       ("pp_rewards", "Large-explore prior IGO"),
                       ("ppc_rewards", "Conservative prior IGO")]:
        rew = results[key]
        if np.all(np.isnan(rew)):
            continue
        print(f"\n== {title}: reward (rows=dt, cols=H) ==")
        header = "  dt\\H |" + "".join(f"{H:>10}" for H in hs) + f"{'best H':>12}"
        print(header)
        print("-" * len(header))
        for i, dt in enumerate(dts):
            row = "".join(f"{rew[i, j]:>10.3f}" for j in range(len(hs)))
            bestH = hs[int(np.nanargmax(rew[i]))]
            print(f"{dt:>6g} |{row}{bestH:>12}")
        # best dt by mean-over-horizons reward
        mean_r = np.nanmean(rew, axis=1)
        best_i = int(np.nanargmax(mean_r))
        print(f"  -> best dt by mean reward: dt={dts[best_i]:g}  (mean {mean_r[best_i]:.3f})")


def main():
    results = load_or_run()
    make_plots(results)
    print_tables(results)


if __name__ == "__main__":
    main()
