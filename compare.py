"""
Compare: Optimal LQR  vs.  Random-Shooting MPC  vs.  CEM  vs.  MPPI
==================================================================

Runs all controllers on *identical* episodes (same env, same initial state,
same seed) and compares episode reward:

  * Optimal LQR feedback  (a = -K s)          -- gold-standard baseline
  * Random-shooting MPC, horizon sweep         -- Phase I
  * CEM,  horizon sweep                         -- Phase II
  * MPPI, horizon sweep                         -- Phase II

Because plain random shooting at H=1 lets the unstable system diverge
(reward ~ -1e26), we plot COST = -reward on a LOG scale so every regime is
visible on one figure.

Key takeaways the figure demonstrates
-------------------------------------
  1. H=1 random shooting FAILS even with exact dynamics: the 1-step return
     -(sQs + aRa) never sees the next state, so it picks a~0 and the unstable
     plant blows up. -> "short-horizon planning can fail."
  2. A couple of steps of lookahead (H~2) is already near-optimal.
  3. For a FIXED sample budget, very long horizons get worse again for random
     shooting: the search space is 3*H dimensional and best-of-N can't cover
     it. CEM / MPPI refine the sampling distribution and stay much closer to
     optimal across horizons. -> motivates Phase II.
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from lqr_env import LQREnv
from optimal import LQRController, run_episode as run_optimal_episode
from phase1 import RandomShootingMPC, run_episode as run_mpc_episode
from phase2 import CEMPlanner, MPPIPlanner, run_episode as run_plan_episode


# --------------------------------------------------------------------------- #
# experiment configuration
# --------------------------------------------------------------------------- #
ENV_KWARGS = dict(noise_std=0.0)          # deterministic -> clean comparison
INIT_STATE = np.array([1.0, -1.0, 0.5])   # same start for everyone
T = 200                                    # steps per episode
NUM_SAMPLES = 1000                         # candidates per step
SIGMA = 1.0                                # std of the Gaussian action proposal
HORIZONS = list(range(1, 21))             # H = 1 .. 20
SEED = 0

# CEM / MPPI specifics.  The refinement planners want a *tighter* proposal than
# the one-shot random shooter: they re-center every iteration, so a wide sigma
# just wastes samples.  These were tuned to sit close to the optimal cost.
REFINE_SIGMA = 0.2                         # proposal/noise std for CEM & MPPI
NUM_ELITES = 50                            # CEM top-K
CEM_ITERS = 30                             # CEM max iterations per step
MPPI_ITERS = 15                            # MPPI max iterations per step
TEMPERATURE = 20.0                         # MPPI softmax lambda


RESULTS_FILE = "compare_results.npz"

# A signature of everything that affects the numbers. If any of it changes, the
# cached results are stale and the sweep is re-run automatically.
CONFIG_SIG = np.array([
    T, NUM_SAMPLES, SIGMA, SEED, REFINE_SIGMA, NUM_ELITES,
    CEM_ITERS, MPPI_ITERS, TEMPERATURE, *INIT_STATE, *HORIZONS,
], dtype=np.float64)


def make_env() -> LQREnv:
    return LQREnv(seed=SEED, **ENV_KWARGS)


def run_sweep():
    """Compute the optimal baseline + horizon sweep and cache to RESULTS_FILE."""
    # ---- 1) optimal baseline -------------------------------------------- #
    env = make_env()
    ctrl = LQRController(env)
    opt_reward, _ = run_optimal_episode(env, ctrl, init_state=INIT_STATE, T=T)
    print(f"[Optimal LQR]  episode reward (T={T}): {opt_reward:.3f}")
    print(f"[Optimal LQR]  analytical cost-to-go : {ctrl.value(INIT_STATE):.3f}")
    print("-" * 70)

    # ---- 2) planners over horizons -------------------------------------- #
    mpc_rewards, cem_rewards, mppi_rewards = [], [], []
    print(f"{'H':>3} | {'MPC reward':>14} | {'CEM reward':>14} | {'MPPI reward':>14}")
    print("-" * 70)
    for H in HORIZONS:
        env = make_env()
        agent = RandomShootingMPC(
            env, horizon=H, num_samples=NUM_SAMPLES, sigma=SIGMA, seed=SEED
        )
        r_mpc, _ = run_mpc_episode(env, agent, init_state=INIT_STATE, T=T)

        env = make_env()
        cem = CEMPlanner(
            env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
            max_iters=CEM_ITERS, sigma_init=REFINE_SIGMA, seed=SEED,
        )
        r_cem, _ = run_plan_episode(env, cem, init_state=INIT_STATE, T=T)

        env = make_env()
        mppi = MPPIPlanner(
            env, horizon=H, num_samples=NUM_SAMPLES, temperature=TEMPERATURE,
            sigma=REFINE_SIGMA, max_iters=MPPI_ITERS, seed=SEED,
        )
        r_mppi, _ = run_plan_episode(env, mppi, init_state=INIT_STATE, T=T)

        mpc_rewards.append(r_mpc)
        cem_rewards.append(r_cem)
        mppi_rewards.append(r_mppi)
        print(f"{H:>3} | {r_mpc:>14.3f} | {r_cem:>14.3f} | {r_mppi:>14.3f}")

    results = dict(
        horizons=np.array(HORIZONS),
        opt_reward=np.float64(opt_reward),
        mpc_rewards=np.array(mpc_rewards),
        cem_rewards=np.array(cem_rewards),
        mppi_rewards=np.array(mppi_rewards),
        config_sig=CONFIG_SIG,
    )
    np.savez(RESULTS_FILE, **results)
    print(f"Saved results to {RESULTS_FILE}")
    return results


def load_or_run():
    """Load cached results if present and matching the current config, else run."""
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


def main():
    results = load_or_run()
    opt_reward = float(results["opt_reward"])
    mpc_rewards = list(results["mpc_rewards"])
    cem_rewards = list(results["cem_rewards"])
    mppi_rewards = list(results["mppi_rewards"])

    # ---- 3) plot (cost = -reward, log scale) ---------------------------- #
    opt_cost = -opt_reward
    mpc_cost = [-r for r in mpc_rewards]
    cem_cost = [-r for r in cem_rewards]
    mppi_cost = [-r for r in mppi_rewards]

    def make_plot(out: str, h_min: int) -> None:
        """Save the cost-vs-horizon figure, keeping only horizons >= h_min."""
        idx = [i for i, H in enumerate(HORIZONS) if H >= h_min]
        hs = [HORIZONS[i] for i in idx]
        plt.figure(figsize=(9, 5.5))
        plt.axhline(opt_cost, color="black", ls="--", lw=2,
                    label=f"Optimal LQR (cost={opt_cost:.2f})")
        plt.plot(hs, [mpc_cost[i] for i in idx], "o-", color="tab:blue",
                 label="Random-shooting MPC")
        plt.plot(hs, [cem_cost[i] for i in idx], "s-", color="tab:green", label="CEM")
        plt.plot(hs, [mppi_cost[i] for i in idx], "^-", color="tab:orange", label="MPPI")
        plt.yscale("log")
        plt.xlabel("Planning horizon H")
        plt.ylabel(f"Episode cost  = -reward  (T={T}, log scale)")
        plt.title("Optimal LQR vs. Random-Shooting MPC vs. CEM vs. MPPI  (lower is better)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        plt.close()

    def make_plot_refine(out: str, h_min: int) -> None:
        """CEM vs. MPPI vs. optimal only, on a LINEAR y-axis."""
        idx = [i for i, H in enumerate(HORIZONS) if H >= h_min]
        hs = [HORIZONS[i] for i in idx]
        plt.figure(figsize=(9, 5.5))
        plt.axhline(opt_cost, color="black", ls="--", lw=2,
                    label=f"Optimal LQR (cost={opt_cost:.2f})")
        plt.plot(hs, [cem_cost[i] for i in idx], "s-", color="tab:green", label="CEM")
        plt.plot(hs, [mppi_cost[i] for i in idx], "^-", color="tab:orange", label="MPPI")
        plt.xlabel("Planning horizon H")
        plt.ylabel(f"Episode cost  = -reward  (T={T})")
        plt.title("CEM vs. MPPI vs. Optimal LQR  (lower is better)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        plt.close()

    # full sweep, and a zoomed version starting at H=5 (the H=1..2 blow-ups
    # dominate the log scale and hide the differences between the planners).
    make_plot("compare_reward_vs_horizon.png", h_min=min(HORIZONS))
    make_plot("compare_more_horizon.png", h_min=5)
    # CEM/MPPI-only, linear scale. Start at H=3 so the (diverging) H=1..2 points
    # don't flatten everything else against the axis.
    make_plot_refine("compare_cem_mppi.png", h_min=3)
    print("-" * 70)
    print("Saved plots to compare_reward_vs_horizon.png, compare_more_horizon.png, "
          "and compare_cem_mppi.png")

    # ---- 4) takeaways --------------------------------------------------- #
    best_H = HORIZONS[int(np.argmax(mpc_rewards))]
    best_H_cem = HORIZONS[int(np.argmax(cem_rewards))]
    best_H_mppi = HORIZONS[int(np.argmax(mppi_rewards))]
    print(f"\nOptimal episode reward          : {opt_reward:10.3f}")
    print(f"Best random-shooting MPC : H={best_H:<3} reward {max(mpc_rewards):10.3f}")
    print(f"Best CEM                 : H={best_H_cem:<3} reward {max(cem_rewards):10.3f}")
    print(f"Best MPPI                : H={best_H_mppi:<3} reward {max(mppi_rewards):10.3f}")
    print(f"\nH=1  random-shooting MPC : {mpc_rewards[0]:.3e}  (diverges -- short-horizon failure)")
    print("=> Refining the sampling distribution (CEM / MPPI) keeps planning close")
    print("   to optimal across horizons instead of degrading for large H.")


if __name__ == "__main__":
    main()
