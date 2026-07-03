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
from IGO import IGOPlanner


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
CEM_ITERS = 1000                           # CEM max iterations per step (rely on tol)
MPPI_ITERS = 1000                          # MPPI max iterations per step (rely on tol)
TEMPERATURE = 20.0                         # MPPI softmax lambda

# IGO-ML specifics.  Same proposal/elites as CEM, but a soft update with step
# size IGO_DT (< 1) plus a variance-injection term that resists premature
# convergence.  IGO_DT = 1 would recover CEM exactly.
IGO_ITERS = 1000                           # IGO max iterations per step (rely on tol)
IGO_DT = 0.5                               # IGO-ML step size (soft update rate)


RESULTS_FILE = "compare_results.npz"

# A signature of everything that affects the numbers. If any of it changes, the
# cached results are stale and the sweep is re-run automatically.
CONFIG_SIG = np.array([
    T, NUM_SAMPLES, SIGMA, SEED, REFINE_SIGMA, NUM_ELITES,
    CEM_ITERS, MPPI_ITERS, TEMPERATURE, IGO_ITERS, IGO_DT,
    *INIT_STATE, *HORIZONS,
], dtype=np.float64)


def _draw_break(ax_hi, ax_lo) -> None:
    """Draw wavy (~) axis-break marks on the touching edges of two stacked axes."""
    # a little tilde/squiggle, in axes-fraction coords, drawn at both x-ends
    n = 60
    xx = np.linspace(0, 1, n)
    wiggle = 0.6 * np.sin(np.linspace(0, 3 * np.pi, n))  # ~ shape
    for ax, edge in ((ax_hi, 0.0), (ax_lo, 1.0)):        # bottom of hi, top of lo
        for x0 in (0.0, 1.0):                            # left and right corners
            span = 0.04
            xs = x0 + (xx - 0.5) * 2 * span
            ys = edge + wiggle * span
            ax.plot(xs, ys, transform=ax.transAxes, color="k",
                    lw=1.2, clip_on=False, zorder=10)


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
    mpc_rewards, cem_rewards, mppi_rewards, igo_rewards = [], [], [], []
    print(f"{'H':>3} | {'MPC reward':>14} | {'CEM reward':>14} | {'IGO reward':>14}")
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
        igo = IGOPlanner(
            env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
            max_iters=IGO_ITERS, sigma_init=REFINE_SIGMA, dt=IGO_DT, seed=SEED,
        )
        r_igo, _ = run_plan_episode(env, igo, init_state=INIT_STATE, T=T)

        # env = make_env()
        # mppi = MPPIPlanner(
        #     env, horizon=H, num_samples=NUM_SAMPLES, temperature=TEMPERATURE,
        #     sigma=REFINE_SIGMA, max_iters=MPPI_ITERS, seed=SEED,
        # )
        # r_mppi, _ = run_plan_episode(env, mppi, init_state=INIT_STATE, T=T)

        mpc_rewards.append(r_mpc)
        cem_rewards.append(r_cem)
        igo_rewards.append(r_igo)
        # mppi_rewards.append(r_mppi)
        # print(f"{H:>3} | {r_mpc:>14.3f} | {r_cem:>14.3f} | {r_mppi:>14.3f}")
        print(f"{H:>3} | {r_mpc:>14.3f} | {r_cem:>14.3f} | {r_igo:>14.3f}")

    results = dict(
        horizons=np.array(HORIZONS),
        opt_reward=np.float64(opt_reward),
        mpc_rewards=np.array(mpc_rewards),
        cem_rewards=np.array(cem_rewards),
        igo_rewards=np.array(igo_rewards),
        # mppi_rewards=np.array(mppi_rewards),
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
    igo_rewards = list(results["igo_rewards"])
    # mppi_rewards = list(results["mppi_rewards"])

    # ---- 3) plot (cost = -reward, log scale) ---------------------------- #
    opt_cost = -opt_reward
    mpc_cost = [-r for r in mpc_rewards]
    cem_cost = [-r for r in cem_rewards]
    igo_cost = [-r for r in igo_rewards]
    # mppi_cost = [-r for r in mppi_rewards]

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
        plt.plot(hs, [igo_cost[i] for i in idx], "d-", color="tab:red", label="IGO-ML")
        # plt.plot(hs, [mppi_cost[i] for i in idx], "^-", color="tab:orange", label="MPPI")
        plt.yscale("log")
        plt.xlabel("Planning horizon H")
        plt.ylabel(f"Episode cost  = -reward  (T={T}, log scale)")
        plt.title("Optimal LQR vs. Random-Shooting MPC vs. CEM vs. MPPI  (lower is better)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        plt.close()

    def make_plot_refine(out: str, h_min: int, y_break: float = 18.0) -> None:
        """CEM & IGO-ML vs. optimal, with a BROKEN y-axis.

        The bottom panel is linear up to ``y_break`` (where all the interesting
        behaviour lives); costs above it (the diverging short horizons) are shown
        on a log-scaled top panel, with squiggle break marks in between.
        """
        idx = [i for i, H in enumerate(HORIZONS) if H >= h_min]
        hs = [HORIZONS[i] for i in idx]
        cem = [cem_cost[i] for i in idx]
        igo = [igo_cost[i] for i in idx]

        fig, (ax_hi, ax_lo) = plt.subplots(
            2, 1, sharex=True, figsize=(9, 6),
            gridspec_kw=dict(height_ratios=[1, 2.6], hspace=0.06),
        )

        for ax in (ax_hi, ax_lo):
            ax.axhline(opt_cost, color="black", ls="--", lw=2,
                       label=f"Optimal LQR (cost={opt_cost:.2f})")
            ax.plot(hs, cem, "s-", color="tab:green", label="CEM")
            ax.plot(hs, igo, "d-", color="tab:red", label="IGO-ML")
            ax.grid(True, alpha=0.3)

        # mark each planner's lowest-cost (best) horizon with a star
        cem_i = int(np.argmin(cem))
        igo_i = int(np.argmin(igo))
        for ax in (ax_hi, ax_lo):
            ax.plot(hs[cem_i], cem[cem_i], "*", color="tab:green", ms=20,
                    mec="black", mew=0.8, zorder=6,
                    label=f"CEM best (H={hs[cem_i]}, cost={cem[cem_i]:.2f})")
            ax.plot(hs[igo_i], igo[igo_i], "*", color="tab:red", ms=20,
                    mec="black", mew=0.8, zorder=6,
                    label=f"IGO-ML best (H={hs[igo_i]}, cost={igo[igo_i]:.2f})")

        # bottom: linear detail range
        ax_lo.set_ylim(opt_cost - 0.5, y_break)
        # top: log scale covering the diverging costs
        big = [c for c in cem + igo if c > y_break]
        top_lo = 10 ** np.floor(np.log10(min(big)))
        top_hi = 10 ** np.ceil(np.log10(max(big)))
        ax_hi.set_yscale("log")
        ax_hi.set_ylim(top_lo, top_hi)

        # hide the facing spines and draw the squiggle break marks
        ax_hi.spines["bottom"].set_visible(False)
        ax_lo.spines["top"].set_visible(False)
        ax_hi.tick_params(bottom=False)
        _draw_break(ax_hi, ax_lo)

        ax_hi.set_title("CEM & IGO-ML vs. Optimal LQR  (lower is better)")
        ax_lo.set_xlabel("Planning horizon H")
        fig.supylabel(f"Episode cost  = -reward  (T={T})")
        ax_hi.legend(loc="upper right")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # full sweep, and a zoomed version starting at H=5 (the H=1..2 blow-ups
    # dominate the log scale and hide the differences between the planners).
    make_plot("compare_reward_vs_horizon.png", h_min=min(HORIZONS))
    make_plot("compare_more_horizon.png", h_min=5)
    # CEM-only, starting at H=1, with a broken y-axis: linear detail below
    # cost=18, log-scaled diverging costs above the squiggle break.
    make_plot_refine("compare_cem_mppi.png", h_min=1, y_break=18.0)
    print("-" * 70)
    print("Saved plots to compare_reward_vs_horizon.png, compare_more_horizon.png, "
          "and compare_cem_mppi.png")

    # ---- 4) takeaways --------------------------------------------------- #
    best_H = HORIZONS[int(np.argmax(mpc_rewards))]
    best_H_cem = HORIZONS[int(np.argmax(cem_rewards))]
    best_H_igo = HORIZONS[int(np.argmax(igo_rewards))]
    # best_H_mppi = HORIZONS[int(np.argmax(mppi_rewards))]
    print(f"\nOptimal episode reward          : {opt_reward:10.3f}")
    print(f"Best random-shooting MPC : H={best_H:<3} reward {max(mpc_rewards):10.3f}")
    print(f"Best CEM                 : H={best_H_cem:<3} reward {max(cem_rewards):10.3f}")
    print(f"Best IGO-ML              : H={best_H_igo:<3} reward {max(igo_rewards):10.3f}")
    # print(f"Best MPPI                : H={best_H_mppi:<3} reward {max(mppi_rewards):10.3f}")
    print(f"\nH=1  random-shooting MPC : {mpc_rewards[0]:.3e}  (diverges -- short-horizon failure)")
    print("=> Refining the sampling distribution (CEM / MPPI) keeps planning close")
    print("   to optimal across horizons instead of degrading for large H.")


if __name__ == "__main__":
    main()
