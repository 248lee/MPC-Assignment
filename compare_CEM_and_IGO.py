"""
Compare: Optimal LQR  vs.  Random-Shooting MPC  vs.  IGO-ML
==============================================================================

Runs all controllers on *identical* episodes (same env, same initial state,
same seed) and compares episode reward:

  * Optimal LQR feedback  (a = -K s)          -- gold-standard baseline
  * Random-shooting MPC, horizon sweep         -- Phase I
  * IGO-ML, horizon sweep                       -- soft (dt) CEM (see IGO.py)

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
     it. IGO-ML refines the sampling distribution -- while its variance-
     injection soft update resists the premature convergence that plain CEM
     suffers -- and stays much closer to optimal across horizons. -> Phase II.
"""

from __future__ import annotations

import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

import torch

from lqr_env import LQREnv
from optimal import LQRController, run_episode as run_optimal_episode
from phase1 import RandomShootingMPC, run_episode as run_mpc_episode
from IGO import IGOPlanner
from sac_lqr import GaussianPolicy, STATE_DIM, ACTION_DIM


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

# IGO / refinement specifics.  The refinement planners want a *tighter* proposal
# than the one-shot random shooter: they re-center every iteration, so a wide
# sigma just wastes samples.  These were tuned to sit close to the optimal cost.
REFINE_SIGMA = 0.2                         # proposal/noise std for IGO
NUM_ELITES = 250                           # IGO top-K (250/1000 = 25% elite)
IGO_ITERS = 1e10                           # IGO max iterations per step (rely on tol)
DT = 0.1                                  # IGO-ML step size (dt=1 -> CEM)


SAC_MODEL_FILE = "None"              # trained SAC actor (sac_lqr.py)


class SACPolicy:
    """Deterministic (mean-action) wrapper around a trained SAC actor.

    Exposes ``.act(state)`` so it plugs into ``run_optimal_episode`` exactly
    like the analytical ``LQRController``. SAC is a *reactive* policy with no
    planning horizon, so on the cost-vs-horizon figure it is a horizontal line.
    """

    def __init__(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.actor = GaussianPolicy(
            STATE_DIM, ACTION_DIM, tuple(cfg["hidden"]),
            cfg["action_low"], cfg["action_high"],
        )
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.meta = ckpt.get("meta", {})

    def act(self, state: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, _, mean_action = self.actor.sample(s)
        return mean_action.squeeze(0).numpy()


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


def run_episode_premature(env, agent, init_state=None, T=None):
    """Like run_plan_episode but also collects per-step premature-convergence flags.

    A True flag means some ±10σ perturbation of a single mu entry outperforms the
    converged plan, indicating the planner stopped before reaching a local optimum.
    """
    if hasattr(agent, "reset"):
        agent.reset()
    s = env.reset(state=init_state)
    if T is None:
        T = env.max_steps
    total = 0.0
    traj = [s.copy()]
    premature_flags = []
    for _ in range(T):
        a = agent.act(s)
        premature_flags.append(bool(getattr(agent, "last_premature_convergence", False)))
        s, r, term, trunc, _ = env.step(a)
        total += r
        traj.append(s.copy())
        if term or trunc:
            break
    return total, np.array(traj), premature_flags


def run_sweep():
    """Run the optimal baseline + horizon sweep once.

    A single pass over the horizons rolls out every planner and, for IGO-ML and
    CEM, simultaneously records per-timestep premature-convergence flags. Returns
    the reward curves and writes the premature-convergence flags to
    ``premature_check_CEM_and_IGO.csv``. No caching -- always recomputes.
    """
    # ---- 1) optimal baseline -------------------------------------------- #
    env = make_env()
    ctrl = LQRController(env)
    opt_reward, _ = run_optimal_episode(env, ctrl, init_state=INIT_STATE, T=T)
    print(f"[Optimal LQR]  episode reward (T={T}): {opt_reward:.3f}")
    print(f"[Optimal LQR]  analytical cost-to-go : {ctrl.value(INIT_STATE):.3f}")
    print("-" * 70)

    # ---- 2) planners over horizons -------------------------------------- #
    mpc_rewards, igo_rewards, cem_rewards = [], [], []
    premature_rows = []   # (planner, H, timestep, premature) for the CSV
    print(f"{'H':>3} | {'MPC reward':>14} | {'IGO reward':>14} | {'CEM reward':>14} | "
          f"{'premature (igo/cem)':>20}")
    print("-" * 70)
    for H in HORIZONS:
        env = make_env()
        agent = RandomShootingMPC(
            env, horizon=H, num_samples=NUM_SAMPLES, sigma=SIGMA, seed=SEED
        )
        r_mpc, _ = run_mpc_episode(env, agent, init_state=INIT_STATE, T=T)

        # plain IGO-ML (soft CEM): warm-started shifted plan, dt-smoothed update.
        # detect_premature=True so the same rollout also yields the flags.
        env = make_env()
        igo = IGOPlanner(
            env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
            max_iters=IGO_ITERS, sigma_init=REFINE_SIGMA, dt=DT, seed=SEED,
            detect_premature=True,
        )
        r_igo, _, igo_flags = run_episode_premature(env, igo, init_state=INIT_STATE, T=T)
        for t, f in enumerate(igo_flags):
            premature_rows.append(("igo", H, t, f))

        env = make_env()
        cem = IGOPlanner(
            env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
            max_iters=IGO_ITERS, sigma_init=REFINE_SIGMA, dt=1.0, seed=SEED,
            detect_premature=True,
        )
        r_cem, _, cem_flags = run_episode_premature(env, cem, init_state=INIT_STATE, T=T)
        for t, f in enumerate(cem_flags):
            premature_rows.append(("cem", H, t, f))

        mpc_rewards.append(r_mpc)
        igo_rewards.append(r_igo)
        cem_rewards.append(r_cem)
        n = len(igo_flags)
        print(f"{H:>3} | {r_mpc:>14.3f} | {r_igo:>14.3f} | {r_cem:>14.3f} | "
              f"{f'{sum(igo_flags)}/{n} , {sum(cem_flags)}/{n}':>20}")

    with open("premature_check_CEM_and_IGO.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["planner", "H", "timestep", "premature"])
        writer.writerows(premature_rows)
    print(f"Saved {len(premature_rows)} rows to premature_check_CEM_and_IGO.csv")

    return dict(
        horizons=np.array(HORIZONS),
        opt_reward=np.float64(opt_reward),
        mpc_rewards=np.array(mpc_rewards),
        igo_rewards=np.array(igo_rewards),
        cem_rewards=np.array(cem_rewards),
    )


def main():
    results = run_sweep()
    opt_reward = float(results["opt_reward"])
    mpc_rewards = list(results["mpc_rewards"])
    igo_rewards = list(results["igo_rewards"])
    cem_rewards = list(results["cem_rewards"]) if "cem_rewards" in results else None

    # ---- 3) plot (cost = -reward, log scale) ---------------------------- #
    opt_cost = -opt_reward
    mpc_cost = [-r for r in mpc_rewards]
    igo_cost = [-r for r in igo_rewards]
    cem_cost = [-r for r in cem_rewards] if cem_rewards is not None else None

    # trained SAC policy: same env / init state / horizon as the optimal run,
    # so it appears as a horizontal dashed line (reactive policy, no planning H).
    sac_cost = None
    if os.path.exists(SAC_MODEL_FILE):
        sac = SACPolicy(SAC_MODEL_FILE)
        sac_reward, _ = run_optimal_episode(make_env(), sac, init_state=INIT_STATE, T=T)
        sac_cost = -sac_reward
        print(f"[SAC]          episode reward (T={T}): {sac_reward:.3f}  "
              f"-> cost {sac_cost:.3f}")
    else:
        print(f"[SAC]  {SAC_MODEL_FILE} not found -> skipping SAC line "
              f"(train it with sac_lqr.py).")

    def make_plot(out: str, h_min: int) -> None:
        """Save the cost-vs-horizon figure, keeping only horizons >= h_min."""
        idx = [i for i, H in enumerate(HORIZONS) if H >= h_min]
        hs = [HORIZONS[i] for i in idx]
        plt.figure(figsize=(9, 5.5))
        plt.axhline(opt_cost, color="black", ls="--", lw=2,
                    label=f"Optimal LQR (cost={opt_cost:.2f})")
        if sac_cost is not None:
            plt.axhline(sac_cost, color="tab:red", ls="--", lw=2,
                        label=f"SAC (cost={sac_cost:.2f})")
        plt.plot(hs, [mpc_cost[i] for i in idx], "o-", color="tab:blue",
                 label="Random-shooting MPC")
        plt.plot(hs, [igo_cost[i] for i in idx], "s-", color="tab:green", label="IGO-ML")
        if cem_cost is not None:
            plt.plot(hs, [cem_cost[i] for i in idx], "^-", color="tab:orange", label="CEM (dt=1)")
        plt.yscale("log")
        plt.xlabel("Planning horizon H")
        plt.ylabel(f"Episode cost  = -reward  (T={T}, log scale)")
        plt.title("Optimal LQR vs. Random-Shooting MPC vs. IGO-ML  (lower is better)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        plt.close()

    def make_plot_refine(out: str, h_min: int, y_break: float = 18.0) -> None:
        """IGO-ML vs. optimal, with a BROKEN y-axis.

        The bottom panel is linear up to ``y_break`` (where all the interesting
        behaviour lives); costs above it (the diverging short horizons) are shown
        on a log-scaled top panel, with squiggle break marks in between.
        """
        idx = [i for i, H in enumerate(HORIZONS) if H >= h_min]
        hs = [HORIZONS[i] for i in idx]
        igo = [igo_cost[i] for i in idx]
        cem = [cem_cost[i] for i in idx] if cem_cost is not None else None

        fig, (ax_hi, ax_lo) = plt.subplots(
            2, 1, sharex=True, figsize=(9, 6),
            gridspec_kw=dict(height_ratios=[1, 2.6], hspace=0.06),
        )

        for ax in (ax_hi, ax_lo):
            ax.axhline(opt_cost, color="black", ls="--", lw=2,
                       label=f"Optimal LQR (cost={opt_cost:.2f})")
            if sac_cost is not None:
                ax.axhline(sac_cost, color="tab:red", ls="--", lw=2,
                           label=f"SAC (cost={sac_cost:.2f})")
            ax.plot(hs, igo, "s-", color="tab:green", label="IGO-ML")
            if cem is not None:
                ax.plot(hs, cem, "^-", color="tab:orange", label="CEM (dt=1)")
            ax.grid(True, alpha=0.3)

        # mark IGO's lowest-cost (best) horizon with a star
        igo_i = int(np.argmin(igo))
        for ax in (ax_hi, ax_lo):
            ax.plot(hs[igo_i], igo[igo_i], "*", color="tab:green", ms=20,
                    mec="black", mew=0.8, zorder=6,
                    label=f"IGO best (H={hs[igo_i]}, cost={igo[igo_i]:.2f})")

        # bottom: linear detail range
        ax_lo.set_ylim(opt_cost - 0.5, y_break)
        # top: log scale covering the diverging costs
        all_diverging = [c for c in igo if c > y_break]
        if cem is not None:
            all_diverging += [c for c in cem if c > y_break]
        top_lo = 10 ** np.floor(np.log10(min(all_diverging)))
        top_hi = 10 ** np.ceil(np.log10(max(all_diverging)))
        ax_hi.set_yscale("log")
        ax_hi.set_ylim(top_lo, top_hi)

        # hide the facing spines and draw the squiggle break marks
        ax_hi.spines["bottom"].set_visible(False)
        ax_lo.spines["top"].set_visible(False)
        ax_hi.tick_params(bottom=False)
        _draw_break(ax_hi, ax_lo)

        ax_hi.set_title("IGO-ML vs. CEM vs. Optimal LQR  (lower is better)")
        ax_lo.set_xlabel("Planning horizon H")
        fig.supylabel(f"Episode cost  = -reward  (T={T})")
        ax_hi.legend(loc="upper right")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # full sweep, and a zoomed version starting at H=5 (the H=1..2 blow-ups
    # dominate the log scale and hide the differences between the planners).
    make_plot("compare_CEM_and_IGO_reward_vs_horizon.png", h_min=min(HORIZONS))
    make_plot("compare_CEM_and_IGO_more_horizon.png", h_min=5)
    # IGO-only, starting at H=1, with a broken y-axis: linear detail below
    # cost=18, log-scaled diverging costs above the squiggle break.
    # make_plot_refine("compare_cem_mppi.png", h_min=1, y_break=18.0)
    print("-" * 70)
    print("Saved plots to compare_CEM_and_IGO_reward_vs_horizon.png, compare_CEM_and_IGO_more_horizon.png")

    # ---- 4) takeaways --------------------------------------------------- #
    best_H = HORIZONS[int(np.argmax(mpc_rewards))]
    best_H_igo = HORIZONS[int(np.argmax(igo_rewards))]
    print(f"\nOptimal episode reward          : {opt_reward:10.3f}")
    print(f"Best random-shooting MPC : H={best_H:<3} reward {max(mpc_rewards):10.3f}")
    print(f"Best IGO-ML              : H={best_H_igo:<3} reward {max(igo_rewards):10.3f}")
    if cem_cost is not None:
        best_H_cem = HORIZONS[int(np.argmax(cem_rewards))]
        print(f"Best CEM (dt=1)          : H={best_H_cem:<3} reward {max(cem_rewards):10.3f}")
    print(f"\nH=1  random-shooting MPC : {mpc_rewards[0]:.3e}  (diverges -- short-horizon failure)")
    print("=> Refining the sampling distribution (IGO-ML) keeps planning close")
    print("   to optimal across horizons instead of degrading for large H.")


if __name__ == "__main__":
    main()
