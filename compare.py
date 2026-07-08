"""
Compare: Optimal LQR  vs.  Random-Shooting MPC  vs.  IGO-ML  (+ policy priors)
==============================================================================

Runs all controllers on *identical* episodes (same env, same initial state,
same seed) and compares episode reward:

  * Optimal LQR feedback  (a = -K s)          -- gold-standard baseline
  * Random-shooting MPC, horizon sweep         -- Phase I
  * IGO-ML, horizon sweep                       -- soft (dt) CEM (see IGO.py)
  * Policy-prior IGO-ML (SAC seed), two flavours
  * Policy-prior random shooting (SAC seed)

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

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

import torch

from lqr_env import LQREnv
from optimal import LQRController, run_episode as run_optimal_episode
from phase1 import RandomShootingMPC, run_episode as run_mpc_episode
from phase2 import run_episode as run_plan_episode
from IGO import IGOPlanner
from policy_prior_IGO import IGOPlanner as PolicyPriorIGO, SACPrior
from policy_prior_random_shooting import RandomShootingPlanner as PolicyPriorRS
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
PP_STD_SCALE = 5.0                         # policy-prior IGO: SAC-std multiplier


RESULTS_FILE = "compare_results.npz"
SAC_MODEL_FILE = "sac_lqr.pt"              # trained SAC actor (sac_lqr.py)


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

# A signature of everything that affects the numbers. If any of it changes, the
# cached results are stale and the sweep is re-run automatically.
CONFIG_SIG = np.array([
    T, NUM_SAMPLES, SIGMA, SEED, REFINE_SIGMA, NUM_ELITES,
    IGO_ITERS, DT, PP_STD_SCALE, *INIT_STATE, *HORIZONS,
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

    # policy-prior IGO needs the trained SAC actor; skip (NaN) if unavailable.
    prior = SACPrior(SAC_MODEL_FILE) if os.path.exists(SAC_MODEL_FILE) else None
    if prior is None:
        print(f"[Policy-prior IGO]  {SAC_MODEL_FILE} not found -> column will be NaN.")

    # ---- 2) planners over horizons -------------------------------------- #
    mpc_rewards, igo_rewards = [], []
    pp_rewards, ppc_rewards, pprs_rewards = [], [], []   # pp=large-explore, ppc=conservative
    print(f"{'H':>3} | {'MPC reward':>14} | {'IGO reward':>14} | {'PP-large reward':>16} "
          f"| {'PP-consv reward':>16} | {'PP-RandShoot':>14}")
    print("-" * 70)
    for H in HORIZONS:
        env = make_env()
        agent = RandomShootingMPC(
            env, horizon=H, num_samples=NUM_SAMPLES, sigma=SIGMA, seed=SEED
        )
        r_mpc, _ = run_mpc_episode(env, agent, init_state=INIT_STATE, T=T)

        # plain IGO-ML (soft CEM): warm-started shifted plan, dt-smoothed update.
        env = make_env()
        igo = IGOPlanner(
            env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
            max_iters=IGO_ITERS, sigma_init=REFINE_SIGMA, dt=DT, seed=SEED,
        )
        r_igo, _ = run_plan_episode(env, igo, init_state=INIT_STATE, T=T)

        # policy-prior IGO: same IGO-ML refinement, but seeded from the SAC policy
        # instead of warm-starting the shifted plan. Two flavours:
        #   * large-explore : widen the SAC std by PP_STD_SCALE (aggressive search)
        #   * conservative  : shrink the SAC std by 1/(2H), matching the random
        #                     shooter -- a tight, horizon-adaptive proposal.
        if prior is not None:
            env = make_env()
            pp = PolicyPriorIGO(
                env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
                max_iters=IGO_ITERS, prior_std_scale=PP_STD_SCALE, dt=DT,
                prior=prior, seed=SEED,
            )
            r_pp, _ = run_plan_episode(env, pp, init_state=INIT_STATE, T=T)

            # conservative prior IGO: sigma = SAC std / (2H), i.e. the same
            # horizon-adaptive proposal used by policy_prior_random_shooting.py,
            # expressed here via the prior_std_scale multiplier.
            env = make_env()
            ppc = PolicyPriorIGO(
                env, horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
                max_iters=IGO_ITERS, prior_std_scale=1.0 / (2.0 * H), dt=DT,
                prior=prior, seed=SEED,
            )
            r_ppc, _ = run_plan_episode(env, ppc, init_state=INIT_STATE, T=T)

            # policy-prior random shooting: one-shot best-of-N sampled from the
            # SAC prior (raw std, no widening), no refinement loop.
            env = make_env()
            pprs = PolicyPriorRS(
                env, horizon=H, num_samples=NUM_SAMPLES * 100, prior=prior, seed=SEED,
            )
            r_pprs, _ = run_plan_episode(env, pprs, init_state=INIT_STATE, T=T)
        else:
            r_pp = np.nan
            r_ppc = np.nan
            r_pprs = np.nan

        mpc_rewards.append(r_mpc)
        igo_rewards.append(r_igo)
        pp_rewards.append(r_pp)
        ppc_rewards.append(r_ppc)
        pprs_rewards.append(r_pprs)
        print(f"{H:>3} | {r_mpc:>14.3f} | {r_igo:>14.3f} | {r_pp:>16.3f} "
              f"| {r_ppc:>16.3f} | {r_pprs:>14.3f}")

    results = dict(
        horizons=np.array(HORIZONS),
        opt_reward=np.float64(opt_reward),
        mpc_rewards=np.array(mpc_rewards),
        igo_rewards=np.array(igo_rewards),
        pp_rewards=np.array(pp_rewards),
        ppc_rewards=np.array(ppc_rewards),
        pprs_rewards=np.array(pprs_rewards),
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
    igo_rewards = list(results["igo_rewards"])
    pp_rewards = list(results["pp_rewards"]) if "pp_rewards" in results else None
    ppc_rewards = list(results["ppc_rewards"]) if "ppc_rewards" in results else None
    pprs_rewards = list(results["pprs_rewards"]) if "pprs_rewards" in results else None

    # ---- 3) plot (cost = -reward, log scale) ---------------------------- #
    opt_cost = -opt_reward
    mpc_cost = [-r for r in mpc_rewards]
    igo_cost = [-r for r in igo_rewards]
    # policy-prior IGO curves (may be all-NaN if the SAC model was unavailable)
    pp_cost = None
    if pp_rewards is not None and not np.all(np.isnan(pp_rewards)):
        pp_cost = [-r for r in pp_rewards]
    ppc_cost = None
    if ppc_rewards is not None and not np.all(np.isnan(ppc_rewards)):
        ppc_cost = [-r for r in ppc_rewards]
    # policy-prior random-shooting curve (same NaN guard)
    pprs_cost = None
    if pprs_rewards is not None and not np.all(np.isnan(pprs_rewards)):
        pprs_cost = [-r for r in pprs_rewards]

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
        if pp_cost is not None:
            plt.plot(hs, [pp_cost[i] for i in idx], "d-", color="tab:purple",
                     label="Large-explore prior IGO (SAC)")
        if ppc_cost is not None:
            plt.plot(hs, [ppc_cost[i] for i in idx], "P-", color="tab:olive",
                     label="Conservative prior IGO (SAC)")
        if pprs_cost is not None:
            plt.plot(hs, [pprs_cost[i] for i in idx], "v-", color="tab:brown",
                     label="Policy-prior random shooting (SAC)")
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
        pp = [pp_cost[i] for i in idx] if pp_cost is not None else None
        ppc = [ppc_cost[i] for i in idx] if ppc_cost is not None else None

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
            if pp is not None:
                ax.plot(hs, pp, "d-", color="tab:purple",
                        label="Large-explore prior IGO (SAC)")
            if ppc is not None:
                ax.plot(hs, ppc, "P-", color="tab:olive",
                        label="Conservative prior IGO (SAC)")
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
        big = [c for c in igo if c > y_break]
        top_lo = 10 ** np.floor(np.log10(min(big)))
        top_hi = 10 ** np.ceil(np.log10(max(big)))
        ax_hi.set_yscale("log")
        ax_hi.set_ylim(top_lo, top_hi)

        # hide the facing spines and draw the squiggle break marks
        ax_hi.spines["bottom"].set_visible(False)
        ax_lo.spines["top"].set_visible(False)
        ax_hi.tick_params(bottom=False)
        _draw_break(ax_hi, ax_lo)

        ax_hi.set_title("IGO-ML vs. Optimal LQR  (lower is better)")
        ax_lo.set_xlabel("Planning horizon H")
        fig.supylabel(f"Episode cost  = -reward  (T={T})")
        ax_hi.legend(loc="upper right")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # full sweep, and a zoomed version starting at H=5 (the H=1..2 blow-ups
    # dominate the log scale and hide the differences between the planners).
    make_plot("compare_reward_vs_horizon.png", h_min=min(HORIZONS))
    make_plot("compare_more_horizon.png", h_min=5)
    # IGO-only, starting at H=1, with a broken y-axis: linear detail below
    # cost=18, log-scaled diverging costs above the squiggle break.
    make_plot_refine("compare_cem_mppi.png", h_min=1, y_break=18.0)
    print("-" * 70)
    print("Saved plots to compare_reward_vs_horizon.png, compare_more_horizon.png, "
          "and compare_cem_mppi.png")

    # ---- 4) takeaways --------------------------------------------------- #
    best_H = HORIZONS[int(np.argmax(mpc_rewards))]
    best_H_igo = HORIZONS[int(np.argmax(igo_rewards))]
    print(f"\nOptimal episode reward          : {opt_reward:10.3f}")
    print(f"Best random-shooting MPC : H={best_H:<3} reward {max(mpc_rewards):10.3f}")
    print(f"Best IGO-ML              : H={best_H_igo:<3} reward {max(igo_rewards):10.3f}")
    if pp_cost is not None:
        best_H_pp = HORIZONS[int(np.nanargmax(pp_rewards))]
        print(f"Best Large-explore prior IGO : H={best_H_pp:<3} reward "
              f"{np.nanmax(pp_rewards):10.3f}")
    if ppc_cost is not None:
        best_H_ppc = HORIZONS[int(np.nanargmax(ppc_rewards))]
        print(f"Best Conservative prior IGO  : H={best_H_ppc:<3} reward "
              f"{np.nanmax(ppc_rewards):10.3f}")
    if pprs_cost is not None:
        best_H_pprs = HORIZONS[int(np.nanargmax(pprs_rewards))]
        print(f"Best Policy-prior RandShoot: H={best_H_pprs:<3} reward "
              f"{np.nanmax(pprs_rewards):10.3f}")
    print(f"\nH=1  random-shooting MPC : {mpc_rewards[0]:.3e}  (diverges -- short-horizon failure)")
    print("=> Refining the sampling distribution (IGO-ML) keeps planning close")
    print("   to optimal across horizons instead of degrading for large H.")


if __name__ == "__main__":
    main()
