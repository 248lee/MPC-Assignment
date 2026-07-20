"""
Compare: Optimal LQR  vs.  Random-Shooting MPC  vs.  IGO-ML  (+ policy priors)
==============================================================================

Runs all controllers on *identical* episodes (same env, same initial state,
same seed) and compares episode reward:

  * Optimal LQR feedback  (a = -K s)          -- gold-standard baseline
  * Random-shooting MPC, horizon sweep         -- Phase I
  * IGO-ML, horizon sweep                       -- soft (dt) CEM (see IGO.py)
  * IGO complete sample-based, horizon sweep    -- weighted-MLE CEM over ALL
                                                   samples (see
                                                   IGO_complete_sample_based.py)
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
from phase2 import run_episode as run_plan_episode
from IGO import IGOPlanner
from IGO_complete_sample_based import IGOPlanner as IGOCompleteSampleBased
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


def make_refinement_planners(prior, H, detect_premature=False):
    """Build the four IGO refinement planners (igo, igo-cs, pp-large, ppc-consv)
    for horizon ``H``.

    Single source of truth for their hyperparameters, shared by ``run_sweep``
    (reward) and ``run_premature_sweep`` (premature-convergence check), so the
    two sweeps can never drift apart. Each planner gets its own fresh env (via
    ``make_env``), matching how ``run_sweep`` used to build them inline. ``pp``
    and ``ppc`` are ``None`` when no SAC ``prior`` is available.
    """
    igo = IGOPlanner(
        make_env(), horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
        max_iters=IGO_ITERS, sigma_init=REFINE_SIGMA, dt=DT, seed=SEED,
        detect_premature=detect_premature,
    )
    # complete sample-based IGO: identical hyperparameters, but re-fits the
    # Gaussian by a weighted MLE over ALL samples each iteration (non-elites
    # kept at weight 1-dt) instead of the variance-injection soft update.
    igo_cs = IGOCompleteSampleBased(
        make_env(), horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
        max_iters=IGO_ITERS, sigma_init=REFINE_SIGMA, dt=DT, seed=SEED,
        detect_premature=detect_premature,
    )
    pp = ppc = None
    if prior is not None:
        # large-explore : widen the SAC std by PP_STD_SCALE (aggressive search)
        pp = PolicyPriorIGO(
            make_env(), horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
            max_iters=IGO_ITERS, prior_std_scale=PP_STD_SCALE, dt=DT,
            prior=prior, seed=SEED, detect_premature=detect_premature,
        )
        # conservative : shrink the SAC std by 1/(2H) -- the same horizon-adaptive
        # proposal used by policy_prior_random_shooting.py, a tight proposal.
        ppc = PolicyPriorIGO(
            make_env(), horizon=H, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES,
            max_iters=IGO_ITERS, prior_std_scale=1.0 / (2.0 * H), dt=DT,
            prior=prior, seed=SEED, detect_premature=detect_premature,
        )
    return igo, igo_cs, pp, ppc


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
    mpc_rewards, igo_rewards, igo_cs_rewards = [], [], []
    pp_rewards, ppc_rewards, pprs_rewards = [], [], []   # pp=large-explore, ppc=conservative
    print(f"{'H':>3} | {'MPC reward':>14} | {'IGO reward':>14} | {'IGO-CS reward':>14} "
          f"| {'PP-large reward':>16} | {'PP-consv reward':>16} | {'PP-RandShoot':>14}")
    print("-" * 70)
    for H in HORIZONS:
        env = make_env()
        agent = RandomShootingMPC(
            env, horizon=H, num_samples=NUM_SAMPLES, sigma=SIGMA, seed=SEED
        )
        r_mpc, _ = run_mpc_episode(env, agent, init_state=INIT_STATE, T=T)

        # plain IGO-ML, the complete sample-based IGO, and the two policy-prior
        # IGO flavours (large-explore, conservative), all built from the shared
        # factory so their hyperparams stay in lockstep with the
        # premature-convergence sweep.
        igo, igo_cs, pp, ppc = make_refinement_planners(prior, H)
        r_igo, _ = run_plan_episode(igo.env, igo, init_state=INIT_STATE, T=T)
        r_igo_cs, _ = run_plan_episode(igo_cs.env, igo_cs, init_state=INIT_STATE, T=T)

        if prior is not None:
            r_pp, _ = run_plan_episode(pp.env, pp, init_state=INIT_STATE, T=T)
            r_ppc, _ = run_plan_episode(ppc.env, ppc, init_state=INIT_STATE, T=T)

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
        igo_cs_rewards.append(r_igo_cs)
        pp_rewards.append(r_pp)
        ppc_rewards.append(r_ppc)
        pprs_rewards.append(r_pprs)
        print(f"{H:>3} | {r_mpc:>14.3f} | {r_igo:>14.3f} | {r_igo_cs:>14.3f} "
              f"| {r_pp:>16.3f} | {r_ppc:>16.3f} | {r_pprs:>14.3f}")

    results = dict(
        horizons=np.array(HORIZONS),
        opt_reward=np.float64(opt_reward),
        mpc_rewards=np.array(mpc_rewards),
        igo_rewards=np.array(igo_rewards),
        igo_cs_rewards=np.array(igo_cs_rewards),
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
    # Every array a downstream plot/printout expects. A cache written before a
    # new planner column was added still matches CONFIG_SIG (that column adds no
    # new hyperparameter), so we also require every key to be present.
    required_keys = (
        "horizons", "opt_reward", "mpc_rewards", "igo_rewards", "igo_cs_rewards",
        "pp_rewards", "ppc_rewards", "pprs_rewards",
    )
    if os.path.exists(RESULTS_FILE):
        data = np.load(RESULTS_FILE)
        if (
            "config_sig" in data
            and data["config_sig"].shape == CONFIG_SIG.shape
            and np.allclose(data["config_sig"], CONFIG_SIG)
            and all(k in data.files for k in required_keys)
        ):
            print(f"Loaded cached results from {RESULTS_FILE} (config matches).")
            return {k: data[k] for k in data.files}
        print(f"Cached results in {RESULTS_FILE} are stale -> re-running sweep.")
    return run_sweep()


def run_episode_premature(env, agent, init_state=None, T=None):
    """Like run_plan_episode but collects per-step premature-convergence flags."""
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


def run_premature_sweep():
    """Run the four IGO planners with premature-convergence detection, save premature_check.csv.

    For each horizon H and each of (IGO-ML, complete sample-based IGO, large-explore prior IGO,
    conservative prior IGO), rolls out a full T-step episode and records per-timestep whether
    premature convergence was detected: a True entry means some perturbation of the converged
    plan (within the Sobol shell) outperforms it, indicating the planner stopped before reaching
    a local optimum.
    """
    print("=" * 70)
    print("Running premature convergence check (4 planners × 20 horizons × T steps)...")
    print("=" * 70)
    prior = SACPrior(SAC_MODEL_FILE) if os.path.exists(SAC_MODEL_FILE) else None
    if prior is None:
        print(f"[Warning] {SAC_MODEL_FILE} not found; pp-large and pp-consv will be skipped.")

    rows = []
    for H in HORIZONS:
        # SAME factory as run_sweep -> identical hyperparameters, only with
        # premature-convergence detection switched on.
        igo, igo_cs, pp, ppc = make_refinement_planners(prior, H, detect_premature=True)
        _, _, igo_flags = run_episode_premature(igo.env, igo, init_state=INIT_STATE, T=T)
        for t, f in enumerate(igo_flags):
            rows.append(("igo", H, t, f))

        _, _, igo_cs_flags = run_episode_premature(igo_cs.env, igo_cs, init_state=INIT_STATE, T=T)
        for t, f in enumerate(igo_cs_flags):
            rows.append(("igo-cs", H, t, f))

        pp_summary = ppc_summary = "N/A (no SAC model)"
        if prior is not None:
            _, _, pp_flags = run_episode_premature(pp.env, pp, init_state=INIT_STATE, T=T)
            for t, f in enumerate(pp_flags):
                rows.append(("pp-large", H, t, f))
            pp_summary = f"{sum(pp_flags)}/{len(pp_flags)}"

            _, _, ppc_flags = run_episode_premature(ppc.env, ppc, init_state=INIT_STATE, T=T)
            for t, f in enumerate(ppc_flags):
                rows.append(("pp-consv", H, t, f))
            ppc_summary = f"{sum(ppc_flags)}/{len(ppc_flags)}"

        n = len(igo_flags)
        print(f"H={H:>2}: igo={sum(igo_flags)}/{n} premature  "
              f"igo-cs={sum(igo_cs_flags)}/{len(igo_cs_flags)}  "
              f"pp-large={pp_summary}  pp-consv={ppc_summary}")

    with open("premature_check.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["planner", "H", "timestep", "premature"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to premature_check.csv")


def main():
    results = load_or_run()
    opt_reward = float(results["opt_reward"])
    mpc_rewards = list(results["mpc_rewards"])
    igo_rewards = list(results["igo_rewards"])
    igo_cs_rewards = list(results["igo_cs_rewards"]) if "igo_cs_rewards" in results else None
    pp_rewards = list(results["pp_rewards"]) if "pp_rewards" in results else None
    ppc_rewards = list(results["ppc_rewards"]) if "ppc_rewards" in results else None
    pprs_rewards = list(results["pprs_rewards"]) if "pprs_rewards" in results else None

    # ---- 3) plot (cost = -reward, log scale) ---------------------------- #
    opt_cost = -opt_reward
    mpc_cost = [-r for r in mpc_rewards]
    igo_cost = [-r for r in igo_rewards]
    # complete sample-based IGO curve (None if it wasn't in the cache)
    igo_cs_cost = None
    if igo_cs_rewards is not None and not np.all(np.isnan(igo_cs_rewards)):
        igo_cs_cost = [-r for r in igo_cs_rewards]
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
        if igo_cs_cost is not None:
            plt.plot(hs, [igo_cs_cost[i] for i in idx], "^-", color="tab:orange",
                     label="IGO complete sample-based")
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
        igo_cs = [igo_cs_cost[i] for i in idx] if igo_cs_cost is not None else None
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
            if igo_cs is not None:
                ax.plot(hs, igo_cs, "^-", color="tab:orange",
                        label="IGO complete sample-based")
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
        # top: log scale covering the diverging costs (across every plotted curve)
        big = [c for c in igo if c > y_break]
        if igo_cs is not None:
            big += [c for c in igo_cs if c > y_break]
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
    if igo_cs_cost is not None:
        best_H_igo_cs = HORIZONS[int(np.nanargmax(igo_cs_rewards))]
        print(f"Best IGO complete sample-based : H={best_H_igo_cs:<3} reward "
              f"{np.nanmax(igo_cs_rewards):10.3f}")
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
    run_premature_sweep()


if __name__ == "__main__":
    main()
