"""
Compare with an EXPONENTIAL sample budget:  N(H) = N_BASE * GROWTH^(H-1)
========================================================================

The fixed-N sweep (`compare.py`) shows every sampling-based planner degrading
at long horizons. Importance-sampling theory says that is exactly what a fixed
budget must do: the sample size needed for a reliable answer grows like
exp(c*H) (Chatterjee & Diaconis 2018; Yoon et al. 2022 for path-integral
planners). This script re-runs the same sweep but actually GIVES every planner
that budget:

    N(H) = round(N_BASE * GROWTH^(H-1))       with GROWTH = sqrt(2)
         = exp(c*H) up to a constant, c = ln(2)/2 ~= 0.347

i.e. the budget doubles every 2 horizon steps, anchored so H=3 gets the old
fixed budget (N=1000) and H=16 gets ~90.5k samples per CEM iteration.
NUM_ELITES stays fixed at 50 so the ONLY thing that changes with H is N.

To keep the run finishable the sweep stops at H_MAX = 16 and the 80
(algorithm, horizon) jobs are split into NUM_SHARDS = 4 shards, balanced by
estimated cost (LPT greedy bin packing over weight ~ iter_factor * N(H) * H).

Usage
-----
    python compare_expN.py --assign     # print the shard assignment + loads
    python compare_expN.py --smoke      # tiny T=10 sanity run of every algo
    python compare_expN.py --shard i    # run shard i (i = 0..3), in parallel
    python compare_expN.py --plot       # merge shard .npz files + plot

Each shard
  * re-saves `compare_expN_shard{i}.npz` after EVERY finished job, so partial
    results survive a crash and can even be plotted mid-run, and
  * rewrites `compare_expN_status_shard{i}.txt` every few episode steps, so
    progress can be tracked from outside while it runs.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


# --------------------------------------------------------------------------- #
# experiment configuration (mirrors compare.py where possible)
# --------------------------------------------------------------------------- #
ENV_KWARGS = dict(noise_std=0.0)          # deterministic -> clean comparison
INIT_STATE = np.array([1.0, -1.0, 0.5])   # same start for everyone
T = 200                                    # steps per episode
SEED = 0

H_MAX = 16
HORIZONS = list(range(1, H_MAX + 1))      # H = 1 .. 16

# exponential sample budget: N(H) = N_BASE * GROWTH**(H-1) = A * exp(c*H),
# c = ln(GROWTH). GROWTH = sqrt(2) doubles the budget every 2 horizon steps.
N_BASE = 500
GROWTH = float(np.sqrt(2.0))

SIGMA = 1.0                                # random-shooting proposal std
REFINE_SIGMA = 0.2                         # CEM sigma_init
NUM_ELITES = 50                            # fixed K -> only N varies with H
CEM_ITERS = 1000                           # iteration cap (planners stop on tol)
PP_STD_SCALE = 5.0                         # large-explore prior CEM widening
PPRS_MULT = 100                            # policy-prior random shooting: N*100
PPRS_CHUNK = 250_000                        # eval pprs in chunks to cap memory

NUM_SHARDS = 4
PROGRESS_EVERY = 10                        # episode steps between status writes

SAC_MODEL_FILE = "sac_lqr.pt"
SHARD_FILE = "compare_expN_shard{i}.npz"
STATUS_FILE = "compare_expN_status_shard{i}.txt"
RESULTS_FILE = "compare_expN_results.npz"

ALGOS = ["mpc", "cem", "pp", "ppc", "pprs"]

# rough relative cost of one (sample x horizon-step) unit per algorithm, used
# ONLY to balance the shards: the CEM-family planners iterate ~tens of times
# per timestep while the shooters evaluate once; pprs multiplies its N by 100.
ITER_WEIGHT = dict(mpc=1.0, cem=30.0, pp=30.0, ppc=30.0, pprs=float(PPRS_MULT))

CONFIG_SIG = np.array([
    T, SEED, N_BASE, GROWTH, SIGMA, REFINE_SIGMA, NUM_ELITES, CEM_ITERS,
    PP_STD_SCALE, PPRS_MULT, *INIT_STATE, *HORIZONS,
], dtype=np.float64)


def num_samples(H: int) -> int:
    """Exponential budget N(H) = N_BASE * GROWTH^(H-1)."""
    return int(round(N_BASE * GROWTH ** (H - 1)))


def job_samples(algo: str, H: int) -> int:
    """Actual per-iteration sample count of a job (pprs keeps its x100)."""
    return num_samples(H) * (PPRS_MULT if algo == "pprs" else 1)


# --------------------------------------------------------------------------- #
# job list + shard assignment
# --------------------------------------------------------------------------- #
def job_weight(algo: str, H: int) -> float:
    return ITER_WEIGHT[algo] * num_samples(H) * H


def shard_assignment() -> list[list[tuple[str, int]]]:
    """Split all (algo, H) jobs into NUM_SHARDS lists balanced by weight.

    Greedy LPT bin packing: place the heaviest remaining job into the lightest
    shard. Deterministic, so every worker process computes the same split.
    Within a shard, jobs run cheapest-first (early progress; the big
    memory-hungry jobs land at the end when shards have naturally staggered).
    """
    shards: list[list[tuple[str, int]]] = [[] for _ in range(NUM_SHARDS)]
    loads = [0.0] * NUM_SHARDS
    for algo, H in sorted(
        ((a, h) for a in ALGOS for h in HORIZONS),
        key=lambda j: (-job_weight(*j), j[0], j[1]),
    ):
        i = int(np.argmin(loads))
        shards[i].append((algo, H))
        loads[i] += job_weight(algo, H)
    for jobs in shards:
        jobs.sort(key=lambda j: job_weight(*j))
    return shards


def print_assignment() -> None:
    shards = shard_assignment()
    total = sum(job_weight(a, h) for a in ALGOS for h in HORIZONS)
    print(f"N(H) = {N_BASE} * {GROWTH:.4f}^(H-1):")
    print("   " + "  ".join(f"H={H}:{num_samples(H)}" for H in HORIZONS))
    print()
    for i, jobs in enumerate(shards):
        load = sum(job_weight(*j) for j in jobs)
        print(f"shard {i}: {len(jobs):>2} jobs, {100 * load / total:5.1f}% of load")
        print("   " + "  ".join(f"{a}:H{h}" for a, h in jobs))


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def make_env():
    from lqr_env import LQREnv
    return LQREnv(seed=SEED, **ENV_KWARGS)


def _load_prior():
    if not os.path.exists(SAC_MODEL_FILE):
        return None
    from policy_prior_CEM import SACPrior
    return SACPrior(SAC_MODEL_FILE)


def _make_chunked_pprs(env, H, N, prior):
    """Policy-prior random shooting, evaluated in chunks so the (N, H, adim)
    arrays never blow up memory at large N (N can reach ~9M at H=16).

    Same argmax-over-N semantics as ``policy_prior_random_shooting`` -- we just
    stream the N samples through the model ``PPRS_CHUNK`` at a time and keep the
    single best first-action seen. Deterministic given the seed.
    """
    from policy_prior_CEM import _rollout_returns
    from policy_prior_random_shooting import RandomShootingPlanner

    class ChunkedPolicyPriorRS(RandomShootingPlanner):
        def plan(self, state):
            H_, N_ = self.horizon, self.num_samples
            adim = self.env.action_dim
            lo, hi = self.env.action_low, self.env.action_high
            mu, sigma = self._init_from_prior(state)
            sigma = sigma / (H_ * 2)                    # match the base planner
            best_ret, best_a0 = -np.inf, None
            remaining = N_
            while remaining > 0:
                c = min(PPRS_CHUNK, remaining)
                noise = self.rng.normal(size=(c, H_, adim))
                actions = np.clip(mu + sigma * noise, lo, hi)
                returns = _rollout_returns(self.env, state, actions, self.gamma)
                k = int(np.argmax(returns))
                if returns[k] > best_ret:
                    best_ret, best_a0 = float(returns[k]), actions[k, 0].copy()
                remaining -= c
            return best_a0

    return ChunkedPolicyPriorRS(env, horizon=H, num_samples=N, prior=prior, seed=SEED)


def build_agent(algo: str, H: int, env, prior):
    N = num_samples(H)
    if algo == "mpc":
        from phase1 import RandomShootingMPC
        return RandomShootingMPC(env, horizon=H, num_samples=N, sigma=SIGMA, seed=SEED)
    if algo == "cem":
        from phase2 import CEMPlanner
        return CEMPlanner(
            env, horizon=H, num_samples=N, num_elites=NUM_ELITES,
            max_iters=CEM_ITERS, sigma_init=REFINE_SIGMA, seed=SEED,
        )
    if prior is None:
        return None
    from policy_prior_CEM import CEMPlanner as PolicyPriorCEM
    if algo == "pp":
        return PolicyPriorCEM(
            env, horizon=H, num_samples=N, num_elites=NUM_ELITES,
            max_iters=CEM_ITERS, prior_std_scale=PP_STD_SCALE, prior=prior, seed=SEED,
        )
    if algo == "ppc":
        return PolicyPriorCEM(
            env, horizon=H, num_samples=N, num_elites=NUM_ELITES,
            max_iters=CEM_ITERS, prior_std_scale=1.0 / (2.0 * H), prior=prior, seed=SEED,
        )
    if algo == "pprs":
        return _make_chunked_pprs(env, H, N * PPRS_MULT, prior)
    raise ValueError(f"unknown algo: {algo!r}")


def run_episode_progress(env, agent, on_step=None, T_steps: int | None = None) -> float:
    """Same rollout as the planners' run_episode, with a step callback."""
    if hasattr(agent, "reset"):
        agent.reset()
    s = env.reset(state=INIT_STATE)
    steps = T if T_steps is None else T_steps
    total = 0.0
    for t in range(steps):
        a = agent.act(s)
        s, r, term, trunc, _ = env.step(a)
        total += r
        if on_step is not None and ((t + 1) % PROGRESS_EVERY == 0 or t + 1 == steps):
            on_step(t + 1)
        if term or trunc:
            break
    return total


def run_shard(i: int) -> None:
    import torch
    torch.set_num_threads(1)   # 4 workers share the CPU; the SAC net is tiny

    jobs = shard_assignment()[i]
    prior = _load_prior()
    if prior is None:
        print(f"[shard {i}] {SAC_MODEL_FILE} not found -> pp/ppc/pprs will be NaN.")

    results = {algo: np.full(len(HORIZONS), np.nan) for algo in ALGOS}
    done_lines: list[str] = []
    t0 = time.time()

    def save() -> None:
        np.savez(
            SHARD_FILE.format(i=i),
            horizons=np.array(HORIZONS),
            n_samples=np.array([num_samples(H) for H in HORIZONS]),
            config_sig=CONFIG_SIG,
            shard=np.int64(i),
            **{f"{algo}_rewards": results[algo] for algo in ALGOS},
        )

    def write_status(current: str) -> None:
        lines = [
            f"shard {i} | {len(done_lines)}/{len(jobs)} jobs done | total {time.time() - t0:.0f}s",
            f"current: {current}",
        ] + [f"done: {ln}" for ln in done_lines]
        with open(STATUS_FILE.format(i=i), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    write_status("starting")
    for j, (algo, H) in enumerate(jobs):
        N = job_samples(algo, H)
        tag = f"{algo} H={H} N={N}"
        print(f"[shard {i}] job {j + 1}/{len(jobs)}: {tag}", flush=True)
        jt0 = time.time()
        env = make_env()
        agent = build_agent(algo, H, env, prior)
        if agent is None:                       # missing SAC checkpoint
            done_lines.append(f"{tag}: SKIPPED (no {SAC_MODEL_FILE})")
            continue

        def on_step(t: int) -> None:
            write_status(f"{tag} | step {t}/{T} | job {time.time() - jt0:.0f}s")

        r = run_episode_progress(env, agent, on_step)
        results[algo][HORIZONS.index(H)] = r
        done_lines.append(f"{tag}: reward={r:.6g} ({time.time() - jt0:.0f}s)")
        save()
        write_status(f"finished {tag}, next job")
        print(f"[shard {i}] done {tag}: reward={r:.6g} "
              f"({time.time() - jt0:.0f}s, total {time.time() - t0:.0f}s)", flush=True)

    write_status("ALL DONE")
    print(f"[shard {i}] ALL DONE in {time.time() - t0:.0f}s "
          f"-> {SHARD_FILE.format(i=i)}", flush=True)


# --------------------------------------------------------------------------- #
# smoke test (tiny T, sanity + speed calibration)
# --------------------------------------------------------------------------- #
def smoke() -> None:
    prior = _load_prior()
    for algo in ALGOS:
        for H in (3, 8):
            env = make_env()
            agent = build_agent(algo, H, env, prior)
            if agent is None:
                print(f"{algo:>5} H={H}: skipped (no SAC checkpoint)")
                continue
            t0 = time.time()
            r = run_episode_progress(env, agent, T_steps=10)
            dt = time.time() - t0
            print(f"{algo:>5} H={H:>2} N={job_samples(algo, H):>7}: "
                  f"reward(T=10)={r:12.4f}  {dt:6.2f}s  "
                  f"(~{dt / 10:.3f}s/step -> full episode ~{dt / 10 * T:.0f}s)",
                  flush=True)


# --------------------------------------------------------------------------- #
# merge + plot
# --------------------------------------------------------------------------- #
def _fmt_n(n: int) -> str:
    if n >= 10000:
        return f"{n / 1000:.0f}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def merge_shards() -> dict[str, np.ndarray]:
    merged = {algo: np.full(len(HORIZONS), np.nan) for algo in ALGOS}
    found = 0
    for i in range(NUM_SHARDS):
        f = SHARD_FILE.format(i=i)
        if not os.path.exists(f):
            print(f"WARNING: {f} missing -> its jobs will be NaN.")
            continue
        data = np.load(f)
        if not (data["config_sig"].shape == CONFIG_SIG.shape
                and np.allclose(data["config_sig"], CONFIG_SIG)):
            print(f"WARNING: {f} has a stale config -> skipped.")
            continue
        found += 1
        for algo in ALGOS:
            arr = data[f"{algo}_rewards"]
            ok = ~np.isnan(arr)
            merged[algo][ok] = arr[ok]
    print(f"Merged {found}/{NUM_SHARDS} shard files.")
    for algo in ALGOS:
        missing = [HORIZONS[k] for k in np.flatnonzero(np.isnan(merged[algo]))]
        if missing:
            print(f"WARNING: {algo} missing H={missing}")
    return merged


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = merge_shards()

    # baselines (fast, computed here rather than in the workers)
    from optimal import LQRController, run_episode as run_optimal_episode
    env = make_env()
    ctrl = LQRController(env)
    opt_reward, _ = run_optimal_episode(env, ctrl, init_state=INIT_STATE, T=T)
    opt_cost = -opt_reward
    print(f"[Optimal LQR] episode reward (T={T}): {opt_reward:.3f}")

    sac_cost = None
    if os.path.exists(SAC_MODEL_FILE):
        from compare import SACPolicy
        sac = SACPolicy(SAC_MODEL_FILE)
        sac_reward, _ = run_optimal_episode(make_env(), sac, init_state=INIT_STATE, T=T)
        sac_cost = -sac_reward
        print(f"[SAC]         episode reward (T={T}): {sac_reward:.3f}")

    np.savez(
        RESULTS_FILE,
        horizons=np.array(HORIZONS),
        n_samples=np.array([num_samples(H) for H in HORIZONS]),
        opt_reward=np.float64(opt_reward),
        sac_reward=np.float64(-sac_cost) if sac_cost is not None else np.float64(np.nan),
        config_sig=CONFIG_SIG,
        **{f"{algo}_rewards": merged[algo] for algo in ALGOS},
    )
    print(f"Saved merged results to {RESULTS_FILE}")

    # results table
    header = f"{'H':>3} {'N(H)':>7} | " + " | ".join(f"{a:>14}" for a in ALGOS)
    print(header)
    print("-" * len(header))
    for k, H in enumerate(HORIZONS):
        row = " | ".join(f"{merged[a][k]:>14.3f}" for a in ALGOS)
        print(f"{H:>3} {num_samples(H):>7} | {row}")

    styles = dict(
        mpc=("o-", "tab:blue", "Random-shooting MPC"),
        cem=("s-", "tab:green", "CEM"),
        pp=("d-", "tab:purple", "Large-explore prior CEM (SAC)"),
        ppc=("P-", "tab:olive", "Conservative prior CEM (SAC)"),
        pprs=("v-", "tab:brown", f"Policy-prior random shooting (N×{PPRS_MULT})"),
    )

    def make_plot(out: str, h_min: int) -> None:
        idx = [k for k, H in enumerate(HORIZONS) if H >= h_min]
        hs = [HORIZONS[k] for k in idx]
        plt.figure(figsize=(9.5, 5.5))
        plt.axhline(opt_cost, color="black", ls="--", lw=2,
                    label=f"Optimal LQR (cost={opt_cost:.2f})")
        if sac_cost is not None:
            plt.axhline(sac_cost, color="tab:red", ls="--", lw=2,
                        label=f"SAC (cost={sac_cost:.2f})")
        for algo, (fmt, color, label) in styles.items():
            cost = [-merged[algo][k] for k in idx]
            plt.plot(hs, cost, fmt, color=color, label=label)
        plt.yscale("log")
        plt.xticks(hs, [f"{H}\n{_fmt_n(num_samples(H))}" for H in hs], fontsize=8)
        plt.xlabel("Planning horizon H   (lower tick row: sample budget N(H))")
        plt.ylabel(f"Episode cost = -reward  (T={T}, log scale)")
        plt.title(f"Exponential budget N(H) = {N_BASE}·√2$^{{H-1}}$"
                  f"  ≈ exp(0.35·H)   (lower is better)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"Saved {out}")

    make_plot("compare_expN_reward_vs_horizon.png", h_min=min(HORIZONS))
    make_plot("compare_expN_zoom.png", h_min=5)

    def make_plot_linear(out: str, h_min: int) -> None:
        """LINEAR-scale zoom of only the near-optimal planners (drop the
        random-shooting MPC, which is 100x off-scale). This is where the actual
        H-vs-budget story lives: how far each refined planner drifts from the
        optimal cost as H grows even though N(H) grows like exp(0.35 H)."""
        idx = [k for k, H in enumerate(HORIZONS) if H >= h_min]
        hs = [HORIZONS[k] for k in idx]
        plt.figure(figsize=(9.5, 5.5))
        plt.axhline(opt_cost, color="black", ls="--", lw=2,
                    label=f"Optimal LQR (cost={opt_cost:.3f})")
        if sac_cost is not None:
            plt.axhline(sac_cost, color="tab:red", ls="--", lw=2,
                        label=f"SAC (cost={sac_cost:.3f})")
        for algo in ("cem", "pp", "ppc", "pprs"):     # skip mpc (off-scale)
            fmt, color, label = styles[algo]
            cost = [-merged[algo][k] for k in idx]
            plt.plot(hs, cost, fmt, color=color, label=label)
        plt.xticks(hs, [f"{H}\n{_fmt_n(num_samples(H))}" for H in hs], fontsize=8)
        plt.xlabel("Planning horizon H   (lower tick row: sample budget N(H))")
        plt.ylabel(f"Episode cost = -reward  (T={T}, linear scale)")
        plt.title("Near-optimal planners under an exponential budget "
                  "(linear zoom, lower is better)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"Saved {out}")

    make_plot_linear("compare_expN_near_optimal.png", h_min=2)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=int, default=None, help="run shard i (0..3)")
    p.add_argument("--plot", action="store_true", help="merge shards + plot")
    p.add_argument("--assign", action="store_true", help="print shard assignment")
    p.add_argument("--smoke", action="store_true", help="tiny T=10 sanity run")
    args = p.parse_args()

    if args.assign:
        print_assignment()
    elif args.smoke:
        smoke()
    elif args.shard is not None:
        run_shard(args.shard)
    elif args.plot:
        plot()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
