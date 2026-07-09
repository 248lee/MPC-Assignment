"""
train_prior.py -- train SAC policy priors on the non-LQR environments
=====================================================================

`sac_lqr.py` trains a SAC prior specifically on the plain LQR env (with an
optimal-controller-based convergence metric). This script does the same thing
for the *harder* registered envs -- ``clqr``, ``switched``, ``terminal`` -- which
have no analytical optimum, so:

  * the behavior (data-collection) policy is still MPPI(H=5) rolled out through
    the env's own model (MPPI is env-agnostic via the batched interface), and
  * convergence is tracked by how much the critic's Q-values stop changing on a
    fixed random reference batch (no optimal gain K needed), plus a max-episode
    cap.

The SAC actor/critic sizes are taken from each env's ``state_dim`` / ``action_dim``
(clqr: 6-D state, 3-D du action; switched: 3-D state, 5-D action = control +
mode-logits; terminal: 3-D / 3-D), and the checkpoint stores those dims so the
prior loaders (``policy_prior_CEM.SACPrior``, ``registry._build_sac``) can rebuild
the network. Each env's prior is saved to ``sac_{env}.pt``.

Usage
-----
    python train_prior.py --env all                 # clqr, switched, terminal
    python train_prior.py --env switched
    python train_prior.py --env clqr --max-episodes 300 --seed 0

NB: LQR is intentionally excluded (train it with ``sac_lqr.py``). Requires torch.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from registry import ENV_REGISTRY
from sac_lqr import SAC, SACConfig, ReplayBuffer, MPPIBehavior


TRAINABLE = ["clqr", "switched", "terminal"]          # every registered env but lqr


# --------------------------------------------------------------------------- #
# generic evaluation: greedy SAC return vs. the MPPI behavior's return
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(agent, env_make, behavior_make, T, n_episodes=10, seed=1234):
    rng = np.random.default_rng(seed)
    sac_returns, mppi_returns = [], []
    for _ in range(n_episodes):
        env = env_make()
        s0 = rng.normal(0.0, getattr(env, "init_state_std", 1.0), size=env.state_dim)

        s = env.reset(state=s0.copy())
        tot = 0.0
        for _ in range(T):
            s, r, term, trunc, _ = env.step(agent.act(s, deterministic=True))
            tot += r
            if term or trunc:
                break
        sac_returns.append(tot)

        env = env_make()
        beh = behavior_make()
        beh.reset()
        s = env.reset(state=s0.copy())
        tot = 0.0
        for _ in range(T):
            s, r, term, trunc, _ = env.step(beh.act(s))
            tot += r
            if term or trunc:
                break
        mppi_returns.append(tot)
    return float(np.mean(sac_returns)), float(np.mean(mppi_returns))


# --------------------------------------------------------------------------- #
# train one env's prior
# --------------------------------------------------------------------------- #
def train_env(
    env_name: str,
    max_episodes: int = 200,
    t_trunc: int | None = None,
    critic_tol: float = 1e-3,
    critic_patience: int = 20,
    warmup_episodes: int = 20,
    grad_steps_per_episode: int = 100,
    explore_std: float = 0.5,
    mppi_num_samples: int = 256,
    buffer_capacity: int = 1_000_000,
    eval_every: int = 25,
    seed: int = 0,
):
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"unknown env {env_name!r}")
    if env_name == "lqr":
        raise ValueError("train LQR with sac_lqr.py, not train_prior.py")

    espec = ENV_REGISTRY[env_name]
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # data-collection truncation: the env's task horizon if finite (terminal),
    # else a short stabilization window.
    probe = espec.make(seed)
    T_trunc = int(t_trunc) if t_trunc is not None else min(getattr(probe, "max_steps", 30), 30)
    sdim, adim = probe.state_dim, probe.action_dim
    a_lo, a_hi = probe.action_low, probe.action_high

    env_make = lambda: espec.make(seed)                      # real env the agent acts in
    model_make = lambda: espec.make(seed + 1)                # deterministic model for MPPI
    behavior_make = lambda: MPPIBehavior(
        model_make(), horizon=5, num_samples=mppi_num_samples,
        explore_std=explore_std, seed=seed + 2,
    )

    cfg = SACConfig(
        state_dim=sdim, action_dim=adim,
        action_low=a_lo, action_high=a_hi,
        target_entropy=-float(adim),
    )
    agent = SAC(cfg, device)
    buffer = ReplayBuffer(buffer_capacity, sdim, adim, device)

    # fixed random reference batch to measure the critic's function-space change
    ref_rng = np.random.default_rng(seed + 123)
    ref_s = ref_rng.normal(0.0, getattr(probe, "init_state_std", 1.0), size=(512, sdim))
    ref_a = ref_rng.uniform(a_lo, a_hi, size=(512, adim))
    ref_s_t = torch.as_tensor(ref_s, dtype=torch.float32, device=device)
    ref_a_t = torch.as_tensor(ref_a, dtype=torch.float32, device=device)
    prev_q_ref = None

    ema_q_delta = None
    ema_beta = 0.9
    below_tol_streak = 0
    t0 = time.time()

    print(f"=== training prior on env '{env_name}' ({espec.label}) ===")
    print(f"state_dim={sdim} action_dim={adim} action_bounds=[{a_lo}, {a_hi}] "
          f"T_trunc={T_trunc} behavior=MPPI(H=5, N={mppi_num_samples}, explore={explore_std})")
    print("-" * 68)

    stop_reason = "max_episodes"
    ep = 0
    for ep in range(1, max_episodes + 1):
        # ---- collect one MPPI-driven episode ------------------------------
        env = env_make()
        behavior = behavior_make()
        behavior.reset()
        s = env.reset()
        for t in range(T_trunc):
            a = behavior.act(s)
            s2, r, term, trunc, _ = env.step(a)
            done = 1.0 if term else 0.0
            buffer.add(s, a, r, s2, done)
            s = s2
            if term or trunc:
                break

        # ---- gradient updates ---------------------------------------------
        if ep > warmup_episodes and len(buffer) >= cfg.batch_size:
            for _ in range(grad_steps_per_episode):
                agent.update(buffer)

            # ---- convergence: relative change of Q on the reference batch --
            q_now = agent.q_reference(ref_s_t, ref_a_t)
            if prev_q_ref is not None:
                dq = (q_now - prev_q_ref).abs().mean().item()
                scale = prev_q_ref.abs().mean().item() + 1e-6
                rel_q = dq / scale
                ema_q_delta = rel_q if ema_q_delta is None else \
                    ema_beta * ema_q_delta + (1 - ema_beta) * rel_q
                below_tol_streak = below_tol_streak + 1 if ema_q_delta < critic_tol else 0
            prev_q_ref = q_now

        # ---- logging + eval -----------------------------------------------
        if ep % eval_every == 0 or ep == 1:
            sac_r, mppi_r = evaluate(agent, env_make, behavior_make, T_trunc)
            q_str = f"{ema_q_delta:.2e}" if ema_q_delta is not None else "  n/a "
            print(f"ep {ep:5d} | buf {len(buffer):7d} | critic_dQ {q_str} "
                  f"| streak {below_tol_streak:3d} | eval SAC {sac_r:10.2f}  MPPI {mppi_r:10.2f} "
                  f"| {time.time()-t0:5.0f}s")

        if ema_q_delta is not None and below_tol_streak >= critic_patience:
            stop_reason = "critic_converged"
            print(f"\nCritic Q-change below tol ({critic_tol:.1e}) for "
                  f"{critic_patience} episodes at episode {ep}.")
            break

    # ---- final eval + save ---------------------------------------------- #
    sac_r, mppi_r = evaluate(agent, env_make, behavior_make, T_trunc, n_episodes=50)
    save_path = f"sac_{env_name}.pt"
    agent.save(save_path, extra=dict(
        stop_reason=stop_reason, episodes=ep, env=env_name, T_trunc=T_trunc,
        eval_sac_reward=sac_r, eval_mppi_reward=mppi_r, seed=seed,
    ))
    print("-" * 68)
    print(f"Stopped ({stop_reason}) after {ep} episodes.")
    print(f"Final eval  SAC {sac_r:.2f}  vs  MPPI {mppi_r:.2f}  (gap {sac_r - mppi_r:+.2f})")
    print(f"Saved SAC prior -> {os.path.abspath(save_path)}\n")
    return save_path


def main():
    p = argparse.ArgumentParser(description="Train SAC policy priors for the non-LQR envs.")
    p.add_argument("--env", default="all",
                   help="'all' (clqr, switched, terminal) or a single env name")
    p.add_argument("--max-episodes", type=int, default=200)
    p.add_argument("--t-trunc", type=int, default=None,
                   help="data-collection episode length (default: env task horizon, capped at 30)")
    p.add_argument("--critic-tol", type=float, default=1e-3)
    p.add_argument("--critic-patience", type=int, default=20)
    p.add_argument("--grad-steps-per-episode", type=int, default=100)
    p.add_argument("--explore-std", type=float, default=0.5)
    p.add_argument("--mppi-num-samples", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    targets = TRAINABLE if args.env == "all" else [args.env]
    for env_name in targets:
        train_env(
            env_name,
            max_episodes=args.max_episodes,
            t_trunc=args.t_trunc,
            critic_tol=args.critic_tol,
            critic_patience=args.critic_patience,
            grad_steps_per_episode=args.grad_steps_per_episode,
            explore_std=args.explore_std,
            mppi_num_samples=args.mppi_num_samples,
            eval_every=args.eval_every,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
