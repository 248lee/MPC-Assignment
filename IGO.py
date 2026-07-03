"""
IGO-ML planner for MPC
======================

A drop-in sibling of the Phase II planners (see ``phase2.py``).  Where CEM
*hard-replaces* the Gaussian each iteration (``mu, sigma <- elite mean/std``),
IGO-ML performs a **soft** (step-size ``dt``) update of the natural parameters
and adds a *variance-injection* term that counteracts premature convergence:

    variance_injection = dt * (1 - dt) * (mu_star - mu)^2
    sigma^2 <- (1 - dt) * sigma^2 + dt * sigma_star^2 + variance_injection
    mu      <- (1 - dt) * mu      + dt * mu_star

This is exactly the update studied on the toy linear problem in
``premature.py`` -- here it is lifted from a scalar to the
(horizon x action_dim) sampling distribution over action sequences and used as
the inner loop of MPC, with the same warm-start / receding-horizon wrapper as
CEM and MPPI.

Note: with ``dt = 1`` the injection term vanishes and the update reduces to
plain CEM (``mu = mu_star``, ``sigma^2 = sigma_star^2``); ``dt < 1`` is what
makes IGO-ML a *smoothed* CEM that resists collapsing the search variance.
"""

from __future__ import annotations

import numpy as np

from lqr_env import LQREnv

# reuse the shared trajectory-evaluation and rollout helpers from Phase II so
# IGO plugs into exactly the same MPC harness as CEM / MPPI.
from phase2 import _rollout_returns, run_episode


# --------------------------------------------------------------------------- #
# IGO-ML
# --------------------------------------------------------------------------- #
class IGOPlanner:
    def __init__(
        self,
        env: LQREnv,
        horizon: int = 15,
        num_samples: int = 1000,
        num_elites: int = 50,
        max_iters: int = 1000,
        sigma_init: float = 0.2,
        dt: float = 0.5,
        tol_mu: float = 1e-3,
        tol_sigma: float = 1e-3,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        horizon      : planning horizon H.
        num_samples  : sequences sampled per iteration (N).
        num_elites   : how many top sequences drive the update (K).
        max_iters    : iteration budget I per timestep.
        sigma_init   : std the Gaussian is (re)initialized to each timestep.
        dt           : IGO-ML step size in (0, 1].  dt=1 recovers CEM; smaller
                       dt gives a slower, variance-preserving update.
        tol_mu       : stop when ||mu - mu_prev|| < tol_mu.
        tol_sigma    : stop when max(sigma) < tol_sigma (distribution collapsed).
        gamma        : discount for the planned return.
        """
        self.env = env
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.max_iters = int(max_iters)
        self.sigma_init = float(sigma_init)
        self.dt = float(dt)
        self.tol_mu = float(tol_mu)
        self.tol_sigma = float(tol_sigma)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Clear the warm-started mean (call between episodes)."""
        self.mu = np.zeros((self.horizon, self.env.action_dim))

    def plan(self, state: np.ndarray) -> np.ndarray:
        H, N, K = self.horizon, self.num_samples, self.num_elites
        adim = self.env.action_dim
        lo, hi = self.env.action_low, self.env.action_high
        dt = self.dt

        mu = self.mu                                   # warm-started mean
        sigma = np.full((H, adim), self.sigma_init)    # reset exploration std

        for times in range(self.max_iters):
            mu_prev = mu

            # 1. sample + clip
            noise = self.rng.normal(size=(N, H, adim))
            actions = np.clip(mu + sigma * noise, lo, hi)

            # 2. evaluate
            returns = _rollout_returns(self.env, state, actions, self.gamma)

            # 3. top-K elites
            elite_idx = np.argpartition(returns, -K)[-K:]
            elites = actions[elite_idx]

            # 4. elite statistics
            mu_star = elites.mean(axis=0)
            sigma_sq_star = elites.var(axis=0)

            # 5. IGO-ML soft update (in variance space) + variance injection
            sigma_sq = sigma ** 2
            variance_injection = dt * (1 - dt) * (mu_star - mu) ** 2
            sigma_sq = (1 - dt) * sigma_sq + dt * sigma_sq_star + variance_injection
            mu = (1 - dt) * mu + dt * mu_star
            sigma = np.sqrt(sigma_sq)

            # convergence
            if np.linalg.norm(mu - mu_prev) < self.tol_mu or sigma.max() < self.tol_sigma:
                break

        action = mu[0].copy()
        if times == self.max_iters - 1:
            print("\nHit Max Iter")
        # warm start: shift the plan forward by one step
        self.mu = np.vstack([mu[1:], np.zeros((1, adim))])
        return action

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.plan(state)


if __name__ == "__main__":
    env = LQREnv(noise_std=0.0, seed=0)
    s0 = np.array([1.0, -1.0, 0.5])

    igo = IGOPlanner(env, horizon=15, num_samples=1000, dt=0.5, seed=0)
    total, traj = run_episode(env, igo, init_state=s0, T=200)
    print(f"IGO-ML (H=15, N=1000, K=50, dt=0.5) reward (T=200): {total:.4f}")
    print(f"  final state: {np.round(traj[-1], 5)}")
