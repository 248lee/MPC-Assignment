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
from scipy.stats import qmc

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
        max_iters: int = 1e10,
        sigma_init: float = 0.2,
        dt: float = 0.5,
        tol_mu: float = 1e-3,
        tol_sigma: float = 1e-3,
        gamma: float = 1.0,
        seed: int | None = None,
        detect_premature: bool = False,
        premature_r_inner: float = 3.0,
        premature_r_outer: float = 5.0,
        premature_samples_per_dim: int = 16,
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
        premature_r_inner : inner shell radius (in sigma) -- samples must lie
                       OUTSIDE this box to count (default 3).
        premature_r_outer : outer shell radius (in sigma) -- samples must lie
                       INSIDE this box (default 5).
        premature_samples_per_dim : Sobol points per plan dimension; the total
                       is rounded up to a power of two (default 16).
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
        self.detect_premature = detect_premature
        self.premature_r_inner = float(premature_r_inner)
        self.premature_r_outer = float(premature_r_outer)
        self.premature_samples_per_dim = int(premature_samples_per_dim)
        # Independent stream for the Sobol premature check so that toggling
        # detection never perturbs the planning RNG (self.rng) used for sampling.
        self.sobol_rng = np.random.default_rng(None if seed is None else seed + 10_000)
        self.last_premature_convergence: bool = False
        self.reset()

    def reset(self) -> None:
        """Clear the warm-started mean (call between episodes)."""
        self.mu = np.zeros((self.horizon, self.env.action_dim))

    def _check_premature(self, state: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> bool:
        """Falsification test via a scrambled-Sobol probe of the box shell.

        We draw quasi-random neighbours of the converged plan ``mu`` and keep
        those in the shell between the inner (``r_inner * sigma``) and outer
        (``r_outer * sigma``) axis-aligned boxes -- a sample is kept iff, for
        every dimension, ``|delta_d| <= r_outer * sigma_d`` (inside the outer
        box, true by construction) AND, for at least one dimension,
        ``|delta_d| > r_inner * sigma_d`` (outside the inner box). If ANY kept
        neighbour beats ``mu``, the plan converged prematurely.

        The sample count grows linearly with the plan dimension ``D = H*adim``
        (``samples_per_dim`` points per dimension) and is rounded UP to a power
        of two so Sobol keeps its low-discrepancy balance. The sequence is
        Owen-scrambled with a per-call seed so successive timesteps probe
        different directions instead of a fixed lattice with blind spots.
        """
        lo, hi = self.env.action_low, self.env.action_high
        H, adim = mu.shape
        D = H * adim
        mu_return = _rollout_returns(self.env, state, mu[None], self.gamma)[0]

        # N = 2**m, the smallest power of two >= samples_per_dim * D.
        m = max(1, int(np.ceil(np.log2(self.premature_samples_per_dim * D))))
        u = qmc.Sobol(d=D, scramble=True, seed=self.sobol_rng).random_base2(m)

        sig = sigma.reshape(D)
        delta = (2.0 * u - 1.0) * (self.premature_r_outer * sig)   # inside +/-r_outer*sigma
        keep = (np.abs(delta) > self.premature_r_inner * sig).any(axis=1)  # outside inner box
        delta = delta[keep]
        if delta.size == 0:
            return False

        cand = np.clip(mu.reshape(D) + delta, lo, hi).reshape(-1, H, adim)
        returns = _rollout_returns(self.env, state, cand, self.gamma)
        return bool((returns > mu_return).any())

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
        if self.detect_premature:
            self.last_premature_convergence = self._check_premature(state, mu, sigma)
        if times == self.max_iters - 1:
            print("\npure IGO Hit Max Iter")
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
