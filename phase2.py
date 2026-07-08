"""
Phase II — CEM and MPPI planners for MPC
========================================

Phase I sampled action sequences from a *fixed* proposal and kept the best one.
Phase II *refines* the sampling distribution across iterations so the samples
concentrate on the high-return action sequences:

  * CEM  -- keep a Gaussian over action sequences; each iteration keep the
            top-K elites and REFIT both the mean and the std to them.
  * MPPI -- keep one nominal control sequence; each iteration update it with a
            softmax (reward-weighted) average of N noisy perturbations.

Both are used as the inner loop of MPC: replan from the current state, execute
only the first action, and WARM START the next timestep with the shifted plan.
"""

from __future__ import annotations

import numpy as np

from lqr_env import LQREnv


# --------------------------------------------------------------------------- #
# shared trajectory evaluation
# --------------------------------------------------------------------------- #
def _rollout_returns(env: LQREnv, state: np.ndarray, actions: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted H-step return of each sequence in ``actions`` (N, H, adim).

    Uses the (known) deterministic model via the env's batched helpers.
    """
    N = actions.shape[0]
    H = actions.shape[1]
    states = np.tile(np.asarray(state, dtype=np.float64), (N, 1))   # (N, n)
    returns = np.zeros(N)
    discount = 1.0
    for h in range(H):
        a = actions[:, h, :]
        returns += discount * env.reward_batch(states, a)
        states = env.dynamics_batch(states, a)
        discount *= gamma
    return returns


# --------------------------------------------------------------------------- #
# CEM
# --------------------------------------------------------------------------- #
class CEMPlanner:
    def __init__(
        self,
        env: LQREnv,
        horizon: int = 15,
        num_samples: int = 1000,
        num_elites: int = 50,
        max_iters: int = 1000,
        sigma_init: float = 0.2,
        tol_mu: float = 1e-3,
        tol_sigma: float = 1e-3,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        horizon      : planning horizon H.
        num_samples  : sequences sampled per CEM iteration (N).
        num_elites   : how many top sequences refit the Gaussian (K).
        max_iters    : iteration budget I per timestep.
        sigma_init   : std the Gaussian is (re)initialized to each timestep.
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

        mu = self.mu                              # warm-started mean
        sigma = np.full((H, adim), self.sigma_init)   # reset exploration

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

            # 4. refit BOTH mean and std
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0)

            # convergence
            if np.linalg.norm(mu - mu_prev) < self.tol_mu or sigma.max() < self.tol_sigma:
                break
            # print(f"\riter {times:4d}  sigma.max={sigma.max():.5f}", end="", flush=True)

        action = mu[0].copy()
        if times == self.max_iters - 1:
            print("\nCEM Hit Max Iter")
        # warm start: shift the plan forward by one step
        self.mu = np.vstack([mu[1:], np.zeros((1, adim))])
        return action

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.plan(state)


# --------------------------------------------------------------------------- #
# MPPI
# --------------------------------------------------------------------------- #
class MPPIPlanner:
    def __init__(
        self,
        env: LQREnv,
        horizon: int = 15,
        num_samples: int = 1000,
        temperature: float = 20.0,
        sigma: float = 0.2,
        max_iters: int = 1000,
        tol: float = 1e-3,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        horizon      : planning horizon H.
        num_samples  : noisy perturbations sampled per iteration (N).
        temperature  : softmax temperature lambda (small -> greedy).
        sigma        : std of the exploration noise around the nominal.
        max_iters    : iteration budget I per timestep.
        tol          : stop when ||mu_0 - mu_0_prev|| < tol (first-action change).
        gamma        : discount for the planned return.
        """
        self.env = env
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.sigma = float(sigma)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Clear the warm-started nominal control sequence (between episodes)."""
        self.mu = np.zeros((self.horizon, self.env.action_dim))

    def plan(self, state: np.ndarray) -> np.ndarray:
        H, N = self.horizon, self.num_samples
        adim = self.env.action_dim
        lo, hi = self.env.action_low, self.env.action_high
        lam = self.temperature

        mu = self.mu                              # warm-started nominal

        for times in range(self.max_iters):
            a0_prev = mu[0].copy()

            # 1. sample noisy perturbations of the nominal + clip
            noise = self.rng.normal(scale=self.sigma, size=(N, H, adim))
            actions = np.clip(mu + noise, lo, hi)

            # 2. evaluate
            returns = _rollout_returns(self.env, state, actions, self.gamma)

            # 3. softmax weights (max-subtracted for stability)
            beta = returns.max()
            weights = np.exp((returns - beta) / lam)
            weights /= weights.sum()

            # 4. reward-weighted average update of the nominal
            mu = np.einsum("n,nhd->hd", weights, actions)

            # convergence on the first (executed) action
            if np.linalg.norm(mu[0] - a0_prev) < self.tol:
                break

        action = mu[0].copy()
        # if times == self.max_iters - 1:
        #     print("\nHit Max Iter")
        # warm start: shift the nominal forward by one step
        self.mu = np.vstack([mu[1:], np.zeros((1, adim))])
        return action

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.plan(state)


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def run_episode(env: LQREnv, agent, init_state=None, T: int | None = None):
    """Roll out a (warm-started) planner; return (total_reward, state_trajectory)."""
    if hasattr(agent, "reset"):
        agent.reset()
    s = env.reset(state=init_state)
    if T is None:
        T = env.max_steps
    total = 0.0
    traj = [s.copy()]
    for _ in range(T):
        a = agent.act(s)
        s, r, term, trunc, _ = env.step(a)
        total += r
        traj.append(s.copy())
        if term or trunc:
            break
    return total, np.array(traj)


if __name__ == "__main__":
    env = LQREnv(noise_std=0.0, seed=0)
    s0 = np.array([1.0, -1.0, 0.5])

    cem = CEMPlanner(env, horizon=15, num_samples=1000, seed=0)
    total, traj = run_episode(env, cem, init_state=s0, T=200)
    print(f"CEM  (H=15, N=1000, K=50) reward (T=200): {total:.4f}")
    print(f"  final state: {np.round(traj[-1], 5)}")

    # mppi = MPPIPlanner(env, horizon=15, num_samples=1000, seed=0)
    # total, traj = run_episode(env, mppi, init_state=s0, T=200)
    # print(f"MPPI (H=15, N=1000, lambda=20) reward (T=200): {total:.4f}")
    # print(f"  final state: {np.round(traj[-1], 5)}")
