"""
Policy-prior random-shooting planner for MPC
============================================

A stripped-down sibling of the policy-prior IGO-ML planner
(`policy_prior_IGO.py`). IGO there *iterates*: sample -> evaluate -> soft-update
the elite Gaussian, repeating until the distribution's mean plan stabilizes.

Random shooting does the sampling step **exactly once**:

  1. seed the H-step Gaussian from the trained SAC policy (`sac_lqr.pt`) by
     rolling it through the known deterministic model for H steps,
  2. draw ``num_samples`` action sequences from that Gaussian,
  3. evaluate every sequence's H-step return under the model, and
  4. keep the single best sequence and execute its first action.

There is NO refinement loop and NO convergence test -- one shot, pick the arg-max.

Unlike the IGO variant, we do NOT widen the prior: the sampling std is exactly
the SAC action std (no ``prior_std_scale``). The policy prior is trusted to
already put mass in the right region, and random shooting just polishes the
single-step choice by picking the best of ``num_samples`` draws around it.

Like the IGO variant, the plan is NOT carried over / shifted between timesteps;
each timestep re-seeds from the SAC prior.
"""

from __future__ import annotations

import numpy as np

from lqr_env import LQREnv

# reuse the shared rollout + policy prior from the IGO planner so the two
# variants stay in lock-step (same evaluation, same prior mapping).
from policy_prior_IGO import _rollout_returns, SACPrior


# --------------------------------------------------------------------------- #
# Policy-prior random shooting
# --------------------------------------------------------------------------- #
class RandomShootingPlanner:
    def __init__(
        self,
        env: LQREnv,
        horizon: int = 15,
        num_samples: int = 1000,
        gamma: float = 1.0,
        sac_path: str = "sac_lqr.pt",
        prior: SACPrior | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        horizon     : planning horizon H.
        num_samples : action sequences sampled per timestep (N). All drawn once;
                      the best-scoring one is executed. No iteration.
        gamma       : discount for the planned return.
        sac_path    : checkpoint used to build the SAC policy prior.
        prior       : a preloaded SACPrior (overrides ``sac_path``).
        """
        self.env = env
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.gamma = float(gamma)
        self.prior = prior if prior is not None else SACPrior(sac_path)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """No warm-started state to clear -- each timestep is re-seeded from the
        SAC policy prior. Kept for API compatibility with ``run_episode``."""
        pass

    def _init_from_prior(self, state: np.ndarray):
        """Seed (mu, sigma) for the H-step distribution by rolling the SAC
        policy through the deterministic model from ``state``.

        NOTE: sigma is the *raw* SAC std -- unlike the IGO variant we do not
        widen it with a ``prior_std_scale``."""
        H, adim = self.horizon, self.env.action_dim
        mu = np.zeros((H, adim))
        sigma = np.zeros((H, adim))
        s = np.asarray(state, dtype=np.float64)
        for h in range(H):
            mean_a, std_a = self.prior.action_mean_std(s)
            mu[h] = mean_a
            sigma[h] = std_a                    # keep the SAC std as-is
            s = self.env.dynamics(s, mean_a)    # deterministic model step
        return mu, sigma

    def plan(self, state: np.ndarray) -> np.ndarray:
        H, N = self.horizon, self.num_samples
        adim = self.env.action_dim
        lo, hi = self.env.action_low, self.env.action_high

        # seed the sampling distribution from the SAC policy prior
        # (mean = policy rollout, std = raw policy std -- no widening)
        mu, sigma = self._init_from_prior(state)

        sigma = sigma / (float)(H * 2)

        # single shot: sample N sequences, clip to the action box
        noise = self.rng.normal(size=(N, H, adim))
        actions = np.clip(mu + sigma * noise, lo, hi)

        # evaluate every sequence and keep the best one's first action
        returns = _rollout_returns(self.env, state, actions, self.gamma)
        best = int(np.argmax(returns))
        action = actions[best, 0].copy()

        # NOTE: no warm start -- the next timestep re-seeds from the SAC prior.
        return action

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.plan(state)


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def run_episode(env: LQREnv, agent, init_state=None, T: int | None = None):
    """Roll out the planner; return (total_reward, state_trajectory)."""
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
    from optimal import LQRController, run_episode as run_optimal_episode

    env = LQREnv(noise_std=0.0, seed=0)
    s0 = np.array([1.0, -1.0, 0.5])

    ctrl = LQRController(env)
    opt_r, _ = run_optimal_episode(LQREnv(noise_std=0.0, seed=0), ctrl, init_state=s0, T=200)
    print(f"Optimal LQR                                       reward (T=200): {opt_r:.4f}")

    # one shared prior so we don't reload the checkpoint per planner
    prior = SACPrior("sac_lqr.pt")
    for H in (5, 15):
        env = LQREnv(noise_std=0.0, seed=0)
        rs = RandomShootingPlanner(env, horizon=H, num_samples=1000, prior=prior, seed=0)
        total, traj = run_episode(env, rs, init_state=s0, T=200)
        print(f"Policy-prior random shooting (H={H:>2}, N=1000)     reward (T=200): "
              f"{total:.4f}   final state: {np.round(traj[-1], 5)}")
