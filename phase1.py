"""
Phase I — MPC with Finite-Horizon Random Shooting
=================================================

At every timestep the controller:

    1. starts from the current state,
    2. samples ``num_samples`` action sequences of length ``horizon`` (H),
    3. evaluates each predicted trajectory with the (known) model,
    4. executes only the first action of the best sequence,
    5. observes the next state,
    6. replans.

"Random shooting" = step 2 samples action sequences from a fixed distribution
(here, uniform over the action bounds) instead of optimizing the distribution.

Optionally a terminal value function V(s_H) can be added to the H-step return.
With the *optimal* LQR value function this turns short-horizon planning into
something near-optimal — which is exactly the point the assignment makes about
"when a terminal value function matters."
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from lqr_env import LQREnv


class RandomShootingMPC:
    def __init__(
        self,
        env: LQREnv,
        horizon: int = 15,
        num_samples: int = 1000,
        sampler: str = "gaussian",
        sigma: float = 1.0,
        terminal_value: Callable[[np.ndarray], np.ndarray] | None = None,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        env            : model used for planning (its A, B, Q, R, bounds).
        horizon        : planning horizon H.
        num_samples    : number of random action sequences sampled per step.
        sampler        : "gaussian" (N(0, sigma^2 I), clipped to bounds) or
                         "uniform" (uniform over the action bounds).
        sigma          : std of the Gaussian proposal (ignored for uniform).
        terminal_value : optional vectorized V(states)->(N,) added after H steps.
        gamma          : discount factor for the planned return.

        Note on `sampler`: a Gaussian proposal centred at 0 concentrates samples
        where good actions actually live (the optimal action here is order 1),
        so best-of-N finds far better sequences than uniform over the wide
        bounds. This is still pure *random shooting* -- the proposal is fixed,
        not refined across iterations (that refinement is what CEM/MPPI add).
        """
        self.env = env
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.sampler = sampler
        self.sigma = float(sigma)
        self.terminal_value = terminal_value
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

    def _sample_actions(self, N: int, H: int, adim: int) -> np.ndarray:
        """Return (N, H, adim) action sequences from the chosen proposal."""
        if self.sampler == "uniform":
            return self.rng.uniform(
                self.env.action_low, self.env.action_high, size=(N, H, adim)
            )
        elif self.sampler == "gaussian":
            a = self.rng.normal(0.0, self.sigma, size=(N, H, adim))
            return np.clip(a, self.env.action_low, self.env.action_high)
        raise ValueError(f"unknown sampler: {self.sampler!r}")

    def plan(self, state: np.ndarray) -> np.ndarray:
        """Return the first action of the best sampled sequence."""
        H, N = self.horizon, self.num_samples
        adim = self.env.action_dim

        actions = self._sample_actions(N, H, adim)   # (N, H, adim)

        states = np.tile(np.asarray(state, dtype=np.float64), (N, 1))   # (N, n)
        returns = np.zeros(N)
        discount = 1.0
        for h in range(H):
            a = actions[:, h, :]
            returns += discount * self.env.reward_batch(states, a)
            states = self.env.dynamics_batch(states, a)
            discount *= self.gamma

        if self.terminal_value is not None:
            returns += discount * self.terminal_value(states)

        best = int(np.argmax(returns))
        return actions[best, 0, :].copy()

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.plan(state)


def run_episode(env: LQREnv, agent: RandomShootingMPC, init_state=None, T: int | None = None):
    """Roll out the MPC agent; return (total_reward, state_trajectory)."""
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


def horizon_sweep(
    env_kwargs: dict,
    horizons,
    num_samples: int = 1000,
    init_state=None,
    T: int = 200,
    seed: int = 0,
):
    """Run one episode per horizon; return dict {H: total_reward}."""
    results = {}
    for H in horizons:
        env = LQREnv(seed=seed, **env_kwargs)
        agent = RandomShootingMPC(env, horizon=H, num_samples=num_samples, seed=seed)
        total, _ = run_episode(env, agent, init_state=init_state, T=T)
        results[H] = total
        print(f"H={H:>3}  episode_reward={total:10.3f}")
    return results


if __name__ == "__main__":
    env = LQREnv(noise_std=0.0, seed=0)
    s0 = np.array([1.0, -1.0, 0.5])

    agent = RandomShootingMPC(env, horizon=15, num_samples=1000, seed=0)
    total, traj = run_episode(env, agent, init_state=s0, T=200)
    print(f"Random-shooting MPC (H=15, N=1000) reward (T=200): {total:.4f}")
    print(f"Final state: {np.round(traj[-1], 5)}")
