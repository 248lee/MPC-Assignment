"""
LQR Environment (3D state, 3D action)
=====================================

Linear dynamics with a quadratic cost, used as the planning testbed for the
MPC / CEM / MPPI assignment.

Dynamics (deterministic):
    s_{t+1} = A s_t + B a_t

Dynamics (stochastic):
    s_{t+1} = A s_t + B a_t + eps_t,   eps_t ~ N(0, noise_std^2 I)

Reward:
    r(s_t, a_t) = -( s_t^T Q s_t + a_t^T R a_t )

The API mirrors the Gymnasium convention (reset / step) so the same object can
be used both as the *real* environment that an agent acts in, and as the
*model* that a planner rolls out internally.
"""

from __future__ import annotations

import numpy as np


class LQREnv:
    """A 3-dimensional Linear-Quadratic-Regulator environment."""

    def __init__(
        self,
        A: np.ndarray | None = None,
        B: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        noise_std: float = 0.0,
        max_steps: int = 10_000,
        action_low: float = -10.0,
        action_high: float = 10.0,
        init_state_std: float = 1.0,
        seed: int | None = None,
    ):
        self.state_dim = 3
        self.action_dim = 3

        # --- system matrices (sensible, mildly unstable defaults) ----------
        # A is slightly unstable so that *not* controlling is costly: this is
        # what makes planning horizon actually matter in later phases.
        if A is None:
            A = np.array(
                [
                    [1.10, 0.10, 0.00],
                    [0.00, 1.05, 0.10],
                    [0.05, 0.00, 1.10],
                ],
                dtype=np.float64,
            )
        if B is None:
            B = np.array(
                [
                    [1.00, 0.00, 0.00],
                    [0.00, 1.00, 0.00],
                    [0.00, 0.00, 1.00],
                ],
                dtype=np.float64,
            )
        if Q is None:
            Q = np.eye(self.state_dim, dtype=np.float64)
        if R is None:
            R = 0.1 * np.eye(self.action_dim, dtype=np.float64)

        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)

        self._check_shapes()

        self.noise_std = float(noise_std)
        self.max_steps = int(max_steps)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.init_state_std = float(init_state_std)

        self.rng = np.random.default_rng(seed)

        self.state: np.ndarray | None = None
        self.t = 0

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def _check_shapes(self) -> None:
        n, m = self.state_dim, self.action_dim
        assert self.A.shape == (n, n), f"A must be {(n, n)}, got {self.A.shape}"
        assert self.B.shape == (n, m), f"B must be {(n, m)}, got {self.B.shape}"
        assert self.Q.shape == (n, n), f"Q must be {(n, n)}, got {self.Q.shape}"
        assert self.R.shape == (m, m), f"R must be {(m, m)}, got {self.R.shape}"

    # ------------------------------------------------------------------ #
    # core gym-style API
    # ------------------------------------------------------------------ #
    def reset(self, state: np.ndarray | None = None, seed: int | None = None):
        """Reset the environment and return the initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if state is None:
            self.state = self.rng.normal(
                loc=0.0, scale=self.init_state_std, size=self.state_dim
            )
        else:
            self.state = np.asarray(state, dtype=np.float64).copy()
            assert self.state.shape == (self.state_dim,)

        self.t = 0
        return self.state.copy()

    def step(self, action: np.ndarray):
        """Apply ``action``, advance one step, return (next_state, reward, terminated, truncated, info)."""
        assert self.state is not None, "Call reset() before step()."
        a = np.clip(
            np.asarray(action, dtype=np.float64), self.action_low, self.action_high
        )
        assert a.shape == (self.action_dim,), f"action must be {(self.action_dim,)}"

        reward = self.reward(self.state, a)

        next_state = self.A @ self.state + self.B @ a
        if self.noise_std > 0.0:
            next_state = next_state + self.rng.normal(
                0.0, self.noise_std, size=self.state_dim
            )

        self.state = next_state
        self.t += 1

        terminated = False
        truncated = self.t >= self.max_steps
        info = {"t": self.t}
        return self.state.copy(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # reward + dynamics helpers (also usable as a model by a planner)
    # ------------------------------------------------------------------ #
    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """r(s, a) = -(s^T Q s + a^T R a)."""
        s = np.asarray(state, dtype=np.float64)
        a = np.asarray(action, dtype=np.float64)
        return float(-(s @ self.Q @ s + a @ self.R @ a))

    def dynamics(self, state: np.ndarray, action: np.ndarray, noise: bool = False) -> np.ndarray:
        """Predict the next state. ``noise=False`` gives the deterministic model."""
        s = np.asarray(state, dtype=np.float64)
        a = np.clip(np.asarray(action, dtype=np.float64), self.action_low, self.action_high)
        ns = self.A @ s + self.B @ a
        if noise and self.noise_std > 0.0:
            ns = ns + self.rng.normal(0.0, self.noise_std, size=self.state_dim)
        return ns

    # ------------------------------------------------------------------ #
    # batched / vectorized helpers (handy for CEM & MPPI later)
    # ------------------------------------------------------------------ #
    def reward_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Vectorized reward for (N, state_dim) states and (N, action_dim) actions."""
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)
        sQs = np.einsum("ni,ij,nj->n", states, self.Q, states)
        aRa = np.einsum("ni,ij,nj->n", actions, self.R, actions)
        return -(sQs + aRa)

    def dynamics_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Vectorized deterministic dynamics for (N, state_dim), (N, action_dim)."""
        states = np.atleast_2d(states)
        actions = np.atleast_2d(np.clip(actions, self.action_low, self.action_high))
        return states @ self.A.T + actions @ self.B.T

    def sample_action(self) -> np.ndarray:
        """Uniformly sample an action from the action bounds."""
        return self.rng.uniform(self.action_low, self.action_high, size=self.action_dim)


if __name__ == "__main__":
    # quick sanity check
    env = LQREnv(noise_std=0.0, seed=0)
    s = env.reset(state=np.array([1.0, -1.0, 0.5]))
    print("initial state:", s)

    total_r = 0.0
    for _ in range(5):
        a = np.zeros(env.action_dim)          # do nothing -> state should blow up
        s, r, term, trunc, info = env.step(a)
        total_r += r
        print(f"t={info['t']:>2}  reward={r:8.3f}  state={np.round(s, 3)}")
    print("total reward (no control):", round(total_r, 3))

    # check batched helpers match the scalar versions
    S = np.random.randn(4, 3)
    Aa = np.random.randn(4, 3)
    rb = env.reward_batch(S, Aa)
    rs = np.array([env.reward(S[i], Aa[i]) for i in range(4)])
    assert np.allclose(rb, rs), "reward_batch mismatch"
    nb = env.dynamics_batch(S, Aa)
    ns = np.array([env.dynamics(S[i], Aa[i]) for i in range(4)])
    assert np.allclose(nb, ns), "dynamics_batch mismatch"
    print("batched helpers OK")
