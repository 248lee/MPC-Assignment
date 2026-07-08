"""
Harder-than-LQR planning environments
======================================

`lqr_env.py`'s plain `LQREnv` is deliberately the *easiest* case: unconstrained,
fully observed, time-invariant, purely quadratic linear-quadratic control. Its
finite-horizon optimal value is exactly one global quadratic form (the Riccati
matrix P), for any horizon -- so with an exact terminal value even H=1 planning
is optimal, and the benchmark can't tell "the method used the terminal value
well" apart from "the problem had no structure to expose a flaw".

This module adds the three environments the accompanying research note
(`planning_env_research.md`) recommends as strictly harder terminal-value stress
tests, each still cleanly matrix-representable and -- crucially -- each exposing
the *same batched interface the planners already use* so no planner code changes:

    state_dim, action_dim, action_low, action_high        (attributes)
    reset(state=None, seed=None) -> state
    step(action) -> (next_state, reward, terminated, truncated, info)
    reward(state, action) -> float                        (scalar)
    dynamics(state, action) -> next_state                 (scalar, deterministic)
    reward_batch(states (N,sd), actions (N,ad)) -> (N,)
    dynamics_batch(states (N,sd), actions (N,ad)) -> (N,sd)

Envs added here:

  * ConstrainedLQREnv     -- LQR with tight input saturation AND rate limits.
                            State is augmented to [x; u_prev]; the action is the
                            control *increment* du (rate-limited), and u itself
                            is saturated inside the dynamics. The optimal policy
                            becomes piecewise-affine and the value piecewise-
                            quadratic -> the doc's #1 terminal-value benchmark.

  * SwitchedLinearEnv     -- M linear modes with mode-dependent (A,B,Q,R). The
                            mode is *chosen by the planner*: the action carries
                            `M` extra continuous mode-logits and the env takes
                            argmax. Continuous samplers (CEM/MPPI/random shoot)
                            thus optimize a relaxed mode selection. The value is
                            a pointwise-min of finitely many quadratics -> the
                            doc's #2 mode-aware terminal benchmark.

  * NonQuadraticTerminalLQREnv -- linear dynamics + quadratic stage cost, but a
                            non-quadratic (Huber / l1) terminal cost applied at a
                            fixed finite task horizon. Only the terminal breaks
                            Riccati closure, so any performance gap isolates the
                            terminal-value representation -> the doc's #3 (cleanest
                            terminal ablation). Exposes `terminal_value(states)`
                            so terminal-aware planners can consume the true tail
                            cost.

All envs are pure NumPy (no torch), so they import and run without the deep-RL
stack. Analytical baselines exist only for plain LQR (see `optimal.py`); these
harder envs deliberately break the single-quadratic closed form.
"""

from __future__ import annotations

import time

import numpy as np


# default (mildly unstable) linear plant shared with lqr_env.LQREnv
_A_DEFAULT = np.array(
    [[1.10, 0.10, 0.00],
     [0.00, 1.05, 0.10],
     [0.05, 0.00, 1.10]],
    dtype=np.float64,
)
_B_DEFAULT = np.eye(3, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Constrained LQR (saturation + rate limits, augmented state [x; u_prev])
# --------------------------------------------------------------------------- #
class ConstrainedLQREnv:
    """LQR with input saturation ``u in [u_min, u_max]`` and rate limits
    ``du in [du_min, du_max]``.

    The planner's *action* is the control increment ``du``; the applied control
    is ``u = clip(u_prev + du, u_min, u_max)``. To keep everything Markov and
    matrix-form the state is augmented to ``z = [x; u_prev]`` (dim 6), and the
    dynamics/reward operate on ``z``:

        u        = clip(u_prev + du, u_min, u_max)          # saturation
        x_{t+1}  = A x_t + B u
        u_prev   <- u
        r        = -(x^T Q x + u^T R u + du^T R_du du)

    ``action_low/high`` are the *rate* limits (bounds on ``du``); the amplitude
    saturation is enforced separately inside the dynamics. Tight defaults are
    chosen so both constraints actually bind on the standard init state (the
    unconstrained optimal control here is order 1).
    """

    def __init__(
        self,
        A=None, B=None, Q=None, R=None, R_du=None,
        u_min: float = -0.8, u_max: float = 0.8,
        du_min: float = -0.3, du_max: float = 0.3,
        noise_std: float = 0.0, max_steps: int = 200,
        init_state_std: float = 1.0, seed: int | None = None,
    ):
        self.A = _A_DEFAULT.copy() if A is None else np.asarray(A, np.float64)
        self.B = _B_DEFAULT.copy() if B is None else np.asarray(B, np.float64)
        self.xdim = self.A.shape[0]
        self.udim = self.B.shape[1]
        self.Q = np.eye(self.xdim) if Q is None else np.asarray(Q, np.float64)
        self.R = 0.1 * np.eye(self.udim) if R is None else np.asarray(R, np.float64)
        self.R_du = 0.1 * np.eye(self.udim) if R_du is None else np.asarray(R_du, np.float64)

        self.u_min, self.u_max = float(u_min), float(u_max)
        self.noise_std = float(noise_std)
        self.max_steps = int(max_steps)
        self.init_state_std = float(init_state_std)

        # planner-facing interface: state is [x; u_prev], action is du
        self.state_dim = self.xdim + self.udim
        self.action_dim = self.udim
        self.action_low = float(du_min)
        self.action_high = float(du_max)

        self.rng = np.random.default_rng(seed)
        self.state = None
        self.t = 0

    # -- helpers ---------------------------------------------------------- #
    def _split(self, z):
        return z[..., :self.xdim], z[..., self.xdim:]

    def _apply_u(self, u_prev, du):
        return np.clip(u_prev + du, self.u_min, self.u_max)

    # -- gym-style API ---------------------------------------------------- #
    def reset(self, state=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if state is None:
            x = self.rng.normal(0.0, self.init_state_std, size=self.xdim)
            z = np.concatenate([x, np.zeros(self.udim)])
        else:
            z = np.asarray(state, np.float64).copy()
            assert z.shape == (self.state_dim,)
        self.state = z
        self.t = 0
        return self.state.copy()

    def step(self, action):
        print("Shout out to CLQR")
        du = np.clip(np.asarray(action, np.float64), self.action_low, self.action_high)
        r = self.reward(self.state, du)
        ns = self.dynamics(self.state, du, noise=self.noise_std > 0.0)
        self.state = ns
        self.t += 1
        return self.state.copy(), r, False, self.t >= self.max_steps, {"t": self.t}

    # -- reward / dynamics (also the planner's model) --------------------- #
    def reward(self, state, action):
        x, u_prev = self._split(np.asarray(state, np.float64))
        du = np.clip(np.asarray(action, np.float64), self.action_low, self.action_high)
        u = self._apply_u(u_prev, du)
        return float(-(x @ self.Q @ x + u @ self.R @ u + du @ self.R_du @ du))

    def dynamics(self, state, action, noise=False):
        x, u_prev = self._split(np.asarray(state, np.float64))
        du = np.clip(np.asarray(action, np.float64), self.action_low, self.action_high)
        u = self._apply_u(u_prev, du)
        xn = self.A @ x + self.B @ u
        if noise and self.noise_std > 0.0:
            xn = xn + self.rng.normal(0.0, self.noise_std, size=self.xdim)
        return np.concatenate([xn, u])

    def reward_batch(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(np.clip(actions, self.action_low, self.action_high))
        x, u_prev = self._split(states)
        u = self._apply_u(u_prev, actions)
        xQx = np.einsum("ni,ij,nj->n", x, self.Q, x)
        uRu = np.einsum("ni,ij,nj->n", u, self.R, u)
        dRd = np.einsum("ni,ij,nj->n", actions, self.R_du, actions)
        return -(xQx + uRu + dRd)

    def dynamics_batch(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(np.clip(actions, self.action_low, self.action_high))
        x, u_prev = self._split(states)
        u = self._apply_u(u_prev, actions)
        xn = x @ self.A.T + u @ self.B.T
        return np.concatenate([xn, u], axis=-1)


# --------------------------------------------------------------------------- #
# Switched linear system (planner selects the mode via continuous logits)
# --------------------------------------------------------------------------- #
class SwitchedLinearEnv:
    """M-mode switched linear system with mode-dependent (A, B, Q, R).

    The mode is a *decision*: the action is ``[u (udim); logits (M)]`` and the
    active mode is ``argmax(logits)``. Continuous planners therefore optimize a
    relaxed (soft) mode choice and the executed mode is the arg-max. Dynamics
    and cost are mode-dependent:

        sigma    = argmax(logits)
        x_{t+1}  = A_sigma x + B_sigma u
        r        = -(x^T Q_sigma x + u^T R_sigma u)

    The finite-horizon value is a pointwise minimum of finitely many quadratics
    (Zhang & Hu), so a single smooth quadratic terminal critic cannot represent
    it -- and a wrong terminal value makes the planner pick the wrong mode early,
    not just misprice the tail.
    """

    def __init__(self, modes=None, noise_std: float = 0.0, max_steps: int = 200,
                 action_bound: float = 10.0, init_state_std: float = 1.0,
                 seed: int | None = None):
        if modes is None:
            # mode 0: unstable plant, strong actuation, expensive control
            # mode 1: stable-ish plant, weak actuation, cheap control
            modes = [
                dict(A=_A_DEFAULT, B=_B_DEFAULT, Q=np.eye(3), R=0.5 * np.eye(3)),
                dict(A=0.95 * np.eye(3), B=0.3 * np.eye(3), Q=np.eye(3), R=0.05 * np.eye(3)),
            ]
        self.modes = modes
        self.M = len(modes)
        self.xdim = modes[0]["A"].shape[0]
        self.udim = modes[0]["B"].shape[1]

        # stacked matrices for vectorized per-row mode indexing
        self.A_stack = np.stack([np.asarray(m["A"], np.float64) for m in modes])   # (M,n,n)
        self.B_stack = np.stack([np.asarray(m["B"], np.float64) for m in modes])   # (M,n,u)
        self.Q_stack = np.stack([np.asarray(m["Q"], np.float64) for m in modes])
        self.R_stack = np.stack([np.asarray(m["R"], np.float64) for m in modes])

        self.noise_std = float(noise_std)
        self.max_steps = int(max_steps)
        self.init_state_std = float(init_state_std)

        self.state_dim = self.xdim
        self.action_dim = self.udim + self.M         # [u ; mode-logits]
        self.action_low = -float(action_bound)
        self.action_high = float(action_bound)

        self.rng = np.random.default_rng(seed)
        self.state = None
        self.t = 0

    def _split(self, action):
        return action[..., :self.udim], action[..., self.udim:]

    def reset(self, state=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if state is None:
            self.state = self.rng.normal(0.0, self.init_state_std, size=self.xdim)
        else:
            self.state = np.asarray(state, np.float64).copy()
            assert self.state.shape == (self.state_dim,)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        print("switched")
        a = np.clip(np.asarray(action, np.float64), self.action_low, self.action_high)
        r = self.reward(self.state, a)
        ns = self.dynamics(self.state, a, noise=self.noise_std > 0.0)
        self.state = ns
        self.t += 1
        return self.state.copy(), r, False, self.t >= self.max_steps, {"t": self.t}

    def reward(self, state, action):
        x = np.asarray(state, np.float64)
        u, logits = self._split(np.clip(np.asarray(action, np.float64),
                                        self.action_low, self.action_high))
        s = int(np.argmax(logits))
        return float(-(x @ self.Q_stack[s] @ x + u @ self.R_stack[s] @ u))

    def dynamics(self, state, action, noise=False):
        x = np.asarray(state, np.float64)
        u, logits = self._split(np.clip(np.asarray(action, np.float64),
                                        self.action_low, self.action_high))
        s = int(np.argmax(logits))
        xn = self.A_stack[s] @ x + self.B_stack[s] @ u
        if noise and self.noise_std > 0.0:
            xn = xn + self.rng.normal(0.0, self.noise_std, size=self.xdim)
        return xn

    def reward_batch(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(np.clip(actions, self.action_low, self.action_high))
        u, logits = self._split(actions)
        s = np.argmax(logits, axis=-1)                       # (N,)
        Q = self.Q_stack[s]                                  # (N,n,n)
        R = self.R_stack[s]                                  # (N,u,u)
        xQx = np.einsum("ni,nij,nj->n", states, Q, states)
        uRu = np.einsum("ni,nij,nj->n", u, R, u)
        return -(xQx + uRu)

    def dynamics_batch(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(np.clip(actions, self.action_low, self.action_high))
        u, logits = self._split(actions)
        s = np.argmax(logits, axis=-1)                       # (N,)
        A = self.A_stack[s]                                  # (N,n,n)
        B = self.B_stack[s]                                  # (N,n,u)
        xn = np.einsum("nij,nj->ni", A, states) + np.einsum("nij,nj->ni", B, u)
        return xn


# --------------------------------------------------------------------------- #
# Linear dynamics + non-quadratic (Huber / l1) terminal cost
# --------------------------------------------------------------------------- #
def _huber(z, delta):
    az = np.abs(z)
    return np.where(az <= delta, 0.5 * az ** 2, delta * (az - 0.5 * delta))


class NonQuadraticTerminalLQREnv:
    """Finite-horizon linear control with a non-quadratic terminal cost.

    Dynamics are the plain linear plant and the stage cost is quadratic, exactly
    as in LQR -- the ONLY departure is the terminal cost applied once at the task
    horizon ``task_horizon``:

        stage:     r_t   = -(x^T Q x + u^T R u)              t < task_horizon
        terminal:  add   -phi(W x_T)      where phi in { l1, huber }

    Because only the terminal breaks Riccati closure, any performance gap vs. a
    terminal-aware planner isolates the *terminal-value representation* rather
    than a dynamics-modelling error. ``terminal_value(states)`` returns the true
    (negated) tail cost ``-phi(W x)`` so terminal-aware planners can consume it.

    NB: the batched ``reward_batch`` returns only the *stage* cost (the planner's
    H-step lookahead sees stage costs; the terminal enters through the explicit
    ``terminal_value`` hook or the episode's final step), so a value-free planner
    naturally ignores the terminal -- which is the whole point of the ablation.
    """

    def __init__(self, A=None, B=None, Q=None, R=None, W=None,
                 terminal_type: str = "huber", huber_delta: float = 0.5,
                 task_horizon: int = 30, noise_std: float = 0.0,
                 init_state_std: float = 1.0, seed: int | None = None):
        self.A = _A_DEFAULT.copy() if A is None else np.asarray(A, np.float64)
        self.B = _B_DEFAULT.copy() if B is None else np.asarray(B, np.float64)
        self.xdim = self.A.shape[0]
        self.udim = self.B.shape[1]
        self.Q = np.eye(self.xdim) if Q is None else np.asarray(Q, np.float64)
        self.R = 0.1 * np.eye(self.udim) if R is None else np.asarray(R, np.float64)
        # terminal weighting matrix (defaults to a mild emphasis on the state)
        self.W = 3.0 * np.eye(self.xdim) if W is None else np.asarray(W, np.float64)
        self.terminal_type = terminal_type
        self.huber_delta = float(huber_delta)

        self.task_horizon = int(task_horizon)
        self.max_steps = int(task_horizon)          # episode == one finite-horizon task
        self.noise_std = float(noise_std)
        self.init_state_std = float(init_state_std)

        self.state_dim = self.xdim
        self.action_dim = self.udim
        self.action_low = -10.0
        self.action_high = 10.0

        self.rng = np.random.default_rng(seed)
        self.state = None
        self.t = 0

    # -- terminal cost ---------------------------------------------------- #
    def _phi(self, x_batch):
        """Non-quadratic terminal COST phi(W x) for (N, xdim) states -> (N,)."""
        z = x_batch @ self.W.T
        if self.terminal_type == "l1":
            return np.sum(np.abs(z), axis=-1)
        if self.terminal_type == "huber":
            return np.sum(_huber(z, self.huber_delta), axis=-1)
        if self.terminal_type == "max_affine":
            return np.max(z, axis=-1)
        raise ValueError(f"unknown terminal_type {self.terminal_type!r}")

    def terminal_value(self, states):
        """Terminal *value* (negated cost) V(x) = -phi(W x), vectorized (N,)."""
        return -self._phi(np.atleast_2d(states))

    # -- gym-style API ---------------------------------------------------- #
    def reset(self, state=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if state is None:
            self.state = self.rng.normal(0.0, self.init_state_std, size=self.xdim)
        else:
            self.state = np.asarray(state, np.float64).copy()
            assert self.state.shape == (self.state_dim,)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        print("terminal")
        a = np.clip(np.asarray(action, np.float64), self.action_low, self.action_high)
        r = self.reward(self.state, a)                       # stage cost
        xn = self.A @ self.state + self.B @ a
        if self.noise_std > 0.0:
            xn = xn + self.rng.normal(0.0, self.noise_std, size=self.xdim)
        self.state = xn
        self.t += 1
        terminated = self.t >= self.task_horizon
        if terminated:
            r += float(self.terminal_value(self.state[None])[0])   # add tail cost
        return self.state.copy(), r, terminated, False, {"t": self.t}

    def reward(self, state, action):
        s = np.asarray(state, np.float64)
        a = np.asarray(action, np.float64)
        return float(-(s @ self.Q @ s + a @ self.R @ a))

    def dynamics(self, state, action, noise=False):
        s = np.asarray(state, np.float64)
        a = np.clip(np.asarray(action, np.float64), self.action_low, self.action_high)
        xn = self.A @ s + self.B @ a
        if noise and self.noise_std > 0.0:
            xn = xn + self.rng.normal(0.0, self.noise_std, size=self.xdim)
        return xn

    def reward_batch(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)
        sQs = np.einsum("ni,ij,nj->n", states, self.Q, states)
        aRa = np.einsum("ni,ij,nj->n", actions, self.R, actions)
        return -(sQs + aRa)

    def dynamics_batch(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(np.clip(actions, self.action_low, self.action_high))
        return states @ self.A.T + actions @ self.B.T


if __name__ == "__main__":
    # sanity: scalar vs batched agreement for each env
    for name, env in [
        ("clqr", ConstrainedLQREnv(seed=0)),
        ("switched", SwitchedLinearEnv(seed=0)),
        ("terminal", NonQuadraticTerminalLQREnv(seed=0)),
    ]:
        rng = np.random.default_rng(1)
        S = rng.normal(size=(5, env.state_dim))
        Acts = rng.normal(size=(5, env.action_dim))
        rb = env.reward_batch(S, Acts)
        rs = np.array([env.reward(S[i], Acts[i]) for i in range(5)])
        nb = env.dynamics_batch(S, Acts)
        ns = np.array([env.dynamics(S[i], Acts[i]) for i in range(5)])
        assert np.allclose(rb, rs), f"{name}: reward mismatch"
        assert np.allclose(nb, ns), f"{name}: dynamics mismatch"
        print(f"{name:9s} OK  state_dim={env.state_dim} action_dim={env.action_dim}")
