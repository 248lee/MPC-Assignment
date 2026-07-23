"""
LQR environment + rollout, JAX edition
=======================================

The functional core of the JAX reconstruction. The NumPy ``LQREnv`` (lqr_env.py)
is a stateful gym-style object; here we split it into:

  * ``LQRParams`` -- an immutable pytree of the model (A, B, Q, R + action bounds)
    that can be closed over by jitted kernels, and
  * pure ``reward`` / ``dynamics`` functions, and
  * ``make_rollout`` -- the parallel evaluator that turns the NumPy Python-loop
    ``phase2._rollout_returns`` into a ``lax.scan`` over the horizon, ``vmap``-ed
    over the N candidate action sequences.

Determinism / dtype
-------------------
We enable float64 (``jax_enable_x64``) so the deterministic (noise_std=0) rollout
matches the NumPy float64 reference to ~1e-6. The matrices are 3x3, so float64 on
the GPU is essentially free; a float32 mode can be enabled later for raw speed.

The reference JAX CEM (jax_playground/jax_cem.py) rolls out *without* a discount;
the NumPy planners apply ``returns += discount * r; discount *= gamma``. We keep
the discount here so the JAX return matches ``phase2._rollout_returns`` exactly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)  # float64 parity with the NumPy reference

import jax.numpy as jnp


class LQRParams(NamedTuple):
    """Immutable LQR model, safe to close over inside jitted kernels."""

    A: jnp.ndarray            # (n, n)
    B: jnp.ndarray            # (n, m)
    Q: jnp.ndarray            # (n, n)
    R: jnp.ndarray            # (m, m)
    action_low: float
    action_high: float
    state_dim: int
    action_dim: int


def from_numpy_env(env) -> LQRParams:
    """Build ``LQRParams`` from a NumPy ``LQREnv`` (or anything with A/B/Q/R)."""
    return LQRParams(
        A=jnp.asarray(env.A),
        B=jnp.asarray(env.B),
        Q=jnp.asarray(env.Q),
        R=jnp.asarray(env.R),
        action_low=float(env.action_low),
        action_high=float(env.action_high),
        state_dim=int(env.state_dim),
        action_dim=int(env.action_dim),
    )


def reward(p: LQRParams, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """r(s, a) = -(s^T Q s + a^T R a). Reward is scored on the *pre-transition*
    state and the (already-clipped) action, matching ``LQREnv.reward``."""
    return -(s @ p.Q @ s + a @ p.R @ a)


def dynamics(p: LQRParams, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """s' = A s + B a, with the action clipped to bounds first (matches
    ``LQREnv.dynamics_batch`` / ``step``, which both clip)."""
    a = jnp.clip(a, p.action_low, p.action_high)
    return p.A @ s + p.B @ a


def make_rollout(params: LQRParams, horizon: int, gamma: float = 1.0):
    """Return ``(rollout, batched_rollout)`` bound to ``params``.

    ``rollout(s0, actions)`` -- discounted H-step return of ONE action sequence
    ``actions`` of shape (H, adim); a ``lax.scan`` over the horizon.

    ``batched_rollout(s0, actions_batch)`` -- the same, ``vmap``-ed over the N
    leading axis of ``actions_batch`` (N, H, adim) -> (N,); s0 is shared.

    Mirrors ``phase2._rollout_returns``: clip the action, score reward on the
    current state, then advance; accumulate with a running discount.
    """
    A, B, Q, R = params.A, params.B, params.Q, params.R
    low, high = params.action_low, params.action_high

    def rollout(s0: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        def step(carry, a):
            s, discount = carry
            a = jnp.clip(a, low, high)
            r = -(s @ Q @ s + a @ R @ a)
            s_next = A @ s + B @ a
            return (s_next, discount * gamma), discount * r

        (_, _), rewards = jax.lax.scan(step, (s0, 1.0), actions)
        return rewards.sum()

    batched_rollout = jax.vmap(rollout, in_axes=(None, 0))
    return rollout, batched_rollout


if __name__ == "__main__":
    # Rollout parity vs the NumPy reference (deterministic model).
    import numpy as np
    from lqr_env import LQREnv
    from phase2 import _rollout_returns

    env = LQREnv(noise_std=0.0, seed=0)
    params = from_numpy_env(env)

    rng = np.random.default_rng(0)
    N, H, adim = 512, 12, env.action_dim
    s0 = np.array([1.0, -1.0, 0.5])
    actions = np.clip(rng.normal(size=(N, H, adim)), env.action_low, env.action_high)

    for gamma in (1.0, 0.95):
        _, batched = make_rollout(params, H, gamma=gamma)
        jax_ret = np.asarray(batched(jnp.asarray(s0), jnp.asarray(actions)))
        np_ret = _rollout_returns(env, s0, actions, gamma)
        max_err = np.max(np.abs(jax_ret - np_ret))
        assert np.allclose(jax_ret, np_ret, atol=1e-6), f"mismatch gamma={gamma}: {max_err}"
        print(f"gamma={gamma}: rollout parity OK  (max abs err {max_err:.2e})")
