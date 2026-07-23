"""
Optimal LQR controller (JAX)
============================

Port of optimal.py: the analytical infinite-horizon LQR feedback ``a = -K s``,
where ``(P, K)`` solve the Discrete Algebraic Riccati Equation by the same
fixed-point iteration as the NumPy version (optimal.solve_dare) -- here as a
``lax.while_loop`` so it is jittable, though it is a one-time host computation.

This is the gold-standard baseline / deterministic anchor: every sampling-based
planner should approach ``value(s0) = -s0^T P s0`` as horizon and samples grow.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from lqr_env_jax import LQRParams


def solve_dare(A, B, Q, R, max_iter: int = 10_000, tol: float = 1e-12):
    """Fixed-point Riccati iteration; returns (P, K). Mirrors optimal.solve_dare
    (P starts at Q; iterate P until it stops changing; K from the final P)."""
    A = jnp.asarray(A); B = jnp.asarray(B); Q = jnp.asarray(Q); R = jnp.asarray(R)

    def gain(P):
        BtP = B.T @ P
        return jnp.linalg.solve(R + BtP @ B, BtP @ A)          # (m, n)

    def riccati(P):
        K = gain(P)
        return Q + A.T @ P @ A - (A.T @ P @ B) @ K

    def cond(carry):
        P, i, err = carry
        return jnp.logical_and(i < max_iter, err >= tol)

    def body(carry):
        P, i, _ = carry
        P_next = riccati(P)
        return P_next, i + 1, jnp.max(jnp.abs(P_next - P))

    P0 = Q
    P, _, _ = jax.lax.while_loop(cond, body, (P0, 0, jnp.inf))
    K = gain(P)
    return P, K


class LQRController:
    """Optimal linear-feedback controller ``a = clip(-K s)`` (matches optimal.LQRController)."""

    def __init__(self, params: LQRParams):
        self.params = params
        self.P, self.K = solve_dare(params.A, params.B, params.Q, params.R)
        self.low, self.high = params.action_low, params.action_high

    def act(self, state) -> jnp.ndarray:
        s = jnp.asarray(state)
        return jnp.clip(-self.K @ s, self.low, self.high)

    def value(self, state) -> float:
        """Optimal cost-to-go reward -s^T P s."""
        s = jnp.asarray(state)
        return float(-(s @ self.P @ s))


if __name__ == "__main__":
    import numpy as np
    from lqr_env import LQREnv
    from optimal import solve_dare as np_solve_dare, LQRController as NpCtrl
    from lqr_env_jax import from_numpy_env

    env = LQREnv(noise_std=0.0, seed=0)
    params = from_numpy_env(env)

    P, K = solve_dare(params.A, params.B, params.Q, params.R)
    Pn, Kn = np_solve_dare(env.A, env.B, env.Q, env.R)
    print(f"P parity: max abs err {np.max(np.abs(np.asarray(P) - Pn)):.2e}")
    print(f"K parity: max abs err {np.max(np.abs(np.asarray(K) - Kn)):.2e}")

    s0 = np.array([1.0, -1.0, 0.5])
    jv = LQRController(params).value(s0)
    nv = NpCtrl(env).value(s0)
    print(f"value(s0): jax {jv:.6f}  numpy {nv:.6f}  err {abs(jv-nv):.2e}")
    assert np.allclose(np.asarray(P), Pn, atol=1e-8)
    assert np.allclose(np.asarray(K), Kn, atol=1e-8)
    print("DARE parity OK")
