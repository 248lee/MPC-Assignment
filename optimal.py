"""
Optimal LQR controller (analytical baseline)
============================================

For the environment

    s_{t+1} = A s_t + B a_t
    cost_t  = s_t^T Q s_t + a_t^T R a_t        (reward = -cost)

the optimal *infinite-horizon* policy is a linear state feedback

    a_t = -K s_t

where K is obtained from the stabilizing solution P of the Discrete-time
Algebraic Riccati Equation (DARE):

    P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
    K = (R + B^T P B)^{-1} B^T P A

This is the best any planner can do in the deterministic, infinite-horizon
limit, so we use it as the gold-standard baseline that MPC should approach as
its planning horizon grows.
"""

from __future__ import annotations

import numpy as np

from lqr_env import LQREnv


def solve_dare(A, B, Q, R, max_iter: int = 10_000, tol: float = 1e-12):
    """Solve the DARE by fixed-point iteration; return (P, K)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    P = Q.copy()
    for _ in range(max_iter):
        BtP = B.T @ P
        S = R + BtP @ B                       # (m, m)
        K = np.linalg.solve(S, BtP @ A)       # (m, n)
        P_next = Q + A.T @ P @ A - (A.T @ P @ B) @ K
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next

    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return P, K


class LQRController:
    """Optimal infinite-horizon linear-feedback controller a = -K s."""

    def __init__(self, env: LQREnv):
        self.env = env
        self.P, self.K = solve_dare(env.A, env.B, env.Q, env.R)

    def act(self, state: np.ndarray) -> np.ndarray:
        a = -self.K @ np.asarray(state, dtype=np.float64)
        return np.clip(a, self.env.action_low, self.env.action_high)

    def value(self, state: np.ndarray) -> float:
        """Optimal cost-to-go reward (negative of s^T P s) for a given state."""
        s = np.asarray(state, dtype=np.float64)
        return float(-(s @ self.P @ s))


def run_episode(env: LQREnv, controller: LQRController, init_state=None, T: int | None = None):
    """Roll out the controller; return (total_reward, state_trajectory)."""
    s = env.reset(state=init_state)
    if T is None:
        T = env.max_steps
    total = 0.0
    traj = [s.copy()]
    for _ in range(T):
        a = controller.act(s)
        s, r, term, trunc, _ = env.step(a)
        total += r
        traj.append(s.copy())
        if term or trunc:
            break
    return total, np.array(traj)


if __name__ == "__main__":
    env = LQREnv(noise_std=0.0, seed=0)
    ctrl = LQRController(env)

    print("Optimal feedback gain K:")
    print(np.round(ctrl.K, 4))
    print("\nClosed-loop eigenvalues (A - B K):")
    eig = np.linalg.eigvals(env.A - env.B @ ctrl.K)
    print(np.round(eig, 4), " (|.| < 1 => stable)")

    s0 = np.array([1.0, -1.0, 0.5])
    total, traj = run_episode(env, ctrl, init_state=s0, T=200)
    print(f"\nEpisode reward (T=200) from s0={s0}: {total:.4f}")
    print(f"Analytical cost-to-go value(s0):        {ctrl.value(s0):.4f}")
    print(f"Final state: {np.round(traj[-1], 5)}")
