"""
Policy-prior CEM planner for MPC
================================

This is a variant of the Phase II CEM planner (`phase2.py`). Vanilla CEM there
does two things between timesteps:

  * it *refines* the sampling Gaussian across iterations (keep top-K elites,
    refit mean + std), and
  * it *warm starts* the next timestep by shifting the converged plan forward
    one step (phase2.py lines 125-126).

Here we KEEP the CEM refinement loop but REPLACE the warm start with a learned
**policy prior**: at every timestep the initial sampling distribution is seeded
by the trained SAC policy (`sac_lqr.pt`) instead of the shifted previous plan.

Concretely, to seed the H-step distribution we roll the SAC policy through the
(known, deterministic) model for H steps from the current state:

    mu[h]    = SAC mean action along that rolled-out trajectory
    sigma[h] = SAC action std at that state, multiplied by ``prior_std_scale``
               (= 5.0 by default -- widen the prior so CEM can still search)

CEM then proceeds exactly as before (sample -> evaluate -> refit elites), but it
starts each timestep from a good, policy-informed guess rather than a shifted
plan, and the plan is NOT carried over / shifted between timesteps.
"""

from __future__ import annotations

import numpy as np
import torch

from lqr_env import LQREnv
from sac_lqr import GaussianPolicy, STATE_DIM, ACTION_DIM


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


def _finite_horizon_P0(env: LQREnv, H: int, gamma: float) -> np.ndarray:
    """Backward Riccati matrix ``P0`` for the H-step (no terminal cost) LQR
    subproblem that CEM optimizes at each timestep.

    The subproblem is
        min  sum_{h=0}^{H-1} gamma^h (s_h^T Q s_h + a_h^T R a_h)
        s.t. s_{h+1} = A s_h + B a_h,   s_0 given,
    with NO cost on the terminal state s_H (the rollout stops after H rewards).
    Its optimal cost is ``s_0^T P0 s_0`` -> optimal *return* is ``-s_0^T P0 s_0``.

    The discount is folded in via the standard change of variables
    ``(A, B) -> (sqrt(gamma) A, sqrt(gamma) B)``, which turns the discounted
    problem into an ordinary (undiscounted) finite-horizon LQR. ``P0`` is
    state-independent, so we compute it once and reuse it every timestep.

    NB: this is the *unconstrained* optimum; it ignores the action box
    ``[action_low, action_high]``, so it is an upper bound on the return any
    (clipped) CEM plan can achieve -- exactly the ceiling the convergence gate
    in ``plan`` measures progress against.
    """
    g = np.sqrt(gamma)
    A, B = g * env.A, g * env.B
    Q, R = env.Q, env.R
    P = np.zeros_like(Q)                       # P_H = 0: no terminal-state cost
    for _ in range(H):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P = Q + A.T @ P @ A - (A.T @ P @ B) @ K
    return P


# --------------------------------------------------------------------------- #
# SAC policy prior
# --------------------------------------------------------------------------- #
class SACPrior:
    """Trained SAC actor exposed as an action *distribution* prior.

    ``action_mean_std(state)`` returns the mean action and its (approximate)
    standard deviation in raw action space. The squashed-Gaussian actor stores
    mean ``mu_u`` and std ``std_u`` in pre-tanh space; we map them through the
    tanh + affine squashing:

        mean = tanh(mu_u) * scale + bias
        std  ~= |d action / d u| * std_u = scale * (1 - tanh(mu_u)^2) * std_u
                (delta-method linearization of the tanh squashing)
    """

    def __init__(self, path: str = "sac_lqr.pt"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        # dims come from the checkpoint when present (priors trained on the
        # harder envs via train_prior.py); fall back to the LQR sizes for older
        # checkpoints that predate the dimension-aware config.
        sdim = cfg.get("state_dim", STATE_DIM)
        adim = cfg.get("action_dim", ACTION_DIM)
        self.actor = GaussianPolicy(
            sdim, adim, tuple(cfg["hidden"]),
            cfg["action_low"], cfg["action_high"],
        )
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()

    @torch.no_grad()
    def action_mean_std(self, state: np.ndarray):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mu_u, log_std = self.actor.forward(s)
        std_u = log_std.exp()
        t = torch.tanh(mu_u)
        scale = self.actor.action_scale
        bias = self.actor.action_bias
        mean_a = t * scale + bias
        std_a = std_u * scale * (1.0 - t.pow(2))       # delta method
        return mean_a.squeeze(0).numpy(), std_a.squeeze(0).numpy()


# --------------------------------------------------------------------------- #
# Policy-prior CEM
# --------------------------------------------------------------------------- #
class CEMPlanner:
    def __init__(
        self,
        env: LQREnv,
        horizon: int = 15,
        num_samples: int = 1000,
        num_elites: int = 50,
        max_iters: int = 1000,
        prior_std_scale: float = 5.0,
        tol_mu: float = 1e-3,
        tol_sigma: float = 1e-3,
        gamma: float = 1.0,
        sac_path: str = "sac_lqr.pt",
        prior: SACPrior | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        horizon         : planning horizon H.
        num_samples     : sequences sampled per CEM iteration (N).
        num_elites      : how many top sequences refit the Gaussian (K).
        max_iters       : iteration budget I per timestep.
        prior_std_scale : multiplier applied to the SAC action std when seeding
                          the initial sampling distribution (5.0 -> widen it).
        tol_mu          : (unused) kept for signature compatibility.
        tol_sigma       : first-level stop gate -- only *consider* stopping once
                          max(sigma) < tol_sigma (distribution collapsed). The
                          loop then actually stops when the mean plan's planned
                          return regresses vs. the previous iteration.
        gamma           : discount for the planned return.
        sac_path        : checkpoint used to build the SAC policy prior.
        prior           : a preloaded SACPrior (overrides ``sac_path``).
        """
        self.env = env
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.max_iters = int(max_iters)
        self.prior_std_scale = float(prior_std_scale)
        self.tol_mu = float(tol_mu)
        self.tol_sigma = float(tol_sigma)
        self.gamma = float(gamma)
        self.prior = prior if prior is not None else SACPrior(sac_path)
        self.rng = np.random.default_rng(seed)
        # closed-form finite-horizon LQR cost matrix for the H-step subproblem;
        # state-independent, so compute it once and reuse every timestep.
        self._P0 = _finite_horizon_P0(self.env, self.horizon, self.gamma)
        self.reset()

    def reset(self) -> None:
        """No warm-started state to clear -- each timestep is re-seeded from the
        SAC policy prior. Kept for API compatibility with ``run_episode``."""
        pass

    def _init_from_prior(self, state: np.ndarray):
        """Seed (mu, sigma) for the H-step distribution by rolling the SAC
        policy through the deterministic model from ``state``."""
        H, adim = self.horizon, self.env.action_dim
        mu = np.zeros((H, adim))
        sigma = np.zeros((H, adim))
        s = np.asarray(state, dtype=np.float64)
        for h in range(H):
            mean_a, std_a = self.prior.action_mean_std(s)
            mu[h] = mean_a
            sigma[h] = std_a * self.prior_std_scale
            s = self.env.dynamics(s, mean_a)   # deterministic model step
        return mu, sigma

    def plan(self, state: np.ndarray) -> np.ndarray:
        H, N, K = self.horizon, self.num_samples, self.num_elites
        adim = self.env.action_dim
        lo, hi = self.env.action_low, self.env.action_high

        # initialize the sampling distribution from the SAC policy prior
        # (mean = policy rollout, std = policy std * prior_std_scale)
        mu, sigma = self._init_from_prior(state)

        # track the previous mean plan and its planned return, for the
        # second-level convergence test below.
        prev_mu = mu.copy()
        prev_mu_return = _rollout_returns(self.env, state, mu[None], self.gamma)[0]

        # Closed-form optimal return of this H-step LQR subproblem (finite-
        # horizon Riccati, no terminal cost): optimal cost = s0^T P0 s0, so the
        # optimal *return* is -s0^T P0 s0. State-dependent but iteration-
        # independent, so compute it once here.
        s0 = np.asarray(state, dtype=np.float64)
        opt_sub_return = -float(s0 @ self._P0 @ s0)

        for times in range(self.max_iters):
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

            # two-level convergence:
            #   level 1 -- the distribution has collapsed (max std small); only
            #              once that holds do we consider stopping, and
            #   level 2 -- the refitted mean plan's planned return REGRESSED
            #              relative to the previous iteration.
            # When both hold, stop and keep the previous (better) plan.
            mu_return = _rollout_returns(self.env, state, mu[None], self.gamma)[0]

            # ``opt_sub_return`` (computed once above) is the closed-form ceiling
            # for this H-step LQR subproblem. ``opt_sub_return - *_return`` is the
            # optimality gap; stop once the distribution has collapsed AND the
            # gap stopped shrinking (current gap > 90% of the previous gap).
            if sigma.max() < self.tol_sigma and (opt_sub_return - mu_return) > 0.9 * (opt_sub_return - prev_mu_return):  # 這個終止條件非常重要，要被寫到新的報告書裡面
                mu = prev_mu
                break
            prev_mu, prev_mu_return = mu.copy(), mu_return

        action = mu[0].copy()
        if times == self.max_iters - 1:
            print("\nHit Max Iter")
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
    print(f"Optimal LQR                                  reward (T=200): {opt_r:.4f}")

    # one shared prior so we don't reload the checkpoint per planner
    prior = SACPrior("sac_lqr.pt")
    for H in (5, 15):
        env = LQREnv(noise_std=0.0, seed=0)
        cem = CEMPlanner(env, horizon=H, num_samples=1000, prior=prior, seed=0)
        total, traj = run_episode(env, cem, init_state=s0, T=200)
        print(f"Policy-prior CEM (H={H:>2}, N=1000, K=50, std*5) reward (T=200): "
              f"{total:.4f}   final state: {np.round(traj[-1], 5)}")

    # NB: like vanilla CEM (phase2.py), open-loop search degrades at large H for
    # a fixed sample budget; the SAC prior makes short horizons essentially
    # optimal (H=3 matches the optimal LQR exactly).
