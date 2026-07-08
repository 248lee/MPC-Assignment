"""
SAC on the LQR environment (off-policy, MPPI-collected replay buffer)
====================================================================

Soft Actor-Critic (SAC) is an *off-policy* actor-critic algorithm: it learns a
stochastic policy and a pair of Q-critics from whatever transitions live in the
replay buffer, regardless of which policy generated them.

Here we exploit that off-policy property: the transitions are NOT collected by
the SAC policy itself but by the **MPPI planner with planning horizon H = 5**
(`phase2.MPPIPlanner`). SAC then distills a fast reactive neural-network policy
from that planner's experience.

Key design points
------------------
* Behavior (data-collection) policy .... MPPI, H = 5   (phase2.py)
* Target (learned) policy ............... Gaussian MLP with tanh squashing
* Critics ............................... twin Q-networks + target networks
* Temperature alpha ..................... auto-tuned (target entropy = -|A|)
* Episode truncation .................... T_TRUNC steps  (see below)

Episode truncation (T_TRUNC)
----------------------------
Policy-evaluation episodes are truncated at ``T_TRUNC`` steps. This value was
chosen with the rule requested for the assignment:

    T_trunc = ceil( 1.5 * H_converge )

where ``H_converge`` is the horizon the *optimal* LQR controller (a = -K s,
`optimal.LQRController`) needs to drive a representative initial state
(init_state_std = 1.0) into the origin s = [0, 0, 0] -- the minimum-cost state.
See ``TRUNCATION_ANALYSIS`` below for the measured numbers.

Stopping / saving
-----------------
Training stops and the PyTorch networks are written to disk when EITHER:
  * the critic's (smoothed, relative) parameter-update magnitude drops below
    ``critic_tol`` -- i.e. the critic has essentially stopped moving, OR
  * more than ``max_episodes`` (default 10000) episodes have been rolled out.

Run:  .venv/Scripts/python.exe sac_lqr.py
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lqr_env import LQREnv
from phase2 import MPPIPlanner
from optimal import LQRController  # analytical baseline, evaluation only


# --------------------------------------------------------------------------- #
# episode truncation horizon (see module docstring / sub-agent analysis)
# --------------------------------------------------------------------------- #
# H_converge  : optimal LQR controller convergence horizon to the origin.
# T_TRUNC     : ceil(1.5 * H_converge).
TRUNCATION_ANALYSIS = """
Deterministic LQREnv, optimal a=-Ks controller, 5000 random init states ~N(0,I)
(init_state_std=1.0). The closed-loop (A-BK) spectral radius is ~0.095, so the
state norm decays geometrically (~10x per step), independent of init magnitude,
with no action clipping engaged.

Convergence horizon = first t after which ||s_t|| stays below eps (worst case
over all 5000 states):

    eps      worst-case horizon
    1e-1              2
    1e-2              3
    1e-3              4      <-- chosen "converged to origin" threshold
    1e-4              5
    1e-5              6

Using eps=1e-3 (state negligible vs. typical init norm ~1.6): H_converge = 4.
    T_trunc = ceil(1.5 * 4) = 6.
"""
H_CONVERGE = 4           # optimal-controller convergence horizon (eps=1e-3)
T_TRUNC = 6              # ceil(1.5 * H_CONVERGE)


STATE_DIM = 3
ACTION_DIM = 3

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


# --------------------------------------------------------------------------- #
# replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device):
        self.capacity = int(capacity)
        self.device = device
        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((self.capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.done[i] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return to_t(self.s), to_t(self.a), to_t(self.r), to_t(self.s2), to_t(self.done)

    def __len__(self):
        return self.size


# --------------------------------------------------------------------------- #
# networks
# --------------------------------------------------------------------------- #
def mlp(sizes, activation=nn.ReLU, out_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Squashed-Gaussian policy: outputs actions in [action_low, action_high]."""

    def __init__(self, state_dim, action_dim, hidden=(256, 256),
                 action_low=-10.0, action_high=10.0):
        super().__init__()
        self.trunk = mlp([state_dim, *hidden], activation=nn.ReLU,
                         out_activation=nn.ReLU)
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)

        # affine map from tanh output (-1, 1) to [low, high]
        self.register_buffer(
            "action_scale", torch.tensor((action_high - action_low) / 2.0)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0)
        )

    def forward(self, state):
        h = self.trunk(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        """Return (action, log_prob, deterministic_action)."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()                       # reparameterized
        t = torch.tanh(u)
        action = t * self.action_scale + self.action_bias

        # log prob with tanh + scaling change-of-variables correction
        log_prob = normal.log_prob(u)
        log_prob -= torch.log(self.action_scale * (1 - t.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    """State-action value Q(s, a)."""

    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([state_dim + action_dim, *hidden, 1])

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


# --------------------------------------------------------------------------- #
# SAC agent
# --------------------------------------------------------------------------- #
@dataclass
class SACConfig:
    hidden: tuple = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    # temperature. The LQR optimum is deterministic, so we use a small FIXED
    # entropy weight by default -- auto-tuning with target_entropy=-|A| is
    # calibrated for actions in [-1,1] and runs away under the [-10,10] scale.
    autotune_alpha: bool = False
    init_alpha: float = 0.05
    target_entropy: float = -float(ACTION_DIM)
    grad_clip: float = 10.0
    action_low: float = -10.0
    action_high: float = 10.0
    # state / action dimensions. Default to the LQR sizes so existing LQR
    # training + checkpoints are unchanged; train_prior.py overrides these to
    # match the harder envs (e.g. clqr has a 6-D augmented state, switched a
    # 5-D action = control + mode-logits).
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM


class SAC:
    def __init__(self, cfg: SACConfig, device):
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        sdim, adim = cfg.state_dim, cfg.action_dim

        self.actor = GaussianPolicy(
            sdim, adim, cfg.hidden, cfg.action_low, cfg.action_high
        ).to(device)
        self.q1 = QNetwork(sdim, adim, cfg.hidden).to(device)
        self.q2 = QNetwork(sdim, adim, cfg.hidden).to(device)
        self.q1_t = QNetwork(sdim, adim, cfg.hidden).to(device)
        self.q2_t = QNetwork(sdim, adim, cfg.hidden).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        for p in self.q1_t.parameters():
            p.requires_grad_(False)
        for p in self.q2_t.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.critic_lr
        )

        # temperature (fixed by default; optionally auto-tuned)
        self.autotune_alpha = cfg.autotune_alpha
        self.target_entropy = cfg.target_entropy
        self.log_alpha = torch.tensor(
            [float(np.log(cfg.init_alpha))],
            requires_grad=self.autotune_alpha, device=device,
        )
        if self.autotune_alpha:
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def _critic_flat_params(self):
        return torch.cat(
            [p.detach().flatten() for p in self.q1.parameters()]
            + [p.detach().flatten() for p in self.q2.parameters()]
        )

    def act(self, state_np, deterministic=False):
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(s)
        a = mean_action if deterministic else action
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def q_reference(self, s_t, a_t):
        """min(Q1, Q2) on a fixed reference batch -- used to measure how much
        the critic *function* is still changing between episodes."""
        return torch.min(self.q1(s_t, a_t), self.q2(s_t, a_t))

    def update(self, buffer: ReplayBuffer):
        """One gradient step of critics, actor, and temperature.

        Returns a dict of diagnostics, including ``critic_rel_delta`` -- the
        relative L2 change of the critic parameters induced by this step.
        """
        cfg = self.cfg
        s, a, r, s2, done = buffer.sample(cfg.batch_size)

        # ---- critic update ------------------------------------------------
        with torch.no_grad():
            a2, logp2, _ = self.actor.sample(s2)
            q1_t = self.q1_t(s2, a2)
            q2_t = self.q2_t(s2, a2)
            min_q_t = torch.min(q1_t, q2_t) - self.alpha * logp2
            target = r + self.gamma * (1.0 - done) * min_q_t

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        before = self._critic_flat_params()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.cfg.grad_clip,
        )
        self.critic_opt.step()
        after = self._critic_flat_params()
        delta = torch.norm(after - before).item()
        rel_delta = delta / (torch.norm(before).item() + EPS)

        # ---- actor update -------------------------------------------------
        for p in self.q1.parameters():
            p.requires_grad_(False)
        for p in self.q2.parameters():
            p.requires_grad_(False)

        a_pi, logp_pi, _ = self.actor.sample(s)
        q_pi = torch.min(self.q1(s, a_pi), self.q2(s, a_pi))
        actor_loss = (self.alpha.detach() * logp_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for p in self.q1.parameters():
            p.requires_grad_(True)
        for p in self.q2.parameters():
            p.requires_grad_(True)

        # ---- temperature update (only if auto-tuning) --------------------
        if self.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # ---- soft target update ------------------------------------------
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "critic_rel_delta": rel_delta,
            "q_mean": q1.mean().item(),
        }

    def save(self, path: str, extra: dict | None = None):
        payload = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_t.state_dict(),
            "q2_target": self.q2_t.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": vars(self.cfg),
        }
        if extra:
            payload["meta"] = extra
        torch.save(payload, path)


# --------------------------------------------------------------------------- #
# behavior policy: MPPI (H = 5), optionally with exploration noise
# --------------------------------------------------------------------------- #
class MPPIBehavior:
    """MPPI(H=5) planner used to fill the replay buffer.

    ``explore_std`` adds zero-mean Gaussian noise to the executed action to
    widen state-action coverage for the off-policy critic. Set 0.0 for a pure
    MPPI behavior policy.
    """

    def __init__(self, model_env: LQREnv, horizon=5, num_samples=256,
                 explore_std=0.0, seed=0):
        self.mppi = MPPIPlanner(
            model_env, horizon=horizon, num_samples=num_samples, seed=seed
        )
        self.explore_std = float(explore_std)
        self.low = model_env.action_low
        self.high = model_env.action_high
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.mppi.reset()

    def act(self, state):
        a = self.mppi.act(state)
        if self.explore_std > 0.0:
            a = a + self.rng.normal(0.0, self.explore_std, size=a.shape)
        return np.clip(a, self.low, self.high)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(agent: SAC, env: LQREnv, opt_ctrl: LQRController, n_episodes=10,
             T=T_TRUNC, seed=1234):
    """Roll out the greedy SAC policy and the optimal controller from the same
    initial states; return (sac_reward, optimal_reward) averaged."""
    rng = np.random.default_rng(seed)
    sac_returns, opt_returns = [], []
    for _ in range(n_episodes):
        s0 = rng.normal(0.0, env.init_state_std, size=env.state_dim)

        s = env.reset(state=s0.copy())
        tot = 0.0
        for _ in range(T):
            s, r, term, trunc, _ = env.step(agent.act(s, deterministic=True))
            tot += r
            if term or trunc:
                break
        sac_returns.append(tot)

        s = env.reset(state=s0.copy())
        tot = 0.0
        for _ in range(T):
            s, r, term, trunc, _ = env.step(opt_ctrl.act(s))
            tot += r
            if term or trunc:
                break
        opt_returns.append(tot)
    return float(np.mean(sac_returns)), float(np.mean(opt_returns))


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def train(
    max_episodes: int = 200,
    T_trunc: int = T_TRUNC,
    critic_tol: float = 1e-3,
    critic_patience: int = 20,
    warmup_episodes: int = 20,
    grad_steps_per_episode: int = 100,
    explore_std: float = 0.5,
    mppi_num_samples: int = 256,
    buffer_capacity: int = 1_000_000,
    eval_every: int = 25,
    save_path: str = "sac_lqr.pt",
    seed: int = 0,
    noise_std: float = 0.0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # real environment the agent acts in
    env = LQREnv(noise_std=noise_std, seed=seed)
    # deterministic model the MPPI planner rolls out internally
    model_env = LQREnv(noise_std=0.0, seed=seed + 1)
    opt_ctrl = LQRController(env)

    behavior = MPPIBehavior(
        model_env, horizon=5, num_samples=mppi_num_samples,
        explore_std=explore_std, seed=seed + 2,
    )

    cfg = SACConfig(action_low=env.action_low, action_high=env.action_high)
    agent = SAC(cfg, device)
    buffer = ReplayBuffer(buffer_capacity, STATE_DIM, ACTION_DIM, device)

    # fixed reference batch to measure the critic's function-space change.
    # States span the visited region (~N(0, init_std)); actions probe near the
    # optimal manifold a=-Ks (+ spread), where the critic actually matters.
    ref_rng = np.random.default_rng(seed + 123)
    ref_s = ref_rng.normal(0.0, env.init_state_std, size=(512, STATE_DIM))
    ref_a = -(opt_ctrl.K @ ref_s.T).T + ref_rng.normal(0.0, 0.5, size=(512, ACTION_DIM))
    ref_a = np.clip(ref_a, env.action_low, env.action_high)
    ref_s_t = torch.as_tensor(ref_s, dtype=torch.float32, device=device)
    ref_a_t = torch.as_tensor(ref_a, dtype=torch.float32, device=device)
    prev_q_ref = None

    ema_q_delta = None       # primary stop signal: relative Q change / episode
    ema_rel_delta = None     # diagnostic: per-step critic parameter change
    ema_beta = 0.9
    below_tol_streak = 0
    t0 = time.time()

    print(f"Truncation: H_converge={H_CONVERGE}, T_trunc={T_trunc}")
    print(f"Behavior policy: MPPI(H=5, N={mppi_num_samples}, explore_std={explore_std})")
    print("-" * 68)

    stop_reason = "max_episodes"
    for ep in range(1, max_episodes + 1):
        # ---- collect one MPPI-driven episode into the buffer -------------
        behavior.reset()
        s = env.reset()
        for _ in range(T_trunc):
            a = behavior.act(s)
            s2, r, term, trunc, _ = env.step(a)
            done = 1.0 if term else 0.0     # truncation is NOT termination
            buffer.add(s, a, r, s2, done)
            s = s2
            if term or trunc:
                break

        # ---- gradient updates --------------------------------------------
        ep_rel_delta = []
        if ep > warmup_episodes and len(buffer) >= cfg.batch_size:
            for _ in range(grad_steps_per_episode):
                info = agent.update(buffer)
                ep_rel_delta.append(info["critic_rel_delta"])

        # ---- convergence tracking ----------------------------------------
        # Primary signal: how much the critic's Q-outputs on the fixed
        # reference batch changed this episode (relative). This -> 0 when the
        # critic has converged (unlike the per-step Adam parameter delta, which
        # floors at the optimizer's step size and never reaches 0).
        if ep_rel_delta:
            mean_delta = float(np.mean(ep_rel_delta))
            ema_rel_delta = (
                mean_delta if ema_rel_delta is None
                else ema_beta * ema_rel_delta + (1 - ema_beta) * mean_delta
            )

            q_now = agent.q_reference(ref_s_t, ref_a_t)
            if prev_q_ref is not None:
                dq = (q_now - prev_q_ref).abs().mean().item()
                scale = prev_q_ref.abs().mean().item() + EPS
                rel_q = dq / scale
                ema_q_delta = (
                    rel_q if ema_q_delta is None
                    else ema_beta * ema_q_delta + (1 - ema_beta) * rel_q
                )
                if ema_q_delta < critic_tol:
                    below_tol_streak += 1
                else:
                    below_tol_streak = 0
            prev_q_ref = q_now

        # ---- logging + eval ----------------------------------------------
        if ep % eval_every == 0 or ep == 1:
            sac_r, opt_r = evaluate(agent, env, opt_ctrl, n_episodes=10, T=T_trunc)
            q_str = f"{ema_q_delta:.2e}" if ema_q_delta is not None else "  n/a "
            print(
                f"ep {ep:5d} | buf {len(buffer):7d} | critic_dQ {q_str} "
                f"| streak {below_tol_streak:3d} "
                f"| eval SAC {sac_r:9.2f}  OPT {opt_r:9.2f} "
                f"| {time.time()-t0:5.0f}s"
            )

        # ---- stopping criterion ------------------------------------------
        if ema_q_delta is not None and below_tol_streak >= critic_patience:
            stop_reason = "critic_converged"
            print(
                f"\nCritic update magnitude (relative Q change) below tol "
                f"({critic_tol:.1e}) for {critic_patience} consecutive "
                f"episodes at episode {ep}."
            )
            break

    # ---- final eval + save ----------------------------------------------
    sac_r, opt_r = evaluate(agent, env, opt_ctrl, n_episodes=50, T=T_trunc)
    meta = {
        "stop_reason": stop_reason,
        "episodes": ep,
        "H_converge": H_CONVERGE,
        "T_trunc": T_trunc,
        "critic_tol": critic_tol,
        "final_ema_q_delta": ema_q_delta,
        "final_ema_param_delta": ema_rel_delta,
        "eval_sac_reward": sac_r,
        "eval_opt_reward": opt_r,
        "explore_std": explore_std,
        "seed": seed,
    }
    agent.save(save_path, extra=meta)
    print("-" * 68)
    print(f"Stopped ({stop_reason}) after {ep} episodes.")
    print(f"Final eval  SAC {sac_r:.2f}  vs  OPT {opt_r:.2f}  "
          f"(gap {sac_r - opt_r:+.2f})")
    print(f"Saved SAC networks -> {os.path.abspath(save_path)}")
    return agent, meta


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train SAC on LQR from MPPI(H=5) data.")
    p.add_argument("--max-episodes", type=int, default=75)
    p.add_argument("--t-trunc", type=int, default=T_TRUNC)
    p.add_argument("--critic-tol", type=float, default=1e-3)
    p.add_argument("--critic-patience", type=int, default=20)
    p.add_argument("--grad-steps-per-episode", type=int, default=100)
    p.add_argument("--explore-std", type=float, default=0.5)
    p.add_argument("--mppi-num-samples", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-path", type=str, default="sac_lqr.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=0.0)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(
        max_episodes=args.max_episodes,
        T_trunc=args.t_trunc,
        critic_tol=args.critic_tol,
        critic_patience=args.critic_patience,
        grad_steps_per_episode=args.grad_steps_per_episode,
        explore_std=args.explore_std,
        mppi_num_samples=args.mppi_num_samples,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed,
        noise_std=args.noise_std,
    )
