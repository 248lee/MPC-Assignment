"""
SAC training in JAX (optax)
===========================

Port of the training half of sac_lqr.py. The networks / forward passes / init /
checkpoint I/O live in sac_jax.py; here we add the off-policy SAC update (twin
critics, delayed targets, fixed-or-auto temperature) as a jitted optax step, the
MPPI(H=5) behavior policy that fills the replay buffer (built on the JAX
build_mppi), evaluation against the optimal controller, and the ``train()`` loop
with the same critic-Q-convergence stopping rule as the original.

The saved checkpoint uses sac_jax.save_ckpt -- the SAME format convert_sac_ckpt.py
emits -- so a freshly trained actor and a converted PyTorch actor are
interchangeable in the driver / SAC-penalized planner.

Run::

    uv run python sac_jax_train.py --max-episodes 200 --save-path sac_lqr_jax.pkl
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp
import optax

from lqr_env import LQREnv
from lqr_env_jax import from_numpy_env
from optimal_jax import LQRController
from planners_jax import build_mppi
from sac_jax import (
    STATE_DIM, ACTION_DIM, EPS,
    policy_sample, q_forward, init_policy_params, init_q_params, save_ckpt,
)

H_CONVERGE = 4
T_TRUNC = 6


@dataclass
class SACConfig:
    hidden: tuple = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    autotune_alpha: bool = False
    init_alpha: float = 0.05
    grad_clip: float = 10.0
    action_low: float = -10.0
    action_high: float = 10.0

    @property
    def action_scale(self):
        return (self.action_high - self.action_low) / 2.0

    @property
    def action_bias(self):
        return (self.action_high + self.action_low) / 2.0


# --------------------------------------------------------------------------- #
# replay buffer (host-side numpy circular buffer, like sac_lqr.ReplayBuffer)
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    def __init__(self, capacity, sdim, adim):
        self.cap = capacity
        self.s = np.zeros((capacity, sdim), np.float32)
        self.a = np.zeros((capacity, adim), np.float32)
        self.r = np.zeros((capacity, 1), np.float32)
        self.s2 = np.zeros((capacity, sdim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.s2[i] = s2; self.done[i] = done
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.size, size=batch)
        return (jnp.asarray(self.s[idx]), jnp.asarray(self.a[idx]), jnp.asarray(self.r[idx]),
                jnp.asarray(self.s2[idx]), jnp.asarray(self.done[idx]))

    def __len__(self):
        return self.size


# --------------------------------------------------------------------------- #
# SAC update (functional / optax)
# --------------------------------------------------------------------------- #
def build_sac_update(cfg: SACConfig, actor_tx, critic_tx):
    scale, bias, gamma, tau, alpha = (cfg.action_scale, cfg.action_bias,
                                      cfg.gamma, cfg.tau, cfg.init_alpha)

    def critic_loss(critic_params, actor, qt, batch, key):
        (q1p, q2p) = critic_params
        (q1t, q2t) = qt
        s, a, r, s2, done = batch
        a2, logp2, _ = policy_sample(actor, s2, key, scale, bias)
        min_q_t = jnp.minimum(q_forward(q1t, s2, a2), q_forward(q2t, s2, a2)) - alpha * logp2
        target = r + gamma * (1.0 - done) * min_q_t
        q1 = q_forward(q1p, s, a); q2 = q_forward(q2p, s, a)
        loss = jnp.mean((q1 - target) ** 2) + jnp.mean((q2 - target) ** 2)
        return loss, jnp.mean(q1)

    def actor_loss(actor, critic_params, s, key):
        (q1p, q2p) = critic_params
        a_pi, logp_pi, _ = policy_sample(actor, s, key, scale, bias)
        q_pi = jnp.minimum(q_forward(q1p, s, a_pi), q_forward(q2p, s, a_pi))
        return jnp.mean(alpha * logp_pi - q_pi)

    @jax.jit
    def update(state, batch, key):
        ck, ak = jax.random.split(key)
        # ---- critic ----
        (closs, qmean), cgrads = jax.value_and_grad(critic_loss, has_aux=True)(
            state["critic"], state["actor"], state["qt"], batch, ck)
        cupd, copt = critic_tx.update(cgrads, state["copt"], state["critic"])
        critic = optax.apply_updates(state["critic"], cupd)
        # ---- actor (uses the just-updated critic, as in the torch version) ----
        aloss, agrads = jax.value_and_grad(actor_loss)(state["actor"], critic, batch[0], ak)
        aupd, aopt = actor_tx.update(agrads, state["aopt"], state["actor"])
        actor = optax.apply_updates(state["actor"], aupd)
        # ---- soft target update ----
        qt = jax.tree_util.tree_map(lambda t, p: (1 - tau) * t + tau * p, state["qt"], critic)
        new_state = {"actor": actor, "critic": critic, "qt": qt, "copt": copt, "aopt": aopt}
        return new_state, {"critic_loss": closs, "actor_loss": aloss, "q_mean": qmean}

    return update


def q_reference(critic, ref_s, ref_a):
    (q1p, q2p) = critic
    return jnp.minimum(q_forward(q1p, ref_s, ref_a), q_forward(q2p, ref_s, ref_a))


# --------------------------------------------------------------------------- #
# MPPI(H=5) behavior policy
# --------------------------------------------------------------------------- #
class MPPIBehavior:
    def __init__(self, params, horizon=5, num_samples=256, n_iters=10,
                 sigma=0.2, temperature=20.0, explore_std=0.0, seed=0):
        self.plan = build_mppi(params, horizon, num_samples, n_iters, 1.0, sigma, temperature)
        self.H, self.adim = horizon, params.action_dim
        self.low, self.high = params.action_low, params.action_high
        self.explore_std = float(explore_std)
        self.rng = np.random.default_rng(seed)
        self.key = jax.random.PRNGKey(seed)
        self.reset()

    def reset(self):
        self.mu = jnp.zeros((self.H, self.adim))

    def act(self, state):
        self.key, sub = jax.random.split(self.key)
        self.mu = self.plan(sub, jnp.asarray(state), self.mu)
        a = np.asarray(self.mu[0])
        if self.explore_std > 0.0:
            a = a + self.rng.normal(0.0, self.explore_std, size=a.shape)
        self.mu = jnp.vstack([self.mu[1:], jnp.zeros((1, self.adim))])
        return np.clip(a, self.low, self.high)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(actor, cfg, env, opt_ctrl, n_episodes=10, T=T_TRUNC, seed=1234):
    from sac_jax import policy_forward
    scale, bias = cfg.action_scale, cfg.action_bias

    def greedy(s):
        m, _ = policy_forward(actor, jnp.asarray(s))          # deterministic mean action
        return np.asarray(jnp.tanh(m) * scale + bias)

    rng = np.random.default_rng(seed)
    sac_ret, opt_ret = [], []
    for _ in range(n_episodes):
        s0 = rng.normal(0.0, env.init_state_std, size=env.state_dim)
        s = env.reset(state=s0.copy()); tot = 0.0
        for _ in range(T):
            s, r, term, trunc, _ = env.step(greedy(s)); tot += r
            if term or trunc:
                break
        sac_ret.append(tot)
        s = env.reset(state=s0.copy()); tot = 0.0
        for _ in range(T):
            s, r, term, trunc, _ = env.step(np.asarray(opt_ctrl.act(s))); tot += r
            if term or trunc:
                break
        opt_ret.append(tot)
    return float(np.mean(sac_ret)), float(np.mean(opt_ret))


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def train(max_episodes=200, T_trunc=T_TRUNC, critic_tol=1e-3, critic_patience=20,
          warmup_episodes=20, grad_steps_per_episode=100, explore_std=0.5,
          mppi_num_samples=256, buffer_capacity=1_000_000, eval_every=25,
          save_path="sac_lqr_jax.pkl", seed=0, noise_std=0.0):
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    env = LQREnv(noise_std=noise_std, seed=seed)
    model_env = LQREnv(noise_std=0.0, seed=seed + 1)
    model_params = from_numpy_env(model_env)
    opt_ctrl = LQRController(from_numpy_env(env))

    behavior = MPPIBehavior(model_params, horizon=5, num_samples=mppi_num_samples,
                            explore_std=explore_std, seed=seed + 2)

    cfg = SACConfig(action_low=env.action_low, action_high=env.action_high)

    key, ka, kq1, kq2 = jax.random.split(key, 4)
    actor = init_policy_params(ka, cfg.hidden)
    q1 = init_q_params(kq1, cfg.hidden); q2 = init_q_params(kq2, cfg.hidden)
    critic = (q1, q2)
    qt = jax.tree_util.tree_map(lambda x: x, critic)  # target = copy of critic

    actor_tx = optax.adam(cfg.actor_lr)
    critic_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.critic_lr))
    state = {"actor": actor, "critic": critic, "qt": qt,
             "aopt": actor_tx.init(actor), "copt": critic_tx.init(critic)}
    update = build_sac_update(cfg, actor_tx, critic_tx)

    buffer = ReplayBuffer(buffer_capacity, STATE_DIM, ACTION_DIM)
    sample_rng = np.random.default_rng(seed + 7)

    ref_rng = np.random.default_rng(seed + 123)
    ref_s = ref_rng.normal(0.0, env.init_state_std, size=(512, STATE_DIM)).astype(np.float32)
    ref_a = -(np.asarray(opt_ctrl.K) @ ref_s.T).T + ref_rng.normal(0.0, 0.5, size=(512, ACTION_DIM))
    ref_a = np.clip(ref_a, env.action_low, env.action_high).astype(np.float32)
    ref_s_j, ref_a_j = jnp.asarray(ref_s), jnp.asarray(ref_a)
    prev_q_ref = None

    ema_q_delta = None; ema_beta = 0.9; below = 0; t0 = time.time()
    print(f"Behavior: MPPI(H=5,N={mppi_num_samples},explore_std={explore_std}) | T_trunc={T_trunc}")
    print("-" * 68)

    stop_reason = "max_episodes"; ep = 0
    for ep in range(1, max_episodes + 1):
        behavior.reset(); s = env.reset()
        for _ in range(T_trunc):
            a = behavior.act(s)
            s2, r, term, trunc, _ = env.step(a)
            buffer.add(s, a, r, s2, 1.0 if term else 0.0)
            s = s2
            if term or trunc:
                break

        if ep > warmup_episodes and len(buffer) >= cfg.batch_size:
            for _ in range(grad_steps_per_episode):
                key, uk = jax.random.split(key)
                batch = buffer.sample(cfg.batch_size, sample_rng)
                state, info = update(state, batch, uk)

            q_now = q_reference(state["critic"], ref_s_j, ref_a_j)
            if prev_q_ref is not None:
                dq = float(jnp.mean(jnp.abs(q_now - prev_q_ref)))
                rel_q = dq / (float(jnp.mean(jnp.abs(prev_q_ref))) + EPS)
                ema_q_delta = rel_q if ema_q_delta is None else ema_beta * ema_q_delta + (1 - ema_beta) * rel_q
                below = below + 1 if ema_q_delta < critic_tol else 0
            prev_q_ref = q_now

        if ep % eval_every == 0 or ep == 1:
            sr, orr = evaluate(state["actor"], cfg, env, opt_ctrl, n_episodes=10, T=T_trunc)
            q_str = f"{ema_q_delta:.2e}" if ema_q_delta is not None else "  n/a "
            print(f"ep {ep:5d} | buf {len(buffer):7d} | critic_dQ {q_str} | streak {below:3d} "
                  f"| eval SAC {sr:9.2f} OPT {orr:9.2f} | {time.time()-t0:5.0f}s")

        if ema_q_delta is not None and below >= critic_patience:
            stop_reason = "critic_converged"
            print(f"\nCritic converged (relative Q change < {critic_tol:.1e} for "
                  f"{critic_patience} eps) at episode {ep}.")
            break

    sr, orr = evaluate(state["actor"], cfg, env, opt_ctrl, n_episodes=50, T=T_trunc)
    config = {"hidden": tuple(cfg.hidden), "action_low": cfg.action_low,
              "action_high": cfg.action_high, "action_scale": cfg.action_scale,
              "action_bias": cfg.action_bias}
    meta = {"stop_reason": stop_reason, "episodes": ep, "eval_sac_reward": sr,
            "eval_opt_reward": orr, "final_ema_q_delta": ema_q_delta, "seed": seed}
    save_ckpt(save_path, state["actor"], config,
              q1=state["critic"][0], q2=state["critic"][1], meta=meta)
    print("-" * 68)
    print(f"[{stop_reason}] saved {save_path} | final eval SAC {sr:.2f} vs OPT {orr:.2f}")
    return state, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-episodes", type=int, default=200)
    p.add_argument("--warmup-episodes", type=int, default=20)
    p.add_argument("--grad-steps-per-episode", type=int, default=100)
    p.add_argument("--explore-std", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-path", type=str, default="sac_lqr_jax.pkl")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train(max_episodes=a.max_episodes, warmup_episodes=a.warmup_episodes,
          grad_steps_per_episode=a.grad_steps_per_episode, explore_std=a.explore_std,
          eval_every=a.eval_every, save_path=a.save_path, seed=a.seed)
