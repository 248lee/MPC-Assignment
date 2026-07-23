"""
SAC in JAX (nets + inference + checkpoint I/O)
==============================================

Port of sac_lqr.py. The networks are expressed as pure-jnp forward functions over
an explicit parameter pytree (dicts of arrays) rather than flax modules -- this
makes the one-time PyTorch->JAX weight copy (convert_sac_ckpt.py) a transparent,
verifiable mapping and gives exact forward parity with the torch actor. optax
drives the training updates (see the trainer section / sac_jax_train.py).

Architecture (must match sac_lqr.GaussianPolicy / QNetwork exactly so the copied
weights reproduce the trained policy):
  * policy trunk : Linear(3,256) ReLU -> Linear(256,256) ReLU
                   -> mu_head Linear(256,3), log_std_head Linear(256,3) [clamp]
  * squash       : a = tanh(mu + std*eps) * action_scale + action_bias
  * Q-net        : Linear(6,256) ReLU -> Linear(256,256) ReLU -> Linear(256,1)

A PyTorch ``nn.Linear`` stores ``weight`` as (out, in); here ``h @ w + b`` wants
``w`` as (in, out), so the converter transposes every weight.
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp

STATE_DIM = 3
ACTION_DIM = 3
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


# --------------------------------------------------------------------------- #
# forward passes (pure jnp over an explicit param pytree)
# --------------------------------------------------------------------------- #
def policy_forward(pp, s):
    """(mu, log_std) for state(s) ``s`` (..., state_dim). ``pp`` is the policy
    param pytree: {'trunk': [ {'w','b'}, ... ], 'mu': {'w','b'}, 'log_std': {'w','b'}}."""
    h = s
    for layer in pp["trunk"]:
        h = jax.nn.relu(h @ layer["w"] + layer["b"])
    mu = h @ pp["mu"]["w"] + pp["mu"]["b"]
    log_std = h @ pp["log_std"]["w"] + pp["log_std"]["b"]
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return mu, log_std


def policy_sample(pp, s, key, action_scale, action_bias):
    """Reparameterized squashed-Gaussian sample; returns (action, log_prob, mean_action),
    matching sac_lqr.GaussianPolicy.sample (log_prob summed over action dim, keepdim)."""
    mu, log_std = policy_forward(pp, s)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mu.shape)
    u = mu + std * eps
    t = jnp.tanh(u)
    action = t * action_scale + action_bias

    # Normal.log_prob(u) - log(scale*(1 - t^2) + EPS), summed over action dim.
    log_prob = -0.5 * ((u - mu) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
    log_prob = log_prob - jnp.log(action_scale * (1 - t ** 2) + EPS)
    log_prob = log_prob.sum(axis=-1, keepdims=True)

    mean_action = jnp.tanh(mu) * action_scale + action_bias
    return action, log_prob, mean_action


def q_forward(qp, s, a):
    """Q(s, a); ``qp`` is a list of {'w','b'} layers (ReLU between, none after last)."""
    x = jnp.concatenate([s, a], axis=-1)
    h = x
    n = len(qp)
    for i, layer in enumerate(qp):
        h = h @ layer["w"] + layer["b"]
        if i < n - 1:
            h = jax.nn.relu(h)
    return h  # (..., 1)


# --------------------------------------------------------------------------- #
# parameter initialization (for training from scratch)
# --------------------------------------------------------------------------- #
def _linear(key, fan_in, fan_out):
    # He-uniform-ish init (matches PyTorch nn.Linear default kaiming_uniform scale).
    lim = 1.0 / jnp.sqrt(fan_in)
    w = jax.random.uniform(key, (fan_in, fan_out), minval=-lim, maxval=lim)
    b = jnp.zeros((fan_out,))
    return {"w": w, "b": b}


def init_policy_params(key, hidden=(256, 256), state_dim=STATE_DIM, action_dim=ACTION_DIM):
    keys = jax.random.split(key, len(hidden) + 2)
    trunk = []
    sizes = [state_dim, *hidden]
    for i in range(len(sizes) - 1):
        trunk.append(_linear(keys[i], sizes[i], sizes[i + 1]))
    mu = _linear(keys[-2], hidden[-1], action_dim)
    log_std = _linear(keys[-1], hidden[-1], action_dim)
    return {"trunk": trunk, "mu": mu, "log_std": log_std}


def init_q_params(key, hidden=(256, 256), state_dim=STATE_DIM, action_dim=ACTION_DIM):
    sizes = [state_dim + action_dim, *hidden, 1]
    keys = jax.random.split(key, len(sizes) - 1)
    return [_linear(keys[i], sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


# --------------------------------------------------------------------------- #
# checkpoint I/O (torch-free; a pickle of numpy arrays + config)
# --------------------------------------------------------------------------- #
def _to_numpy(tree):
    import numpy as np
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def save_ckpt(path, actor, config, q1=None, q2=None, meta=None):
    """Persist a JAX SAC checkpoint. ``actor`` is the policy pytree; ``config``
    holds hidden/action_low/action_high; q1/q2/meta are optional. Both the
    trainer and convert_sac_ckpt.py emit this same format."""
    payload = {
        "actor": _to_numpy(actor),
        "config": dict(config),
        "q1": _to_numpy(q1) if q1 is not None else None,
        "q2": _to_numpy(q2) if q2 is not None else None,
        "meta": meta or {},
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def load_ckpt(path):
    """Load a JAX SAC checkpoint -> (actor_pytree, config, extras). No torch."""
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    actor = _to_jax(payload["actor"])
    config = payload["config"]
    extras = {k: (_to_jax(payload[k]) if payload.get(k) is not None else None)
              for k in ("q1", "q2")}
    extras["meta"] = payload.get("meta", {})
    return actor, config, extras


# --------------------------------------------------------------------------- #
# SAC policy sampler (for the SAC-penalized planner and evaluation)
# --------------------------------------------------------------------------- #
class SACPolicy:
    """Trained JAX actor exposed as a batched sampler / deterministic controller.

    ``sample_actions(states, key)`` draws one squashed-Gaussian action per state
    (used by the SAC-penalized rollout); ``act(state)`` returns the deterministic
    mean action (used to evaluate the policy like optimal_jax.LQRController)."""

    def __init__(self, path: str):
        self.actor, self.config, _ = load_ckpt(path)
        self.action_scale = float(self.config["action_scale"])
        self.action_bias = float(self.config["action_bias"])

    def sample_actions(self, states, key):
        a, _, _ = policy_sample(self.actor, states, key, self.action_scale, self.action_bias)
        return a

    def forward(self, states):
        return policy_forward(self.actor, states)

    def act(self, state):
        mu, _ = policy_forward(self.actor, jnp.asarray(state))
        return jnp.tanh(mu) * self.action_scale + self.action_bias


if __name__ == "__main__":
    # smoke: init, forward, sample, save/load round-trip
    key = jax.random.PRNGKey(0)
    pp = init_policy_params(key)
    s = jnp.ones((5, STATE_DIM))
    mu, log_std = policy_forward(pp, s)
    a, lp, ma = policy_sample(pp, s, key, 10.0, 0.0)
    print("policy_forward:", mu.shape, log_std.shape, "| sample:", a.shape, lp.shape)
    save_ckpt("/tmp/_sac_test.pkl", pp, {"hidden": (256, 256), "action_scale": 10.0, "action_bias": 0.0})
    actor2, cfg2, _ = load_ckpt("/tmp/_sac_test.pkl")
    mu2, _ = policy_forward(actor2, s)
    import numpy as np
    print("save/load round-trip max err:", float(np.max(np.abs(np.asarray(mu - mu2)))))
