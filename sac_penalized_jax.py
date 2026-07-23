"""
SAC-penalized IGO planner (JAX)
===============================

Port of IGO_SAC_penalized.py. Same weighted-MLE refit as the complete
sample-based variant, but the rollouts that *score* each sample mix in
SAC-policy actions: at horizon step ``h`` every sample keeps its random-shoot
action with probability ``kappa**h`` (so h=0 always keeps it, later steps defer
to the learned policy), otherwise it takes an action sampled from the SAC actor
at that sample's current rolled-out state. The MLE is then fit to the *mixed*
executed actions, not the raw shoots.

Because the SAC action depends on each sample's own rolled-out state, the rollout
is a ``lax.scan`` over the horizon operating on the full (N, state_dim) batch (it
cannot reuse the single-sequence vmap rollout). Everything stays jitted on-GPU;
the SAC actor is the pure-jnp ``policy_sample`` from sac_jax.py loaded from the
(converted or trained) JAX checkpoint.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lqr_env_jax import LQRParams
from updates_jax import igo_weighted_mle
from sac_jax import policy_sample


def build_sac_penalized_planner(params: LQRParams, sac_actor, action_scale,
                                action_bias, horizon: int, num_samples: int,
                                num_elites: int, n_iters: int, gamma: float,
                                kappa: float, sigma_init: float):
    """Jitted ``plan(key, s0, mu, dt) -> (mu, sigma)`` for the SAC-penalized variant."""
    A, B, Q, R = params.A, params.B, params.Q, params.R
    low, high = params.action_low, params.action_high
    N, K, adim, sdim = num_samples, num_elites, params.action_dim, params.state_dim
    H = horizon

    def mixed_rollout(key, s0, random_shoots):
        """(actions (N,H,adim), returns (N,)) for the shoot/SAC mixture."""
        states0 = jnp.broadcast_to(s0, (N, sdim))
        hs = jnp.arange(H)
        shoots_t = jnp.swapaxes(random_shoots, 0, 1)  # (H, N, adim)

        def step(carry, inp):
            states, discount, key = carry
            h_idx, shoots_h = inp
            key, ck, sk = jax.random.split(key, 3)
            keep = jax.random.bernoulli(ck, jnp.power(kappa, h_idx), (N,))
            sac_a = jnp.clip(policy_sample(sac_actor, states, sk, action_scale, action_bias)[0],
                             low, high)
            shoots_h = jnp.clip(shoots_h, low, high)
            a = jnp.where(keep[:, None], shoots_h, sac_a)
            r = -(jnp.einsum("ni,ij,nj->n", states, Q, states)
                  + jnp.einsum("ni,ij,nj->n", a, R, a))
            states_next = states @ A.T + a @ B.T
            return (states_next, discount * gamma, key), (a, discount * r)

        (_, _, _), (acts, rews) = jax.lax.scan(step, (states0, 1.0, key), (hs, shoots_t))
        return jnp.swapaxes(acts, 0, 1), rews.sum(axis=0)

    @jax.jit
    def plan(key, s0, mu, dt):
        sigma0 = jnp.full((H, adim), sigma_init)

        def iter_body(carry, _):
            key, mu, sigma = carry
            key, nk, rk = jax.random.split(key, 3)
            noise = jax.random.normal(nk, (N, H, adim))
            shoots = jnp.clip(mu + sigma * noise, low, high)
            actions, returns = mixed_rollout(rk, s0, shoots)
            mu, sigma = igo_weighted_mle(actions, returns, mu, sigma, dt, K)
            return (key, mu, sigma), None

        (key, mu, sigma), _ = jax.lax.scan(iter_body, (key, mu, sigma0), None, length=n_iters)
        return mu, sigma

    return plan


if __name__ == "__main__":
    import numpy as np
    from lqr_env import LQREnv
    from lqr_env_jax import from_numpy_env
    from planners_jax import nk_for_horizon
    from sac_jax import load_ckpt

    env = LQREnv(noise_std=0.0, seed=0)
    params = from_numpy_env(env)
    actor, config, _ = load_ckpt("sac_lqr_jax.pkl")
    scale, bias = config["action_scale"], config["action_bias"]

    H = 10
    N, Kk = nk_for_horizon(H, 1000, 250)
    plan = build_sac_penalized_planner(params, actor, scale, bias, H, N, Kk,
                                       n_iters=int(3e5 / 1000), gamma=1.0,
                                       kappa=0.8, sigma_init=0.2)
    key = jax.random.PRNGKey(0)
    s = env.reset(state=np.array([1.0, -1.0, 0.5])); mu = jnp.zeros((H, 3)); total = 0.0
    for _ in range(200):
        key, sub = jax.random.split(key)
        mu, sigma = plan(sub, jnp.asarray(s), mu, 0.1)
        s, r, _, tr, _ = env.step(np.asarray(mu[0])); total += r
        mu = jnp.vstack([mu[1:], jnp.zeros((1, 3))])
    print(f"SAC-penalized IGO (H={H}) reward T=200: {total:.3f}  final {np.round(s,4)}")
