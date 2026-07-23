"""
Premature-convergence detection (fully JAX)
===========================================

A falsification probe of a converged plan ``mu``: draw neighbours in the box
shell between ``r_inner * sigma`` and ``r_outer * sigma`` and, if ANY of them
beats ``mu`` under the deterministic model, the planner is flagged as having
stopped short of a local optimum.

Difference from the NumPy version (IGO_complete_sample_based._check_premature):
that one uses a scrambled **scipy Sobol** sequence and host-side filtering to a
variable row count. Per the user's decision this is reimplemented **fully in
JAX** with ``jax.random.uniform`` and a fixed probe size -- out-of-shell
candidates are masked by setting their return to ``-inf`` instead of being
dropped, keeping the kernel shape static (one compile per horizon). The number
of probe points matches the NumPy convention (``2**ceil(log2(spd * D))``,
``D = H*adim``) so the statistical power is comparable, but the sampler (uniform
i.i.d. vs low-discrepancy Sobol) -- and therefore individual flags -- will not
match the NumPy CSV row-for-row. This is expected.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from lqr_env_jax import LQRParams, make_rollout


def make_premature_check(params: LQRParams, horizon: int, gamma: float = 1.0,
                         r_inner: float = 3.0, r_outer: float = 5.0,
                         samples_per_dim: int = 16):
    """Return a jitted ``check(key, state, mu, sigma) -> bool`` for this horizon."""
    _, batched_rollout = make_rollout(params, horizon, gamma)
    adim = params.action_dim
    low, high = params.action_low, params.action_high
    D = horizon * adim
    m = max(1, int(math.ceil(math.log2(samples_per_dim * D))))
    n_probe = 2 ** m  # matches the NumPy Sobol power-of-two probe count

    @jax.jit
    def check(key, state, mu, sigma):
        mu_return = batched_rollout(state, mu[None])[0]
        u = jax.random.uniform(key, (n_probe, horizon, adim))
        delta = (2.0 * u - 1.0) * (r_outer * sigma)              # inside +/- r_outer*sigma
        keep = (jnp.abs(delta) > r_inner * sigma).reshape(n_probe, -1).any(axis=1)
        cand = jnp.clip(mu + delta, low, high)                   # (n_probe, H, adim)
        returns = batched_rollout(state, cand)
        returns = jnp.where(keep, returns, -jnp.inf)
        return (returns > mu_return).any()

    return check


if __name__ == "__main__":
    import numpy as np
    from lqr_env import LQREnv
    from lqr_env_jax import from_numpy_env
    from planners_jax import build_planner
    from updates_jax import igo_variance_injection

    env = LQREnv(noise_std=0.0, seed=0)
    params = from_numpy_env(env)
    H = 10
    plan = build_planner(params, H, 1000, 250, 100, 1.0, igo_variance_injection, 0.2)
    check = make_premature_check(params, H)

    key = jax.random.PRNGKey(0)
    s = env.reset(state=np.array([1.0, -1.0, 0.5]))
    mu = jnp.zeros((H, 3)); n_flag = 0
    for t in range(50):
        key, k1, k2 = jax.random.split(key, 3)
        mu, sigma = plan(k1, jnp.asarray(s), mu, 0.1)
        n_flag += int(bool(check(k2, jnp.asarray(s), mu, sigma)))
        s, r, _, tr, _ = env.step(np.asarray(mu[0]))
        mu = jnp.vstack([mu[1:], jnp.zeros((1, 3))])
    print(f"premature flags over 50 steps (H={H}): {n_flag}/50")
