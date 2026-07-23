"""
Sampling-based planners (JAX, GPU)
==================================

The heart of the reconstruction: the *sample -> roll out -> score -> refit* loop
fused into one jitted kernel. The rollout is a ``lax.scan`` over the horizon
(lqr_env_jax.make_rollout), ``vmap``-ed over the N candidate sequences; the CEM /
IGO iterations are a ``lax.scan`` over ``n_iters``; the per-iteration refit is a
pluggable ``update_fn`` from updates_jax.py.

Design choices (see the plan):
  * ``lax.scan`` can't ``break``, so we run a FIXED ``n_iters`` (set generously so
    the NumPy tol_mu/tol_sigma would have fired). The 3x3 matmuls make extra
    constant-work iterations essentially free and keep the kernel fused.
  * Horizon ``H`` (and ``N``/``K`` for the IGO variants, which rescale by
    ``floor(N*log(H+2)*2)``) are STATIC per build -> the sweep recompiles once per
    H, then reuses the kernel across all T timesteps.
  * ``dt`` is a runtime arg so the CEM / IGO variants that differ only in dt can
    share a compiled kernel (the update_fn is the compile-time distinction).

Every ``plan`` takes the warm-started ``mu`` as input and returns the refined
``(mu, sigma)`` (or just the first action, for random shooting); the receding-
horizon driver executes ``mu[0]`` and shifts ``mu`` on the host.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from lqr_env_jax import LQRParams, make_rollout


def nk_for_horizon(H: int, N: int, K: int) -> tuple[int, int]:
    """The IGO variants rescale the sample/elite budget by ``floor(N*log(H+2)*2)``
    (IGO_complete_sample_based.py / IGO_SAC_penalized.py). Pure function of the
    STATIC horizon, so resolve it in Python before building the kernel."""
    scale = math.log(H + 2) * 2.0
    return int(math.floor(N * scale)), int(math.floor(K * scale))


def build_planner(params: LQRParams, horizon: int, num_samples: int,
                  num_elites: int, n_iters: int, gamma: float, update_fn,
                  sigma_init: float):
    """CEM / IGO planner: jitted ``plan(key, s0, mu, dt) -> (mu, sigma)``.

    ``mu`` (H, adim) is warm-started by the caller; ``sigma`` is reset to
    ``sigma_init`` at the start of every plan call (matching the NumPy planners).
    ``dt`` is a runtime scalar; ``update_fn`` and all shapes are compile-time.
    """
    _, batched_rollout = make_rollout(params, horizon, gamma)
    low, high = params.action_low, params.action_high
    N, K, adim = num_samples, num_elites, params.action_dim

    @jax.jit
    def plan(key, s0, mu, dt):
        sigma0 = jnp.full((horizon, adim), sigma_init)

        def iter_body(carry, _):
            key, mu, sigma = carry
            key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, (N, horizon, adim))
            samples = jnp.clip(mu + sigma * noise, low, high)
            returns = batched_rollout(s0, samples)
            mu, sigma = update_fn(samples, returns, mu, sigma, dt, K)
            return (key, mu, sigma), None

        (key, mu, sigma), _ = jax.lax.scan(
            iter_body, (key, mu, sigma0), None, length=n_iters
        )
        return mu, sigma

    return plan


def build_random_shooting(params: LQRParams, horizon: int, num_samples: int,
                          gamma: float, sigma: float):
    """Phase-I random shooting: sample N sequences from N(0, sigma^2) (centred at
    zero, NOT warm-started), score, return the best sequence's first action.
    Matches phase1.RandomShootingMPC with sampler='gaussian'."""
    _, batched_rollout = make_rollout(params, horizon, gamma)
    low, high = params.action_low, params.action_high
    N, adim = num_samples, params.action_dim

    @jax.jit
    def plan(key, s0):
        noise = jax.random.normal(key, (N, horizon, adim))
        actions = jnp.clip(sigma * noise, low, high)
        returns = batched_rollout(s0, actions)
        best = jnp.argmax(returns)
        return actions[best, 0, :]

    return plan


def build_mppi(params: LQRParams, horizon: int, num_samples: int, n_iters: int,
               gamma: float, sigma: float, temperature: float):
    """Phase-II MPPI: warm-started nominal, softmax(reward/temperature)-weighted
    average update. Matches phase2.MPPIPlanner. Returns the refined nominal mu."""
    _, batched_rollout = make_rollout(params, horizon, gamma)
    low, high = params.action_low, params.action_high
    N, adim = num_samples, params.action_dim
    lam = temperature

    @jax.jit
    def plan(key, s0, mu):
        def iter_body(carry, _):
            key, mu = carry
            key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, (N, horizon, adim)) * sigma
            actions = jnp.clip(mu + noise, low, high)
            returns = batched_rollout(s0, actions)
            beta = returns.max()
            w = jnp.exp((returns - beta) / lam)
            w = w / w.sum()
            mu = jnp.einsum("n,nhd->hd", w, actions)
            return (key, mu), None

        (key, mu), _ = jax.lax.scan(iter_body, (key, mu), None, length=n_iters)
        return mu

    return plan
