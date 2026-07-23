"""
Distribution-update rules for the sampling-based planners (JAX)
===============================================================

Each planner iteration is *sample -> roll out -> score -> refit the Gaussian*.
Only the refit differs between CEM and the IGO variants, so we factor it into
pluggable pure functions with a common signature::

    update_fn(samples, returns, mu, sigma, dt, K) -> (mu_new, sigma_new)

  * ``samples`` : (N, H, adim) clipped action sequences drawn this iteration.
  * ``returns`` : (N,) discounted returns from ``batched_rollout``.
  * ``mu, sigma``: (H, adim) current Gaussian parameters.
  * ``dt``      : IGO step size in (0, 1]; ``dt=1`` recovers CEM for the IGO
                  variants. Ignored by ``refit_hard``.
  * ``K``       : elite count (STATIC -- feeds ``lax.top_k``).

``lax.top_k`` returns an ordered top-K, whereas the NumPy planners use
``np.argpartition`` (unordered); the *set* of elites is identical, so the elite
mean/std/weights match regardless of order (returns are continuous -> no ties).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def refit_hard(samples, returns, mu, sigma, dt, K):
    """CEM (phase2.CEMPlanner): hard elite refit of both mean and std.

    dt is unused (CEM has no soft update). Population std (ddof=0), matching
    NumPy ``elites.std(axis=0)``; no epsilon floor (kept for exact parity)."""
    _, idx = jax.lax.top_k(returns, K)
    elites = samples[idx]
    return elites.mean(axis=0), elites.std(axis=0)


def igo_variance_injection(samples, returns, mu, sigma, dt, K):
    """IGO-ML (IGO.py): soft update in variance space + variance injection.

        var_star  = Var(elites);  mu_star = Mean(elites)
        var_inj   = dt*(1-dt)*(mu_star - mu)^2
        var_new   = (1-dt)*sigma^2 + dt*var_star + var_inj
        mu_new    = (1-dt)*mu      + dt*mu_star
    """
    _, idx = jax.lax.top_k(returns, K)
    elites = samples[idx]
    mu_star = elites.mean(axis=0)
    var_star = elites.var(axis=0)
    var = sigma ** 2
    var_inj = dt * (1.0 - dt) * (mu_star - mu) ** 2
    var_new = (1.0 - dt) * var + dt * var_star + var_inj
    mu_new = (1.0 - dt) * mu + dt * mu_star
    return mu_new, jnp.sqrt(var_new)


def igo_weighted_mle(samples, returns, mu, sigma, dt, K):
    """IGO complete-sample-based (IGO_complete_sample_based.py): weighted MLE
    over ALL samples. Non-elites weight ``1-dt``; elites get ``+dt*N/K``;
    normalize; then weighted mean/variance become the new mu/sigma. ``dt=1``
    zeroes the non-elite weight -> plain CEM."""
    N = samples.shape[0]
    _, idx = jax.lax.top_k(returns, K)
    weights = jnp.full((N,), 1.0 - dt)
    weights = weights.at[idx].add(dt * N / K)
    weights = weights / weights.sum()
    w = weights[:, None, None]
    mu_new = (w * samples).sum(axis=0)
    var_new = (w * (samples - mu_new) ** 2).sum(axis=0)
    return mu_new, jnp.sqrt(var_new)
