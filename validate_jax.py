"""
Validation + benchmark for the JAX reconstruction
==================================================

Because JAX's PRNG differs from NumPy's, we validate in LAYERS rather than by
bit-equality of sampled actions:

  1. Rollout parity (exact)   -- JAX batched_rollout vs phase2._rollout_returns
                                 on identical action sequences (noise_std=0).
  2. Update-rule parity(exact)-- each update_fn vs its NumPy counterpart on
                                 identical (samples, returns).
  3. DARE parity (exact)      -- optimal_jax.solve_dare vs optimal.solve_dare.
  4. Aggregate agreement      -- full T=200 episodes: JAX planners land near the
                                 optimal cost floor value(s0).
  5. SAC forward parity(exact)-- torch GaussianPolicy vs jnp policy_forward on
                                 the converted weights (needs the [convert] extra
                                 + sac_lqr.pt; skipped otherwise).

``--bench`` additionally times the fused JAX CEM kernel vs the vectorized NumPy
planner across sample counts.

Run:  uv run python validate_jax.py [--bench]
"""

from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from lqr_env import LQREnv
from lqr_env_jax import from_numpy_env, make_rollout
from phase2 import _rollout_returns
from optimal import solve_dare as np_dare
from optimal_jax import solve_dare as jax_dare, LQRController
from updates_jax import refit_hard, igo_variance_injection, igo_weighted_mle
from planners_jax import build_planner


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


def test_rollout():
    print("1. Rollout parity (exact)")
    env = LQREnv(noise_std=0.0, seed=0); params = from_numpy_env(env)
    rng = np.random.default_rng(0); ok = True
    for H, gamma in [(6, 1.0), (15, 0.95), (20, 1.0)]:
        acts = np.clip(rng.normal(size=(400, H, 3)), env.action_low, env.action_high)
        _, batched = make_rollout(params, H, gamma)
        j = np.asarray(batched(jnp.asarray(np.array([1., -1., .5])), jnp.asarray(acts)))
        n = _rollout_returns(env, np.array([1., -1., .5]), acts, gamma)
        err = float(np.max(np.abs(j - n)))
        ok &= check(f"H={H:>2} gamma={gamma}", np.allclose(j, n, atol=1e-6), f"max err {err:.1e}")
    return ok


def test_updates():
    print("2. Update-rule parity (exact)")
    def np_hard(s, r, mu, sig, dt, K):
        e = s[np.argpartition(r, -K)[-K:]]; return e.mean(0), e.std(0)
    def np_vi(s, r, mu, sig, dt, K):
        e = s[np.argpartition(r, -K)[-K:]]; ms = e.mean(0); vs = e.var(0)
        v = (1-dt)*sig**2 + dt*vs + dt*(1-dt)*(ms-mu)**2
        return (1-dt)*mu+dt*ms, np.sqrt(v)
    def np_wm(s, r, mu, sig, dt, K):
        N = s.shape[0]; idx = np.argpartition(r, -K)[-K:]
        w = np.full(N, 1.-dt); w[idx] += dt*N/K; w /= w.sum(); ww = w[:, None, None]
        m = (ww*s).sum(0); return m, np.sqrt((ww*(s-m)**2).sum(0))

    rng = np.random.default_rng(1)
    s = rng.normal(size=(1000, 8, 3)); r = rng.normal(size=1000)
    mu = rng.normal(size=(8, 3)); sig = np.abs(rng.normal(size=(8, 3))) + .1
    ok = True
    for name, jf, nf in [("refit_hard", refit_hard, np_hard),
                         ("variance_injection", igo_variance_injection, np_vi),
                         ("weighted_mle", igo_weighted_mle, np_wm)]:
        worst = 0.0
        for dt in (0.1, 0.5, 1.0):
            jm, js = jf(jnp.asarray(s), jnp.asarray(r), jnp.asarray(mu), jnp.asarray(sig), dt, 250)
            nm, ns = nf(s, r, mu, sig, dt, 250)
            worst = max(worst, float(np.max(np.abs(np.asarray(jm)-nm))),
                        float(np.max(np.abs(np.asarray(js)-ns))))
        ok &= check(name, worst < 1e-9, f"max err {worst:.1e}")
    return ok


def test_dare():
    print("3. DARE parity (exact)")
    env = LQREnv(seed=0); params = from_numpy_env(env)
    P, K = jax_dare(params.A, params.B, params.Q, params.R)
    Pn, Kn = np_dare(env.A, env.B, env.Q, env.R)
    eP = float(np.max(np.abs(np.asarray(P)-Pn))); eK = float(np.max(np.abs(np.asarray(K)-Kn)))
    return check("solve_dare (P,K)", eP < 1e-8 and eK < 1e-8, f"P err {eP:.1e}, K err {eK:.1e}")


def test_aggregate():
    print("4. Aggregate agreement (near optimal cost floor)")
    env = LQREnv(noise_std=0.0, seed=0); params = from_numpy_env(env)
    s0 = np.array([1., -1., .5]); opt = -LQRController(params).value(s0)
    ok = True
    for name, upd, dt in [("CEM(dt=1)", refit_hard, 1.0),
                          ("IGO-ML", igo_variance_injection, 0.1)]:
        plan = build_planner(params, 8, 1000, 250, 60, 1.0, upd, 0.2)
        key = jax.random.PRNGKey(0); mu = jnp.zeros((8, 3)); s = env.reset(state=s0); tot = 0.0
        for _ in range(200):
            key, sub = jax.random.split(key)
            mu, sig = plan(sub, jnp.asarray(s), mu, dt)
            s, r, _, tr, _ = env.step(np.asarray(mu[0])); tot += r
            mu = jnp.vstack([mu[1:], jnp.zeros((1, 3))])
        cost = -tot
        ok &= check(name, cost >= opt - 1e-3 and cost < opt * 3,
                    f"cost {cost:.3f} vs optimal floor {opt:.3f}")
    return ok


def test_sac_forward():
    print("5. SAC forward parity (exact; needs torch + sac_lqr.pt)")
    import os
    if not os.path.exists("sac_lqr_jax.pkl") or not os.path.exists("sac_lqr.pt"):
        print("  [SKIP] sac_lqr.pt / sac_lqr_jax.pkl not present")
        return True
    try:
        import torch
    except ImportError:
        print("  [SKIP] torch not installed (run with --extra convert)")
        return True
    from sac_lqr import GaussianPolicy, STATE_DIM, ACTION_DIM
    from sac_jax import policy_forward, load_ckpt
    ck = torch.load("sac_lqr.pt", map_location="cpu", weights_only=False); cfg = ck["config"]
    actor = GaussianPolicy(STATE_DIM, ACTION_DIM, tuple(cfg["hidden"]), cfg["action_low"], cfg["action_high"])
    actor.load_state_dict(ck["actor"]); actor.eval()
    pp, _, _ = load_ckpt("sac_lqr_jax.pkl")
    S = np.random.default_rng(0).normal(size=(64, STATE_DIM))
    with torch.no_grad():
        mt, lt = actor.forward(torch.as_tensor(S, dtype=torch.float32))
    mj, lj = policy_forward(pp, jnp.asarray(S))
    e = max(float(np.max(np.abs(np.asarray(mj)-mt.numpy()))),
            float(np.max(np.abs(np.asarray(lj)-lt.numpy()))))
    return check("policy_forward", e < 1e-4, f"max err {e:.1e}")


def benchmark():
    import time
    print("\nBenchmark: fused JAX CEM kernel vs vectorized NumPy")
    env = LQREnv(seed=0); params = from_numpy_env(env)
    H, K, n_iters = 15, 128, 8
    s0 = jnp.asarray(env.reset(state=np.array([1., -1., .5])))
    for N in (256, 2048, 8192, 32768):
        plan = build_planner(params, H, N, K, n_iters, 1.0, refit_hard, 1.0)
        key = jax.random.PRNGKey(0); mu = jnp.zeros((H, 3))
        m, _ = plan(key, s0, mu, 1.0); m.block_until_ready()  # compile
        t0 = time.perf_counter()
        for _ in range(10):
            m, _ = plan(key, s0, mu, 1.0); m.block_until_ready()
        jt = (time.perf_counter() - t0) / 10

        rng = np.random.default_rng(0)
        A, B, Q, R = env.A, env.B, env.Q, env.R; s0n = np.array([1., -1., .5])
        t0 = time.perf_counter()
        for _ in range(3):
            mu_n = np.zeros((H, 3)); sig = np.ones((H, 3))
            for _ in range(n_iters):
                noise = rng.standard_normal((N, H, 3)); samp = np.clip(mu_n + sig*noise, -10, 10)
                st = np.tile(s0n, (N, 1)); tot = np.zeros(N)
                for t in range(H):
                    a = samp[:, t, :]
                    tot += -(np.einsum("ni,ij,nj->n", st, Q, st) + np.einsum("ni,ij,nj->n", a, R, a))
                    st = st @ A.T + a @ B.T
                idx = np.argpartition(tot, -K)[-K:]; mu_n = samp[idx].mean(0); sig = samp[idx].std(0)+1e-6
        nt = (time.perf_counter() - t0) / 3
        print(f"  N={N:6d} | JAX {jt*1e3:8.2f} ms | NumPy {nt*1e3:9.2f} ms | speedup {nt/jt:7.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    print("=" * 70)
    results = [test_rollout(), test_updates(), test_dare(), test_aggregate(), test_sac_forward()]
    print("=" * 70)
    print(f"PARITY: {sum(results)}/{len(results)} groups passed"
          + ("  ALL PASS" if all(results) else "  *** FAILURES ***"))
    if args.bench:
        benchmark()


if __name__ == "__main__":
    main()
