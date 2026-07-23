from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import qmc

from lqr_env import LQREnv
from sac_lqr import GaussianPolicy, STATE_DIM, ACTION_DIM

# reuse the MPC harness from Phase II; ``_plain_rollout_returns`` is the pure
# (no policy mixing) evaluator, kept for the premature-convergence probe below.
from phase2 import _rollout_returns as _plain_rollout_returns, run_episode


# --------------------------------------------------------------------------- #
# SAC policy sampler
# --------------------------------------------------------------------------- #
class SACPolicy:
    """Trained SAC actor exposed as a batched action *sampler*.

    ``sample_actions(states)`` draws one squashed-Gaussian action per state
    (``(N, state_dim) -> (N, adim)``), reproducing the actor's tanh + affine
    squashing (see ``GaussianPolicy.sample``). A private torch generator keeps
    the sampling reproducible and never perturbs global torch RNG.
    """

    def __init__(self, path: str = "sac_lqr.pt", seed: int | None = None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.actor = GaussianPolicy(
            STATE_DIM, ACTION_DIM, tuple(cfg["hidden"]),
            cfg["action_low"], cfg["action_high"],
        )
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(int(seed))

    @torch.no_grad()
    def sample_actions(self, states: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(np.asarray(states, dtype=np.float32))
        mu_u, log_std = self.actor.forward(s)
        std = log_std.exp()
        eps = torch.randn(mu_u.shape, generator=self.gen)
        u = mu_u + std * eps
        a = torch.tanh(u) * self.actor.action_scale + self.actor.action_bias
        return a.numpy()


# --------------------------------------------------------------------------- #
# mixed random-shoot / SAC-policy rollout
# --------------------------------------------------------------------------- #
def _rollout_returns(env, state, random_shoots, policy, kappa, gamma, rng):
    """Roll out a per-timestep MIXTURE of the random shoots and the SAC policy.

    ``random_shoots`` (N, H, adim) are the Gaussian samples produced the CEM/IGO
    way (``clip(mu + sigma * noise)``). At timestep ``h`` each sample keeps its
    random-shoot action with probability ``kappa ** h`` -- so ``h = 0`` always
    keeps it and later steps increasingly defer to the learned policy --
    otherwise it takes an action sampled from the SAC policy at that sample's
    current rolled-out state.

    Returns ``(actions, returns)`` where ``actions`` (N, H, adim) is the actually
    executed mixture -- a fraction drawn from ``random_shoots`` and the rest from
    the SAC policy -- and ``returns`` is each mixed sequence's discounted H-step
    return under the known deterministic model.
    """
    N, H, _ = random_shoots.shape
    lo, hi = env.action_low, env.action_high
    states = np.tile(np.asarray(state, dtype=np.float64), (N, 1))   # (N, n)
    actions = np.empty_like(random_shoots)
    returns = np.zeros(N)
    discount = 1.0
    for h in range(H):
        # per-sample coin: prob kappa**h keep the shoot, else use a SAC action
        keep_shoot = rng.random(N) < kappa ** h
        sac_a = np.clip(policy.sample_actions(states), lo, hi)
        a = np.where(keep_shoot[:, None], random_shoots[:, h, :], sac_a)
        actions[:, h, :] = a
        returns += discount * env.reward_batch(states, a)
        states = env.dynamics_batch(states, a)
        discount *= gamma
    return actions, returns


# --------------------------------------------------------------------------- #
# IGO-ML
# --------------------------------------------------------------------------- #
class IGOPlanner:
    def __init__(
        self,
        env: LQREnv,
        horizon: int = 15,
        num_samples: int = 1000,
        num_elites: int = 50,
        max_iters: int = 1e3,
        sigma_init: float = 0.2,
        dt: float = 0.5,
        tol_mu: float = 1e-3,
        tol_sigma: float = 1e-3,
        stop_snr: float = 2.0,
        gamma: float = 1.0,
        kappa: float = 0.9,
        sac_path: str = "sac_lqr.pt",
        policy: "SACPolicy | None" = None,
        seed: int | None = None,
        detect_premature: bool = False,
        premature_r_inner: float = 3.0,
        premature_r_outer: float = 5.0,
        premature_samples_per_dim: int = 16,
    ):
        """
        Parameters
        ----------
        horizon      : planning horizon H.
        num_samples  : sequences sampled per iteration (N).
        num_elites   : how many top sequences drive the update (K).
        max_iters    : iteration budget I per timestep.
        sigma_init   : std the Gaussian is (re)initialized to each timestep.
        dt           : step size in (0, 1].  Non-elite samples enter the
                       weighted MLE with weight (1 - dt); dt=1 recovers CEM
                       (elites only), smaller dt keeps more non-elite mass and
                       preserves variance.
        tol_mu       : legacy; superseded by the stop_snr gate below.
        tol_sigma    : legacy; a variance/elite-std collapse gate never fires
                       once Q is mixed into the sampling measure (the Gaussian
                       cannot collapse), so it is no longer the stop criterion.
        stop_snr     : convergence gate.  Stop once the per-step move, measured
                       in the Fisher/KL metric, drops below stop_snr times its
                       pure-Monte-Carlo-noise floor -- i.e. the update is
                       statistically indistinguishable from sampling jitter.
                       stop_snr=1 sits right at the noise floor; >1 adds slack.
        gamma        : discount for the planned return.
        kappa        : decay base in (0, 1] for the random-shoot / SAC mixture.
                       At horizon step h a sample keeps its random shoot with
                       probability kappa**h, else uses a SAC-policy action.
        sac_path     : checkpoint used to build the SAC policy sampler.
        policy       : a preloaded SACPolicy (overrides ``sac_path``).
        premature_r_inner : inner shell radius (in sigma) -- samples must lie
                       OUTSIDE this box to count (default 3).
        premature_r_outer : outer shell radius (in sigma) -- samples must lie
                       INSIDE this box (default 5).
        premature_samples_per_dim : Sobol points per plan dimension; the total
                       is rounded up to a power of two (default 16).
        """
        self.env = env
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.max_iters = int(max_iters)
        self.sigma_init = float(sigma_init)
        self.dt = float(dt)
        self.tol_mu = float(tol_mu)
        self.tol_sigma = float(tol_sigma)
        self.stop_snr = float(stop_snr)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.policy = policy if policy is not None else SACPolicy(sac_path, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.detect_premature = detect_premature
        self.premature_r_inner = float(premature_r_inner)
        self.premature_r_outer = float(premature_r_outer)
        self.premature_samples_per_dim = int(premature_samples_per_dim)
        # Independent stream for the Sobol premature check so that toggling
        # detection never perturbs the planning RNG (self.rng) used for sampling.
        self.sobol_rng = np.random.default_rng(None if seed is None else seed + 10_000)
        self.last_premature_convergence: bool = False
        self.reset()

    def reset(self) -> None:
        """Clear the warm-started mean (call between episodes)."""
        self.mu = np.zeros((self.horizon, self.env.action_dim))

    def _check_premature(self, state: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> bool:
        """Falsification test via a scrambled-Sobol probe of the box shell.

        We draw quasi-random neighbours of the converged plan ``mu`` and keep
        those in the shell between the inner (``r_inner * sigma``) and outer
        (``r_outer * sigma``) axis-aligned boxes -- a sample is kept iff, for
        every dimension, ``|delta_d| <= r_outer * sigma_d`` (inside the outer
        box, true by construction) AND, for at least one dimension,
        ``|delta_d| > r_inner * sigma_d`` (outside the inner box). If ANY kept
        neighbour beats ``mu``, the plan converged prematurely.

        The sample count grows linearly with the plan dimension ``D = H*adim``
        (``samples_per_dim`` points per dimension) and is rounded UP to a power
        of two so Sobol keeps its low-discrepancy balance. The sequence is
        Owen-scrambled with a per-call seed so successive timesteps probe
        different directions instead of a fixed lattice with blind spots.
        """
        lo, hi = self.env.action_low, self.env.action_high
        H, adim = mu.shape
        D = H * adim
        mu_return = _plain_rollout_returns(self.env, state, mu[None], self.gamma)[0]

        # N = 2**m, the smallest power of two >= samples_per_dim * D.
        m = max(1, int(np.ceil(np.log2(self.premature_samples_per_dim * D))))
        u = qmc.Sobol(d=D, scramble=True, seed=self.sobol_rng).random_base2(m)

        sig = sigma.reshape(D)
        delta = (2.0 * u - 1.0) * (self.premature_r_outer * sig)   # inside +/-r_outer*sigma
        keep = (np.abs(delta) > self.premature_r_inner * sig).any(axis=1)  # outside inner box
        delta = delta[keep]
        if delta.size == 0:
            return False

        cand = np.clip(mu.reshape(D) + delta, lo, hi).reshape(-1, H, adim)
        returns = _plain_rollout_returns(self.env, state, cand, self.gamma)
        return bool((returns > mu_return).any())

    def plan(self, state: np.ndarray) -> np.ndarray:
        H, N, K = self.horizon, self.num_samples, self.num_elites
        N = math.floor(N * np.log(H + 2) * 2)
        K = math.floor(K * np.log(H + 2) * 2)
        adim = self.env.action_dim
        lo, hi = self.env.action_low, self.env.action_high
        dt = self.dt

        mu = self.mu                                   # warm-started mean
        sigma = np.full((H, adim), self.sigma_init)    # reset exploration std
        
        # --- [新增] 位移與路徑長度比 (Displacement vs. Path Length Ratio) 的緩衝區 ---
        # 建議 window_size 設為 4 或 5 (可透過 self.stop_window 設定)
        window_size = getattr(self, 'stop_window', 5)
        mu_history = [mu.copy()]   # 記錄視窗內的 mu 點，注意必須 .copy()
        step_distances = []        # 記錄視窗內的單步距離 ||d_mu||
        # -------------------------------------------------------------------------

        for times in range(self.max_iters):
            # 1. sample the random shoots (CEM/IGO Gaussian) + clip
            noise = self.rng.normal(size=(N, H, adim))
            random_shoots = np.clip(mu + sigma * noise, lo, hi)

            # 2. evaluate a per-timestep mixture of the shoots and the SAC policy;
            #    the returned actions (part shoots, part SAC) feed the MLE below.
            actions, returns = _rollout_returns(
                self.env, state, random_shoots, self.policy, self.kappa,
                self.gamma, self.rng,
            )

            # 3. top-K elites
            elite_idx = np.argpartition(returns, -K)[-K:]

            # remember theta^t before the update, to measure the step below
            mu_prev = mu
            sigma_sq_prev = sigma ** 2

            # 4. weighted MLE over ALL samples
            weights = np.full(N, 1.0 - dt)
            weights[elite_idx] = weights[elite_idx] + dt * N / K
            weights /= weights.sum()
            w = weights[:, None, None]

            mu = (w * actions).sum(axis=0)
            sigma_sq = (w * (actions - mu) ** 2).sum(axis=0)
            sigma = np.sqrt(sigma_sq)

            # --- [修改區塊: 總位移與路徑比 收斂檢測] ---
            # 1. 計算並記錄當前單步的距離
            step_dist = np.linalg.norm(mu - mu_prev)
            step_distances.append(step_dist)
            mu_history.append(mu.copy())  # 把更新後的 mu 加進歷史軌跡

            # 2. 維持滑動視窗的大小不大於 window_size
            if len(step_distances) > window_size:
                step_distances.pop(0)
                mu_history.pop(0)

            # 3. 當視窗填滿時，開始進行收斂檢測
            if len(step_distances) == window_size:
                # 總位移 D (Displacement): 視窗內起點與終點的直線距離
                # 首尾相減，mu_history[-1] 是當前 mu，mu_history[0] 是 W 步前的 mu
                D = np.linalg.norm(mu_history[-1] - mu_history[0])
                
                # 路徑總長 L (Path Length): 視窗內每一步的距離總和
                L = sum(step_distances)

                # 計算比值 R (加上 1e-12 避免除以零)
                R = D / (L + 1e-12)

                # 當比值 R 小於設定閾值時，判定為已進入原地震盪
                # 建議的 stop_disp_ratio 約為 0.15 ~ 0.25 之間
                if R < getattr(self, 'stop_disp_ratio', 0.2):
                    # print(f"Converged at iter {times} with Ratio R: {R:.4f}")
                    break
            # ---------------------------------------

        action = mu[0].copy()
        if self.detect_premature:
            self.last_premature_convergence = self._check_premature(state, mu, sigma)
        if times == self.max_iters - 1:
            print("\nIGO SAC-penalized Hit Max Iter")
        # warm start: shift the plan forward by one step
        self.mu = np.vstack([mu[1:], np.zeros((1, adim))])
        return action

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.plan(state)


if __name__ == "__main__":
    env = LQREnv(noise_std=0.0, seed=0)
    s0 = np.array([1.0, -1.0, 0.5])

    igo = IGOPlanner(env, horizon=15, num_samples=2000, num_elites=500, dt=0.5, seed=0)
    total, traj = run_episode(env, igo, init_state=s0, T=200)
    print(f"IGO-ML (H=15, N=1000, K=50, dt=0.5) reward (T=200): {total:.4f}")
    print(f"  final state: {np.round(traj[-1], 5)}")
