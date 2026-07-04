import numpy as np
import matplotlib.pyplot as plt


class PureCEM:
    """Cross-Entropy Method optimizer for custom objective function."""

    def __init__(self, H, adim, num_samples=100, num_elites=20,
                 max_iters=100, sigma_init=1.0, tol_mu=1e-3, tol_sigma=1e-3,
                 seed=None):
        """
        Args:
            H: planning horizon
            adim: action dimension
            num_samples: number of samples per iteration
            num_elites: number of top samples to keep
            max_iters: maximum number of iterations
            sigma_init: initial standard deviation
            tol_mu: convergence tolerance for mean
            tol_sigma: convergence tolerance for std
            seed: random seed
        """
        self.H = H
        self.adim = adim
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.sigma_init = sigma_init
        self.tol_mu = tol_mu
        self.tol_sigma = tol_sigma
        self.rng = np.random.RandomState(seed)

        # Initialize distribution (centered at origin)
        self.mu = np.zeros((H, adim)) + 0.1
        self.sigma = np.full((H, adim), sigma_init)

    def objective(self, x):
        """
        Objective function: f(x) = sum(weight_i * x_i)

        Properties:
        - Maximum at origin: f(0) = 0
        - Penalizes deviation in any direction
        - If x_i > 0: weight = -w
        - If x_i < 0: weight = +w
        - w = 0.1 if ALL x_j >= 0 (first quadrant), else w = 1.0

        Args:
            x: array of shape (H, adim)

        Returns:
            scalar objective value
        """
        # Determine w: 0.1 if all dimensions non-negative, else 1.0
        if np.all(x >= 0):
            w = 0.00001
        else:
            w = 10000.0

        # Assign weights based on sign of each element
        weights = np.where(x > 0, -w, np.where(x < 0, w, 0))

        # Sum weighted values
        value = np.sum(weights * x)

        return value

    def optimize(self, verbose=True):
        """
        Run CEM optimization.

        Returns:
            optimal_x: shape (H, adim)
            history: list of dicts with iteration statistics
        """
        H, N, K = self.H, self.num_samples, self.num_elites
        adim = self.adim

        history = []

        # Record the starting point: f(mu) at the initial mu (origin -> 0.0)
        history.append({
            'iteration': -1,
            'objective_at_mu': self.objective(self.mu),
            'best_objective': self.objective(self.mu),
            'mu_norm': np.linalg.norm(self.mu),
            'sigma_max': self.sigma.max(),
            'mu_change': 0.0,
        })

        for iteration in range(self.max_iters):
            mu_prev = self.mu.copy()

            # 1. Sample from Gaussian: N(mu, sigma^2)
            noise = self.rng.normal(size=(N, H, adim))
            samples = self.mu + self.sigma * noise

            # 2. Evaluate objective for each sample
            objectives = np.array([self.objective(x) for x in samples])

            # 3. Select top-K elites (highest objective values)
            elite_idx = np.argpartition(objectives, -K)[-K:]
            elites = samples[elite_idx]
            elite_objs = objectives[elite_idx]

            # 4. Refit distribution from elites
            self.mu = elites.mean(axis=0)
            self.sigma = elites.std(axis=0)

            # Track statistics
            best_obj = elite_objs.max()
            mu_change = np.linalg.norm(self.mu - mu_prev)

            history.append({
                'iteration': iteration,
                'objective_at_mu': self.objective(self.mu),   # f evaluated at the refit mu
                'best_objective': best_obj,
                'mu_norm': np.linalg.norm(self.mu),
                'sigma_max': self.sigma.max(),
                'mu_change': mu_change,
            })

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration + 1:4d}: best_obj={best_obj:8.4f}, "
                      f"sigma.max={self.sigma.max():.5f}, mu_change={mu_change:.5f}")

            # 5. Check convergence
            # if mu_change < self.tol_mu or self.sigma.max() < self.tol_sigma:
            #     if verbose:
            #         print(f"Converged at iteration {iteration}")
            #     break
        else:
            if verbose:
                print("Hit max iterations")

        return self.mu, history

    def plot_convergence(self, history, save_path=None):
        """
        Plot optimization convergence.

        Args:
            history: list of dicts from optimize()
            save_path: optional path to save figure
        """
        iterations = [h['iteration'] for h in history]
        obj_at_mu = [h['objective_at_mu'] for h in history]
        best_objs = [h['best_objective'] for h in history]
        sigma_maxs = [h['sigma_max'] for h in history]
        mu_norms = [h['mu_norm'] for h in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: f(mu) at each iteration's mean (starts at 0.0, the optimum)
        axes[0].plot(iterations, obj_at_mu, 'b-o', linewidth=2, markersize=4,
                     label='f(μ)')
        axes[0].plot(iterations, best_objs, 'c--s', linewidth=1, markersize=3,
                     alpha=0.6, label='best elite')
        axes[0].axhline(0.0, color='k', linestyle=':', alpha=0.5,
                        label='optimum (0)')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Objective Value')
        axes[0].set_title('f(μ) at Each Iteration')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Maximum sigma over iterations
        axes[1].plot(iterations, sigma_maxs, 'g-o', linewidth=2, markersize=4)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Max σ')
        axes[1].set_title('Maximum Std Dev Over Iterations')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Norm of mean over iterations
        axes[2].plot(iterations, mu_norms, 'r-o', linewidth=2, markersize=4)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('||μ||')
        axes[2].set_title('Norm of Mean Over Iterations')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Figure saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Example: optimize for H=10 horizon, adim=2 actions
    H = 10
    adim = 2

    optimizer = PureCEM(
        H=H, adim=adim,
        num_samples=100, num_elites=20,
        max_iters=100, sigma_init=1.0,
        seed=42
    )

    optimal_x, history = optimizer.optimize()

    print(f"\n=== Optimization Results ===")
    print(f"Optimal solution shape: {optimal_x.shape}")
    print(f"Optimal solution:\n{optimal_x}")
    print(f"Objective value: {optimizer.objective(optimal_x):.6f}")
    print(f"(Target: should be close to 0)")

    # Plot convergence
    optimizer.plot_convergence(history)
