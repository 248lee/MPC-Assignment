import numpy as np
import matplotlib.pyplot as plt

def linear_function(x):
    """A simple linear objective function."""
    return x

def get_variance_trajectory(dt, q=0.25, N=2000, iterations=50):
    """
    Runs IGO-ML and returns the history of variance.
    """
    mu = 0.0
    sigma_sq = 1.0
    num_elites = max(1, int(N * q))
    
    variance_history = [sigma_sq]
    
    for _ in range(iterations):
        # 1. Sample N points
        samples = np.random.normal(mu, np.sqrt(sigma_sq), N)
        
        # 2. Evaluate fitness
        fitness = linear_function(samples)
        
        # 3. Select elites
        elite_indices = np.argsort(fitness)[::-1][:num_elites]
        elites = samples[elite_indices]
        
        # 4. Elite statistics
        mu_star = np.mean(elites)
        sigma_sq_star = np.var(elites)
        
        # 5. IGO-ML parameter updates
        variance_injection = dt * (1 - dt) * (mu_star - mu)**2
        sigma_sq = (1 - dt) * sigma_sq + dt * sigma_sq_star + variance_injection
        mu = (1 - dt) * mu + dt * mu_star
        
        variance_history.append(sigma_sq)
        
    return variance_history

# --- Plotting Setup ---
maxdt = 0.55
dts = np.arange(0.05, maxdt, 0.05)
q = 0.25
iterations = 50

plt.figure(figsize=(10, 6))
colormap = plt.get_cmap('viridis')

for i, dt in enumerate(dts):
    # Get the color from the colormap based on the index
    color = colormap(i / len(dts))
    
    # Run simulation
    history = get_variance_trajectory(dt, q=q, iterations=iterations)
    
    # Plot the line
    plt.plot(range(iterations + 1), history, color=color, linewidth=1.5)

# Create a colorbar to act as a legend for \delta t
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0.05, vmax=maxdt))
sm.set_array([]) # Dummy array for the colorbar
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r'Step Size ($\delta t$)', fontsize=12)

plt.title(r'IGO-ML Variance Trajectories over Iterations ($q=0.25$)', fontsize=14)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel(r'Variance ($\sigma^2$)', fontsize=12)
plt.yscale('log') # Log scale helps visualize both stabilizing and exploding variances
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()