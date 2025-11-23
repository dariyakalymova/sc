"""
monte_carlo_gbm.py
Monte Carlo simulation of Geometric Brownian Motion (GBM) using Euler-Maruyama.

Model (GBM):
    dS_t = mu * S_t dt + sigma * S_t dW_t,   S_0 given
Analytical solution:
    S_t = S_0 * exp((mu - 0.5 sigma^2) t + sigma W_t)

This script:
 - Simulates paths with Euler-Maruyama (vectorized)
 - Compares Monte Carlo estimates (mean, variance) to analytical values
 - Performs a weak convergence study (error of expected S_T) vs dt
 - Estimates Value-at-Risk (VaR) and Expected Shortfall (ES)
 - Saves plots to mc_output/

Requires: numpy, matplotlib, pandas (optional)
Run: python monte_carlo_gbm.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# -----------------------
# Parameters
# -----------------------
S0 = 100.0         # initial stock price
mu = 0.05          # drift
sigma = 0.2        # volatility
T = 1.0            # years
seed = 42
np.random.seed(seed)

output_dir = "mc_output"
os.makedirs(output_dir, exist_ok=True)

# Utility: analytical moments for GBM
def gbm_exact_moments(S0, mu, sigma, T):
    """Return exact E[S_T] and Var[S_T] for GBM."""
    m = S0 * np.exp(mu * T)
    var = (S0**2) * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1)
    return m, var

# Euler-Maruyama GBM simulation (vectorized)
def simulate_gbm_em(S0, mu, sigma, T, n_steps, n_paths):
    """
    Simulate GBM using Euler-Maruyama.
    Returns array of shape (n_paths, n_steps+1) with time-0 included.
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    times = np.linspace(0, T, n_steps + 1)
    S = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    S[:, 0] = S0
    for i in range(n_steps):
        Z = np.random.normal(size=n_paths)
        S[:, i+1] = S[:, i] + mu * S[:, i] * dt + sigma * S[:, i] * Z * sqrt_dt
    return times, S

# Strong check: compare single path EM vs exact (for illustration)
def simulate_single_path_compare(S0, mu, sigma, T, n_steps):
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    S_em = np.empty(n_steps + 1)
    S_em[0] = S0
    W = 0.0
    for i in range(n_steps):
        z = np.random.normal()
        W += z * sqrt_dt
        S_em[i+1] = S_em[i] + mu * S_em[i] * dt + sigma * S_em[i] * z * sqrt_dt
    # exact single path using same W increments: reconstruct W increments cumulatively
    # Note: for exact we need full Brownian increments used above; here we cannot reconstruct,
    # so this function is mainly illustrative for plotting EM paths.
    return S_em

# Convergence (weak): measure error in expectation E[S_T]
def weak_convergence_study(S0, mu, sigma, T, n_paths, n_steps_list):
    exact_mean, _ = gbm_exact_moments(S0, mu, sigma, T)
    results = []
    for n_steps in n_steps_list:
        # reset seed for reproducibility across n_steps if desired:
        np.random.seed(seed)
        _, S = simulate_gbm_em(S0, mu, sigma, T, n_steps, n_paths)
        S_T = S[:, -1]
        mc_mean = S_T.mean()
        mc_var = S_T.var(ddof=1)
        err = abs(mc_mean - exact_mean)
        results.append((n_steps, mc_mean, mc_var, err))
    return results, exact_mean

# VaR and ES estimation
def compute_var_es(S_T, alpha=0.05):
    """
    Compute portfolio loss VaR and Expected Shortfall at level alpha.
    Here we treat loss = S0 - S_T (loss positive if S_T < S0).
    VaR_alpha = quantile of loss at level alpha.
    ES = expected loss conditional on loss >= VaR_alpha
    """
    losses = S0 - S_T
    q = np.quantile(losses, alpha)
    es = losses[losses >= q].mean() if np.any(losses >= q) else q
    return q, es

# Plotting utilities
def plot_histogram_ST(S_T, exact_mean, filename):
    plt.figure(figsize=(7,4))
    plt.hist(S_T, bins=60, density=True, alpha=0.6, label="MC histogram")
    # overlay approximate lognormal pdf using sample mean/var of log if desired
    plt.axvline(exact_mean, color='red', linestyle='--', label=f"Exact E[S_T]={exact_mean:.3f}")
    plt.xlabel("S_T")
    plt.ylabel("Density")
    plt.title("Histogram of S_T (Monte Carlo)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_convergence(results, exact_mean, filename):
    n_steps = [r[0] for r in results]
    errs = [r[3] for r in results]
    plt.figure(figsize=(6.5,4))
    # matplotlib removed 'basex'/'basey' kwargs in newer versions; default base=10 is used
    plt.loglog(1/np.array(n_steps), errs, 'o-')
    plt.xlabel(r'$\Delta t$ (log scale), $\Delta t = T / n\_steps$')
    plt.ylabel('Absolute error in E[S_T]')
    plt.title('Weak convergence (error of expectation) vs dt')
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

# -----------------------
# Main experiment
# -----------------------
def main():
    # Monte Carlo config
    n_paths = 200_000      # increase if you have time/CPU; keep moderate for speed
    n_steps = 100
    times, S = simulate_gbm_em(S0, mu, sigma, T, n_steps, n_paths)
    S_T = S[:, -1]

    # Analytical moments
    exact_mean, exact_var = gbm_exact_moments(S0, mu, sigma, T)
    mc_mean = S_T.mean()
    mc_var = S_T.var(ddof=1)
    print("Monte Carlo (n_paths={}, n_steps={}):".format(n_paths, n_steps))
    print(f"  MC mean = {mc_mean:.6f}, Exact mean = {exact_mean:.6f}, abs error = {abs(mc_mean-exact_mean):.6e}")
    print(f"  MC var  = {mc_var:.6f}, Exact var  = {exact_var:.6f}, rel error = {abs(mc_var-exact_var)/exact_var:.6e}")

    # Save histogram
    hist_file = os.path.join(output_dir, f"histogram_n{n_paths}_steps{n_steps}.png")
    plot_histogram_ST(S_T, exact_mean, hist_file)
    print(f"Saved histogram to {hist_file}")

    # Compute VaR and ES at 5% and 1%
    for alpha in (0.05, 0.01):
        q, es = compute_var_es(S_T, alpha=alpha)
        print(f"VaR (alpha={alpha*100:.0f}%): {q:.4f}, ES: {es:.4f}")

    # Convergence study (weak)
    n_steps_list = [10, 20, 40, 80, 160]   # dt halves each time
    print("\nRunning weak convergence study (this will resimulate for reproducibility)...")
    conv_results, exact_mean = weak_convergence_study(S0, mu, sigma, T, n_paths//10, n_steps_list)
    # note: using fewer paths for convergence to reduce runtime (n_paths//10)
    for r in conv_results:
        print(f"n_steps={r[0]:4d}, MC mean={r[1]:.6f}, MC var={r[2]:.6f}, abs_err={r[3]:.6e}")

    conv_file = os.path.join(output_dir, "convergence_weak.png")
    plot_convergence(conv_results, exact_mean, conv_file)
    print(f"Saved convergence plot to {conv_file}")

    # Example sample paths (few) for visualization
    n_paths_plot = 8
    times_plot, S_plot = simulate_gbm_em(S0, mu, sigma, T, n_steps=500, n_paths=n_paths_plot)
    plt.figure(figsize=(7,4))
    for i in range(n_paths_plot):
        plt.plot(times_plot, S_plot[i], lw=1)
    plt.xlabel("t")
    plt.ylabel("S_t")
    plt.title("Sample GBM paths (Euler-Maruyama)")
    plt.grid(True)
    sample_paths_file = os.path.join(output_dir, "sample_paths.png")
    plt.tight_layout()
    plt.savefig(sample_paths_file, dpi=200)
    plt.close()
    print(f"Saved sample paths to {sample_paths_file}")

if __name__ == "__main__":
    main()
