import numpy as np
import matplotlib.pyplot as plt
import numba

# --- Main Simulation Function (Optimized with Numba) ---

@numba.jit(nopython=True)
def fast_simulate_alignment(p, n, sigma, lambda_vals, num_trials):
    """
    Simulates eigenvector alignment using the dual matrix method and Numba.

    Args:
        p (int): Number of features.
        n (int): Number of samples.
        sigma (float): Standard deviation of the isotropic noise.
        lambda_vals (np.array): Array of spike strengths (lambda) to test.
        num_trials (int): Number of trials to average over for each lambda.

    Returns:
        np.array: Averaged squared cosine similarities for each lambda.
    """
    
    # Define a random population leading eigenvector
    # Note: Numba requires arrays to be created outside the loop if possible
    v = np.random.randn(p, 1)
    v = v / np.linalg.norm(v)

    cos_sq_vals = np.zeros(len(lambda_vals))

    # Loop over each lambda value
    for i in range(len(lambda_vals)):
        lam = lambda_vals[i]
        trial_cos_sq = np.zeros(num_trials)

        # Pre-calculate terms for the data generation step
        sqrt_term1 = np.sqrt(lam + sigma**2) - sigma
        sqrt_term2 = sigma

        # Loop over the number of trials for averaging
        for k in range(num_trials):
            # Generate standard normal data
            Z = np.random.randn(p, n)

            # Transform data to have the spiked covariance structure (efficiently)
            X = (sqrt_term1 * v * (v.T @ Z)) + (sqrt_term2 * Z)

            # --- OPTIMIZED EIGENVECTOR CALCULATION ---
            # Form the small n x n dual matrix
            S_dual = (1 / n) * (X.T @ X)

            # Find the leading eigenvector of the small dual matrix
            # Numba supports np.linalg.eigh
            _, dual_eigenvectors = np.linalg.eigh(S_dual)
            u_hat = dual_eigenvectors[:, -1].reshape(n, 1)

            # Reconstruct the leading eigenvector of the large matrix S
            v_hat = X @ u_hat
            v_hat = v_hat / np.linalg.norm(v_hat)

            # Calculate squared cosine similarity
            cos_sq = (v.T @ v_hat)[0, 0]**2
            trial_cos_sq[k] = cos_sq
        
        cos_sq_vals[i] = np.mean(trial_cos_sq)

    return cos_sq_vals

# --- Plotting Function (No Numba needed) ---

def plot_results(p, n, sigma, lambda_vals, cos_sq_vals):
    """
    Plots the simulation results and the theoretical prediction.
    """
    gamma = p / n
    
    # Theory predicts a phase transition for eigenvector alignment
    lambda_c = sigma**2 * np.sqrt(gamma)
    
    # Theoretical alignment (squared cosine) for lambda > lambda_c
    theory_cos_sq = np.maximum(0, 1 - (sigma**4 * gamma) / lambda_vals**2)
    theory_cos_sq[lambda_vals <= lambda_c] = 0

    plt.figure(figsize=(12, 8))
    plt.plot(lambda_vals, cos_sq_vals, 'o-', label='Simulated Alignment', markersize=5)
    plt.plot(lambda_vals, theory_cos_sq, '--', label='Theoretical Alignment', linewidth=2)
    
    plt.axvline(x=lambda_c, color='red', linestyle='--',
                label=rf'Phase Transition ($\lambda_c = \sigma^2 \sqrt{{p/n}} \approx {lambda_c:.2f}$)')

    plt.xlabel(r'Signal Strength ($\lambda$)')
    plt.ylabel('Squared Cosine Similarity')
    plt.title(f'Fast Eigenvector Alignment Simulation (p={p}, n={n})')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution Block ---

if __name__ == '__main__':
    # --- Simulation Parameters ---
    p = 10000   # Number of features (p >> n)
    n = 100     # Number of samples
    sigma = 1.0 # Isotropic noise standard deviation

    # --- Adaptive Range for Lambda ---
    # The phase transition depends on p and n, so we adapt our range to it.
    gamma = p / n
    lambda_c = sigma**2 * np.sqrt(gamma)
    
    # Simulate a range of lambda values spanning the critical point
    print(f"Setting p={p}, n={n}. The theoretical phase transition is at lambda_c = {lambda_c:.2f}")
    lambda_vals = np.linspace(1e-6, 10 * lambda_c, 50)
    
    num_trials = 20 # Number of trials to average for smoothness

    # --- Run Simulation and Plot ---
    print("Starting Numba-optimized simulation...")
    print("This may take a moment as Numba compiles the function for the first time.")
    print("No progress will be shown during the run.")
    
    simulated_cos_sq = fast_simulate_alignment(p, n, sigma, lambda_vals, num_trials)
    
    print("Simulation finished. Plotting results...")
    plot_results(p, n, sigma, lambda_vals, simulated_cos_sq)
