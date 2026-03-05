import numpy as np
import matplotlib.pyplot as plt

def simulate_eigenvector_alignment(p, n, sigma, lambda_vals, num_trials):
    """
    Simulates the alignment between population and sample leading eigenvectors
    in the spiked covariance model for the p >> n regime.

    Args:
        p (int): Number of features.
        n (int): Number of samples.
        sigma (float): Variance of the isotropic noise.
        lambda_vals (np.array): Array of spike strengths (lambda) to test.
        num_trials (int): Number of trials to average over for each lambda.

    Returns:
        np.array: Averaged squared cosine similarities for each lambda.
    """
    
    # Define a random population leading eigenvector
    v = np.random.randn(p, 1)
    v = v / np.linalg.norm(v)

    cos_sq_vals = []

    for lam in lambda_vals:
        trial_cos_sq = []
        print(f"Simulating for lambda = {lam:.2f}...")

        # The population covariance matrix is C = lam * v*v^T + sigma^2 * I
        # Its eigenvalues are (lam + sigma^2) and (sigma^2, ..., sigma^2)
        # The matrix square root C_sqrt is used to generate data
        # C_sqrt = sqrt(lam + sigma^2) * v*v^T + sigma * (I - v*v^T)
        
        # Pre-calculate terms for C_sqrt
        sqrt_term1 = np.sqrt(lam + sigma**2) - sigma
        sqrt_term2 = sigma

        for _ in range(num_trials):
            # Generate standard normal data
            Z = np.random.randn(p, n)

            # Gemini suggested this method to generate the data
            # Transform data to have the spiked covariance structure
            # X = C_sqrt @ Z
            # This is an efficient way to compute C_sqrt @ Z without forming C_sqrt
            X = (sqrt_term1 * v * (v.T @ Z)) + (sqrt_term2 * Z)

            # Compute sample covariance matrix
            S = (1 / n) * (X @ X.T)

            # Find the leading eigenvector of the sample matrix
            # eigh is preferred for hermitian matrices
            _, eigenvectors = np.linalg.eigh(S)
            v_hat = eigenvectors[:, -1].reshape(p, 1) # Eigenvector for largest eigenvalue

            # Calculate squared cosine similarity
            cos_sq = (v.T @ v_hat)**2
            trial_cos_sq.append(cos_sq)
        
        cos_sq_vals.append(np.mean(trial_cos_sq))

    return np.array(cos_sq_vals)

def plot_results(p, n, sigma, lambda_vals, cos_sq_vals):
    """
    Plots the simulation results and the theoretical prediction.
    """
    gamma = p / n
    
    # Theory predicts a phase transition for eigenvector alignment
    # The variance of the noise is sigma^2
    lambda_c = sigma**2 * np.sqrt(gamma)
    
    # Theoretical alignment (squared cosine) for lambda > lambda_c
    # This is a good approximation for the high-dimensional limit
    theory_cos_sq = np.maximum(0, 1 - (sigma**4 * gamma) / lambda_vals**2)
    theory_cos_sq[lambda_vals <= lambda_c] = 0

    plt.figure(figsize=(12, 8))
    plt.plot(lambda_vals, cos_sq_vals, 'o-', label='Simulated Alignment', markersize=5)
    plt.plot(lambda_vals, theory_cos_sq, '--', label='Theoretical Alignment', linewidth=2)
    
    # Add a line for the phase transition point
    plt.axvline(x=lambda_c, color='red', linestyle='--',
                label=rf'Phase Transition ($\lambda_c = \sigma^2 \sqrt{{p/n}} \approx {lambda_c:.2f}$)')

    plt.xlabel(r'Signal Strength ($\lambda$)')
    plt.ylabel('Squared Cosine Similarity')
    plt.title(f'Eigenvector Alignment in Spiked Model (p={p}, n={n})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # --- Simulation Parameters ---
    p = 1000   # Number of features (p >> n)
    n = 100     # Number of samples
    sigma = 1.0 # Isotropic noise level

    # Range of lambda values to simulate, spanning the critical point
    # Start from a small epsilon to avoid division by zero in theoretical curve
    lambda_vals = np.linspace(1e-6, 5, 50)
    
    num_trials = 20 # Number of trials to average for smoothness

    # --- Run Simulation and Plot ---
    simulated_cos_sq = simulate_eigenvector_alignment(p, n, sigma, lambda_vals, num_trials)
    plot_results(p, n, sigma, lambda_vals, simulated_cos_sq)
