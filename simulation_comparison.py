import numpy as np
import matplotlib.pyplot as plt

# Import the newly created functions
from utils import generate_spiked_data
from sparse_pca import sparse_pca_johnstone_lu

def get_standard_pca_eigenvector(X):
    """
    Calculates the leading eigenvector of the data matrix X using the
    efficient dual-matrix method.
    
    Args:
        X (np.ndarray): Data matrix of shape (p, n).

    Returns:
        np.ndarray: The leading eigenvector of shape (p, 1).
    """
    p, n = X.shape
    
    # Form the small n x n dual matrix
    S_dual = (1 / n) * (X.T @ X)

    # Find the leading eigenvector of the small dual matrix
    _, dual_eigenvectors = np.linalg.eigh(S_dual)
    u_hat = dual_eigenvectors[:, -1].reshape(n, 1)

    # Reconstruct the leading eigenvector of the large matrix S = (1/n)XX.T
    v_hat = X @ u_hat
    v_hat = v_hat / np.linalg.norm(v_hat)
    
    return v_hat

def align_vectors(v_true, v_est):
    """Aligns v_est to v_true by flipping its sign if necessary."""
    if (v_true.T @ v_est)[0, 0] < 0:
        return -v_est
    return v_est

def plot_eigenvector_comparison(v_true, v_hat_std, v_hat_sparse, p):
    """
    Plots the true, standard PCA, and sparse PCA eigenvectors.
    """
    plt.figure(figsize=(18, 10))
    
    # For clarity, only plot a subset of the features if p is large
    plot_p = min(p, 500)
    indices = np.linspace(0, p - 1, plot_p, dtype=int)

    # Marker size
    ms = 4

    plt.plot(indices, v_true[indices], 'o-', label=f'True Eigenvector (v)', color='black', ms=ms)
    plt.plot(indices, v_hat_std[indices], 'x-', label=f'Standard PCA (v_hat)', color='red', alpha=0.7, ms=ms)
    plt.plot(indices, v_hat_sparse[indices], 's-', label=f'Sparse PCA (v_hat_sparse)', color='blue', alpha=0.7, ms=ms)
    
    plt.title(f'Eigenvector Comparison (Plotting {plot_p} of {p} features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    # --- Simulation Parameters ---
    p = 10000   # Number of features
    n = 200    # Number of samples
    sigma = 1.0 # Isotropic noise standard deviation
    
    # Sparsity settings for the true eigenvector
    is_sparse = True
    sparsity_frac = 0.001 # 5% of features are non-zero

    # --- Set Signal Strength (Lambda) ---
    # We choose a single lambda value in an interesting regime for comparison.
    # The theoretical phase transition for alignment occurs at lambda_c.
    gamma = p / n
    lambda_c = sigma**2 * np.sqrt(gamma)
    
    # Let's test a value 50% above the critical point
    lam = lambda_c * 0.8
    
    print("--- Simulation Setup ---")
    print(f"p = {p}, n = {n}, gamma = {gamma:.2f}")
    print(f"True eigenvector is {'SPARSE' if is_sparse else 'DENSE'} with {sparsity_frac*100}% non-zero features.")
    print(f"Theoretical phase transition lambda_c = {lambda_c:.2f}")
    print(f"Using signal strength lambda = {lam:.2f}\n")

    # --- 1. Generate Data ---
    print("1. Generating spiked model data...")
    X, v_true = generate_spiked_data(p, n, sigma, lam, sparse=is_sparse, sparsity_fraction=sparsity_frac)
    print("   Data generated.")

    # --- 2. Standard PCA ---
    print("2. Running standard PCA...")
    v_hat_std = get_standard_pca_eigenvector(X)
    # Align for correct comparison
    v_hat_std = align_vectors(v_true, v_hat_std)
    cos_sq_std = (v_true.T @ v_hat_std)[0, 0]**2
    print(f"   Standard PCA alignment (cos^2): {cos_sq_std:.4f}")

    # --- 3. Sparse PCA (Johnstone & Lu) ---
    print("3. Running Sparse PCA...")
    v_hat_sparse, k_selected = sparse_pca_johnstone_lu(X)
    # Align for correct comparison
    v_hat_sparse = align_vectors(v_true, v_hat_sparse)
    cos_sq_sparse = (v_true.T @ v_hat_sparse)[0, 0]**2
    print(f"   Sparse PCA selected k = {k_selected} features.")
    print(f"   Sparse PCA alignment (cos^2):   {cos_sq_sparse:.4f}\n")

    # --- 4. Visualize Results ---
    print("4. Plotting eigenvector comparison...")
    plot_eigenvector_comparison(v_true, v_hat_std, v_hat_sparse, p)
