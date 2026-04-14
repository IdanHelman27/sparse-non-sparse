import numpy as np
import matplotlib.pyplot as plt
from utils import generate_mixed_spiked_data
from sparse_pca import sparse_pca_johnstone_lu

def run_mixed_simulation():
    print("--- Mixed Spiked Covariance Simulation ---")
    
    # 1. Simulation Parameters
    p = 10000            # Number of features
    n = 200             # Number of samples
    sigma = 1.0         # Noise standard deviation
    
    lam1 = 15.0         # Signal strength of dense spike (v)
    lam2 = 5.0          # Signal strength of sparse spike (u)
    sparsity_frac = 0.001
    
    print(f"Dimensions: p = {p}, n = {n}")
    print(f"Parameters: lambda_1 (dense) = {lam1}, lambda_2 (sparse) = {lam2}, sigma^2 = {sigma**2}")
    print(f"Sparsity fraction for u: {sparsity_frac*100}%\n")
    
    if not (lam1 > lam2 > sigma**2):
        print("Warning: Constraints lam1 > lam2 > sigma^2 are not fully met!\n")
    
    # 2. Generate Data
    print("1. Generating mixed sparse-non-sparse data...")
    X, v_true, u_true = generate_mixed_spiked_data(p, n, sigma, lam1, lam2, sparsity_frac)
    
    # 3. Run Sparse PCA to recover u
    print("2. Running Sparse PCA directly on data X...")
    u_hat, k_selected, lambda_hat = sparse_pca_johnstone_lu(X)
    
    # 4. Compare Results
    print("\n3. Comparing Results...")
    
    # Align u_hat to u_true for meaningful comparison
    if (u_true.T @ u_hat)[0, 0] < 0:
        u_hat = -u_hat
        
    cos_sq_u = (u_true.T @ u_hat)[0, 0]**2
    
    print(f"   -> Sparse feature selection k = {k_selected}")
    print(f"   -> Alignment (cos^2) between true 'u' and 'u_hat': {cos_sq_u:.4f}")
    
    # Eigenvalue comparison
    # Keep in mind that a sample eigenvalue theoretically estimates lambda + sigma^2 
    # (plus potential noise inflation depending on the aspect ratio p/n).
    print(f"   -> True sparse eigenvalue (lambda_2): {lam2:.4f}")
    print(f"   -> Estimated eigenvalue (lambda_hat): {lambda_hat:.4f}")

def run_lambda_sweep():
    print("\n--- Running Lambda Sweep Experiment ---")
    p = 10000            # Reduced slightly from 10000 for faster multi-trial loops
    n = 200
    sigma = 1.0
    sparsity_frac = 0.001
    num_trials = 5
    
    # 1. Sweep lam1 while keeping lam2 fixed
    lam2_fixed = 5.0
    lam1_vals = np.linspace(6.0, 500.0, 10)
    
    cos_sq_u_lam1, cos_sq_v_lam1 = [], []
    k_sel_lam1 = []
    
    print(f"1. Sweeping lambda_1 from {lam1_vals[0]} to {lam1_vals[-1]} (fixed lambda_2 = {lam2_fixed})...")
    for lam1 in lam1_vals:
        trial_cos_u, trial_cos_v, trial_k = [], [], []
        for _ in range(num_trials):
            X, v_true, u_true = generate_mixed_spiked_data(p, n, sigma, lam1, lam2_fixed, sparsity_frac)
            u_hat, k_selected, _ = sparse_pca_johnstone_lu(X)
            trial_cos_u.append((u_true.T @ u_hat)[0, 0]**2)
            trial_cos_v.append((v_true.T @ u_hat)[0, 0]**2)
            trial_k.append(k_selected)
        cos_sq_u_lam1.append(np.mean(trial_cos_u))
        cos_sq_v_lam1.append(np.mean(trial_cos_v))
        k_sel_lam1.append(np.mean(trial_k))
        
    # 2. Sweep lam2 while keeping lam1 fixed
    lam1_fixed = 15.0
    lam2_vals = np.linspace(2.0, 14.0, 10)
    
    cos_sq_u_lam2, cos_sq_v_lam2 = [], []
    k_sel_lam2 = []
    
    print(f"2. Sweeping lambda_2 from {lam2_vals[0]} to {lam2_vals[-1]} (fixed lambda_1 = {lam1_fixed})...")
    for lam2 in lam2_vals:
        trial_cos_u, trial_cos_v, trial_k = [], [], []
        for _ in range(num_trials):
            X, v_true, u_true = generate_mixed_spiked_data(p, n, sigma, lam1_fixed, lam2, sparsity_frac)
            u_hat, k_selected, _ = sparse_pca_johnstone_lu(X)
            trial_cos_u.append((u_true.T @ u_hat)[0, 0]**2)
            trial_cos_v.append((v_true.T @ u_hat)[0, 0]**2)
            trial_k.append(k_selected)
        cos_sq_u_lam2.append(np.mean(trial_cos_u))
        cos_sq_v_lam2.append(np.mean(trial_cos_v))
        k_sel_lam2.append(np.mean(trial_k))
        
    # 3. Plotting
    print("Plotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    true_k = int(p * sparsity_frac)
    
    axes[0, 0].plot(lam1_vals, cos_sq_u_lam1, 's-', label=r'Alignment with sparse $u$')
    axes[0, 0].plot(lam1_vals, cos_sq_v_lam1, 'o-', label=r'Alignment with dense $v$')
    axes[0, 0].set_xlabel(r'$\lambda_1$ (Dense Spike Strength)')
    axes[0, 0].set_ylabel(r'Squared Cosine Similarity ($\cos^2$)')
    axes[0, 0].set_title(rf'Varying $\lambda_1$ ($\lambda_2$={lam2_fixed})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1, 0].plot(lam1_vals, k_sel_lam1, '^-', color='purple', label='Selected features ($k$)')
    axes[1, 0].axhline(y=true_k, color='red', linestyle='--', alpha=0.6, label='True sparsity')
    axes[1, 0].set_xlabel(r'$\lambda_1$ (Dense Spike Strength)')
    axes[1, 0].set_ylabel('Number of Features ($k$)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    axes[0, 1].plot(lam2_vals, cos_sq_u_lam2, 's-', label=r'Alignment with sparse $u$')
    axes[0, 1].plot(lam2_vals, cos_sq_v_lam2, 'o-', label=r'Alignment with dense $v$')
    axes[0, 1].set_xlabel(r'$\lambda_2$ (Sparse Spike Strength)')
    axes[0, 1].set_ylabel(r'Squared Cosine Similarity ($\cos^2$)')
    axes[0, 1].set_title(rf'Varying $\lambda_2$ ($\lambda_1$={lam1_fixed})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    axes[1, 1].plot(lam2_vals, k_sel_lam2, '^-', color='purple', label='Selected features ($k$)')
    axes[1, 1].axhline(y=true_k, color='red', linestyle='--', alpha=0.6, label='True sparsity')
    axes[1, 1].set_xlabel(r'$\lambda_2$ (Sparse Spike Strength)')
    axes[1, 1].set_ylabel('Number of Features ($k$)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def run_eigenvector_order_experiment():
    print("\n--- Testing Eigenvector Order Hypothesis ---")
    print("Hypothesis: When lambda_1 >> lambda_2, the reduced dense spike 'v' may dominate")
    print("the first principal component in Sparse PCA, pushing the true sparse spike 'u'")
    print("to the second principal component.")
    
    p = 10000
    n = 200
    sigma = 1.0
    sparsity_frac = 0.001
    num_trials = 10
    
    lam2_fixed = 5.0
    # Sweep lambda_1 up to very large values to trigger the swap
    lam1_vals = np.linspace(100, 1000, 15)
    
    cos_sq_u_pc1, cos_sq_u_pc2 = [], []
    cos_sq_v_pc1, cos_sq_v_pc2 = [], []
    
    print(f"Sweeping lambda_1 from {lam1_vals[0]} to {lam1_vals[-1]} (fixed lambda_2 = {lam2_fixed})...")
    for lam1 in lam1_vals:
        trial_u_pc1, trial_u_pc2 = [], []
        trial_v_pc1, trial_v_pc2 = [], []
        for _ in range(num_trials):
            X, v_true, u_true = generate_mixed_spiked_data(p, n, sigma, lam1, lam2_fixed, sparsity_frac)
            # Request the top 2 components
            V_hat, k_selected, _ = sparse_pca_johnstone_lu(X, num_components=2)
            
            pc1 = V_hat[:, 0:1]
            pc2 = V_hat[:, 1:2]
            
            trial_u_pc1.append((u_true.T @ pc1)[0, 0]**2)
            trial_u_pc2.append((u_true.T @ pc2)[0, 0]**2)
            
            trial_v_pc1.append((v_true.T @ pc1)[0, 0]**2)
            trial_v_pc2.append((v_true.T @ pc2)[0, 0]**2)
            
        cos_sq_u_pc1.append(np.mean(trial_u_pc1))
        cos_sq_u_pc2.append(np.mean(trial_u_pc2))
        cos_sq_v_pc1.append(np.mean(trial_v_pc1))
        cos_sq_v_pc2.append(np.mean(trial_v_pc2))
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(lam1_vals, cos_sq_u_pc1, 'b^-', label=r'Sparse $u$ align with PC1', linewidth=2)
    plt.plot(lam1_vals, cos_sq_u_pc2, 'b--', label=r'Sparse $u$ align with PC2', alpha=0.7)
    plt.plot(lam1_vals, cos_sq_v_pc1, 'rs-', label=r'Dense $v$ align with PC1', linewidth=2)
    plt.plot(lam1_vals, cos_sq_v_pc2, 'r--', label=r'Dense $v$ align with PC2', alpha=0.7)
    plt.xlabel(r'$\lambda_1$ (Dense Spike Strength)')
    plt.ylabel(r'Squared Cosine Similarity ($\cos^2$)')
    plt.title(f'Eigenvector Order Swap in Sparse PCA ($\lambda_2$={lam2_fixed})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def run_sparse_eigenvalue_estimation_experiment():
    print("\n--- Testing Sparse Eigenvalue Estimation ---")
    print("Hypothesis: Sparse PCA reduces the dimensionality from p to k, so the")
    print("estimated eigenvalue should have significantly less bias than Standard PCA,")
    print("closely matching the theoretical population eigenvalue (lambda_2 + sigma^2).")
    
    p = 10000
    n = 200
    sigma = 1.0
    sparsity_frac = 0.001
    num_trials = 10
    
    lam1_fixed = 20.0
    # Sweep lambda_2 to see how well its eigenvalue is estimated
    lam2_vals = np.linspace(3.0, 18.0, 10)
    
    estimated_eigenvalues = []
    estimated_eigenvalues_std = []
    
    print(f"Sweeping lambda_2 from {lam2_vals[0]} to {lam2_vals[-1]} (fixed lambda_1 = {lam1_fixed})...")
    for lam2 in lam2_vals:
        trial_eigenvalues = []
        for _ in range(num_trials):
            X, _, _ = generate_mixed_spiked_data(p, n, sigma, lam1_fixed, lam2, sparsity_frac)
            
            # We request only 1 component, which Sparse PCA will map to the sparse spike
            # because the sparse spike's variance is highly concentrated in the selected k features
            _, _, lambda_hat = sparse_pca_johnstone_lu(X, num_components=1)
            
            val = lambda_hat[0] if isinstance(lambda_hat, np.ndarray) else lambda_hat
            trial_eigenvalues.append(val)
            
        estimated_eigenvalues.append(np.mean(trial_eigenvalues))
        estimated_eigenvalues_std.append(np.std(trial_eigenvalues))
        
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(lam2_vals, estimated_eigenvalues, yerr=estimated_eigenvalues_std, 
                 fmt='bo-', label='Sparse PCA Estimated Eigenvalue', linewidth=2, capsize=5)
                 
    # Theoretical Population Eigenvalue: lambda_2 + sigma^2
    plt.plot(lam2_vals, lam2_vals + sigma**2, 'r--', label=r'Theoretical Population ($\lambda_2 + \sigma^2$)', linewidth=2)
    
    plt.xlabel(r'$\lambda_2$ (Sparse Spike Strength)')
    plt.ylabel('Eigenvalue Estimate')
    plt.title(f'Sparse Eigenvalue Recovery via Sparse PCA ($\lambda_1$={lam1_fixed})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

def run_standard_pca_dense_recovery_experiment():
    print("\n--- Testing Standard PCA Dense Recovery ---")
    print("Hypothesis: When lambda_1 >> lambda_2, standard PCA should easily recover")
    print("the dense spike 'v' and accurately estimate its eigenvalue, matching BBP theory.")
    
    p = 10000
    n = 200
    sigma = 1.0
    sparsity_frac = 0.001
    num_trials = 10
    
    lam2_fixed = 5.0
    lam1_vals = np.linspace(100, 1000, 15)
    
    cos_sq_v_vals = []
    estimated_eigenvalues = []
    
    print(f"Sweeping lambda_1 from {lam1_vals[0]} to {lam1_vals[-1]} (fixed lambda_2 = {lam2_fixed})...")
    for lam1 in lam1_vals:
        trial_cos_v = []
        trial_lam = []
        for _ in range(num_trials):
            X, v_true, _ = generate_mixed_spiked_data(p, n, sigma, lam1, lam2_fixed, sparsity_frac)
            
            # --- Efficient Standard PCA (Dual Matrix) ---
            S_dual = (1 / n) * (X.T @ X)
            evals, evecs = np.linalg.eigh(S_dual)
            lambda_hat = evals[-1]
            u_dual = evecs[:, -1].reshape(n, 1)
            v_hat = X @ u_dual
            v_hat = v_hat / np.linalg.norm(v_hat)
            
            trial_cos_v.append((v_true.T @ v_hat)[0, 0]**2)
            trial_lam.append(lambda_hat)
            
        cos_sq_v_vals.append(np.mean(trial_cos_v))
        estimated_eigenvalues.append(np.mean(trial_lam))
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].plot(lam1_vals, cos_sq_v_vals, 'rs-', label=r'Dense $v$ align with PC1', linewidth=2)
    axes[0].set_xlabel(r'$\lambda_1$ (Dense Spike Strength)')
    axes[0].set_ylabel(r'Squared Cosine Similarity ($\cos^2$)')
    axes[0].set_title(f'Standard PCA Eigenvector Recovery ($\lambda_2$={lam2_fixed})')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].plot(lam1_vals, estimated_eigenvalues, 'bo-', label='Estimated Sample Eigenvalue', linewidth=2)
    
    # Theoretical Population Eigenvalue
    pop_eigenvalues = [lam1 + sigma**2 for lam1 in lam1_vals]
    axes[1].plot(lam1_vals, pop_eigenvalues, 'k--', label=r'Population Eigenvalue ($\lambda_1 + \sigma^2$)', linewidth=2)
    
    # BBP Predicted Sample Eigenvalue (approximate inflation for strong spikes)
    # lambda_sample ~ (lambda + sigma^2) * (1 + (p/n)*(sigma^2/lambda))
    bbp_predictions = [(lam1 + sigma**2) * (1 + (p / n) * (sigma**2 / lam1)) for lam1 in lam1_vals]
    axes[1].plot(lam1_vals, bbp_predictions, 'g:', label=r'BBP Predicted Sample Eigenvalue', linewidth=2)
    
    axes[1].set_xlabel(r'$\lambda_1$ (Dense Spike Strength)')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title(f'Standard PCA Eigenvalue Estimation ($\lambda_2$={lam2_fixed})')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    np.random.seed(42) # For reproducible results
    run_mixed_simulation()
    run_lambda_sweep()
    run_eigenvector_order_experiment()
    run_sparse_eigenvalue_estimation_experiment()
    run_standard_pca_dense_recovery_experiment()