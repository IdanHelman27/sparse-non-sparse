import numpy as np
import matplotlib.pyplot as plt
from utils import generate_spiked_data

def run_top_k_variance_experiment():
    # --- Simulation Parameters ---
    p = 10000          # Number of features
    sigma = 1.0       # Isotropic noise standard deviation
    lam = 5.0         # Signal strength
    
    sparsity_fraction = 0.001
    k = int(p * sparsity_fraction) # The actual number of non-zero coordinates
    
    # We will vary the number of samples (n/m) to see how variance estimation improves
    sample_sizes = [50, 100, 200, 500, 1000, 2000]
    num_trials = 50
    
    avg_overlaps = []
    avg_true_ranks = []
    avg_median_vars = []
    avg_true_vars = []
    
    print("--- Top-k Variance Recovery Experiment ---")
    print(f"Features (p) = {p}, Signal (k) = {k}, Lambda = {lam}")
    print(f"{'Samples (n)':<12} | {'lambda_c':<10} | {'Avg Recovery':<14} | {'Avg Rank':<10} | {'Med Var':<9} | {'True Var':<9}")
    print("-" * 76)
    
    for n in sample_sizes:
        trial_overlaps = []
        trial_ranks = []
        trial_median_vars = []
        trial_true_vars = []
        
        for _ in range(num_trials):
            # 1. Generate data
            X, v_true = generate_spiked_data(p, n, sigma, lam, sparse=True, sparsity_fraction=sparsity_fraction)
            
            # 2. Extract ground truth indices (where true eigenvector is non-zero)
            true_indices = set(np.where(v_true[:, 0] != 0)[0])
            
            # 3. Calculate sample variance for each feature (row)
            sample_variances = np.var(X, axis=1)
            
            # 4. Pick the top 'k' coordinates with the highest variance
            # np.argsort sorts ascending, so we take the last 'k' elements
            top_k_indices = set(np.argsort(sample_variances)[-k:])
            
            # 5. Check what fraction of the actual 'k' coordinates were correctly identified
            overlap = len(true_indices.intersection(top_k_indices)) / k
            trial_overlaps.append(overlap)
            
            # 6. Analyze ranks and variances
            true_indices_list = list(true_indices)
            
            # Ranks (0 is highest variance, p-1 is lowest)
            sorted_indices = np.argsort(sample_variances)[::-1]
            ranks = np.empty(p, dtype=int)
            ranks[sorted_indices] = np.arange(p)
            trial_ranks.append(np.mean(ranks[true_indices_list]))
            
            # Variances
            trial_median_vars.append(np.median(sample_variances))
            trial_true_vars.append(np.mean(sample_variances[true_indices_list]))
            
        avg_overlap = np.mean(trial_overlaps)
        avg_overlaps.append(avg_overlap)
        
        avg_rank = np.mean(trial_ranks)
        avg_true_ranks.append(avg_rank)
        
        avg_med_var = np.mean(trial_median_vars)
        avg_median_vars.append(avg_med_var)
        
        avg_true_var = np.mean(trial_true_vars)
        avg_true_vars.append(avg_true_var)
        
        # Calculate the BBP phase transition threshold for Standard PCA
        lambda_c = sigma**2 * np.sqrt(p / n)
        print(f"{n:<12} | {lambda_c:<10.2f} | {avg_overlap:<14.4f} | {avg_rank:<10.1f} | {avg_med_var:<9.4f} | {avg_true_var:<9.4f}")
        
    # --- Plotting the Results ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    
    # Plot 1: Recovery Fraction
    axes[0].plot(sample_sizes, avg_overlaps, marker='o', linestyle='-', color='b', label='Recovery Fraction')
    axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.6, label='Perfect Recovery (1.0)')
    axes[0].set_ylabel('Fraction Recovered')
    axes[0].set_title(f'Top-{k} Variance Coordinate Recovery vs. Sample Size')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].legend()
    
    # Plot 2: True Coordinate Ranks
    ideal_rank = (k - 1) / 2
    axes[1].plot(sample_sizes, avg_true_ranks, marker='s', linestyle='-', color='g', label='Avg True Coord Rank')
    axes[1].axhline(y=ideal_rank, color='r', linestyle='--', alpha=0.6, label=f'Ideal Avg Rank ({ideal_rank})')
    axes[1].set_ylabel('Average Rank (Lower is Better)')
    axes[1].set_title('Average Rank of True Signal Coordinates')
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend()
    
    # Plot 3: Variances
    axes[2].plot(sample_sizes, avg_true_vars, marker='^', linestyle='-', color='purple', label='Mean Var of True Coords')
    axes[2].plot(sample_sizes, avg_median_vars, marker='v', linestyle='-', color='orange', label='Median Var of All Coords')
    axes[2].set_xlabel('Number of Samples (n)')
    axes[2].set_ylabel('Sample Variance')
    axes[2].set_title('Variance Comparison: True Signal vs Median Noise')
    axes[2].set_xscale('log')
    axes[2].grid(True, which="both", ls="--", alpha=0.5)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_top_k_variance_experiment()