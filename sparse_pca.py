import numpy as np
from numpy.linalg import eigh

def _hard_threshold(data, threshold):
    """Apply hard thresholding to the data."""
    return data * (np.abs(data) >= threshold)

def sparse_pca_johnstone_lu(X, alpha=2.0, num_components=1):
    """
    Performs sparse PCA based on the Johnstone & Lu (2009) paper,
    assuming the standard basis is appropriate (no wavelet transform).

    Args:
        X (np.ndarray): Data matrix with shape (p, n), where p is features
                        and n is samples.
        alpha (float): A tuning parameter that controls the strictness of the
                       variance threshold. Lower values are less strict.
        num_components (int): The number of top principal components to return.

    Returns:
        tuple: (v_hat_sparse, k)
            - v_hat_sparse (np.ndarray): The estimated sparse eigenvector(s) shape (p, num_components).
            - k (int): The number of components selected.
            - lambda_hat (float or np.ndarray): The estimated eigenvalue(s) of the selected components.
    """
    p, n = X.shape

    # --- Step 1: Subset Selection (Data-Based Choice of k) ---
    
    # Calculate sample variance for each feature (row)
    sample_variances = np.var(X, axis=1)
    
    # Estimate noise level sigma_hat^2 as the median of sample variances
    # This is from the paper, Section 4.2, Equation (14)
    sigma_hat_sq = np.median(sample_variances)

    # Determine k using the variance threshold rule from Section 3.3, Eq. (11).
    # This selects features whose variance is statistically significant above the noise floor.
    # The threshold is T = sigma_hat^2 * (1 + alpha_n)
    
    # The paper suggests alpha > sqrt(12) for the proof. We use the default alpha=2.0 as a practical value. 
    alpha_n = alpha * np.sqrt(np.log(max(p, n)) / n)
    variance_threshold = sigma_hat_sq * (1 + alpha_n)

    # Select indices of features that pass the threshold
    top_k_indices = np.where(sample_variances >= variance_threshold)[0]
    
    k = len(top_k_indices)

    # If no features are selected, fall back to selecting the one with the highest variance
    if k == 0:
        top_k_indices = np.array([np.argmax(sample_variances)])
        k = 1


    # --- Step 2: Reduced PCA ---
    # Create a reduced dataset with only the top k features
    reduced_dataset = X[top_k_indices, :]
    
    # Calculate the (k x k) covariance matrix of the reduced dataset
    cov_reduced = (1 / n) * (reduced_dataset @ reduced_dataset.T)
    
    # Find the leading eigenvector of this smaller covariance matrix
    eigenvalues, p_tilde_full = eigh(cov_reduced)

    # --- Step 3: Reconstruction & Thresholding ---
    
    noise_std_dev_in_pc = np.sqrt(sigma_hat_sq / n) # Heuristic for noise in PCs
    universal_threshold = noise_std_dev_in_pc * np.sqrt(2 * np.log(k))
    
    if num_components == 1:
        p_tilde = p_tilde_full[:, -1] # This is a k-dimensional vector
        lambda_hat = eigenvalues[-1]

        p_tilde_thresholded = _hard_threshold(p_tilde, universal_threshold)

        # Reconstruct the full p-dimensional eigenvector
        v_hat_sparse = np.zeros((p, 1))
        v_hat_sparse[top_k_indices] = p_tilde_thresholded.reshape(-1, 1)
        
        # Normalize the final vector
        norm = np.linalg.norm(v_hat_sparse)
        if norm > 0:
            v_hat_sparse = v_hat_sparse / norm

        return v_hat_sparse, k, lambda_hat
    else:
        actual_num_components = min(num_components, k)
        top_eigenvalues = eigenvalues[-actual_num_components:][::-1]
        top_p_tilde = p_tilde_full[:, -actual_num_components:][:, ::-1]
        
        V_hat_sparse = np.zeros((p, num_components))
        for i in range(actual_num_components):
            p_tilde_thresholded = _hard_threshold(top_p_tilde[:, i], universal_threshold)
            v_hat = np.zeros((p, 1))
            v_hat[top_k_indices] = p_tilde_thresholded.reshape(-1, 1)
            norm = np.linalg.norm(v_hat)
            if norm > 0:
                v_hat = v_hat / norm
            V_hat_sparse[:, i:i+1] = v_hat
            
        return V_hat_sparse, k, top_eigenvalues
