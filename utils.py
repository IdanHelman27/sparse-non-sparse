import numpy as np

def generate_spiked_data(p, n, sigma, lam, sparse=True, sparsity_fraction=0.1):
    """
    Generates a data matrix X with a spiked covariance structure.

    The population covariance is S = lambda * v * v.T + sigma^2 * I.
    The data is generated efficiently without forming the large covariance matrix.

    Args:
        p (int): Number of features.
        n (int): Number of samples.
        sigma (float): Standard deviation of the isotropic noise.
        lam (float): The spike strength (eigenvalue of the signal part).
        sparse (bool): If True, the true eigenvector 'v' is sparse. 
                       If False, it's a dense Gaussian vector.
        sparsity_fraction (float): If sparse, the fraction of non-zero elements in 'v'.

    Returns:
        tuple: (X, v)
            - X (np.ndarray): The generated data matrix of shape (p, n).
            - v (np.ndarray): The true leading eigenvector of shape (p, 1).
    """
    # --- Define the true leading eigenvector 'v' ---
    if sparse:
        # Create a sparse vector
        v = np.zeros((p, 1))
        num_nonzero = int(p * sparsity_fraction)
        if num_nonzero > 0:
            # Randomly choose indices to be non-zero
            nonzero_indices = np.random.choice(p, num_nonzero, replace=False)
            # Assign random +/- 1 values to these indices (Rademacher distribution).
            # This ensures no non-zero coordinate is "too small" and that the 
            # signal variance is spread perfectly evenly across the true support.
            v[nonzero_indices] = np.random.choice([-1.0, 1.0], size=(num_nonzero, 1))
    else:
        # Create a dense Gaussian vector
        v = np.random.randn(p, 1)
    
    # Normalize to unit norm
    v = v / np.linalg.norm(v)

    # --- Generate the data matrix X ---
    # X = sqrt(lambda) * v * y.T + sigma * Z
    # where y ~ N(0, 1) and Z is a matrix of standard normals.
    # An equivalent and more direct way from fast_eigenvector_simulation.py:
    
    Z = np.random.randn(p, n)
    
    # These terms define the transformation on Z to get the desired covariance
    sqrt_term1 = np.sqrt(lam + sigma**2) - sigma
    sqrt_term2 = sigma

    # Transform Z to get X
    X = (sqrt_term1 * v * (v.T @ Z)) + (sqrt_term2 * Z)

    return X, v

def generate_mixed_spiked_data(p, n, sigma, lam1, lam2, sparsity_fraction=0.1):
    """
    Generates a data matrix X with a mixed spiked covariance structure containing
    both a dense and a sparse signal component.

    The population covariance is S = lam1 * v * v.T + lam2 * u * u.T + sigma^2 * I.
    The vectors u and v are orthonormal, where u is sparse and v is dense.
    The constraints lam1 > lam2 > sigma^2 should ideally hold.

    Args:
        p (int): Number of features.
        n (int): Number of samples.
        sigma (float): Standard deviation of the isotropic noise.
        lam1 (float): The spike strength for the dense vector v.
        lam2 (float): The spike strength for the sparse vector u.
        sparsity_fraction (float): The fraction of non-zero elements in 'u'.

    Returns:
        tuple: (X, v, u)
            - X (np.ndarray): The generated data matrix of shape (p, n).
            - v (np.ndarray): The true dense leading eigenvector of shape (p, 1).
            - u (np.ndarray): The true sparse second eigenvector of shape (p, 1).
    """
    # --- 1. Define the sparse vector 'u' ---
    u = np.zeros((p, 1))
    num_nonzero = int(p * sparsity_fraction)
    if num_nonzero > 0:
        nonzero_indices = np.random.choice(p, num_nonzero, replace=False)
        u[nonzero_indices] = np.random.choice([-1.0, 1.0], size=(num_nonzero, 1))
    
    norm_u = np.linalg.norm(u)
    if norm_u > 0:
        u = u / norm_u
    elif p > 0:
        u[0, 0] = 1.0  # Fallback to prevent divide-by-zero if sparsity_fraction is 0.0

    # --- 2. Define the dense Gaussian vector 'v' ---
    v = np.random.randn(p, 1)
    # Make v orthogonal to u using Gram-Schmidt
    # v = v - (u.T @ v)[0, 0] * u
    v = v / np.linalg.norm(v)

    # --- 3. Generate the data matrix X ---
    Z = np.random.randn(p, n)
    
    # We want S^{1/2} = (sqrt(lam1 + sigma^2) - sigma)*v*v.T + (sqrt(lam2 + sigma^2) - sigma)*u*u.T + sigma*I
    sqrt_term1 = np.sqrt(lam1 + sigma**2) - sigma
    sqrt_term2 = np.sqrt(lam2 + sigma**2) - sigma

    # Transform Z to get X efficiently without forming large covariance matrices
    X = (sqrt_term1 * v * (v.T @ Z)) + (sqrt_term2 * u * (u.T @ Z)) + (sigma * Z)

    return X, v, u