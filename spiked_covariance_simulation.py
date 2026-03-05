import numpy as np
import matplotlib.pyplot as plt

# 1. Set the dimensions
N = 500    # Number of features/dimensions
T = 2000   # Number of samples
gamma = N / T  # Aspect ratio (0.25)

# Calculate theoretical boundaries
l_c = 1 + np.sqrt(gamma)          # Critical threshold (1.5)
lambda_plus = (1 + np.sqrt(gamma))**2 # Bulk upper edge (2.25)

# 2. Define the population spikes
l_strong = 3.0  # > 1.5 (Strong signal, will separate)
l_weak = 1.2    # < 1.5 (Weak signal, will be absorbed)

# Theoretical prediction for the separated empirical eigenvalue
predicted_spike = l_strong + (gamma * l_strong) / (l_strong - 1) 

# 3. Generate the Data
# Start with standard Gaussian noise Z ~ N(0, 1)
np.random.seed(45)
X = np.random.randn(N, T)

# Inject the spikes by scaling the variance of the first two dimensions
X[0, :] = X[0, :] * np.sqrt(l_strong)
X[1, :] = X[1, :] * np.sqrt(l_weak)

# 4. Compute the Empirical Covariance Matrix and its Eigenvalues
S = (X @ X.T) / T
eigenvalues = np.real(np.linalg.eigvals(S))

# 5. Visualize the Phase Transition
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues, bins=60, density=True, alpha=0.7, color='steelblue', label='Empirical Eigenvalues')

# Add markers for our theoretical boundaries and predictions
plt.axvline(x=lambda_plus, color='red', linestyle='--', linewidth=2, label=rf'MP Bulk Edge ($\lambda_+={lambda_plus:.2f}$)')
plt.axvline(x=predicted_spike, color='green', linestyle='--', linewidth=2, label=f'Predicted Strong Spike ({predicted_spike:.2f})')

plt.title('Simulating the Spiked Covariance Model')
plt.xlabel('Eigenvalue Magnitude')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()