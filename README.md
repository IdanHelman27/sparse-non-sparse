# Spiked Covariance Model Simulations

This repository contains Python scripts for simulating and visualizing phenomena associated with the spiked covariance model from Random Matrix Theory (RMT). These simulations explore how signals can be detected in high-dimensional, noisy data.
It compares the theoretical limits of standard Principal Component Analysis (PCA) against the recovery capabilities of Sparse PCA.

## Scripts

- **`spiked_covariance_simulation.py`**: This script simulates the Baik-Ben Arous-Péché (BBP) phase transition for eigenvalues. It shows how a "strong" spike in the population covariance matrix creates an outlier eigenvalue in the sample covariance matrix, while a "weak" spike is absorbed into the Marchenko-Pastur bulk.

- **`eigenvector_alignment_simulation.py`**: This script explores the `p >> n` (high-dimensionality) regime. It simulates the phase transition for eigenvector alignment, plotting the squared cosine similarity between the true population eigenvector and the sample eigenvector as a function of signal strength (`lambda`).

- **`fast_eigenvector_simulation.py`**: A high-performance version of the eigenvector alignment simulation. This script is optimized for very large `p` using:
    1.  The "dual covariance" method to avoid forming large `p x p` matrices.
    2.  The Numba JIT compiler for significant speed improvements.

- **`sparse_pca.py`**: Implements Sparse PCA based on the Johnstone & Lu (2009) algorithm, which performs subset selection based on coordinate variance followed by reduced PCA and hard thresholding.

- **`simulation_comparison.py`**: A direct comparison between Standard PCA and Sparse PCA. It generates a sparse spiked signal, runs both algorithms, and plots the recovered eigenvectors to visually demonstrate how Sparse PCA succeeds in the `p >> n` regime where Standard PCA fails.

- **`top_k_variance_experiment.py`**: An empirical experiment isolating the subset selection step of Sparse PCA. It verifies whether the `k` coordinates with the highest variance successfully identify the true non-zero components of a sparse signal across varying sample sizes.

## Setup and Installation

To run these simulations, you'll need Python 3 and the packages listed in `requirements.txt`.

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/IdanHelman27/sparse-non-sparse.git
    cd sparse-non-sparse
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Simulations

With your virtual environment activated, you can run any of the simulation scripts.

### Eigenvalue Phase Transition
This simulation shows the distribution of eigenvalues.
```bash
python spiked_covariance_simulation.py
```

### Eigenvector Alignment Simulation
This simulation is optimized for speed and shows the phase transition for eigenvector recovery. It will first calculate the theoretical transition point based on the parameters (`p`, `n`) and then simulate across that range.
```bash
python fast_eigenvector_simulation.py
```

The scripts will run the simulation and display a Matplotlib plot of the results.