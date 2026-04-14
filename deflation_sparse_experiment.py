import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_mixed_spiked_data
from sparse_pca import sparse_pca_johnstone_lu


def squared_cosine(a, b):
    """Return squared cosine similarity between two vectors in [0, 1]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    val = float((a.T @ b)[0, 0] / denom)
    return val * val


def get_leading_eigenvector_dual(X):
    """Estimate the leading sample eigenvector using the dual covariance matrix."""
    n = X.shape[1]
    s_dual = (1.0 / n) * (X.T @ X)
    _, evecs = np.linalg.eigh(s_dual)
    u_dual = evecs[:, -1].reshape(n, 1)
    v_hat = X @ u_dual
    norm = np.linalg.norm(v_hat)
    if norm > 0:
        v_hat = v_hat / norm
    return v_hat


def deflate_top_direction(X, v_hat):
    """Deflate the top direction: x_i <- x_i - v_hat v_hat^T x_i."""
    return X - v_hat @ (v_hat.T @ X)


def run_experiment(
    p=10000,
    n=200,
    sigma=1.0,
    lam2=5.0,
    sparsity_frac=0.001,
    lam1_min=10.0,
    lam1_max=800.0,
    num_lam1=12,
    num_trials=10,
):
    lam1_vals = np.linspace(lam1_min, lam1_max, num_lam1)

    direct_u_align, direct_v_align = [], []
    deflated_u_align, deflated_v_align = [], []

    print("--- Deflation + Sparse PCA experiment ---")
    print(f"p={p}, n={n}, sigma={sigma}, lambda_2={lam2}, sparsity={sparsity_frac}")
    print(f"Sweeping lambda_1 in [{lam1_min}, {lam1_max}] with {num_lam1} points")
    total_tests = len(lam1_vals) * num_trials
    completed_tests = 0

    for lam1 in lam1_vals:
        trial_direct_u, trial_direct_v = [], []
        trial_defl_u, trial_defl_v = [], []

        for _ in range(num_trials):
            X, v_true, u_true = generate_mixed_spiked_data(
                p, n, sigma, lam1, lam2, sparsity_frac
            )

            # Baseline: direct Sparse PCA
            u_hat_direct, _, _ = sparse_pca_johnstone_lu(X)
            trial_direct_u.append(squared_cosine(u_true, u_hat_direct))
            trial_direct_v.append(squared_cosine(v_true, u_hat_direct))

            # Proposed: remove top dense direction first, then Sparse PCA
            v_hat_top = get_leading_eigenvector_dual(X)
            X_deflated = deflate_top_direction(X, v_hat_top)
            u_hat_deflated, _, _ = sparse_pca_johnstone_lu(X_deflated)
            trial_defl_u.append(squared_cosine(u_true, u_hat_deflated))
            trial_defl_v.append(squared_cosine(v_true, u_hat_deflated))

            completed_tests += 1
            pct = 100.0 * completed_tests / total_tests
            print(f"Progress: {completed_tests}/{total_tests} ({pct:5.1f}%)")

        direct_u_align.append(np.mean(trial_direct_u))
        direct_v_align.append(np.mean(trial_direct_v))
        deflated_u_align.append(np.mean(trial_defl_u))
        deflated_v_align.append(np.mean(trial_defl_v))

    return {
        "lam1_vals": lam1_vals,
        "direct_u_align": np.array(direct_u_align),
        "direct_v_align": np.array(direct_v_align),
        "deflated_u_align": np.array(deflated_u_align),
        "deflated_v_align": np.array(deflated_v_align),
    }


def plot_results(results, lam2):
    lam1_vals = results["lam1_vals"]
    direct_u_align = results["direct_u_align"]
    direct_v_align = results["direct_v_align"]
    deflated_u_align = results["deflated_u_align"]
    deflated_v_align = results["deflated_v_align"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(lam1_vals, direct_u_align, "o-", label="Direct Sparse PCA: align with sparse u")
    axes[0].plot(
        lam1_vals,
        deflated_u_align,
        "s-",
        label="Deflation + Sparse PCA: align with sparse u",
    )
    axes[0].set_xlabel(r"$\lambda_1$ (Dense spike strength)")
    axes[0].set_ylabel(r"Squared cosine with true sparse $u$")
    axes[0].set_title(rf"Sparse recovery vs. $\lambda_1$ (fixed $\lambda_2={lam2}$)")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend()

    axes[1].plot(lam1_vals, direct_v_align, "o-", label="Direct Sparse PCA: align with dense v")
    axes[1].plot(
        lam1_vals,
        deflated_v_align,
        "s-",
        label="Deflation + Sparse PCA: align with dense v",
    )
    axes[1].set_xlabel(r"$\lambda_1$ (Dense spike strength)")
    axes[1].set_ylabel(r"Squared cosine with true dense $v$")
    axes[1].set_title(r"Dense leakage into estimated sparse vector")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare direct Sparse PCA vs. PCA-deflation + Sparse PCA."
    )
    parser.add_argument("--p", type=int, default=10000)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--lam2", type=float, default=5.0)
    parser.add_argument("--sparsity-frac", type=float, default=0.001)
    parser.add_argument("--lam1-min", type=float, default=10.0)
    parser.add_argument("--lam1-max", type=float, default=800.0)
    parser.add_argument("--num-lam1", type=int, default=12)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    results = run_experiment(
        p=args.p,
        n=args.n,
        sigma=args.sigma,
        lam2=args.lam2,
        sparsity_frac=args.sparsity_frac,
        lam1_min=args.lam1_min,
        lam1_max=args.lam1_max,
        num_lam1=args.num_lam1,
        num_trials=args.num_trials,
    )

    print("\nSummary (average squared cosine):")
    print("lambda_1 | direct_u | deflated_u | direct_v | deflated_v")
    for i, lam1 in enumerate(results["lam1_vals"]):
        print(
            f"{lam1:8.2f} | "
            f"{results['direct_u_align'][i]:8.4f} | "
            f"{results['deflated_u_align'][i]:10.4f} | "
            f"{results['direct_v_align'][i]:8.4f} | "
            f"{results['deflated_v_align'][i]:10.4f}"
        )

    if not args.no_plot:
        plot_results(results, args.lam2)


if __name__ == "__main__":
    main()

