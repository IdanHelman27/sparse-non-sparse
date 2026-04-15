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


def peel_direction(X, d_hat):
    """Peel/remove one direction from X: X <- X - d_hat d_hat^T X."""
    return X - d_hat @ (d_hat.T @ X)


def run_pipeline_once(X):
    """Run: sparse PCA -> peel sparse -> PCA -> peel dense from original -> sparse PCA."""
    u_hat_1, _, _ = sparse_pca_johnstone_lu(X)
    X_prime = peel_direction(X, u_hat_1)

    v_hat = get_leading_eigenvector_dual(X_prime)
    X_double_prime = peel_direction(X, v_hat)

    u_hat_2, _, _ = sparse_pca_johnstone_lu(X_double_prime)
    return u_hat_1, v_hat, u_hat_2


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

    u1_u_align = np.zeros(num_lam1)
    u1_v_align = np.zeros(num_lam1)
    vhat_v_align = np.zeros(num_lam1)
    vhat_u_align = np.zeros(num_lam1)
    u2_u_align = np.zeros(num_lam1)
    u2_v_align = np.zeros(num_lam1)
    u2_u_ci95 = np.zeros(num_lam1)

    print("--- Sparse -> Dense -> Sparse pipeline experiment ---")
    print(f"p={p}, n={n}, sigma={sigma}, lambda_2={lam2}, sparsity={sparsity_frac}")
    print(f"Sweeping lambda_1 in [{lam1_min}, {lam1_max}] with {num_lam1} points")

    total_tests = num_lam1 * num_trials
    completed_tests = 0

    for j, lam1 in enumerate(lam1_vals):
        trial_u1_u = []
        trial_u1_v = []
        trial_vhat_v = []
        trial_vhat_u = []
        trial_u2_u = []
        trial_u2_v = []

        for _ in range(num_trials):
            X, v_true, u_true = generate_mixed_spiked_data(
                p, n, sigma, lam1, lam2, sparsity_frac
            )
            u_hat_1, v_hat, u_hat_2 = run_pipeline_once(X)

            trial_u1_u.append(squared_cosine(u_true, u_hat_1))
            trial_u1_v.append(squared_cosine(v_true, u_hat_1))
            trial_vhat_v.append(squared_cosine(v_true, v_hat))
            trial_vhat_u.append(squared_cosine(u_true, v_hat))
            trial_u2_u.append(squared_cosine(u_true, u_hat_2))
            trial_u2_v.append(squared_cosine(v_true, u_hat_2))

            completed_tests += 1
            pct = 100.0 * completed_tests / total_tests
            print(f"Progress: {completed_tests}/{total_tests} ({pct:5.1f}%)")

        u1_u_align[j] = float(np.mean(trial_u1_u))
        u1_v_align[j] = float(np.mean(trial_u1_v))
        vhat_v_align[j] = float(np.mean(trial_vhat_v))
        vhat_u_align[j] = float(np.mean(trial_vhat_u))
        u2_u_align[j] = float(np.mean(trial_u2_u))
        u2_v_align[j] = float(np.mean(trial_u2_v))

        std_u2_u = float(np.std(trial_u2_u, ddof=1)) if num_trials > 1 else 0.0
        u2_u_ci95[j] = 1.96 * std_u2_u / np.sqrt(num_trials) if num_trials > 1 else 0.0

    return {
        "lam1_vals": lam1_vals,
        "u1_u_align": u1_u_align,
        "u1_v_align": u1_v_align,
        "vhat_v_align": vhat_v_align,
        "vhat_u_align": vhat_u_align,
        "u2_u_align": u2_u_align,
        "u2_v_align": u2_v_align,
        "u2_u_ci95": u2_u_ci95,
    }


def plot_results(results, lam2):
    lam1_vals = results["lam1_vals"]
    u1_u_align = results["u1_u_align"]
    u2_u_align = results["u2_u_align"]
    u2_u_ci95 = results["u2_u_ci95"]
    vhat_v_align = results["vhat_v_align"]
    vhat_u_align = results["vhat_u_align"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(lam1_vals, u1_u_align, "o-", label=r"Step 1 $\hat{u}_1$ vs true $u$")
    axes[0].plot(lam1_vals, u2_u_align, "s-", label=r"Step 5 $\hat{u}_2$ vs true $u$")
    lower = np.clip(u2_u_align - u2_u_ci95, 0.0, 1.0)
    upper = np.clip(u2_u_align + u2_u_ci95, 0.0, 1.0)
    axes[0].fill_between(lam1_vals, lower, upper, alpha=0.2)
    axes[0].set_xlabel(r"$\lambda_1$ (Dense spike strength)")
    axes[0].set_ylabel(r"Squared cosine with true sparse $u$")
    axes[0].set_title(rf"Sparse recovery before vs after full pipeline ($\lambda_2={lam2}$)")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend()

    axes[1].plot(lam1_vals, vhat_v_align, "o-", label=r"Step 3 $\hat{v}$ vs true dense $v$")
    axes[1].plot(lam1_vals, vhat_u_align, "s-", label=r"Step 3 $\hat{v}$ vs true sparse $u$")
    axes[1].set_xlabel(r"$\lambda_1$ (Dense spike strength)")
    axes[1].set_ylabel(r"Squared cosine")
    axes[1].set_title(r"Dense estimate quality after peeling Step-1 sparse estimate")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Test sparse->dense->sparse peeling pipeline on mixed sparse+dense spikes."
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
    print("lambda_1 | u1_u | u1_v | vhat_v | vhat_u | u2_u (+/-95%CI) | u2_v")
    for j, lam1 in enumerate(results["lam1_vals"]):
        print(
            f"{lam1:8.2f} | "
            f"{results['u1_u_align'][j]:6.4f} | "
            f"{results['u1_v_align'][j]:6.4f} | "
            f"{results['vhat_v_align'][j]:7.4f} | "
            f"{results['vhat_u_align'][j]:7.4f} | "
            f"{results['u2_u_align'][j]:.4f} +/- {results['u2_u_ci95'][j]:.4f} | "
            f"{results['u2_v_align'][j]:6.4f}"
        )

    if not args.no_plot:
        plot_results(results, args.lam2)


if __name__ == "__main__":
    main()

