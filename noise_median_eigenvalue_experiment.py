import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_mixed_spiked_data


def get_nonzero_sample_eigenvalues(X):
    """Return non-zero sample covariance eigenvalues using the dual covariance trick."""
    n = X.shape[1]
    s_dual = (1.0 / n) * (X.T @ X)
    eigvals = np.linalg.eigvalsh(s_dual)
    return eigvals


def summarize_sigma_estimates(sigma_estimates, sigma_vals, num_trials):
    """Compute mean/std/CI/bias/RMSE summaries for sigma estimates."""
    sigma_hat_mean = np.mean(sigma_estimates, axis=1)
    sigma_hat_std = (
        np.std(sigma_estimates, axis=1, ddof=1)
        if num_trials > 1
        else np.zeros_like(sigma_hat_mean)
    )
    sigma_hat_ci95 = (
        1.96 * sigma_hat_std / np.sqrt(num_trials)
        if num_trials > 1
        else np.zeros_like(sigma_hat_mean)
    )
    sigma_hat_bias = sigma_hat_mean - sigma_vals
    sigma_hat_rmse = np.sqrt(np.mean((sigma_estimates - sigma_vals[:, None]) ** 2, axis=1))
    return {
        "mean": sigma_hat_mean,
        "std": sigma_hat_std,
        "ci95": sigma_hat_ci95,
        "bias": sigma_hat_bias,
        "rmse": sigma_hat_rmse,
    }


def run_noise_median_experiment(
    p=10000,
    n=200,
    sigma_values=None,
    lam1=20.0,
    lam2=5.0,
    sparsity_frac=0.001,
    num_trials=20,
):
    if sigma_values is None:
        sigma_values = [0.5, 1.0, 1.5, 2.0]

    sigma_vals = np.array(sigma_values, dtype=float)
    sigma_raw_estimates = np.zeros((len(sigma_vals), num_trials))
    sigma_scaled_estimates = np.zeros((len(sigma_vals), num_trials))
    sigma_sq_raw_estimates = np.zeros((len(sigma_vals), num_trials))
    sigma_sq_scaled_estimates = np.zeros((len(sigma_vals), num_trials))

    print("--- Noise estimation from median sample eigenvalue (raw vs scaled) ---")
    print(
        f"p={p}, n={n}, lambda_1={lam1}, lambda_2={lam2}, sparsity={sparsity_frac}, trials={num_trials}"
    )
    print(f"Trying sigma values: {sigma_vals.tolist()}")

    total_tests = len(sigma_vals) * num_trials
    completed_tests = 0

    for i, sigma_true in enumerate(sigma_vals):
        for t in range(num_trials):
            X, _, _ = generate_mixed_spiked_data(
                p, n, sigma_true, lam1, lam2, sparsity_frac
            )
            eigvals = get_nonzero_sample_eigenvalues(X)
            median_eig = float(np.median(eigvals))
            sigma_sq_hat_raw = median_eig
            sigma_sq_hat_scaled = (n / p) * median_eig
            sigma_hat_raw = float(np.sqrt(max(sigma_sq_hat_raw, 0.0)))
            sigma_hat_scaled = float(np.sqrt(max(sigma_sq_hat_scaled, 0.0)))

            sigma_sq_raw_estimates[i, t] = sigma_sq_hat_raw
            sigma_sq_scaled_estimates[i, t] = sigma_sq_hat_scaled
            sigma_raw_estimates[i, t] = sigma_hat_raw
            sigma_scaled_estimates[i, t] = sigma_hat_scaled

            completed_tests += 1
            pct = 100.0 * completed_tests / total_tests
            print(f"Progress: {completed_tests}/{total_tests} ({pct:5.1f}%)")

    raw_stats = summarize_sigma_estimates(sigma_raw_estimates, sigma_vals, num_trials)
    scaled_stats = summarize_sigma_estimates(sigma_scaled_estimates, sigma_vals, num_trials)
    sigma_sq_raw_mean = np.mean(sigma_sq_raw_estimates, axis=1)
    sigma_sq_scaled_mean = np.mean(sigma_sq_scaled_estimates, axis=1)

    return {
        "sigma_vals": sigma_vals,
        "raw_mean": raw_stats["mean"],
        "raw_std": raw_stats["std"],
        "raw_ci95": raw_stats["ci95"],
        "raw_bias": raw_stats["bias"],
        "raw_rmse": raw_stats["rmse"],
        "scaled_mean": scaled_stats["mean"],
        "scaled_std": scaled_stats["std"],
        "scaled_ci95": scaled_stats["ci95"],
        "scaled_bias": scaled_stats["bias"],
        "scaled_rmse": scaled_stats["rmse"],
        "sigma_sq_raw_mean": sigma_sq_raw_mean,
        "sigma_sq_scaled_mean": sigma_sq_scaled_mean,
    }


def plot_noise_estimation_results(results):
    sigma_vals = results["sigma_vals"]
    raw_mean = results["raw_mean"]
    raw_ci95 = results["raw_ci95"]
    scaled_mean = results["scaled_mean"]
    scaled_ci95 = results["scaled_ci95"]

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_vals, sigma_vals, "k--", label="Ideal: estimate = true sigma")
    plt.errorbar(
        sigma_vals,
        raw_mean,
        yerr=raw_ci95,
        fmt="o-",
        capsize=4,
        label=r"Raw: $\hat{\sigma}=\sqrt{\mathrm{median}(\lambda_i)}$",
    )
    plt.errorbar(
        sigma_vals,
        scaled_mean,
        yerr=scaled_ci95,
        fmt="s-",
        capsize=4,
        label=r"Scaled: $\hat{\sigma}=\sqrt{\frac{n}{p}\,\mathrm{median}(\lambda_i)}$",
    )
    plt.xlabel(r"True noise level $\sigma$")
    plt.ylabel(r"Estimated noise level $\hat{\sigma}$")
    plt.title(r"Noise estimation: raw vs scaled median eigenvalue")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare raw and (n/p)-scaled median-eigenvalue sigma estimators in the mixed sparse+dense spike model."
    )
    parser.add_argument("--p", type=int, default=10000)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument(
        "--sigma-values",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0],
        help="List of true sigma values to test.",
    )
    parser.add_argument("--lam1", type=float, default=20.0)
    parser.add_argument("--lam2", type=float, default=5.0)
    parser.add_argument("--sparsity-frac", type=float, default=0.001)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    results = run_noise_median_experiment(
        p=args.p,
        n=args.n,
        sigma_values=args.sigma_values,
        lam1=args.lam1,
        lam2=args.lam2,
        sparsity_frac=args.sparsity_frac,
        num_trials=args.num_trials,
    )

    print("\nSummary:")
    print(
        "sigma_true | raw sigma_hat(mean +/- CI) | scaled sigma_hat(mean +/- CI) | "
        "raw bias | scaled bias | raw RMSE | scaled RMSE | raw mean sigma_sq_hat | scaled mean sigma_sq_hat"
    )
    for i, sigma_true in enumerate(results["sigma_vals"]):
        print(
            f"{sigma_true:9.4f} | "
            f"{results['raw_mean'][i]:.4f} +/- {results['raw_ci95'][i]:.4f} | "
            f"{results['scaled_mean'][i]:.4f} +/- {results['scaled_ci95'][i]:.4f} | "
            f"{results['raw_bias'][i]:+.4f} | "
            f"{results['scaled_bias'][i]:+.4f} | "
            f"{results['raw_rmse'][i]:.4f} | "
            f"{results['scaled_rmse'][i]:.4f} | "
            f"{results['sigma_sq_raw_mean'][i]:.4f} | "
            f"{results['sigma_sq_scaled_mean'][i]:.4f}"
        )

    if not args.no_plot:
        plot_noise_estimation_results(results)


if __name__ == "__main__":
    main()

