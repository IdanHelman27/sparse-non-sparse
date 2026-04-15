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


def run_deflation_sparse_pca(X):
    """Run the proposed pipeline: top-direction deflation followed by Sparse PCA."""
    v_hat_top = get_leading_eigenvector_dual(X)
    X_deflated = deflate_top_direction(X, v_hat_top)
    u_hat_deflated, _, _ = sparse_pca_johnstone_lu(X_deflated)
    return u_hat_deflated


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
            u_hat_deflated = run_deflation_sparse_pca(X)
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


def run_lambda2_deflation_experiment(
    p=10000,
    n=200,
    sigma=1.0,
    sparsity_frac=0.001,
    lam1_min=10.0,
    lam1_max=800.0,
    num_lam1=12,
    lam2_values=None,
    num_trials=10,
):
    """Sweep lambda_1 and compare deflated sparse recovery across lambda_2 values."""
    if lam2_values is None:
        lam2_values = [2.0, 5.0, 10.0, 20.0]

    lam1_vals = np.linspace(lam1_min, lam1_max, num_lam1)
    lam2_vals = np.array(lam2_values, dtype=float)
    deflated_u_align = np.zeros((len(lam2_vals), len(lam1_vals)))
    deflated_u_std = np.zeros((len(lam2_vals), len(lam1_vals)))
    deflated_u_ci95 = np.zeros((len(lam2_vals), len(lam1_vals)))

    print("--- Deflation success vs lambda_2 experiment ---")
    print(f"p={p}, n={n}, sigma={sigma}, sparsity={sparsity_frac}")
    print(f"Sweeping lambda_1 in [{lam1_min}, {lam1_max}] with {num_lam1} points")
    print(f"Trying lambda_2 values: {lam2_vals.tolist()}")

    total_tests = len(lam2_vals) * len(lam1_vals) * num_trials
    completed_tests = 0

    for i, lam2 in enumerate(lam2_vals):
        for j, lam1 in enumerate(lam1_vals):
            trial_defl_u = []
            for _ in range(num_trials):
                X, _, u_true = generate_mixed_spiked_data(
                    p, n, sigma, lam1, lam2, sparsity_frac
                )
                u_hat_deflated = run_deflation_sparse_pca(X)
                trial_defl_u.append(squared_cosine(u_true, u_hat_deflated))

                completed_tests += 1
                pct = 100.0 * completed_tests / total_tests
                print(f"Progress: {completed_tests}/{total_tests} ({pct:5.1f}%)")

            trial_arr = np.array(trial_defl_u, dtype=float)
            mean_val = float(np.mean(trial_arr))
            std_val = float(np.std(trial_arr, ddof=1)) if num_trials > 1 else 0.0
            ci95_halfwidth = 1.96 * std_val / np.sqrt(num_trials) if num_trials > 1 else 0.0

            deflated_u_align[i, j] = mean_val
            deflated_u_std[i, j] = std_val
            deflated_u_ci95[i, j] = ci95_halfwidth

    return {
        "lam1_vals": lam1_vals,
        "lam2_vals": lam2_vals,
        "deflated_u_align": deflated_u_align,
        "deflated_u_std": deflated_u_std,
        "deflated_u_ci95": deflated_u_ci95,
    }


def run_top_u_alignment_experiment(
    p=10000,
    n=200,
    sigma=1.0,
    sparsity_frac=0.001,
    lam1_min=10.0,
    lam1_max=800.0,
    num_lam1=12,
    lam2_values=None,
    num_trials=10,
):
    """Sweep lambda_1 and measure alignment between v_hat_top and u_true across lambda_2."""
    if lam2_values is None:
        lam2_values = [2.0, 5.0, 10.0, 20.0]

    lam1_vals = np.linspace(lam1_min, lam1_max, num_lam1)
    lam2_vals = np.array(lam2_values, dtype=float)
    top_u_align = np.zeros((len(lam2_vals), len(lam1_vals)))

    print("--- Alignment of v_hat_top with u_true vs lambda_2 experiment ---")
    print(f"p={p}, n={n}, sigma={sigma}, sparsity={sparsity_frac}")
    print(f"Sweeping lambda_1 in [{lam1_min}, {lam1_max}] with {num_lam1} points")
    print(f"Trying lambda_2 values: {lam2_vals.tolist()}")

    total_tests = len(lam2_vals) * len(lam1_vals) * num_trials
    completed_tests = 0

    for i, lam2 in enumerate(lam2_vals):
        for j, lam1 in enumerate(lam1_vals):
            trial_top_u = []
            for _ in range(num_trials):
                X, _, u_true = generate_mixed_spiked_data(
                    p, n, sigma, lam1, lam2, sparsity_frac
                )
                v_hat_top = get_leading_eigenvector_dual(X)
                trial_top_u.append(squared_cosine(v_hat_top, u_true))

                completed_tests += 1
                pct = 100.0 * completed_tests / total_tests
                print(f"Progress: {completed_tests}/{total_tests} ({pct:5.1f}%)")

            top_u_align[i, j] = float(np.mean(trial_top_u))

    return {
        "lam1_vals": lam1_vals,
        "lam2_vals": lam2_vals,
        "top_u_align": top_u_align,
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


def plot_lambda2_deflation_results(results):
    lam1_vals = results["lam1_vals"]
    lam2_vals = results["lam2_vals"]
    deflated_u_align = results["deflated_u_align"]
    deflated_u_ci95 = results["deflated_u_ci95"]

    plt.figure(figsize=(10, 6))
    for i, lam2 in enumerate(lam2_vals):
        line, = plt.plot(
            lam1_vals,
            deflated_u_align[i],
            "o-",
            label=rf"Deflated Sparse PCA ($\lambda_2={lam2:.2f}$)",
        )
        lower = np.clip(deflated_u_align[i] - deflated_u_ci95[i], 0.0, 1.0)
        upper = np.clip(deflated_u_align[i] + deflated_u_ci95[i], 0.0, 1.0)
        plt.fill_between(
            lam1_vals,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.2,
        )

    plt.xlabel(r"$\lambda_1$ (Dense spike strength)")
    plt.ylabel(r"Squared cosine with true sparse $u$")
    plt.title(r"Deflation recovery vs. $\lambda_1$ for multiple $\lambda_2$")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_top_u_alignment_results(results):
    lam1_vals = results["lam1_vals"]
    lam2_vals = results["lam2_vals"]
    top_u_align = results["top_u_align"]

    plt.figure(figsize=(10, 6))
    for i, lam2 in enumerate(lam2_vals):
        plt.plot(
            lam1_vals,
            top_u_align[i],
            "o-",
            label=rf"Top-PCA alignment with $u$ ($\lambda_2={lam2:.2f}$)",
        )

    plt.xlabel(r"$\lambda_1$ (Dense spike strength)")
    plt.ylabel(r"Squared cosine with true sparse $u$")
    plt.title(r"Alignment of $\hat{v}_{top}$ with sparse $u$ across $\lambda_2$")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare direct Sparse PCA vs. PCA-deflation + Sparse PCA."
    )
    parser.add_argument(
        "--experiment",
        choices=["original", "lam2-sweep", "top-u-alignment"],
        default="original",
        help="Experiment mode: original baseline comparison, lambda_2 sweep for deflation, or top-vs-u alignment.",
    )
    parser.add_argument("--p", type=int, default=10000)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--lam2", type=float, default=5.0)
    parser.add_argument(
        "--lam2-values",
        nargs="+",
        type=float,
        default=[2.0, 5.0, 10.0, 20.0],
        help="List of lambda_2 values used by --experiment lam2-sweep.",
    )
    parser.add_argument("--sparsity-frac", type=float, default=0.001)
    parser.add_argument("--lam1-min", type=float, default=10.0)
    parser.add_argument("--lam1-max", type=float, default=800.0)
    parser.add_argument("--num-lam1", type=int, default=12)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.experiment == "original":
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
    elif args.experiment == "lam2-sweep":
        results = run_lambda2_deflation_experiment(
            p=args.p,
            n=args.n,
            sigma=args.sigma,
            sparsity_frac=args.sparsity_frac,
            lam1_min=args.lam1_min,
            lam1_max=args.lam1_max,
            num_lam1=args.num_lam1,
            lam2_values=args.lam2_values,
            num_trials=args.num_trials,
        )

        print("\nSummary (mean +/- 95% CI half-width for squared cosine with true sparse u):")
        header = "lambda_1 | " + " | ".join(
            [f"lam2={lam2:.2f}" for lam2 in results["lam2_vals"]]
        )
        print(header)
        for j, lam1 in enumerate(results["lam1_vals"]):
            row_vals = " | ".join(
                [
                    (
                        f"{results['deflated_u_align'][i, j]:.4f} "
                        f"+/- {results['deflated_u_ci95'][i, j]:.4f}"
                    )
                    for i in range(len(results["lam2_vals"]))
                ]
            )
            print(f"{lam1:8.2f} | {row_vals}")

        if not args.no_plot:
            plot_lambda2_deflation_results(results)
    else:
        results = run_top_u_alignment_experiment(
            p=args.p,
            n=args.n,
            sigma=args.sigma,
            sparsity_frac=args.sparsity_frac,
            lam1_min=args.lam1_min,
            lam1_max=args.lam1_max,
            num_lam1=args.num_lam1,
            lam2_values=args.lam2_values,
            num_trials=args.num_trials,
        )

        print("\nSummary (average squared cosine between v_hat_top and true sparse u):")
        header = "lambda_1 | " + " | ".join(
            [f"lam2={lam2:.2f}" for lam2 in results["lam2_vals"]]
        )
        print(header)
        for j, lam1 in enumerate(results["lam1_vals"]):
            row_vals = " | ".join(
                [f"{results['top_u_align'][i, j]:8.4f}" for i in range(len(results["lam2_vals"]))]
            )
            print(f"{lam1:8.2f} | {row_vals}")

        if not args.no_plot:
            plot_top_u_alignment_results(results)


if __name__ == "__main__":
    main()

