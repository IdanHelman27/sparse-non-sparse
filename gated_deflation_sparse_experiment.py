import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_mixed_spiked_data
from sparse_pca import sparse_pca_johnstone_lu
from eigenvalue_gap_distinguisher import classify_lambda_gap_from_data


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
    """Deflate one direction from X: X <- X - v_hat v_hat^T X."""
    return X - v_hat @ (v_hat.T @ X)


def run_direct_sparse_pca(X):
    """Run direct sparse PCA on X."""
    u_hat, _, _ = sparse_pca_johnstone_lu(X)
    return u_hat


def run_deflated_sparse_pca(X):
    """Run always-deflate then sparse PCA."""
    v_hat_top = get_leading_eigenvector_dual(X)
    X_deflated = deflate_top_direction(X, v_hat_top)
    u_hat, _, _ = sparse_pca_johnstone_lu(X_deflated)
    return u_hat


def run_gated_sparse_pca(
    X,
    p,
    n,
    num_top_trim=2,
    edge_buffer=0.25,
    z_threshold=2.0,
):
    """
    Run gated pipeline:
      1) classify lambda-gap regime from sample eigenvalues
      2) if 'far' -> deflate top sample direction
      3) run sparse PCA on chosen matrix
    """
    gate = classify_lambda_gap_from_data(
        X=X,
        p=p,
        n=n,
        num_top_trim=num_top_trim,
        edge_buffer=edge_buffer,
        z_threshold=z_threshold,
    )
    if gate["label"] == "far":
        v_hat_top = get_leading_eigenvector_dual(X)
        X_used = deflate_top_direction(X, v_hat_top)
        used_deflation = True
    else:
        X_used = X
        used_deflation = False

    u_hat, _, _ = sparse_pca_johnstone_lu(X_used)
    return u_hat, gate, used_deflation


def mean_ci95(values):
    """Return mean and 95% CI half-width."""
    vals = np.array(values, dtype=float)
    mean_val = float(np.mean(vals))
    if vals.size <= 1:
        return mean_val, 0.0
    std_val = float(np.std(vals, ddof=1))
    return mean_val, 1.96 * std_val / np.sqrt(vals.size)


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
    num_top_trim=2,
    edge_buffer=0.25,
    z_threshold=2.0,
):
    """Sweep lambda_1 (fixed lambda_2) and compare direct/deflated/gated pipelines."""
    lam1_vals = np.linspace(lam1_min, lam1_max, num_lam1)

    direct_u_mean, direct_u_ci95 = [], []
    deflated_u_mean, deflated_u_ci95 = [], []
    gated_u_mean, gated_u_ci95 = [], []
    gated_deflate_rate = []
    gated_far_rate = []

    print("--- Gated Deflation + Sparse PCA experiment ---")
    print(
        f"p={p}, n={n}, sigma={sigma}, lambda_2={lam2}, sparsity={sparsity_frac}, "
        f"edge_buffer={edge_buffer}, z_threshold={z_threshold}"
    )
    print(f"Sweeping lambda_1 in [{lam1_min}, {lam1_max}] with {num_lam1} points")

    total_tests = num_lam1 * num_trials
    completed_tests = 0

    for lam1 in lam1_vals:
        trial_direct_u = []
        trial_defl_u = []
        trial_gated_u = []
        trial_gate_is_far = []
        trial_used_deflation = []

        for _ in range(num_trials):
            X, _, u_true = generate_mixed_spiked_data(
                p=p,
                n=n,
                sigma=sigma,
                lam1=lam1,
                lam2=lam2,
                sparsity_fraction=sparsity_frac,
            )

            u_hat_direct = run_direct_sparse_pca(X)
            u_hat_deflated = run_deflated_sparse_pca(X)
            u_hat_gated, gate, used_deflation = run_gated_sparse_pca(
                X=X,
                p=p,
                n=n,
                num_top_trim=num_top_trim,
                edge_buffer=edge_buffer,
                z_threshold=z_threshold,
            )

            trial_direct_u.append(squared_cosine(u_true, u_hat_direct))
            trial_defl_u.append(squared_cosine(u_true, u_hat_deflated))
            trial_gated_u.append(squared_cosine(u_true, u_hat_gated))
            trial_gate_is_far.append(1.0 if gate["label"] == "far" else 0.0)
            trial_used_deflation.append(1.0 if used_deflation else 0.0)

            completed_tests += 1
            pct = 100.0 * completed_tests / total_tests
            print(f"Progress: {completed_tests}/{total_tests} ({pct:5.1f}%)")

        direct_mean, direct_ci = mean_ci95(trial_direct_u)
        defl_mean, defl_ci = mean_ci95(trial_defl_u)
        gated_mean, gated_ci = mean_ci95(trial_gated_u)

        direct_u_mean.append(direct_mean)
        direct_u_ci95.append(direct_ci)
        deflated_u_mean.append(defl_mean)
        deflated_u_ci95.append(defl_ci)
        gated_u_mean.append(gated_mean)
        gated_u_ci95.append(gated_ci)
        gated_far_rate.append(float(np.mean(trial_gate_is_far)))
        gated_deflate_rate.append(float(np.mean(trial_used_deflation)))

    return {
        "lam1_vals": lam1_vals,
        "direct_u_mean": np.array(direct_u_mean),
        "direct_u_ci95": np.array(direct_u_ci95),
        "deflated_u_mean": np.array(deflated_u_mean),
        "deflated_u_ci95": np.array(deflated_u_ci95),
        "gated_u_mean": np.array(gated_u_mean),
        "gated_u_ci95": np.array(gated_u_ci95),
        "gated_far_rate": np.array(gated_far_rate),
        "gated_deflate_rate": np.array(gated_deflate_rate),
    }


def run_lambda2_sweep_experiment(
    p=10000,
    n=200,
    sigma=1.0,
    sparsity_frac=0.001,
    lam1_min=10.0,
    lam1_max=800.0,
    num_lam1=12,
    lam2_values=None,
    num_trials=10,
    num_top_trim=2,
    edge_buffer=0.25,
    z_threshold=2.0,
):
    """Sweep lambda_1 for multiple lambda_2 values and compare all three methods."""
    if lam2_values is None:
        lam2_values = [2.0, 5.0, 10.0, 20.0]

    lam1_vals = np.linspace(lam1_min, lam1_max, num_lam1)
    lam2_vals = np.array(lam2_values, dtype=float)

    direct_u_mean = np.zeros((len(lam2_vals), len(lam1_vals)))
    deflated_u_mean = np.zeros((len(lam2_vals), len(lam1_vals)))
    gated_u_mean = np.zeros((len(lam2_vals), len(lam1_vals)))
    gated_u_ci95 = np.zeros((len(lam2_vals), len(lam1_vals)))
    gated_deflate_rate = np.zeros((len(lam2_vals), len(lam1_vals)))

    print("--- Gated lambda_2 sweep experiment ---")
    print(
        f"p={p}, n={n}, sigma={sigma}, sparsity={sparsity_frac}, "
        f"edge_buffer={edge_buffer}, z_threshold={z_threshold}"
    )
    print(f"Sweeping lambda_1 in [{lam1_min}, {lam1_max}] with {num_lam1} points")
    print(f"Trying lambda_2 values: {lam2_vals.tolist()}")

    total_tests = len(lam2_vals) * len(lam1_vals) * num_trials
    completed_tests = 0

    for i, lam2 in enumerate(lam2_vals):
        for j, lam1 in enumerate(lam1_vals):
            trial_direct_u = []
            trial_defl_u = []
            trial_gated_u = []
            trial_used_deflation = []

            for _ in range(num_trials):
                X, _, u_true = generate_mixed_spiked_data(
                    p=p,
                    n=n,
                    sigma=sigma,
                    lam1=lam1,
                    lam2=lam2,
                    sparsity_fraction=sparsity_frac,
                )

                u_hat_direct = run_direct_sparse_pca(X)
                u_hat_deflated = run_deflated_sparse_pca(X)
                u_hat_gated, _, used_deflation = run_gated_sparse_pca(
                    X=X,
                    p=p,
                    n=n,
                    num_top_trim=num_top_trim,
                    edge_buffer=edge_buffer,
                    z_threshold=z_threshold,
                )

                trial_direct_u.append(squared_cosine(u_true, u_hat_direct))
                trial_defl_u.append(squared_cosine(u_true, u_hat_deflated))
                trial_gated_u.append(squared_cosine(u_true, u_hat_gated))
                trial_used_deflation.append(1.0 if used_deflation else 0.0)

                completed_tests += 1
                pct = 100.0 * completed_tests / total_tests
                print(f"Progress: {completed_tests}/{total_tests} ({pct:5.1f}%)")

            direct_u_mean[i, j] = float(np.mean(trial_direct_u))
            deflated_u_mean[i, j] = float(np.mean(trial_defl_u))
            gated_mean, gated_ci = mean_ci95(trial_gated_u)
            gated_u_mean[i, j] = gated_mean
            gated_u_ci95[i, j] = gated_ci
            gated_deflate_rate[i, j] = float(np.mean(trial_used_deflation))

    return {
        "lam1_vals": lam1_vals,
        "lam2_vals": lam2_vals,
        "direct_u_mean": direct_u_mean,
        "deflated_u_mean": deflated_u_mean,
        "gated_u_mean": gated_u_mean,
        "gated_u_ci95": gated_u_ci95,
        "gated_deflate_rate": gated_deflate_rate,
    }


def plot_results(results, lam2):
    """Plot fixed-lambda_2 sweep results."""
    lam1_vals = results["lam1_vals"]
    direct_u_mean = results["direct_u_mean"]
    direct_u_ci95 = results["direct_u_ci95"]
    deflated_u_mean = results["deflated_u_mean"]
    deflated_u_ci95 = results["deflated_u_ci95"]
    gated_u_mean = results["gated_u_mean"]
    gated_u_ci95 = results["gated_u_ci95"]
    gated_deflate_rate = results["gated_deflate_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for y, ci, marker, label in [
        (direct_u_mean, direct_u_ci95, "o-", "Direct Sparse PCA"),
        (deflated_u_mean, deflated_u_ci95, "s-", "Always Deflate + Sparse PCA"),
        (gated_u_mean, gated_u_ci95, "^-", "Gated Deflate + Sparse PCA"),
    ]:
        line, = axes[0].plot(lam1_vals, y, marker, label=label)
        lower = np.clip(y - ci, 0.0, 1.0)
        upper = np.clip(y + ci, 0.0, 1.0)
        axes[0].fill_between(lam1_vals, lower, upper, color=line.get_color(), alpha=0.15)

    axes[0].set_xlabel(r"$\lambda_1$ (Dense spike strength)")
    axes[0].set_ylabel(r"Squared cosine with true sparse $u$")
    axes[0].set_title(rf"Sparse recovery vs. $\lambda_1$ (fixed $\lambda_2={lam2}$)")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend()

    axes[1].plot(lam1_vals, gated_deflate_rate, "d-", color="tab:purple", label="Gated deflation rate")
    axes[1].set_xlabel(r"$\lambda_1$ (Dense spike strength)")
    axes[1].set_ylabel("Fraction of trials with deflation")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("How often the gate chooses deflation")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_lambda2_sweep_results(results):
    """Plot lambda_2 sweep results for gated method and gate behavior."""
    lam1_vals = results["lam1_vals"]
    lam2_vals = results["lam2_vals"]
    gated_u_mean = results["gated_u_mean"]
    gated_u_ci95 = results["gated_u_ci95"]
    gated_deflate_rate = results["gated_deflate_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    for i, lam2 in enumerate(lam2_vals):
        line, = axes[0].plot(
            lam1_vals,
            gated_u_mean[i],
            "o-",
            label=rf"Gated ($\lambda_2={lam2:.2f}$)",
        )
        lower = np.clip(gated_u_mean[i] - gated_u_ci95[i], 0.0, 1.0)
        upper = np.clip(gated_u_mean[i] + gated_u_ci95[i], 0.0, 1.0)
        axes[0].fill_between(lam1_vals, lower, upper, color=line.get_color(), alpha=0.18)

    axes[0].set_xlabel(r"$\lambda_1$ (Dense spike strength)")
    axes[0].set_ylabel(r"Squared cosine with true sparse $u$")
    axes[0].set_title("Gated sparse recovery across lambda_2")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend()

    for i, lam2 in enumerate(lam2_vals):
        axes[1].plot(
            lam1_vals,
            gated_deflate_rate[i],
            "d-",
            label=rf"Deflation rate ($\lambda_2={lam2:.2f}$)",
        )
    axes[1].set_xlabel(r"$\lambda_1$ (Dense spike strength)")
    axes[1].set_ylabel("Fraction of trials with deflation")
    axes[1].set_title("Gate decisions across lambda_2")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gated sparse recovery experiment: use eigenvalue-gap distinguisher to "
            "choose between direct Sparse PCA and Deflation+Sparse PCA."
        )
    )
    parser.add_argument(
        "--experiment",
        choices=["original", "lam2-sweep"],
        default="original",
        help="Experiment mode: fixed lambda_2 sweep or lambda_2 sweep grid.",
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
    parser.add_argument("--num-top-trim", type=int, default=2)
    parser.add_argument("--edge-buffer", type=float, default=0.25)
    parser.add_argument("--z-threshold", type=float, default=2.0)
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
            num_top_trim=args.num_top_trim,
            edge_buffer=args.edge_buffer,
            z_threshold=args.z_threshold,
        )

        print("\nSummary (mean +/- 95% CI for squared cosine with true sparse u):")
        print("lambda_1 | direct | always_deflated | gated | gated_deflate_rate")
        for j, lam1 in enumerate(results["lam1_vals"]):
            print(
                f"{lam1:8.2f} | "
                f"{results['direct_u_mean'][j]:.4f} +/- {results['direct_u_ci95'][j]:.4f} | "
                f"{results['deflated_u_mean'][j]:.4f} +/- {results['deflated_u_ci95'][j]:.4f} | "
                f"{results['gated_u_mean'][j]:.4f} +/- {results['gated_u_ci95'][j]:.4f} | "
                f"{results['gated_deflate_rate'][j]:.3f}"
            )

        if not args.no_plot:
            plot_results(results, args.lam2)
        return

    results = run_lambda2_sweep_experiment(
        p=args.p,
        n=args.n,
        sigma=args.sigma,
        sparsity_frac=args.sparsity_frac,
        lam1_min=args.lam1_min,
        lam1_max=args.lam1_max,
        num_lam1=args.num_lam1,
        lam2_values=args.lam2_values,
        num_trials=args.num_trials,
        num_top_trim=args.num_top_trim,
        edge_buffer=args.edge_buffer,
        z_threshold=args.z_threshold,
    )

    print("\nSummary (gated_u mean +/- 95% CI and deflation rate):")
    header = "lambda_1 | " + " | ".join([f"lam2={lam2:.2f}" for lam2 in results["lam2_vals"]])
    print(header)
    for j, lam1 in enumerate(results["lam1_vals"]):
        row_vals = " | ".join(
            [
                (
                    f"{results['gated_u_mean'][i, j]:.4f} +/- {results['gated_u_ci95'][i, j]:.4f} "
                    f"(defl={results['gated_deflate_rate'][i, j]:.3f})"
                )
                for i in range(len(results["lam2_vals"]))
            ]
        )
        print(f"{lam1:8.2f} | {row_vals}")

    if not args.no_plot:
        plot_lambda2_sweep_results(results)


if __name__ == "__main__":
    main()
