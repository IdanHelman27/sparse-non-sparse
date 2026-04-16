import argparse
import numpy as np

from utils import generate_mixed_spiked_data


def get_nonzero_sample_eigenvalues_dual(X):
    """Return non-zero sample covariance eigenvalues via the dual covariance matrix."""
    n = X.shape[1]
    s_dual = (1.0 / n) * (X.T @ X)
    return np.linalg.eigvalsh(s_dual)


def mp_upper_edge(gamma):
    """Return the Marchenko-Pastur upper edge for sigma^2=1."""
    return (1.0 + np.sqrt(gamma)) ** 2


def mp_median_factor(gamma, grid_size=40000):
    """
    Estimate the median of the non-zero MP eigenvalue law for sigma^2=1.

    This median factor m_gamma is used in:
      sigma_hat^2 = median(bulk eigenvalues) / m_gamma.

    We target the high-dimensional regime used in this repo (typically p >= n, gamma = p/n >= 1),
    with non-zero sample eigenvalues of (1/n)XX^T.
    """
    if gamma < 1.0:
        raise ValueError("mp_median_factor currently expects gamma = p/n >= 1.")

    sqrt_gamma = np.sqrt(gamma)
    a = (sqrt_gamma - 1.0) ** 2
    b = (sqrt_gamma + 1.0) ** 2
    eps = 1e-10
    x = np.linspace(a + eps, b - eps, grid_size)
    density = np.sqrt(np.maximum(b - x, 0.0) * np.maximum(x - a, 0.0)) / (2.0 * np.pi * x)

    dx = x[1] - x[0]
    cdf = np.zeros_like(x)
    cdf[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * dx)
    if cdf[-1] <= 0:
        raise RuntimeError("Failed to compute MP CDF normalization.")
    cdf /= cdf[-1]

    return float(np.interp(0.5, cdf, x))


def estimate_sigma_sq_from_bulk_median(eigvals, gamma, num_top_trim=2):
    """
    Estimate sigma^2 from bulk median with MP calibration.

    We drop the top num_top_trim eigenvalues to reduce spike contamination.
    """
    if eigvals.size <= num_top_trim:
        raise ValueError("Not enough eigenvalues after trimming top outliers.")

    bulk = eigvals[:-num_top_trim]
    median_bulk = float(np.median(bulk))
    m_gamma = mp_median_factor(gamma)
    sigma_sq_hat = median_bulk / m_gamma
    if sigma_sq_hat <= 0:
        raise RuntimeError("Estimated sigma^2 is non-positive.")
    return float(sigma_sq_hat), float(median_bulk), float(m_gamma)


def invert_population_eigenvalue_from_outlier(sample_eig_normalized, gamma):
    """
    Invert d = lambda * (1 + gamma/(lambda-1)) for lambda > 1.

    sample_eig_normalized is d = sample_eig / sigma_hat^2.
    Returns (lambda_hat, detectable_flag).
    """
    edge = mp_upper_edge(gamma)
    if sample_eig_normalized <= edge:
        return np.nan, False

    a = sample_eig_normalized - (1.0 + gamma)
    discriminant = a * a - 4.0 * gamma
    if discriminant <= 0:
        return np.nan, False

    theta_hat = 0.5 * (a + np.sqrt(discriminant))
    lambda_hat = 1.0 + theta_hat
    return float(lambda_hat), True


def psi_prime(lambda_pop, gamma):
    """Derivative of psi(lambda)=lambda*(1+gamma/(lambda-1)) for lambda>1."""
    return 1.0 - gamma / ((lambda_pop - 1.0) ** 2)


def asymptotic_sample_outlier_variance(lambda_pop, gamma, n):
    """
    Approximate asymptotic variance of the normalized outlier sample eigenvalue.

    Formula used:
      Var(d) ≈ (2/n) * lambda^2 * (1 - gamma/(lambda-1)^2), for supercritical lambda.
    """
    margin = 1.0 - gamma / ((lambda_pop - 1.0) ** 2)
    if margin <= 0:
        return np.nan
    return float((2.0 / n) * (lambda_pop ** 2) * margin)


def asymptotic_population_estimate_variance(lambda_pop, gamma, n):
    """Delta-method variance for lambda_hat derived from d_hat."""
    var_d = asymptotic_sample_outlier_variance(lambda_pop, gamma, n)
    if not np.isfinite(var_d):
        return np.nan

    deriv = psi_prime(lambda_pop, gamma)
    if deriv <= 0:
        return np.nan
    return float(var_d / (deriv ** 2))


def classify_from_top2_normalized(d1, d2, gamma, n, edge_buffer=0.25, z_threshold=2.0):
    """
    Classify whether lambda_1 and lambda_2 are 'far' or 'close_or_undetectable'.

    Decision rule:
    1) If top outlier is near/below MP edge, return close_or_undetectable.
    2) If top is clear but second is not, treat second as an edge-buffer surrogate.
    3) Invert to lambda-hats and use a gap Z-score:
         Z = (lambda1_hat - lambda2_hat)/sqrt(var1 + var2).
       Z >= z_threshold => far, otherwise close_or_undetectable.
    """
    edge = mp_upper_edge(gamma)
    edge_cut = edge + edge_buffer

    if d1 <= edge_cut:
        return {
            "label": "close_or_undetectable",
            "reason": "top edge test",
            "d1": float(d1),
            "d2": float(d2),
            "edge": float(edge),
            "edge_cut": float(edge_cut),
        }

    lambda1_hat, ok1 = invert_population_eigenvalue_from_outlier(d1, gamma)
    if not ok1:
        return {
            "label": "close_or_undetectable",
            "reason": "top inversion failed",
            "d1": float(d1),
            "d2": float(d2),
            "edge": float(edge),
            "edge_cut": float(edge_cut),
        }

    d2_effective = float(d2) if d2 > edge_cut else float(edge_cut)
    second_mode = "observed" if d2 > edge_cut else "edge_surrogate"
    lambda2_hat, ok2 = invert_population_eigenvalue_from_outlier(d2_effective, gamma)
    if not ok2:
        return {
            "label": "close_or_undetectable",
            "reason": "second inversion failed",
            "d1": float(d1),
            "d2": float(d2),
            "d2_effective": float(d2_effective),
            "second_mode": second_mode,
            "edge": float(edge),
            "edge_cut": float(edge_cut),
            "lambda1_hat": float(lambda1_hat),
            "lambda2_hat": np.nan,
            "gap_z": np.nan,
        }

    var1 = asymptotic_population_estimate_variance(lambda1_hat, gamma, n)
    var2 = asymptotic_population_estimate_variance(lambda2_hat, gamma, n)

    if not np.isfinite(var1) or not np.isfinite(var2) or (var1 + var2) <= 0:
        return {
            "label": "close_or_undetectable",
            "reason": "variance model unstable",
            "d1": float(d1),
            "d2": float(d2),
            "d2_effective": float(d2_effective),
            "second_mode": second_mode,
            "edge": float(edge),
            "edge_cut": float(edge_cut),
            "lambda1_hat": float(lambda1_hat),
            "lambda2_hat": float(lambda2_hat),
            "gap_z": np.nan,
            "var1": float(var1) if np.isfinite(var1) else np.nan,
            "var2": float(var2) if np.isfinite(var2) else np.nan,
        }

    gap_z = (lambda1_hat - lambda2_hat) / np.sqrt(var1 + var2)
    label = "far" if gap_z >= z_threshold else "close_or_undetectable"
    reason = "outlier gap z-test"
    if second_mode == "edge_surrogate":
        reason = "edge-surrogate gap z-test"
    return {
        "label": label,
        "reason": reason,
        "d1": float(d1),
        "d2": float(d2),
        "d2_effective": float(d2_effective),
        "second_mode": second_mode,
        "edge": float(edge),
        "edge_cut": float(edge_cut),
        "lambda1_hat": float(lambda1_hat),
        "lambda2_hat": float(lambda2_hat),
        "var1": float(var1),
        "var2": float(var2),
        "gap_z": float(gap_z),
    }


def classify_lambda_gap_from_eigenvalues(
    eigvals,
    p,
    n,
    num_top_trim=2,
    edge_buffer=0.25,
    z_threshold=2.0,
):
    """Full pipeline classification starting from dual sample eigenvalues."""
    gamma = p / n
    sigma_sq_hat, median_bulk, m_gamma = estimate_sigma_sq_from_bulk_median(
        eigvals=eigvals,
        gamma=gamma,
        num_top_trim=num_top_trim,
    )
    eigvals_norm = eigvals / sigma_sq_hat
    d1 = float(eigvals_norm[-1])
    d2 = float(eigvals_norm[-2]) if eigvals_norm.size >= 2 else np.nan

    result = classify_from_top2_normalized(
        d1=d1,
        d2=d2,
        gamma=gamma,
        n=n,
        edge_buffer=edge_buffer,
        z_threshold=z_threshold,
    )
    result["sigma_sq_hat"] = float(sigma_sq_hat)
    result["median_bulk"] = float(median_bulk)
    result["m_gamma"] = float(m_gamma)
    result["sample_top2"] = (float(eigvals[-1]), float(eigvals[-2]))
    return result


def classify_lambda_gap_from_data(
    X,
    p,
    n,
    num_top_trim=2,
    edge_buffer=0.25,
    z_threshold=2.0,
):
    """Convenience wrapper: data matrix -> dual eigenvalues -> classification."""
    eigvals = get_nonzero_sample_eigenvalues_dual(X)
    return classify_lambda_gap_from_eigenvalues(
        eigvals=eigvals,
        p=p,
        n=n,
        num_top_trim=num_top_trim,
        edge_buffer=edge_buffer,
        z_threshold=z_threshold,
    )


def run_validation_sweep(
    p=1200,
    n=300,
    sigma=1.0,
    sparsity_frac=0.01,
    num_trials=25,
    num_top_trim=2,
    edge_buffer=0.25,
    z_threshold=2.0,
):
    """
    Validate the distinguisher across both 'close' and 'far' settings.

    Returns summary rows for quick inspection of empirical behavior.
    """
    scenarios = [
        {"lam1": 40.0, "lam2": 7.0, "expected": "far"},
        {"lam1": 25.0, "lam2": 20.0, "expected": "close_or_undetectable"},
        {"lam1": 35.0, "lam2": 15.0, "expected": "far"},
        {"lam1": 22.0, "lam2": 19.0, "expected": "close_or_undetectable"},
    ]

    rows = []
    total = len(scenarios) * num_trials
    done = 0

    print("--- Eigenvalue-gap distinguisher validation sweep ---")
    print(
        f"p={p}, n={n}, gamma={p/n:.3f}, sigma={sigma}, sparsity={sparsity_frac}, trials={num_trials}"
    )
    print(
        f"thresholds: edge_buffer={edge_buffer}, z_threshold={z_threshold}, num_top_trim={num_top_trim}"
    )

    for scenario in scenarios:
        lam1 = scenario["lam1"]
        lam2 = scenario["lam2"]
        expected = scenario["expected"]

        label_counts = {"far": 0, "close_or_undetectable": 0}
        sigma_sq_hats = []
        gap_z_vals = []

        for _ in range(num_trials):
            X, _, _ = generate_mixed_spiked_data(
                p=p,
                n=n,
                sigma=sigma,
                lam1=lam1,
                lam2=lam2,
                sparsity_fraction=sparsity_frac,
            )
            result = classify_lambda_gap_from_data(
                X=X,
                p=p,
                n=n,
                num_top_trim=num_top_trim,
                edge_buffer=edge_buffer,
                z_threshold=z_threshold,
            )
            label = result["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            sigma_sq_hats.append(result["sigma_sq_hat"])
            if np.isfinite(result.get("gap_z", np.nan)):
                gap_z_vals.append(result["gap_z"])

            done += 1
            pct = 100.0 * done / total
            print(f"Progress: {done}/{total} ({pct:5.1f}%)")

        observed = (
            "far"
            if label_counts["far"] >= label_counts["close_or_undetectable"]
            else "close_or_undetectable"
        )
        hit_rate = label_counts.get(expected, 0) / num_trials

        rows.append(
            {
                "lam1": lam1,
                "lam2": lam2,
                "expected": expected,
                "observed_majority": observed,
                "far_rate": label_counts["far"] / num_trials,
                "close_rate": label_counts["close_or_undetectable"] / num_trials,
                "hit_rate": hit_rate,
                "sigma_sq_hat_mean": float(np.mean(sigma_sq_hats)),
                "sigma_sq_hat_std": float(np.std(sigma_sq_hats, ddof=1)) if num_trials > 1 else 0.0,
                "gap_z_mean": float(np.mean(gap_z_vals)) if len(gap_z_vals) > 0 else np.nan,
            }
        )

    return rows


def print_validation_table(rows):
    """Pretty-print sweep results."""
    def fmt_num(value, width=8, precision=3):
        if np.isfinite(value):
            return f"{value:{width}.{precision}f}"
        return f"{'nan':>{width}}"

    print("\nSummary:")
    header = (
        f"{'lam1':>6} | {'lam2':>6} | {'expected':>21} | {'observed_majority':>21} | "
        f"{'far_rate':>8} | {'close_rate':>10} | {'hit_rate':>8} | {'sigma_sq_hat(mean+-std)':>24} | {'gap_z_mean':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sigma_summary = f"{r['sigma_sq_hat_mean']:.3f}+-{r['sigma_sq_hat_std']:.3f}"
        print(
            f"{r['lam1']:6.1f} | {r['lam2']:6.1f} | {r['expected']:>21s} | {r['observed_majority']:>21s} | "
            f"{fmt_num(r['far_rate'])} | {fmt_num(r['close_rate'], width=10)} | {fmt_num(r['hit_rate'])} | "
            f"{sigma_summary:>24} | {fmt_num(r['gap_z_mean'], width=10)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Distinguish whether top two normalized spikes are far apart or close/undetectable "
            "using MP edge checks and an outlier-gap z-test."
        )
    )
    parser.add_argument("--mode", choices=["single", "validate"], default="validate")

    # Shared args.
    parser.add_argument("--p", type=int, default=1200)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--sparsity-frac", type=float, default=0.01)
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("--num-top-trim", type=int, default=2)
    parser.add_argument("--edge-buffer", type=float, default=0.25)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)

    # Single-instance args.
    parser.add_argument("--lam1", type=float, default=30.0)
    parser.add_argument("--lam2", type=float, default=10.0)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.mode == "single":
        X, _, _ = generate_mixed_spiked_data(
            p=args.p,
            n=args.n,
            sigma=args.sigma,
            lam1=args.lam1,
            lam2=args.lam2,
            sparsity_fraction=args.sparsity_frac,
        )
        result = classify_lambda_gap_from_data(
            X=X,
            p=args.p,
            n=args.n,
            num_top_trim=args.num_top_trim,
            edge_buffer=args.edge_buffer,
            z_threshold=args.z_threshold,
        )
        print("--- Single-run classification ---")
        print(
            f"p={args.p}, n={args.n}, gamma={args.p/args.n:.3f}, sigma={args.sigma}, lam1={args.lam1}, lam2={args.lam2}"
        )
        print(f"label={result['label']} (reason: {result['reason']})")
        print(
            f"sigma_sq_hat={result['sigma_sq_hat']:.4f}, "
            f"sample_top2=({result['sample_top2'][0]:.4f}, {result['sample_top2'][1]:.4f}), "
            f"normalized_top2=({result['d1']:.4f}, {result['d2']:.4f}), "
            f"d2_effective={result.get('d2_effective', np.nan):.4f}, "
            f"second_mode={result.get('second_mode', 'n/a')}, "
            f"edge={result['edge']:.4f}, edge_cut={result['edge_cut']:.4f}, "
            f"lambda1_hat={result.get('lambda1_hat', np.nan):.4f}, "
            f"lambda2_hat={result.get('lambda2_hat', np.nan):.4f}, "
            f"gap_z={result.get('gap_z', np.nan):.4f}"
        )
        return

    rows = run_validation_sweep(
        p=args.p,
        n=args.n,
        sigma=args.sigma,
        sparsity_frac=args.sparsity_frac,
        num_trials=args.num_trials,
        num_top_trim=args.num_top_trim,
        edge_buffer=args.edge_buffer,
        z_threshold=args.z_threshold,
    )
    print_validation_table(rows)


if __name__ == "__main__":
    main()
