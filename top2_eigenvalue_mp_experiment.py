import argparse
import numpy as np

from utils import generate_mixed_spiked_data


def get_top_sample_eigenvalues_dual(X, top_k=2):
    """Return top non-zero sample eigenvalues of (1/n)XX^T via dual covariance."""
    n = X.shape[1]
    s_dual = (1.0 / n) * (X.T @ X)
    eigvals = np.linalg.eigvalsh(s_dual)
    return eigvals[-top_k:][::-1]


def mp_upper_edge(sigma_sq, gamma):
    """Marchenko-Pastur upper bulk edge."""
    return sigma_sq * (1.0 + np.sqrt(gamma)) ** 2


def invert_spike_from_sample_eigenvalue(sample_eig, sigma_sq, gamma):
    """
    Invert the spiked-covariance mapping:
      sample_eig = (sigma^2 + theta)(1 + gamma*sigma^2/theta),
    and return estimated population eigenvalue sigma^2 + theta.
    """
    edge = mp_upper_edge(sigma_sq, gamma)
    if sample_eig <= edge:
        return np.nan, False

    a = sample_eig - sigma_sq * (1.0 + gamma)
    disc = a * a - 4.0 * gamma * (sigma_sq ** 2)
    if disc < 0.0:
        return np.nan, False

    theta_hat = 0.5 * (a + np.sqrt(disc))
    pop_eig_hat = sigma_sq + theta_hat
    return float(pop_eig_hat), True


def summarize_component(estimates, true_value):
    detectable = ~np.isnan(estimates)
    detect_rate = float(np.mean(detectable))
    if np.any(detectable):
        vals = estimates[detectable]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(vals.size) if vals.size > 1 else 0.0
        bias = mean - true_value
        rmse = float(np.sqrt(np.mean((vals - true_value) ** 2)))
    else:
        mean = np.nan
        std = np.nan
        ci95 = np.nan
        bias = np.nan
        rmse = np.nan

    return {
        "detect_rate": detect_rate,
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "bias": bias,
        "rmse": rmse,
    }


def run_experiment(
    p=10000,
    n=200,
    sigma=1.0,
    lam1=20.0,
    lam2=5.0,
    sparsity_frac=0.001,
    num_trials=30,
):
    sigma_sq = sigma ** 2
    gamma = p / n
    edge = mp_upper_edge(sigma_sq, gamma)

    pop1_true = sigma_sq + lam1
    pop2_true = sigma_sq + lam2

    est_pop1 = np.full(num_trials, np.nan)
    est_pop2 = np.full(num_trials, np.nan)
    sample1_vals = np.zeros(num_trials)
    sample2_vals = np.zeros(num_trials)

    print("--- Top-2 population eigenvalue estimation via MP inversion ---")
    print(
        f"p={p}, n={n}, gamma={gamma:.4f}, sigma={sigma}, lambda_1={lam1}, lambda_2={lam2}, "
        f"sparsity={sparsity_frac}, trials={num_trials}"
    )
    print(f"MP upper edge: lambda_plus={edge:.4f}")

    for t in range(num_trials):
        X, _, _ = generate_mixed_spiked_data(p, n, sigma, lam1, lam2, sparsity_frac)
        top2_sample = get_top_sample_eigenvalues_dual(X, top_k=2)
        sample1_vals[t], sample2_vals[t] = top2_sample[0], top2_sample[1]

        est_pop1[t], _ = invert_spike_from_sample_eigenvalue(top2_sample[0], sigma_sq, gamma)
        est_pop2[t], _ = invert_spike_from_sample_eigenvalue(top2_sample[1], sigma_sq, gamma)

        pct = 100.0 * (t + 1) / num_trials
        print(f"Progress: {t + 1}/{num_trials} ({pct:5.1f}%)")

    comp1 = summarize_component(est_pop1, pop1_true)
    comp2 = summarize_component(est_pop2, pop2_true)

    return {
        "pop1_true": pop1_true,
        "pop2_true": pop2_true,
        "edge": edge,
        "sample1_mean": float(np.mean(sample1_vals)),
        "sample2_mean": float(np.mean(sample2_vals)),
        "comp1": comp1,
        "comp2": comp2,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate top-2 population eigenvalues using MP inverse-spike formulas with known sigma."
    )
    parser.add_argument("--p", type=int, default=10000)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--lam1", type=float, default=20.0)
    parser.add_argument("--lam2", type=float, default=5.0)
    parser.add_argument("--sparsity-frac", type=float, default=0.001)
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    results = run_experiment(
        p=args.p,
        n=args.n,
        sigma=args.sigma,
        lam1=args.lam1,
        lam2=args.lam2,
        sparsity_frac=args.sparsity_frac,
        num_trials=args.num_trials,
    )

    print("\nSummary:")
    print(f"True population eigenvalues: lambda1_true={results['pop1_true']:.4f}, lambda2_true={results['pop2_true']:.4f}")
    print(f"Mean sample top-2 eigenvalues: sample1_mean={results['sample1_mean']:.4f}, sample2_mean={results['sample2_mean']:.4f}")
    print(f"MP edge: lambda_plus={results['edge']:.4f}")
    print("component | detect_rate | estimate(mean +/- 95% CI) | bias | RMSE")
    print(
        f"top-1     | {results['comp1']['detect_rate']:.3f}      | "
        f"{results['comp1']['mean']:.4f} +/- {results['comp1']['ci95']:.4f} | "
        f"{results['comp1']['bias']:+.4f} | {results['comp1']['rmse']:.4f}"
    )
    print(
        f"top-2     | {results['comp2']['detect_rate']:.3f}      | "
        f"{results['comp2']['mean']:.4f} +/- {results['comp2']['ci95']:.4f} | "
        f"{results['comp2']['bias']:+.4f} | {results['comp2']['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()

