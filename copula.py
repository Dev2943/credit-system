"""
Layer 2, Day 1: Correlated Defaults & the Gaussian Copula.

Implements the one-factor Gaussian copula to simulate correlated defaults
across a credit portfolio, builds the portfolio loss distribution, and shows
how the tail fattens as default correlation rises.


Run with: python3 copula.py
"""

import numpy as np
from scipy.stats import norm


# ----------------------------- PORTFOLIO -----------------------------
def make_sample_portfolio(n_names=100, seed=7):
    """Build a sample portfolio: each name has a PD, exposure, and recovery.

    In the full system these PDs come from Layer 1 (Merton / reduced-form).
    Here we generate a realistic spread of credit qualities.
    """
    rng = np.random.default_rng(seed)
    # PDs spanning investment-grade (~0.3%) to high-yield (~8%)
    pds = rng.uniform(0.003, 0.08, n_names)
    # Exposures: most names similar size, a few large (concentration)
    exposures = rng.lognormal(mean=0.0, sigma=0.6, size=n_names)
    exposures = exposures / exposures.sum() * 100.0   # normalize to $100M book
    # Recovery rate ~40% typical, with some variation
    recoveries = np.clip(rng.normal(0.40, 0.10, n_names), 0.1, 0.7)
    return {"pd": pds, "exposure": exposures, "recovery": recoveries}


# ----------------------------- COPULA SIMULATION -----------------------------
def simulate_losses(portfolio, rho, n_scenarios=50_000, seed=42):
    """One-factor Gaussian copula simulation of portfolio losses.

    For each scenario:
      X_i = sqrt(rho) * M + sqrt(1 - rho) * Z_i
      name i defaults if X_i < threshold_i = Phi^{-1}(PD_i)
      loss = sum over defaulted names of exposure_i * (1 - recovery_i)
    """
    pd = portfolio["pd"]
    exposure = portfolio["exposure"]
    recovery = portfolio["recovery"]
    n_names = len(pd)

    rng = np.random.default_rng(seed)

    # Compute the default thresholds c_i = Phi^{-1}(PD_i).
    # A name defaults when its latent variable falls below this threshold.
   
    thresholds = norm.ppf(pd)   # shape (n_names,)

    # Draw the common market factor M: one per scenario
    M = rng.standard_normal(n_scenarios)
    # Draw idiosyncratic shocks Z: one per name per scenario
    Z = rng.standard_normal((n_scenarios, n_names))

    # Build the latent variables X for all scenarios and names.
    # X has shape (n_scenarios, n_names).
    # M needs to broadcast across names -> use M[:, None].
   
    X = np.sqrt(rho) * M[:, None] + np.sqrt(1 - rho) * Z
    

    # Determine defaults: name i defaults in a scenario if X < threshold.
    # Then compute the loss per scenario.
    
    defaults = X < thresholds
    loss_per_name = exposure * (1 - recovery)
    losses = (defaults * loss_per_name).sum(axis=1)
    return losses


# ----------------------------- ANALYSIS -----------------------------
def describe_distribution(losses, label):
    """Print summary stats of a loss distribution."""
    mean = losses.mean()
    p95 = np.percentile(losses, 95)
    p99 = np.percentile(losses, 99)
    p999 = np.percentile(losses, 99.9)
    max_l = losses.max()
    print(f"  {label}")
    print(f"    Expected loss:     ${mean:>7.3f}M")
    print(f"    95th percentile:   ${p95:>7.3f}M")
    print(f"    99th percentile:   ${p99:>7.3f}M")
    print(f"    99.9th percentile: ${p999:>7.3f}M")
    print(f"    Worst scenario:    ${max_l:>7.3f}M")
    return {"mean": mean, "p99": p99, "p999": p999}


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    portfolio = make_sample_portfolio(n_names=100)

    print("=" * 60)
    print("PORTFOLIO")
    print("=" * 60)
    print(f"  Names:            {len(portfolio['pd'])}")
    print(f"  Total exposure:   ${portfolio['exposure'].sum():.1f}M")
    print(f"  Average PD:       {portfolio['pd'].mean():.2%}")
    print(f"  Expected defaults (independent): {portfolio['pd'].sum():.1f}")

    print("\n" + "=" * 60)
    print("LOSS DISTRIBUTION vs CORRELATION (the key result)")
    print("=" * 60)
    print("  As rho rises, expected loss stays ~same but the TAIL fattens.\n")

    results = {}
    for rho in [0.0, 0.10, 0.30, 0.50]:
        losses = simulate_losses(portfolio, rho=rho)
        stats = describe_distribution(losses, f"rho = {rho:.2f}")
        results[rho] = stats
        print()

    # Show the tail-fattening explicitly.
    # Compute the ratio of the 99.9th percentile to the mean for each rho,
    # and print it. This ratio should RISE with rho (fatter tail).
    
    print("  Tail ratio (99.9th percentile / expected loss):")
    for rho, s in results.items():
        ratio = s["p999"] / s["mean"]
        print(f"    rho={rho:.2f}: {ratio:.1f}x")

    print("\n  The independent case (rho=0) has a thin tail — large joint losses")
    print("  almost never happen. As correlation rises, the 99.9th percentile")
    print("  pulls far above the mean: that gap is the concentration risk")
    print("  that economic capital must cover.")
