"""
Layer 1, Day 3: The First-Passage (Black-Cox) Model.

Lets default happen the first time asset value touches the barrier at ANY time
before maturity — not just at T. Fixes Merton's short-horizon PD understatement.

Provides a closed-form first-passage probability (via the reflection principle),
validates it against Monte Carlo, and re-runs the three real firms.


Run with: python3 first_passage.py
"""

import numpy as np
from scipy.stats import norm

from merton_model import solve_asset_value_and_vol, distance_to_default, probability_of_default


# ----------------------------- CLOSED-FORM FIRST PASSAGE -----------------------------
def first_passage_pd(V0, sigma_V, B, mu, T):
    """Probability that asset value V hits barrier B (below V0) before time T.

    Closed form via the reflection principle for GBM:

        m = mu - 0.5 * sigma^2              (drift of log-assets)
        a = ln(B / V0)                      (log-distance to barrier, negative)

        P(hit) = N( (a - m*T) / (sigma*sqrt(T)) )
               + exp(2*m*a / sigma^2) * N( (a + m*T) / (sigma*sqrt(T)) )

    (The second term is the extra mass from paths that hit the barrier early.)
    """
    sqrtT = np.sqrt(T)
    m = mu - 0.5 * sigma_V**2
    a = np.log(B / V0)  # negative since B < V0

    # Implement the two-term first-passage probability.
    
    term1 = norm.cdf((a - m * T) / (sigma_V * sqrtT))
    term2 = np.exp(2 * m * a / sigma_V**2) * norm.cdf((a + m * T) / (sigma_V * sqrtT))
    return term1 + term2


# ----------------------------- MONTE CARLO VALIDATOR -----------------------------
def first_passage_pd_mc(V0, sigma_V, B, mu, T, n_paths=50_000, n_steps=252, seed=42):
    """Estimate first-passage PD by simulation: fraction of paths that ever touch B."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (mu - 0.5 * sigma_V**2) * dt
    vol = sigma_V * np.sqrt(dt)

    # Simulate log-asset paths and check if each ever hits the barrier
    log_V0 = np.log(V0)
    log_B = np.log(B)

    # Simulate n_paths GBM paths over n_steps and count barrier hits.
    # A path "defaults" if its running minimum log-value ever <= log_B.
    
    shocks = rng.standard_normal((n_paths, n_steps))
    increments = drift + vol * shocks
    log_paths = log_V0 + np.cumsum(increments, axis=1)
    min_log = log_paths.min(axis=1)
    hit = (min_log <= log_B)
    return hit.mean()


# ----------------------------- COMPARE ON A FIRM -----------------------------
def compare_models(name, E, sigma_E, D, r=0.045, T=1.0, mu=0.08):
    """Run Merton and first-passage side by side for one firm."""
    V, sigma_V = solve_asset_value_and_vol(E, sigma_E, D, r, T)

    # Merton PD (default only at maturity)
    dd = distance_to_default(V, sigma_V, D, mu, T)
    merton_pd = probability_of_default(dd)

    # First-passage PD (barrier = debt level)
    B = D
    fp_pd = first_passage_pd(V, sigma_V, B, mu, T)
    fp_pd_mc = first_passage_pd_mc(V, sigma_V, B, mu, T)

    print(f"\n{name}")
    print(f"    Asset value V:        {V:>14,.0f}")
    print(f"    Asset vol:            {sigma_V:>14.1%}")
    print(f"    Merton PD (at T):     {merton_pd:>14.3%}")
    print(f"    First-passage PD:     {fp_pd:>14.3%}   (closed form)")
    print(f"    First-passage PD:     {fp_pd_mc:>14.3%}   (Monte Carlo)")
    if merton_pd < 1e-6:
        ratio_str = "n/a (both ~0)"
    else:
        ratio_str = f"{fp_pd/merton_pd:.1f}x"
    print(f"    Ratio FP/Merton:      {ratio_str:>14}")
    return {"name": name, "merton_pd": merton_pd, "fp_pd": fp_pd, "fp_pd_mc": fp_pd_mc}


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("=" * 64)
    print("FIRST-PASSAGE vs MERTON")
    print("=" * 64)

    # Controlled firms first (known behavior)
    compare_models("Healthy Corp (low leverage)", E=100_000, sigma_E=0.25, D=40_000)
    compare_models("Stressed Corp (high leverage)", E=20_000, sigma_E=0.55, D=90_000)

    # Validate: closed form should match Monte Carlo
    print("\n" + "=" * 64)
    print("VALIDATION: closed-form vs Monte Carlo (should match closely)")
    print("=" * 64)
    V, sV = solve_asset_value_and_vol(50_000, 0.40, 60_000, 0.045, 1.0)
    analytic = first_passage_pd(V, sV, 60_000, 0.08, 1.0)
    mc = first_passage_pd_mc(V, sV, 60_000, 0.08, 1.0)
    print(f"  Analytic:    {analytic:.4%}")
    print(f"  Monte Carlo: {mc:.4%}")
    print(f"  Difference:  {abs(analytic-mc):.4%}")

    #  Assert/print whether they agree within a small tolerance (e.g. 0.5%).
    # This is the cross-validation that confirms the closed-form formula.
   
    if abs(analytic - mc) < 0.005:
        print("  PASS — closed form matches simulation")
    else:
        print("  MISMATCH — check the formula")

    print("\n" + "=" * 64)
    print("ECONOMIC CHECK: first-passage PD should exceed Merton PD,")
    print("with a bigger gap for riskier firms.")
    print("=" * 64)
    print("(See the two firms above — Stressed should show a larger FP/Merton ratio.)")
