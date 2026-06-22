"""
Layer 1, Day 1: The Merton Structural Credit Model.

Estimates a firm's probability of default and credit spread by treating
equity as a call option on the firm's assets. Solves the two-equation
nonlinear system for the latent asset value and asset volatility, then
derives distance-to-default, PD, and the implied credit spread.


Run with: python3 merton_model.py
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


# ----------------------------- BLACK-SCHOLES ON ASSETS -----------------------------
def merton_equity_value(V, sigma_V, D, r, T):
    """Equity as a Black-Scholes call on firm assets V, struck at debt D.

    E = V * N(d1) - D * exp(-rT) * N(d2)
    """
    # Implement the Black-Scholes call, with:
    #   underlying = V (asset value), strike = D (debt), vol = sigma_V
    
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    E = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
    return E
   


def merton_d1(V, sigma_V, D, r, T):
    """The d1 term — needed for the equity-vol relationship."""
    return (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))


# ----------------------------- SOLVE FOR LATENT V, sigma_V -----------------------------
def solve_asset_value_and_vol(E, sigma_E, D, r, T):
    """Solve the 2-equation nonlinear system for asset value V and asset vol sigma_V.

    Equation 1 (equity = call on assets):
        E = V * N(d1) - D * exp(-rT) * N(d2)
    Equation 2 (equity vol from asset vol, via Ito):
        sigma_E * E = N(d1) * sigma_V * V

    Returns (V, sigma_V).
    """
    def equations(unknowns):
        V, sigma_V = unknowns
        # Guard against invalid values during the solve
        if V <= 0 or sigma_V <= 0:
            return [1e6, 1e6]

        d1 = merton_d1(V, sigma_V, D, r, T)

        # Build the two residuals that should both equal zero.
        # eq1: model equity minus observed equity E
        # eq2: model equity-vol relationship minus observed sigma_E * E
        
        model_E = merton_equity_value(V, sigma_V, D, r, T)
        eq1 = model_E - E
        eq2 = norm.cdf(d1) * sigma_V * V - sigma_E * E
        return [eq1, eq2]
        

    # Better initial guess: assets ~ equity + PV of debt; asset vol from leverage scaling
    V0 = E + D * np.exp(-r * T)
    sigma_V0 = sigma_E * E / V0
    V, sigma_V = fsolve(equations, [V0, sigma_V0], full_output=False)
    return V, sigma_V


# ----------------------------- CREDIT METRICS -----------------------------
def distance_to_default(V, sigma_V, D, mu, T):
    """Distance to default: standard deviations of asset value from the default point.

    DD = [ln(V/D) + (mu - sigma_V^2/2) T] / (sigma_V sqrt(T))
    Uses real-world drift mu (not r).
    """
    # Implement distance to default.
 
    dd = (np.log(V / D) + (mu - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    return dd
   


def probability_of_default(dd):
    """PD = N(-DD)."""
    return norm.cdf(-dd)


def credit_spread(V, E, D, r, T):
    """Implied credit spread on the firm's debt.

    Risky debt value = V - E.  Risk-free debt value = D * exp(-rT).
    spread = -(1/T) * ln(risky_debt / risk_free_debt)
    """
    # Implement the credit spread.
    
    risky_debt = V - E
    risk_free_debt = D * np.exp(-r * T)
    spread = -(1.0 / T) * np.log(risky_debt / risk_free_debt)
    return spread
    


# ----------------------------- ANALYSIS -----------------------------
def analyze_firm(name, E, sigma_E, D, r=0.04, T=1.0, mu=0.08):
    """Run the full Merton analysis for a firm and print the credit profile."""
    print(f"\n{'='*60}")
    print(f"MERTON CREDIT ANALYSIS: {name}")
    print(f"{'='*60}")
    print(f"  Inputs:")
    print(f"    Equity (market cap) E:  {E:>14,.0f}")
    print(f"    Equity volatility:      {sigma_E:>14.1%}")
    print(f"    Debt (face) D:          {D:>14,.0f}")
    print(f"    Leverage (D/(E+D)):     {D/(E+D):>14.1%}")
    print(f"    r={r:.1%}, T={T:.1f}y, mu={mu:.1%}")

    V, sigma_V = solve_asset_value_and_vol(E, sigma_E, D, r, T)
    dd = distance_to_default(V, sigma_V, D, mu, T)
    pd_val = probability_of_default(dd)
    spread = credit_spread(V, E, D, r, T)

    print(f"\n  Solved latent variables:")
    print(f"    Asset value V:          {V:>14,.0f}")
    print(f"    Asset volatility:       {sigma_V:>14.1%}")
    print(f"\n  Credit metrics:")
    print(f"    Distance to default:    {dd:>14.2f}  sigma")
    print(f"    Probability of default: {pd_val:>14.2%}")
    print(f"    Implied credit spread:  {spread*1e4:>14.0f}  bps")

    return {"V": V, "sigma_V": sigma_V, "dd": dd, "pd": pd_val, "spread": spread}


if __name__ == "__main__":
    # A healthy, low-leverage firm (investment-grade-like)
    analyze_firm("Healthy Corp (low leverage)",
                 E=100_000, sigma_E=0.25, D=40_000)

    # A stressed, high-leverage firm
    analyze_firm("Stressed Corp (high leverage)",
                 E=20_000, sigma_E=0.55, D=90_000)

    # Comparative statics: PD should RISE with leverage, vol, and horizon
    print(f"\n{'='*60}")
    print("COMPARATIVE STATICS (sanity checks)")
    print(f"{'='*60}")

    print("\n  PD vs leverage (fixed assets V=150k, sigma_V=25%, rising debt):")
    print("    [Correct experiment: hold the SAME firm's assets fixed, vary its debt.")
    print("     Holding market cap fixed instead is ill-posed — it implies the assets")
    print("     grew to absorb the debt, so PD would stay flat.]")
    V_fixed, sigma_V_fixed = 150_000, 0.25
    for D in [30_000, 60_000, 90_000, 120_000, 140_000]:
        dd = distance_to_default(V_fixed, sigma_V_fixed, D, 0.08, 1.0)
        print(f"    D={D:>7,}: DD={dd:.2f}  PD={probability_of_default(dd):.2%}")

    print("\n  PD vs equity volatility:")
    for sE in [0.20, 0.35, 0.50, 0.65]:
        V, sV = solve_asset_value_and_vol(100_000, sE, 60_000, 0.04, 1.0)
        dd = distance_to_default(V, sV, 60_000, 0.08, 1.0)
        print(f"    sigma_E={sE:.0%}: PD={probability_of_default(dd):.2%}")

    print("\n  PD vs time horizon:")
    for T in [0.5, 1.0, 2.0, 5.0]:
        V, sV = solve_asset_value_and_vol(100_000, 0.30, 60_000, 0.04, T)
        dd = distance_to_default(V, sV, 60_000, 0.08, T)
        print(f"    T={T:.1f}y: PD={probability_of_default(dd):.2%}")
