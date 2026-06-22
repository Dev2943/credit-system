"""
Layer 2, Day 2: Credit VaR, Expected Shortfall & Concentration.

Extracts tail-risk metrics (credit VaR, expected shortfall) from the copula
loss distribution, separates expected from unexpected loss, and computes
concentration analytics (Herfindahl name + sector concentration, per-name
risk contributions) — the core of what the Credit Portfolio Group monitors.


Run with: python3 portfolio_risk.py
"""

import numpy as np
from scipy.stats import norm

from copula import make_sample_portfolio, simulate_losses


# ----------------------------- VaR & ES -----------------------------
def credit_var(losses, alpha=0.99):
    """Credit VaR: the loss exceeded only (1 - alpha) of the time."""
    # Return the alpha-quantile of the loss distribution.
    return np.percentile(losses, alpha * 100)
    


def expected_shortfall(losses, alpha=0.99):
    """Expected shortfall: average loss in the worst (1 - alpha) tail."""
    var = np.percentile(losses, alpha * 100)

    # Average all losses that exceed the VaR threshold.
    tail = losses[losses >= var]
    return tail.mean()
   


def decompose_loss(losses, alpha=0.99):
    """Split into expected loss (reserves) and unexpected loss (capital)."""
    el = losses.mean()
    var = credit_var(losses, alpha)
    es = expected_shortfall(losses, alpha)
    unexpected = var - el          # capital covers losses beyond the expected
    return {"expected_loss": el, "var": var, "es": es, "unexpected_loss": unexpected}


# ----------------------------- NAME CONCENTRATION -----------------------------
def herfindahl(exposures):
    """Herfindahl-Hirschman Index of exposure concentration.

    HHI = sum of squared exposure weights. 1/HHI = effective number of names.
    """
    # Compute HHI from the exposure array.
    weights = exposures / exposures.sum()
    return float((weights ** 2).sum())
   


# ----------------------------- SECTOR CONCENTRATION -----------------------------
def sector_concentration(exposures, sectors):
    """Group exposures by sector and compute sector-level HHI."""
    unique_sectors = sorted(set(sectors))
    sector_exposure = {}
    for s in unique_sectors:
        mask = np.array([sec == s for sec in sectors])
        sector_exposure[s] = exposures[mask].sum()

    sector_exp = np.array(list(sector_exposure.values()))

    # Compute the sector-level HHI from sector_exp.
    sector_hhi = herfindahl(sector_exp)
    return sector_hhi, sector_exposure


# ----------------------------- RISK CONTRIBUTIONS -----------------------------
def risk_contributions(portfolio, rho, alpha=0.99, n_scenarios=50_000, seed=42):
    """Each name's contribution to portfolio expected shortfall.

    Method: identify the tail scenarios (portfolio loss >= VaR), then measure
    how much each name's losses contribute to the average loss in those scenarios.
    A name's risk-contribution share can differ a lot from its exposure share.
    """
    pd = portfolio["pd"]
    exposure = portfolio["exposure"]
    recovery = portfolio["recovery"]
    n_names = len(pd)

    rng = np.random.default_rng(seed)
    thresholds = norm.ppf(pd)
    M = rng.standard_normal(n_scenarios)
    Z = rng.standard_normal((n_scenarios, n_names))
    X = np.sqrt(rho) * M[:, None] + np.sqrt(1 - rho) * Z
    defaults = X < thresholds                       # (n_scenarios, n_names)
    loss_per_name = exposure * (1 - recovery)
    name_losses = defaults * loss_per_name          # (n_scenarios, n_names)
    port_losses = name_losses.sum(axis=1)

    var = np.percentile(port_losses, alpha * 100)
    tail_mask = port_losses >= var                  # the tail scenarios

    # Each name's risk contribution = its average loss IN the tail scenarios.
    # Then convert to a share of total tail loss.
   
    name_tail_loss = name_losses[tail_mask].mean(axis=0)
    risk_share = name_tail_loss / name_tail_loss.sum()

    exposure_share = exposure / exposure.sum()
    return exposure_share, risk_share


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    portfolio = make_sample_portfolio(n_names=100)
    RHO = 0.20

    # Assign each name to a sector (for sector concentration) — deliberately
    # make one sector over-represented to show concentration.
    rng = np.random.default_rng(3)
    sectors_list = ["Energy", "Tech", "Financials", "Healthcare", "Industrials"]
    # 40% of names in Energy (concentrated), rest spread
    sectors = (["Energy"] * 40 +
               list(rng.choice(sectors_list[1:], size=60)))
    portfolio["sectors"] = sectors

    print("=" * 64)
    print(f"PORTFOLIO TAIL RISK  (rho = {RHO})")
    print("=" * 64)
    losses = simulate_losses(portfolio, rho=RHO)
    d = decompose_loss(losses, alpha=0.99)
    d999 = decompose_loss(losses, alpha=0.999)
    print(f"  Expected loss (reserves):      ${d['expected_loss']:>7.3f}M")
    print(f"  99% Credit VaR:                ${d['var']:>7.3f}M")
    print(f"  99% Expected Shortfall:        ${d['es']:>7.3f}M")
    print(f"  99% Unexpected loss (capital): ${d['unexpected_loss']:>7.3f}M")
    print(f"  99.9% Credit VaR:              ${d999['var']:>7.3f}M")
    print(f"  99.9% Expected Shortfall:      ${d999['es']:>7.3f}M")

    print("\n" + "=" * 64)
    print("NAME CONCENTRATION")
    print("=" * 64)
    hhi = herfindahl(portfolio["exposure"])
    print(f"  Herfindahl index (HHI):        {hhi:.4f}")
    print(f"  Effective number of names:     {1/hhi:.1f}  (of {len(portfolio['pd'])} held)")
    top5 = np.sort(portfolio["exposure"])[::-1][:5]
    print(f"  Top 5 names = {top5.sum():.1f}% of book")

    print("\n" + "=" * 64)
    print("SECTOR CONCENTRATION")
    print("=" * 64)
    sector_hhi, sector_exp = sector_concentration(portfolio["exposure"], sectors)
    print(f"  Sector HHI:                    {sector_hhi:.4f}")
    print(f"  Effective number of sectors:   {1/sector_hhi:.1f}")
    for s, e in sorted(sector_exp.items(), key=lambda x: -x[1]):
        print(f"    {s:14s} ${e:>6.1f}M  ({e/portfolio['exposure'].sum():.0%})")

    print("\n" + "=" * 64)
    print("RISK CONTRIBUTION vs EXPOSURE  (the hedging signal)")
    print("=" * 64)
    exp_share, risk_share = risk_contributions(portfolio, rho=RHO)
    # Show the names whose risk share most exceeds their exposure share
    excess = risk_share - exp_share
    order = np.argsort(excess)[::-1][:8]
    print(f"  {'Name':>5}  {'Exposure%':>10}  {'Risk%':>8}  {'Excess':>8}  {'PD':>6}")
    for i in order:
        print(f"  {i:>5}  {exp_share[i]*100:>9.2f}%  {risk_share[i]*100:>7.2f}%  "
              f"{excess[i]*100:>+7.2f}%  {portfolio['pd'][i]:>5.1%}")
    print("\n  Names where Risk% > Exposure% punch above their weight in tail risk —")
    print("  these are the priority hedging targets for Layer 3.")
