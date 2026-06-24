"""
Layer 3, Day 1: Portfolio Optimization & Hedging.

Chooses CDS protection per name to minimize portfolio tail risk (expected
shortfall) subject to a hedging budget. Builds a greedy allocator and a proper
constrained optimizer, and traces the hedging efficient frontier.

Brings together all three layers: Layer 1 prices the CDS (hedge cost),
Layer 2 supplies the correlated loss distribution and risk contributions,
Layer 3 optimizes the hedge.


Run with: python3 optimization.py
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from copula import make_sample_portfolio
from reduced_form import cds_fair_spread
from portfolio_risk import expected_shortfall


# ----------------------------- POST-HEDGE LOSS SIMULATION -----------------------------
def simulate_hedged_losses(portfolio, hedges, rho, n_scenarios=20_000, seed=42):
    """Simulate portfolio losses AFTER applying CDS hedges.

    hedges[i] = notional of CDS protection bought on name i (0 <= hedges[i] <= exposure[i]).
    When name i defaults, the loss is reduced by the protection payout:
        net loss_i = (exposure_i - hedges_i) * (1 - recovery_i)
    (Protection pays (1-R) of the hedged notional on default, offsetting the loss.)
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
    defaults = X < thresholds

    # Net loss per name after hedging.
    # Unhedged exposure = exposure - hedges; loss on default = that * (1 - recovery).
    
    net_exposure = exposure - hedges
    loss_per_name = net_exposure * (1 - recovery)
    losses = (defaults * loss_per_name).sum(axis=1)

    return losses


# ----------------------------- HEDGE COST -----------------------------
def hedge_cost(hedges, spreads):
    """Annual premium cost of the hedges: sum of notional * spread."""
    return float(np.sum(hedges * spreads))


def compute_spreads(portfolio, R=0.40, T=5.0):
    """CDS spread for each name from its PD (Layer 1 reduced-form pricer).

    Convert each name's PD to a hazard rate, then price the CDS spread.
    """
    pd = portfolio["pd"]
    spreads = np.zeros(len(pd))
    for i, p in enumerate(pd):
        lam = -np.log(1 - p) / 1.0          # 1-year PD -> hazard rate
        spreads[i] = cds_fair_spread(lam, R=R, T=T)
    return spreads


# ----------------------------- GREEDY ALLOCATOR -----------------------------
def greedy_hedge(portfolio, spreads, budget, rho, risk_share):
    """Greedy: hedge names in order of risk-reduction per dollar of premium.

    Efficiency proxy: a name's risk share (tail-risk contribution) divided by
    its per-dollar hedge cost (spread). Hedge highest-efficiency names fully
    until the budget is exhausted.
    """
    exposure = portfolio["exposure"]
    n_names = len(exposure)
    hedges = np.zeros(n_names)

    # Rank names by efficiency = risk_share / (spread * exposure-ish cost).
    # Use risk_share / spread as the efficiency score (risk per unit premium rate).
    # Then greedily hedge each full exposure until budget runs out.
    
    efficiency = risk_share / np.maximum(spreads, 1e-6)
    order = np.argsort(efficiency)[::-1]
    spent = 0.0
    for i in order:
        cost_i = exposure[i] * spreads[i]
        if spent + cost_i <= budget:
            hedges[i] = exposure[i]
            spent += cost_i

    return hedges


# ----------------------------- NUMERICAL OPTIMIZER -----------------------------
def optimize_hedge(portfolio, spreads, budget, rho, n_scenarios=8_000):
    """Constrained optimization: minimize post-hedge ES subject to budget.

    Uses a fixed scenario set (common random numbers) so the objective is smooth
    across hedge choices.
    """
    exposure = portfolio["exposure"]
    pd = portfolio["pd"]
    recovery = portfolio["recovery"]
    n_names = len(exposure)

    # Pre-draw a FIXED scenario set (common random numbers) for a stable objective
    rng = np.random.default_rng(123)
    thresholds = norm.ppf(pd)
    M = rng.standard_normal(n_scenarios)
    Z = rng.standard_normal((n_scenarios, n_names))
    X = np.sqrt(rho) * M[:, None] + np.sqrt(1 - rho) * Z
    defaults = X < thresholds          # fixed across optimization

    def post_hedge_es(hedges):
        net_exposure = exposure - hedges
        loss_per_name = net_exposure * (1 - recovery)
        losses = (defaults * loss_per_name).sum(axis=1)
        return expected_shortfall(losses, alpha=0.99)

    # Set up bounds and the budget constraint, then minimize.
    # Bounds: 0 <= hedges[i] <= exposure[i].
    # Constraint: sum(hedges * spreads) <= budget  (inequality >= 0 form).
    
    bounds = [(0, exposure[i]) for i in range(n_names)]
    constraint = {"type": "ineq",
                  "fun": lambda h: budget - np.sum(h * spreads)}
    x0 = np.zeros(n_names)
    res = minimize(post_hedge_es, x0, method="SLSQP",
                   bounds=bounds, constraints=[constraint],
                   options={"maxiter": 100, "ftol": 1e-4})
    return res.x


# ----------------------------- EFFICIENT FRONTIER -----------------------------
def hedging_frontier(portfolio, spreads, rho, risk_share, n_points=10):
    """Trace tail risk vs hedging budget (the risk-reduction frontier)."""
    exposure = portfolio["exposure"]
    max_cost = float(np.sum(exposure * spreads))   # cost to hedge everything

    budgets = np.linspace(0, max_cost, n_points)
    frontier = []
    for b in budgets:
        hedges = greedy_hedge(portfolio, spreads, b, rho, risk_share)
        losses = simulate_hedged_losses(portfolio, hedges, rho)

        # Record (budget, post-hedge 99% ES) for this point.
       
        es = expected_shortfall(losses, 0.99)
        frontier.append((b, es))

    return frontier


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    portfolio = make_sample_portfolio(n_names=100)
    RHO = 0.20

    spreads = compute_spreads(portfolio)
    print("=" * 64)
    print("SETUP")
    print("=" * 64)
    print(f"  Portfolio: {len(portfolio['pd'])} names, "
          f"${portfolio['exposure'].sum():.1f}M exposure")
    print(f"  Avg CDS spread: {spreads.mean()*1e4:.0f} bps")
    print(f"  Cost to hedge everything: ${np.sum(portfolio['exposure']*spreads):.3f}M/yr")

    # Risk contributions (from Layer 2) to drive the greedy allocator
    from portfolio_risk import risk_contributions
    _, risk_share = risk_contributions(portfolio, rho=RHO)

    # Unhedged baseline
    unhedged = simulate_hedged_losses(portfolio, np.zeros(len(spreads)), RHO)
    base_es = expected_shortfall(unhedged, 0.99)
    print(f"\n  Unhedged 99% ES: ${base_es:.3f}M")

    # A budget = enough to hedge ~20% of the full-hedge cost
    BUDGET = 0.20 * float(np.sum(portfolio["exposure"] * spreads))
    print(f"  Hedging budget:  ${BUDGET:.3f}M/yr")

    print("\n" + "=" * 64)
    print("GREEDY vs OPTIMIZED HEDGE")
    print("=" * 64)
    greedy = greedy_hedge(portfolio, spreads, BUDGET, RHO, risk_share)
    greedy_losses = simulate_hedged_losses(portfolio, greedy, RHO)
    greedy_es = expected_shortfall(greedy_losses, 0.99)
    print(f"  Greedy:    ES ${greedy_es:.3f}M  "
          f"(cost ${hedge_cost(greedy, spreads):.3f}M, "
          f"reduction {(1-greedy_es/base_es)*100:.1f}%)")

    opt = optimize_hedge(portfolio, spreads, BUDGET, RHO)
    opt_losses = simulate_hedged_losses(portfolio, opt, RHO)
    opt_es = expected_shortfall(opt_losses, 0.99)
    print(f"  Optimized: ES ${opt_es:.3f}M  "
          f"(cost ${hedge_cost(opt, spreads):.3f}M, "
          f"reduction {(1-opt_es/base_es)*100:.1f}%)")

    print("\n" + "=" * 64)
    print("HEDGING EFFICIENT FRONTIER (greedy)")
    print("=" * 64)
    frontier = hedging_frontier(portfolio, spreads, RHO, risk_share)
    print(f"  {'Budget $M':>10}  {'99% ES $M':>10}  {'Reduction':>10}")
    for b, es in frontier:
        print(f"  {b:>10.3f}  {es:>10.3f}  {(1-es/base_es)*100:>9.1f}%")
    print("\n  Steep at first (hedging worst risk-contributors), then flattening")
    print("  (diminishing returns). The knee is the cost-effective stopping point.")
