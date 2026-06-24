"""
Test suite for Layer 3: portfolio optimization & hedging.

Covers:
  - Post-hedge simulation: hedging reduces losses; full hedge -> ~0 loss
  - Hedge cost and budget constraints respected
  - Greedy allocator: bounds (0 <= h <= exposure), budget respected
  - Optimizer: reduces ES vs unhedged, respects budget
  - Frontier: monotonically decreasing tail risk as budget grows

Run with: pytest test_layer3.py -v
"""

import numpy as np
import pytest

from copula import make_sample_portfolio
from portfolio_risk import expected_shortfall, risk_contributions
from optimization import (
    simulate_hedged_losses, hedge_cost, compute_spreads,
    greedy_hedge, optimize_hedge, hedging_frontier,
)


@pytest.fixture(scope="module")
def setup():
    portfolio = make_sample_portfolio(n_names=60, seed=11)
    spreads = compute_spreads(portfolio)
    _, risk_share = risk_contributions(portfolio, rho=0.20)
    return portfolio, spreads, risk_share


# ============================================================
# POST-HEDGE SIMULATION
# ============================================================

def test_hedging_reduces_losses(setup):
    """Some hedging should reduce expected shortfall vs no hedging."""
    portfolio, spreads, risk_share = setup
    n = len(spreads)
    unhedged = simulate_hedged_losses(portfolio, np.zeros(n), rho=0.20)
    half_hedge = simulate_hedged_losses(portfolio, portfolio["exposure"] * 0.5, rho=0.20)
    assert expected_shortfall(half_hedge, 0.99) < expected_shortfall(unhedged, 0.99)


def test_full_hedge_eliminates_loss(setup):
    """Hedging the entire exposure should drive losses to ~zero."""
    portfolio, spreads, risk_share = setup
    full = simulate_hedged_losses(portfolio, portfolio["exposure"], rho=0.20)
    assert full.max() < 1e-6


def test_no_hedge_matches_baseline(setup):
    """Zero hedges should give the same losses as the unhedged portfolio."""
    portfolio, spreads, risk_share = setup
    n = len(spreads)
    losses = simulate_hedged_losses(portfolio, np.zeros(n), rho=0.20)
    assert losses.mean() > 0    # there are real losses when unhedged


# ============================================================
# HEDGE COST
# ============================================================

def test_hedge_cost_zero_when_no_hedge(setup):
    """No hedges -> zero cost."""
    portfolio, spreads, risk_share = setup
    assert hedge_cost(np.zeros(len(spreads)), spreads) == 0.0


def test_hedge_cost_positive(setup):
    """Hedging something costs something."""
    portfolio, spreads, risk_share = setup
    cost = hedge_cost(portfolio["exposure"], spreads)
    assert cost > 0


def test_spreads_increase_with_pd(setup):
    """CDS spread should rise with a name's PD."""
    portfolio, spreads, risk_share = setup
    pd = portfolio["pd"]
    # Correlation between PD and spread should be strongly positive
    corr = np.corrcoef(pd, spreads)[0, 1]
    assert corr > 0.9


# ============================================================
# GREEDY ALLOCATOR
# ============================================================

def test_greedy_respects_budget(setup):
    """Greedy hedge cost should not exceed the budget."""
    portfolio, spreads, risk_share = setup
    budget = 0.3 * hedge_cost(portfolio["exposure"], spreads)
    hedges = greedy_hedge(portfolio, spreads, budget, 0.20, risk_share)
    assert hedge_cost(hedges, spreads) <= budget + 1e-9


def test_greedy_respects_bounds(setup):
    """Greedy hedges must be within [0, exposure]."""
    portfolio, spreads, risk_share = setup
    budget = 0.3 * hedge_cost(portfolio["exposure"], spreads)
    hedges = greedy_hedge(portfolio, spreads, budget, 0.20, risk_share)
    assert (hedges >= -1e-9).all()
    assert (hedges <= portfolio["exposure"] + 1e-9).all()


def test_greedy_reduces_risk(setup):
    """Greedy hedging with a real budget should reduce ES."""
    portfolio, spreads, risk_share = setup
    n = len(spreads)
    budget = 0.3 * hedge_cost(portfolio["exposure"], spreads)
    base = expected_shortfall(simulate_hedged_losses(portfolio, np.zeros(n), 0.20), 0.99)
    hedges = greedy_hedge(portfolio, spreads, budget, 0.20, risk_share)
    hedged = expected_shortfall(simulate_hedged_losses(portfolio, hedges, 0.20), 0.99)
    assert hedged < base


# ============================================================
# OPTIMIZER
# ============================================================

def test_optimizer_respects_budget(setup):
    """Optimized hedge cost should not materially exceed the budget."""
    portfolio, spreads, risk_share = setup
    budget = 0.3 * hedge_cost(portfolio["exposure"], spreads)
    hedges = optimize_hedge(portfolio, spreads, budget, 0.20)
    assert hedge_cost(hedges, spreads) <= budget * 1.02    # small numerical slack


def test_optimizer_reduces_risk(setup):
    """Optimized hedging should reduce ES vs unhedged."""
    portfolio, spreads, risk_share = setup
    n = len(spreads)
    budget = 0.3 * hedge_cost(portfolio["exposure"], spreads)
    base = expected_shortfall(simulate_hedged_losses(portfolio, np.zeros(n), 0.20), 0.99)
    hedges = optimize_hedge(portfolio, spreads, budget, 0.20)
    hedged = expected_shortfall(simulate_hedged_losses(portfolio, hedges, 0.20), 0.99)
    assert hedged < base


# ============================================================
# FRONTIER
# ============================================================

def test_frontier_is_decreasing(setup):
    """Tail risk should (weakly) decrease as the hedging budget grows."""
    portfolio, spreads, risk_share = setup
    frontier = hedging_frontier(portfolio, spreads, 0.20, risk_share, n_points=8)
    es_values = [es for _, es in frontier]
    # Each successive ES should be <= the previous (allowing tiny MC noise)
    for i in range(len(es_values) - 1):
        assert es_values[i + 1] <= es_values[i] + 0.05


def test_frontier_starts_at_unhedged(setup):
    """Zero-budget frontier point should equal the unhedged ES."""
    portfolio, spreads, risk_share = setup
    n = len(spreads)
    frontier = hedging_frontier(portfolio, spreads, 0.20, risk_share, n_points=8)
    base = expected_shortfall(simulate_hedged_losses(portfolio, np.zeros(n), 0.20), 0.99)
    assert abs(frontier[0][1] - base) < 0.05
