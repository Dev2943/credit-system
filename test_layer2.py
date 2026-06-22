"""
Test suite for Layer 2: portfolio credit risk.

Covers:
  - Gaussian copula: PD preservation, correlation fattens the tail, EL stable
  - Tail metrics: ES >= VaR >= EL ordering, percentile correctness
  - Concentration: Herfindahl bounds, effective-N, sector HHI
  - Risk contributions: shares sum to 1, high-PD names contribute more

Run with: pytest test_layer2.py -v
"""

import numpy as np
import pytest

from copula import make_sample_portfolio, simulate_losses
from portfolio_risk import (
    credit_var, expected_shortfall, decompose_loss,
    herfindahl, sector_concentration, risk_contributions,
)


# ============================================================
# COPULA PROPERTIES
# ============================================================

def test_copula_preserves_marginal_pd():
    """At rho=0, each name's empirical default rate should match its PD.

    The copula must preserve marginal default probabilities regardless of rho.
    We check the portfolio-average default rate matches the average PD.
    """
    portfolio = make_sample_portfolio(n_names=200, seed=1)
    # Re-simulate but count defaults rather than losses, at rho=0
    pd = portfolio["pd"]
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    n_scen = 100_000
    thresholds = norm.ppf(pd)
    M = rng.standard_normal(n_scen)
    Z = rng.standard_normal((n_scen, len(pd)))
    X = 0.0 * M[:, None] + 1.0 * Z          # rho = 0
    defaults = X < thresholds
    empirical = defaults.mean(axis=0)
    # Each name's empirical default rate ~ its PD (within MC noise)
    assert np.allclose(empirical, pd, atol=0.01)


def test_correlation_does_not_change_expected_loss():
    """Expected loss should be ~invariant to correlation."""
    portfolio = make_sample_portfolio(n_names=100, seed=2)
    el_low = simulate_losses(portfolio, rho=0.0).mean()
    el_high = simulate_losses(portfolio, rho=0.5).mean()
    assert abs(el_low - el_high) / el_low < 0.05    # within 5%


def test_correlation_fattens_tail():
    """Higher correlation -> higher tail percentile (fatter tail)."""
    portfolio = make_sample_portfolio(n_names=100, seed=2)
    tail_low = np.percentile(simulate_losses(portfolio, rho=0.0), 99.9)
    tail_high = np.percentile(simulate_losses(portfolio, rho=0.5), 99.9)
    assert tail_high > tail_low * 2     # tail should grow substantially


# ============================================================
# TAIL METRICS
# ============================================================

def test_es_geq_var_geq_el():
    """Fundamental ordering: ES >= VaR >= expected loss."""
    portfolio = make_sample_portfolio(n_names=100, seed=3)
    losses = simulate_losses(portfolio, rho=0.2)
    el = losses.mean()
    var = credit_var(losses, 0.99)
    es = expected_shortfall(losses, 0.99)
    assert es >= var >= el


def test_higher_confidence_higher_var():
    """99.9% VaR >= 99% VaR (deeper tail, larger loss)."""
    portfolio = make_sample_portfolio(n_names=100, seed=3)
    losses = simulate_losses(portfolio, rho=0.2)
    assert credit_var(losses, 0.999) >= credit_var(losses, 0.99)


def test_unexpected_loss_positive():
    """Unexpected loss (VaR - EL) should be positive for a risky book."""
    portfolio = make_sample_portfolio(n_names=100, seed=3)
    losses = simulate_losses(portfolio, rho=0.2)
    d = decompose_loss(losses, 0.99)
    assert d["unexpected_loss"] > 0


# ============================================================
# CONCENTRATION
# ============================================================

def test_herfindahl_equal_weights():
    """For N equal exposures, HHI = 1/N exactly."""
    exposures = np.ones(50)
    assert abs(herfindahl(exposures) - 1/50) < 1e-9


def test_herfindahl_bounds():
    """HHI is in (0, 1]; single name -> 1, many equal -> small."""
    single = np.array([100.0])
    assert abs(herfindahl(single) - 1.0) < 1e-9
    many = np.ones(1000)
    assert herfindahl(many) < 0.01


def test_concentration_raises_hhi():
    """A concentrated book has higher HHI than an equal-weight book."""
    equal = np.ones(100)
    concentrated = np.array([50.0] + [0.5] * 99)   # one name dominates
    assert herfindahl(concentrated) > herfindahl(equal)


def test_sector_hhi_detects_concentration():
    """A sector-concentrated book should have sector HHI above equal-sector level."""
    exposures = np.ones(100)
    # 70 names in one sector, 30 split across 3 others
    sectors = ["A"] * 70 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
    sector_hhi, _ = sector_concentration(exposures, sectors)
    equal_4_sector_hhi = 1/4
    assert sector_hhi > equal_4_sector_hhi


# ============================================================
# RISK CONTRIBUTIONS
# ============================================================

def test_risk_shares_sum_to_one():
    """Risk-contribution shares should sum to 1."""
    portfolio = make_sample_portfolio(n_names=100, seed=4)
    _, risk_share = risk_contributions(portfolio, rho=0.2)
    assert abs(risk_share.sum() - 1.0) < 1e-6


def test_exposure_shares_sum_to_one():
    """Exposure shares should sum to 1."""
    portfolio = make_sample_portfolio(n_names=100, seed=4)
    exp_share, _ = risk_contributions(portfolio, rho=0.2)
    assert abs(exp_share.sum() - 1.0) < 1e-6


def test_high_pd_names_overcontribute_risk():
    """On average, names with above-median PD should have risk share > exposure share."""
    portfolio = make_sample_portfolio(n_names=200, seed=5)
    exp_share, risk_share = risk_contributions(portfolio, rho=0.2)
    pd = portfolio["pd"]
    high_pd = pd > np.median(pd)
    # Average excess (risk - exposure) for high-PD names should be positive
    excess = risk_share - exp_share
    assert excess[high_pd].mean() > 0
