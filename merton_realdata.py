"""
Layer 1, Day 2: Merton on Real Companies.

Pulls real market data (market cap, price history, balance-sheet debt) for
three firms spanning the credit ladder — Microsoft (AAA), Ford (BBB),
Carnival (junk) — and runs the Day 1 Merton engine on each, then checks
that the model ranks their default risk correctly.


Run with: python3 merton_realdata.py
"""

import numpy as np
import pandas as pd
import yfinance as yf

# Import the Day 1 engine
from merton_model import (
    solve_asset_value_and_vol,
    distance_to_default,
    probability_of_default,
    credit_spread,
)


# Three firms spanning the credit ladder
FIRMS = {
    "MSFT": "Microsoft (AAA)",
    "F":    "Ford (BBB)",
    "CCL":  "Carnival (junk)",
}

R = 0.045   # ~1y risk-free (short Treasury)
T = 1.0     # 1-year horizon
MU = 0.08   # real-world asset drift assumption


# ----------------------------- EQUITY VOLATILITY -----------------------------
def equity_volatility(prices: pd.Series) -> float:
    """Annualized equity volatility from daily log returns.

    sigma_E = std(daily log returns) * sqrt(252)
    """
    # Compute annualized vol from the price series.
    
    log_returns = np.log(prices / prices.shift(1)).dropna()
    sigma_E = log_returns.std() * np.sqrt(252)
    return float(sigma_E)
   


# ----------------------------- DEBT FROM BALANCE SHEET -----------------------------
def get_total_debt(ticker_obj) -> float:
    """Extract total debt from the balance sheet.

    yfinance balance_sheet is a DataFrame with line items as the index and
    dates as columns. We want the most recent column (iloc[:, 0]).
    Common row label: 'Total Debt'. Fall back to summing short + long term debt.
    """
    bs = ticker_obj.balance_sheet
    if bs is None or bs.empty:
        raise ValueError("No balance sheet available")

    most_recent = bs.iloc[:, 0]  # most recent reporting date

    # Pull total debt from the most_recent column.
    # Try 'Total Debt' first; if absent, sum 'Current Debt' + 'Long Term Debt'.
    # Use .get() with defaults and handle NaN.
    
    if "Total Debt" in most_recent.index and not pd.isna(most_recent["Total Debt"]):
        return float(most_recent["Total Debt"])
    short = most_recent.get("Current Debt", 0)
    long = most_recent.get("Long Term Debt", 0)
    short = 0 if pd.isna(short) else short
    long = 0 if pd.isna(long) else long
    return float(short + long)


# ----------------------------- PULL ONE FIRM -----------------------------
def pull_firm_data(ticker: str) -> dict:
    """Pull market cap, equity vol, and debt for one ticker."""
    tk = yf.Ticker(ticker)

    # Market cap (equity value E)
    info = tk.info
    market_cap = info.get("marketCap", None)
    if market_cap is None:
        raise ValueError(f"No market cap for {ticker}")

    # Price history -> equity vol (1 year of daily closes)
    hist = tk.history(period="1y")
    if hist.empty:
        raise ValueError(f"No price history for {ticker}")
    sigma_E = equity_volatility(hist["Close"])

    # Debt
    debt = get_total_debt(tk)

    return {
        "ticker": ticker,
        "E": float(market_cap),
        "sigma_E": sigma_E,
        "D": debt,
    }


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("Pulling real market data (this hits the network)...\n")

    results = []
    for ticker, name in FIRMS.items():
        try:
            data = pull_firm_data(ticker)
        except Exception as e:
            print(f"  {ticker}: FAILED to pull data — {e}")
            continue

        # INSPECT the raw inputs before trusting them (the lesson!)
        print(f"{name} [{ticker}]")
        print(f"    Market cap E:       ${data['E']/1e9:>10,.1f} B")
        print(f"    Equity volatility:  {data['sigma_E']:>11.1%}")
        print(f"    Total debt D:       ${data['D']/1e9:>10,.1f} B")
        print(f"    Leverage D/(E+D):   {data['D']/(data['E']+data['D']):>11.1%}")

        # Run the Merton engine
        V, sigma_V = solve_asset_value_and_vol(
            data["E"], data["sigma_E"], data["D"], R, T)
        dd = distance_to_default(V, sigma_V, data["D"], MU, T)
        pd_val = probability_of_default(dd)
        spread = credit_spread(V, data["E"], data["D"], R, T)

        print(f"    -> Asset vol:        {sigma_V:>11.1%}")
        print(f"    -> Distance to def:  {dd:>11.2f}  sigma")
        print(f"    -> PD (1y):          {pd_val:>11.2%}")
        print(f"    -> Implied spread:   {spread*1e4:>11.0f}  bps")
        print()

        results.append({
            "name": name, "ticker": ticker,
            "leverage": data["D"]/(data["E"]+data["D"]),
            "dd": dd, "pd": pd_val, "spread": spread,
        })

    # ----------------------------- RANKING CHECK -----------------------------
    print("=" * 60)
    print("CREDIT RANKING (the validation)")
    print("=" * 60)

    # Sort results by PD ascending and print the ranking.
    # Then check whether the order is MSFT < Ford < Carnival.
   
    ranked = sorted(results, key=lambda x: x["pd"])
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['name']:24s}  PD={r['pd']:.2%}  spread={r['spread']*1e4:.0f}bps")
    order = [r["ticker"] for r in ranked]
    expected = ["MSFT", "F", "CCL"]
    if order == expected:
        print("\n  PASS — model ranks MSFT < Ford < Carnival, as credit ratings imply")
    else:
        print(f"\n  Order was {order}; expected {expected}")

    print("\nNote: Merton understates absolute 1y PD (default-only-at-maturity,")
    print("lognormal assets). The RANKING and relative spreads are the valid signal.")
