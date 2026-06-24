"""
Layer 4, Day 1: The AI / LLM Layer.

An agent that sits on top of the credit system: runs the full pipeline
(Layers 1-3), serializes the portfolio risk state, and generates a desk-ready
natural-language briefing with flagged concentrations and a hedge recommendation.

Design principle: THE MODELS COMPUTE, THE LLM COMMUNICATES.
Every number is produced by the tested quant engine; the LLM only translates
verified outputs into language. Calls a real LLM if an API key is present,
otherwise uses a grounded template generator so the pipeline always runs.


Run with: python3 ai_agent.py
"""

import os
import numpy as np

from copula import make_sample_portfolio
from portfolio_risk import (
    decompose_loss, herfindahl, sector_concentration, risk_contributions,
)
from copula import simulate_losses
from optimization import (
    compute_spreads, greedy_hedge, optimize_hedge,
    simulate_hedged_losses, hedge_cost,
)
from portfolio_risk import expected_shortfall


# ----------------------------- GATHER PORTFOLIO STATE -----------------------------
def gather_state(portfolio, rho=0.20, budget_frac=0.20):
    """Run the full pipeline and collect every figure the briefing will need.

    This is the 'compute' half — all numbers come from the tested engine.
    """
    exposure = portfolio["exposure"]
    pd = portfolio["pd"]
    sectors = portfolio["sectors"]

    # Tail risk
    losses = simulate_losses(portfolio, rho=rho)
    decomp = decompose_loss(losses, alpha=0.99)

    # Concentration
    name_hhi = herfindahl(exposure)
    sector_hhi, sector_exp = sector_concentration(exposure, sectors)

    # Risk contributions
    exp_share, risk_share = risk_contributions(portfolio, rho=rho)
    excess = risk_share - exp_share
    top_risk_idx = np.argsort(excess)[::-1][:5]

    # Hedge optimization
    spreads = compute_spreads(portfolio)
    full_cost = float(np.sum(exposure * spreads))
    budget = budget_frac * full_cost
    base_es = expected_shortfall(simulate_hedged_losses(portfolio, np.zeros(len(pd)), rho), 0.99)
    hedges = greedy_hedge(portfolio, spreads, budget, rho, risk_share)
    hedged_es = expected_shortfall(simulate_hedged_losses(portfolio, hedges, rho), 0.99)
    hedged_names = np.where(hedges > 1e-9)[0]

    return {
        "n_names": len(pd),
        "total_exposure": float(exposure.sum()),
        "avg_pd": float(pd.mean()),
        "expected_loss": decomp["expected_loss"],
        "var99": decomp["var"],
        "es99": decomp["es"],
        "unexpected_loss": decomp["unexpected_loss"],
        "name_hhi": name_hhi,
        "eff_names": 1.0 / name_hhi,
        "sector_hhi": sector_hhi,
        "eff_sectors": 1.0 / sector_hhi,
        "sector_exp": sector_exp,
        "top_risk_idx": top_risk_idx,
        "exp_share": exp_share,
        "risk_share": risk_share,
        "pd": pd,
        "sectors": sectors,
        "budget": budget,
        "base_es": base_es,
        "hedged_es": hedged_es,
        "hedge_cost": hedge_cost(hedges, spreads),
        "hedged_names": hedged_names,
        "es_reduction_pct": (1 - hedged_es / base_es) * 100,
    }


# ----------------------------- SERIALIZE STATE -> CONTEXT -----------------------------
def serialize_state(s):
    """Turn the computed state into a labeled text context block for the LLM.

    This is the grounding: the LLM sees ONLY these verified numbers.
    """
    # Top sector
    top_sector = max(s["sector_exp"].items(), key=lambda x: x[1])

    # Build a structured, labeled context string from the state dict `s`.
    # Include: portfolio size/exposure, EL/VaR/ES/unexpected loss, name & sector
    # concentration, the top sector and its share, and the hedge recommendation.
   
    lines = []
    lines.append(f"PORTFOLIO: {s['n_names']} names, ${s['total_exposure']:.1f}M exposure, avg PD {s['avg_pd']:.2%}")
    lines.append(f"TAIL RISK: expected loss ${s['expected_loss']:.2f}M, 99% VaR ${s['var99']:.2f}M, "
                 f"99% ES ${s['es99']:.2f}M, unexpected loss (capital) ${s['unexpected_loss']:.2f}M")
    lines.append(f"NAME CONCENTRATION: HHI {s['name_hhi']:.4f}, effective {s['eff_names']:.0f} of {s['n_names']} names")
    lines.append(f"SECTOR CONCENTRATION: HHI {s['sector_hhi']:.4f}, effective {s['eff_sectors']:.1f} sectors; "
                 f"largest sector {top_sector[0]} at ${top_sector[1]:.1f}M ({top_sector[1]/s['total_exposure']:.0%})")
    top_lines = ", ".join(f"name {i} (PD {s['pd'][i]:.1%}, risk share {s['risk_share'][i]:.1%} vs exposure {s['exp_share'][i]:.1%})"
                          for i in s['top_risk_idx'])
    lines.append(f"TOP RISK CONTRIBUTORS: {top_lines}")
    lines.append(f"HEDGE RECOMMENDATION: spend ${s['hedge_cost']:.2f}M/yr hedging {len(s['hedged_names'])} names, "
                 f"reducing 99% ES from ${s['base_es']:.2f}M to ${s['hedged_es']:.2f}M ({s['es_reduction_pct']:.0f}% reduction)")
    return "\n".join(lines)
    


# ----------------------------- BUILD PROMPT -----------------------------
def build_prompt(context, task="briefing"):
    """Construct the task-specific prompt around the grounded context."""
    tasks = {
        "briefing": "Write a concise risk briefing for the desk head. "
                    "Summarize the portfolio's tail risk, flag the most important "
                    "concentration, and state the recommended hedge with its rationale.",
        "alert": "Write a short concentration alert flagging the single most "
                 "important risk in this portfolio and why it matters.",
    }
    instruction = tasks.get(task, tasks["briefing"])

    # Assemble the full prompt: a system framing + the grounded context + the task.
    # Emphasize that the model must ONLY use the numbers provided (no inventing figures).
    
    prompt = (
        "You are a credit portfolio risk analyst. Using ONLY the verified figures "
        "below (do not invent any numbers), write the requested output in clear, "
        "professional language for a trading desk.\n\n"
        f"=== VERIFIED PORTFOLIO DATA ===\n{context}\n\n"
        f"=== TASK ===\n{instruction}\n"
    )
    return prompt
    


# ----------------------------- LLM CALL (with fallback) -----------------------------
def call_llm(prompt):
    """Call a real LLM if an API key is configured; else return None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        print(f"  [LLM call failed: {e}; using template fallback]")
        return None


def template_briefing(s):
    """Deterministic grounded briefing — used when no LLM is configured.

    Still grounded: every number comes from the computed state `s`.
    """
    top_sector = max(s["sector_exp"].items(), key=lambda x: x[1])

    # Write a readable multi-paragraph briefing from the state `s`.
    # Cover: tail risk (ES, unexpected loss), the key concentration (top sector),
    # the top risk contributors, and the hedge recommendation. Plain professional prose.
   
    para1 = (f"Portfolio risk briefing: the book holds {s['n_names']} names totaling "
             f"${s['total_exposure']:.0f}M with an average default probability of {s['avg_pd']:.1%}. "
             f"Expected loss is ${s['expected_loss']:.1f}M (covered by reserves), but the 99% "
             f"expected shortfall is ${s['es99']:.1f}M, implying ${s['unexpected_loss']:.1f}M of "
             f"unexpected loss that economic capital must absorb.")
    para2 = (f"The dominant concentration is sector-based: {top_sector[0]} represents "
             f"${top_sector[1]:.0f}M ({top_sector[1]/s['total_exposure']:.0%}) of the book, leaving an "
             f"effective {s['eff_sectors']:.1f} sectors of diversification. While the book holds "
             f"{s['n_names']} names (effective {s['eff_names']:.0f}), same-sector names default together, "
             f"so this concentration drives the tail.")
    names_str = ", ".join(str(i) for i in s['top_risk_idx'])
    para3 = (f"Names {names_str} contribute disproportionately to tail risk relative to their exposure. "
             f"The recommended hedge spends ${s['hedge_cost']:.2f}M/yr on CDS protection across "
             f"{len(s['hedged_names'])} names, cutting 99% ES from ${s['base_es']:.1f}M to "
             f"${s['hedged_es']:.1f}M — a {s['es_reduction_pct']:.0f}% reduction in tail risk.")
    return "\n\n".join([para1, para2, para3])
    


def generate_briefing(s, task="briefing"):
    """Generate the briefing: real LLM if available, else grounded template."""
    context = serialize_state(s)
    prompt = build_prompt(context, task)
    llm_output = call_llm(prompt)
    if llm_output is not None:
        return llm_output, "LLM"
    return template_briefing(s), "template (no API key)"


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    portfolio = make_sample_portfolio(n_names=100)
    rng = np.random.default_rng(3)
    sectors = ["Energy"] * 40 + list(rng.choice(
        ["Tech", "Financials", "Healthcare", "Industrials"], size=60))
    portfolio["sectors"] = sectors

    print("Running full pipeline (Layers 1-3)...")
    state = gather_state(portfolio)

    print("\n" + "=" * 64)
    print("GROUNDED CONTEXT (what the LLM sees)")
    print("=" * 64)
    print(serialize_state(state))

    print("\n" + "=" * 64)
    print("DESK BRIEFING")
    print("=" * 64)
    briefing, source = generate_briefing(state)
    print(f"[generated via: {source}]\n")
    print(briefing)
