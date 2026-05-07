"""
FUNDAMENTAL AGENT
─────────────────────────────────────────────────────────────────────────────
Role       : Quant Researcher
Objective  : Analyze momentum & book-to-market factors, assign Predictive
             Alpha Score (1–10), and apply Greenwashing Penalty via local LLM.
LLM Backend: Ollama (local, air-gapped) at OLLAMA_BASE_URL
Data Source: Day 2 — yfinance live data via data_ingestion.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import requests
from datetime import datetime
from typing import Any

import numpy as np

from data_ingestion import fetch_ticker_data, get_available_tickers

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3:8b")


# ─── Greenwashing Detection ──────────────────────────────────────────────────

def _compute_greenwashing_delta(ticker_data: dict[str, Any]) -> dict[str, Any]:
    """
    Compare self-reported Scope 1 emissions against NASA satellite-detected
    methane hotspot anomalies. A divergence > 20% triggers a penalty.
    """
    self_rep  = ticker_data["self_reported_scope1_mt"]
    satellite = ticker_data["nasa_satellite_methane_mt"]
    delta_pct = ((satellite - self_rep) / self_rep) * 100 if self_rep > 0 else 0

    penalty_applied   = delta_pct > 20
    penalty_magnitude = min(round(delta_pct / 10, 1), 3.0)

    return {
        "self_reported_scope1_mt"  : self_rep,
        "nasa_satellite_methane_mt": satellite,
        "delta_pct"                : round(delta_pct, 2),
        "greenwashing_flag"        : penalty_applied,
        "penalty_points"           : penalty_magnitude if penalty_applied else 0.0,
        "verdict"                  : (
            f"⚠️  GREENWASHING DETECTED — satellite emissions are "
            f"{delta_pct:.1f}% above self-reported figures."
            if penalty_applied else
            f"✅  Emissions reporting within tolerance ({delta_pct:.1f}% delta)."
        ),
    }


# ─── Factor Scoring ──────────────────────────────────────────────────────────

def _factor_score(ticker_data: dict[str, Any]) -> float:
    mom_score = np.clip((ticker_data["momentum_12m"] + 0.3) / 0.6 * 10, 1, 10)
    btm_score = np.clip(ticker_data["book_to_market"] / 0.8 * 10, 1, 10)
    return round(0.6 * mom_score + 0.4 * btm_score, 2)


# ─── Local LLM Call ──────────────────────────────────────────────────────────

def _call_local_llm(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model" : OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[LOCAL LLM OFFLINE] Cannot reach Ollama. Ensure container is running."
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"[ERROR] {exc}"


# ─── Main Agent Function ─────────────────────────────────────────────────────

def run_fundamental_agent(ticker: str) -> dict[str, Any]:
    """
    Full pipeline:
      1. Fetch live ticker data via yfinance (data_ingestion.py)
      2. Compute greenwashing delta
      3. Generate raw quant factor score
      4. Send prompt to local LLM for deliberation
      5. Return structured result
    """
    # ── Live data fetch (Day 2) ──────────────────────────────────────────────
    data = fetch_ticker_data(ticker)

    gw        = _compute_greenwashing_delta(data)
    raw_score = _factor_score(data)

    data_note = (
        f"[LIVE: yfinance, fetched {data.get('_fetched_at','')[:10]}]"
        if data.get("_data_source") == "yfinance_live"
        else "[SYNTHETIC FALLBACK — yfinance unavailable]"
    )

    prompt = f"""
You are a Quantitative ESG Analyst at a top-tier London hedge fund.
Analyze the following data for {ticker} ({data['name']}) and produce a
final "Predictive Alpha Score" on a scale of 1 (sell) to 10 (strong buy).

## Factor Data  {data_note}
- 12-Month Price Momentum : {data['momentum_12m']*100:.1f}%
- Book-to-Market Ratio    : {data['book_to_market']:.2f}
- Market Cap              : £{data['market_cap_bn']:.1f}bn
- Sector                  : {data['sector']}
- Raw Quant Factor Score  : {raw_score}/10

## ESG Integrity Check (NASA Methane Hotspot Cross-Reference)
- Self-Reported Scope 1 Emissions : {gw['self_reported_scope1_mt']} Mt CO2e
- NASA Satellite-Detected Methane  : {gw['nasa_satellite_methane_mt']} Mt CO2e
- Divergence                       : {gw['delta_pct']}%
- CSRD Compliance Score            : {data['csrd_compliance_score']*100:.0f}%
- GRI Disclosure Completeness      : {data['gri_disclosure_completeness']*100:.0f}%
- Greenwashing Verdict             : {gw['verdict']}
- Greenwashing Penalty (deduct)    : -{gw['penalty_points']} pts from Alpha Score

## Instructions
1. Consider momentum, value, and ESG integrity together.
2. Apply the greenwashing penalty if flagged.
3. Explain your reasoning in 3 bullet points.
4. Conclude with EXACTLY this line:
   PREDICTIVE_ALPHA_SCORE: <integer 1-10>

Be rigorous. Hedge fund LPs are watching.
""".strip()

    llm_response = _call_local_llm(prompt)

    final_score = raw_score
    for line in llm_response.splitlines():
        if "PREDICTIVE_ALPHA_SCORE:" in line:
            try:
                final_score = float(line.split(":")[1].strip())
                final_score = max(1.0, min(10.0, final_score))
            except ValueError:
                pass

    result = {
        "agent"            : "Fundamental Agent",
        "ticker"           : ticker,
        "company"          : data["name"],
        "sector"           : data["sector"],
        "raw_quant_score"  : raw_score,
        "greenwashing"     : gw,
        "predictive_alpha" : final_score,
        "llm_reasoning"    : llm_response,
        "data_source"      : data.get("_data_source", "unknown"),
        "timestamp"        : datetime.utcnow().isoformat(),
    }

    logger.info("FundamentalAgent | %s | Alpha: %s | Source: %s",
                ticker, final_score, result["data_source"])
    return result
