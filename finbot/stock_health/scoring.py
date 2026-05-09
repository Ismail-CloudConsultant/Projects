"""
Deterministic composite scorer for the finbot stock-health pipeline.
No LLM calls — pure Python arithmetic only.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from finbot.config import config
from finbot.shared.state_keys import SCORE_BREAKDOWN, TICKER

logger = logging.getLogger(__name__)

_NEUTRAL = 5.0  # fallback score for any unavailable metric


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _n(val: Any, default: float = _NEUTRAL) -> float:
    """Return val as a Python float, or default if None/NaN/invalid."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def _parse(v: Any) -> dict:
    """Convert a JSON string or dict to dict; return empty dict on failure."""
    if not v:
        return {}
    if isinstance(v, dict):
        return v
    try:
        return json.loads(str(v))
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Valuation sub-scorers
# ---------------------------------------------------------------------------

def _score_pe(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "P/E not available"
    v = _n(val)
    if v < 10:   return 10.0, f"P/E {v:.1f}x (deeply undervalued)"
    if v < 15:   return  8.0, f"P/E {v:.1f}x (undervalued)"
    if v < 25:   return  6.0, f"P/E {v:.1f}x (fairly valued)"
    if v < 40:   return  4.0, f"P/E {v:.1f}x (premium)"
    return 2.0, f"P/E {v:.1f}x (expensive)"


def _score_forward_pe(val: float | None) -> tuple[float, str]:
    score, sig = _score_pe(val)
    return score, sig.replace("P/E", "Fwd P/E")


def _score_peg(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "PEG not available"
    v = _n(val)
    if v < 1.0:  return 10.0, f"PEG {v:.2f} (undervalued relative to growth)"
    if v < 1.5:  return  7.0, f"PEG {v:.2f} (reasonable growth price)"
    if v < 2.0:  return  5.0, f"PEG {v:.2f} (fair)"
    if v < 3.0:  return  3.0, f"PEG {v:.2f} (pricey for growth)"
    return 1.0, f"PEG {v:.2f} (growth fully priced in)"


def _score_pb(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "P/B not available"
    v = _n(val)
    if v < 1:   return 10.0, f"P/B {v:.1f}x (below book value)"
    if v < 2:   return  8.0, f"P/B {v:.1f}x (moderate premium to book)"
    if v < 4:   return  5.0, f"P/B {v:.1f}x (fair premium)"
    if v < 8:   return  3.0, f"P/B {v:.1f}x (high premium)"
    return 1.0, f"P/B {v:.1f}x (very high premium to book)"


def _score_ev_ebitda(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "EV/EBITDA not available"
    v = _n(val)
    if v < 8:   return 10.0, f"EV/EBITDA {v:.1f}x (cheap)"
    if v < 12:  return  7.0, f"EV/EBITDA {v:.1f}x (reasonable)"
    if v < 18:  return  5.0, f"EV/EBITDA {v:.1f}x (fair)"
    if v < 25:  return  3.0, f"EV/EBITDA {v:.1f}x (premium)"
    return 1.0, f"EV/EBITDA {v:.1f}x (expensive)"


def _dividend_bonus(val: float | None) -> tuple[float, str | None]:
    """Return an additive bonus (0, 1, or 2) and a signal label."""
    if val is None:
        return 0.0, None
    v = _n(val, 0.0)
    if v > 0.04:
        return 2.0, f"Dividend yield {v*100:.1f}% — strong income signal"
    if v > 0.02:
        return 1.0, f"Dividend yield {v*100:.1f}% — moderate income"
    return 0.0, None


# ---------------------------------------------------------------------------
# Risk sub-scorers
# ---------------------------------------------------------------------------

def _score_volatility(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "Volatility not available"
    v = _n(val)
    if v < 0.15:  return 10.0, f"Volatility {v*100:.1f}% (low)"
    if v < 0.25:  return  7.0, f"Volatility {v*100:.1f}% (moderate)"
    if v < 0.40:  return  5.0, f"Volatility {v*100:.1f}% (elevated)"
    return 3.0, f"Volatility {v*100:.1f}% (high)"


def _score_sharpe(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "Sharpe ratio not available"
    v = _n(val)
    if v > 2:    return 10.0, f"Sharpe {v:.2f} (excellent risk-adjusted return)"
    if v > 1:    return  7.0, f"Sharpe {v:.2f} (good)"
    if v > 0.5:  return  5.0, f"Sharpe {v:.2f} (adequate)"
    if v > 0:    return  3.0, f"Sharpe {v:.2f} (poor)"
    return 1.0, f"Sharpe {v:.2f} (negative risk-adjusted return)"


def _score_sortino(val: float | None) -> tuple[float, str]:
    score, sig = _score_sharpe(val)
    return score, sig.replace("Sharpe", "Sortino")


def _score_max_drawdown(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "Max drawdown not available"
    v = _n(val)
    if v > -0.10:  return 10.0, f"Max drawdown {v*100:.1f}% (mild)"
    if v > -0.20:  return  7.0, f"Max drawdown {v*100:.1f}% (moderate)"
    if v > -0.35:  return  5.0, f"Max drawdown {v*100:.1f}% (significant)"
    return 3.0, f"Max drawdown {v*100:.1f}% (severe)"


def _score_var95(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "VaR not available"
    v = _n(val)
    if v > -0.02:  return 10.0, f"VaR 95% {v*100:.2f}% (low daily tail risk)"
    if v > -0.03:  return  7.0, f"VaR 95% {v*100:.2f}% (moderate)"
    if v > -0.05:  return  5.0, f"VaR 95% {v*100:.2f}% (elevated)"
    return 3.0, f"VaR 95% {v*100:.2f}% (high daily tail risk)"


# ---------------------------------------------------------------------------
# Momentum sub-scorers
# ---------------------------------------------------------------------------

def _score_rsi(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "RSI not available"
    v = _n(val)
    if 40 <= v <= 60:  return 10.0, f"RSI {v:.1f} (neutral zone — balanced)"
    if 35 <= v <= 70:  return  7.0, f"RSI {v:.1f} (healthy range)"
    if 30 <= v <= 80:  return  5.0, f"RSI {v:.1f} (approaching extreme)"
    return 2.0, f"RSI {v:.1f} (overbought/oversold extreme)"


def _score_macd_crossover(val: str | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "MACD crossover not available"
    v = str(val).lower()
    if v == "bullish":  return 8.0, "MACD bullish crossover"
    if v == "neutral":  return 5.0, "MACD neutral"
    if v == "bearish":  return 2.0, "MACD bearish crossover"
    return _NEUTRAL, f"MACD crossover: {val}"


def _score_sma_cross(golden: bool | None, death: bool | None) -> tuple[float, str]:
    g = bool(golden) if golden is not None else False
    d = bool(death) if death is not None else False
    if g:   return 9.0, "Golden cross (SMA50 > SMA200 — bullish long-term)"
    if d:   return 2.0, "Death cross (SMA50 < SMA200 — bearish long-term)"
    return 5.0, "No SMA crossover signal"


def _score_vs_sma200(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "Price vs SMA200 not available"
    v = _n(val)
    if v > 0.10:   return 9.0, f"Price {v*100:.1f}% above SMA200 (strong uptrend)"
    if v >= 0:     return 7.0, f"Price {v*100:.1f}% above SMA200"
    if v > -0.10:  return 5.0, f"Price {v*100:.1f}% below SMA200"
    return 2.0, f"Price {v*100:.1f}% below SMA200 (downtrend)"


# ---------------------------------------------------------------------------
# Returns sub-scorers
# ---------------------------------------------------------------------------

def _score_return_1y(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "1Y return not available"
    v = _n(val)
    if v > 0.30:   return 10.0, f"1Y return {v*100:.1f}% (exceptional)"
    if v > 0.15:   return  8.0, f"1Y return {v*100:.1f}% (strong)"
    if v >= 0:     return  6.0, f"1Y return {v*100:.1f}% (positive)"
    if v > -0.10:  return  4.0, f"1Y return {v*100:.1f}% (mild loss)"
    return 2.0, f"1Y return {v*100:.1f}% (significant loss)"


def _score_alpha(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "Alpha not available"
    v = _n(val)
    if v > 0.05:   return 10.0, f"Alpha {v*100:.1f}% (strong outperformance vs market)"
    if v >= 0:     return  7.0, f"Alpha {v*100:.1f}% (market outperformer)"
    if v > -0.05:  return  4.0, f"Alpha {v*100:.1f}% (slight underperformance)"
    return 2.0, f"Alpha {v*100:.1f}% (significant underperformance)"


def _score_cagr_3y(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "CAGR 3Y not available"
    v = _n(val)
    if v > 0.30:   return 10.0, f"CAGR 3Y {v*100:.1f}% (exceptional)"
    if v > 0.15:   return  8.0, f"CAGR 3Y {v*100:.1f}% (strong)"
    if v >= 0:     return  6.0, f"CAGR 3Y {v*100:.1f}% (positive)"
    if v > -0.10:  return  4.0, f"CAGR 3Y {v*100:.1f}% (mild loss)"
    return 2.0, f"CAGR 3Y {v*100:.1f}% (significant loss)"


# ---------------------------------------------------------------------------
# Sentiment sub-scorers
# ---------------------------------------------------------------------------

def _score_recommendation(val: str | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "Analyst recommendation not available"
    v = str(val).lower().replace(" ", "_").replace("-", "_")
    _MAP = {
        "strong_buy":   (10.0, "Analyst consensus: Strong Buy"),
        "buy":          ( 8.0, "Analyst consensus: Buy"),
        "hold":         ( 5.0, "Analyst consensus: Hold"),
        "underperform": ( 3.0, "Analyst consensus: Underperform"),
        "sell":         ( 2.0, "Analyst consensus: Sell"),
        "strong_sell":  ( 1.0, "Analyst consensus: Strong Sell"),
    }
    return _MAP.get(v, (_NEUTRAL, f"Analyst recommendation: {val}"))


def _score_price_target_upside(
    current: float | None, target: float | None
) -> tuple[float, str]:
    if current is None or target is None or current <= 0:
        return _NEUTRAL, "Price target upside not available"
    upside = (target - current) / current
    if upside > 0.20:  return 10.0, f"Price target {upside*100:.1f}% upside (strong)"
    if upside > 0.10:  return  8.0, f"Price target {upside*100:.1f}% upside (moderate)"
    if upside >= 0:    return  6.0, f"Price target {upside*100:.1f}% upside (limited)"
    return 3.0, f"Price target implies {upside*100:.1f}% downside"


def _score_short_float(val: float | None) -> tuple[float, str]:
    if val is None:
        return _NEUTRAL, "Short interest not available"
    v = _n(val)
    if v < 0.02:  return 10.0, f"Short interest {v*100:.1f}% (low)"
    if v < 0.05:  return  8.0, f"Short interest {v*100:.1f}% (modest)"
    if v < 0.10:  return  5.0, f"Short interest {v*100:.1f}% (elevated)"
    if v < 0.20:  return  3.0, f"Short interest {v*100:.1f}% (high short pressure)"
    return 1.0, f"Short interest {v*100:.1f}% (extreme short pressure)"


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

def score_to_grade(score: float) -> str:
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    if score >= 35: return "D"
    return "F"


# ---------------------------------------------------------------------------
# Core computation (pure Python, no ADK, no I/O — fully testable)
# ---------------------------------------------------------------------------

def _compute_composite_score(
    returns: dict,
    risk: dict,
    valuation: dict,
    momentum: dict,
    fundamentals: dict,
    sentiment: dict,
) -> dict:
    """
    Compute the composite score from six pre-fetched metric dicts.
    Returns the full result dict without touching session state.
    """
    all_signals: list[tuple[float, str]] = []

    def _avg(subs: list[tuple[float, str]]) -> float:
        return sum(s for s, _ in subs) / len(subs) if subs else _NEUTRAL

    # ---- Valuation ----
    val_subs: list[tuple[float, str]] = [
        _score_pe(valuation.get("trailing_pe")),
        _score_forward_pe(valuation.get("forward_pe")),
        _score_peg(valuation.get("peg_ratio")),
        _score_pb(valuation.get("price_to_book")),
        _score_ev_ebitda(valuation.get("ev_to_ebitda")),
    ]
    val_avg = _avg(val_subs)
    div_bonus, div_sig = _dividend_bonus(valuation.get("dividend_yield"))
    val_score = min(val_avg + div_bonus, 10.0)
    val_signals = list(val_subs)
    if div_sig:
        bonus_rank = 9.0 if div_bonus >= 2 else 7.0
        val_signals.append((bonus_rank, div_sig))
    all_signals.extend(val_signals)

    # ---- Risk ----
    risk_subs: list[tuple[float, str]] = [
        _score_volatility(risk.get("annual_volatility")),
        _score_sharpe(risk.get("sharpe_ratio")),
        _score_sortino(risk.get("sortino_ratio")),
        _score_max_drawdown(risk.get("max_drawdown")),
        _score_var95(risk.get("var_95")),
    ]
    risk_score = _avg(risk_subs)
    all_signals.extend(risk_subs)

    # ---- Momentum ----
    mom_subs: list[tuple[float, str]] = [
        _score_rsi(momentum.get("rsi_14")),
        _score_macd_crossover(momentum.get("macd_crossover")),
        _score_sma_cross(momentum.get("golden_cross"), momentum.get("death_cross")),
        _score_vs_sma200(momentum.get("price_vs_sma200_pct")),
    ]
    mom_score = _avg(mom_subs)
    all_signals.extend(mom_subs)

    # ---- Fundamentals — use pre-computed scores directly ----
    fund_subs: list[tuple[float, str]] = []
    for key, data in fundamentals.items():
        if isinstance(data, dict):
            s = data.get("score")
            if s is not None:
                label = f"{key.replace('_', ' ').title()}: {s}/10"
                fund_subs.append((float(s), label))
    fund_score = _avg(fund_subs) if fund_subs else _NEUTRAL
    all_signals.extend(fund_subs)

    # ---- Returns ----
    ret_subs: list[tuple[float, str]] = [
        _score_return_1y(returns.get("return_1y")),
        _score_alpha(returns.get("alpha_1y")),
        _score_cagr_3y(returns.get("cagr_3y")),
    ]
    ret_score = _avg(ret_subs)
    all_signals.extend(ret_subs)

    # ---- Sentiment ----
    sent_subs: list[tuple[float, str]] = [
        _score_recommendation(sentiment.get("analyst_recommendation")),
        _score_price_target_upside(
            sentiment.get("current_price"),
            sentiment.get("analyst_target_mean_price"),
        ),
        _score_short_float(sentiment.get("short_percent_of_float")),
    ]
    sent_score = _avg(sent_subs)
    all_signals.extend(sent_subs)

    # ---- Weighted overall score ----
    weights = {
        "valuation":    config.weight_valuation,
        "risk":         config.weight_risk,
        "momentum":     config.weight_momentum,
        "fundamentals": config.weight_fundamentals,
        "returns":      config.weight_returns,
        "sentiment":    config.weight_sentiment,
    }
    cat_raw = {
        "valuation":    (val_score,  val_signals),
        "risk":         (risk_score, risk_subs),
        "momentum":     (mom_score,  mom_subs),
        "fundamentals": (fund_score, fund_subs),
        "returns":      (ret_score,  ret_subs),
        "sentiment":    (sent_score, sent_subs),
    }
    category_scores: dict = {}
    overall_raw = 0.0
    for cat, (score, sigs) in cat_raw.items():
        w = weights[cat]
        ws = round(score * w, 4)
        overall_raw += ws
        category_scores[cat] = {
            "score": round(score, 2),
            "weight": w,
            "weighted_score": ws,
            "signals": [lbl for _, lbl in sigs],
        }

    overall_score = round(overall_raw * 10, 2)  # scale to 0-100

    # Top 3 strengths / concerns across all signals
    ranked = sorted(all_signals, key=lambda x: x[0], reverse=True)
    top_strengths = [lbl for _, lbl in ranked[:3]]
    top_concerns  = [lbl for _, lbl in ranked[-3:][::-1]]

    return {
        "overall_score": overall_score,
        "grade": score_to_grade(overall_score),
        "category_scores": category_scores,
        "top_strengths": top_strengths,
        "top_concerns": top_concerns,
    }


# ---------------------------------------------------------------------------
# ADK tool — reads from session state, calls _compute_composite_score
# ---------------------------------------------------------------------------

def compute_composite_score(tool_context: ToolContext) -> dict:
    """
    Read all category metrics from session state and compute the composite
    health score. Writes the full score breakdown to state[SCORE_BREAKDOWN].
    Returns a dict with overall_score, grade, category_scores, top_strengths,
    and top_concerns.
    """
    state = tool_context.state
    valuation_data = _parse(state.get("valuation_metrics"))
    result = _compute_composite_score(
        returns=_parse(state.get("returns_metrics")),
        risk=_parse(state.get("risk_metrics")),
        valuation=valuation_data,
        momentum=_parse(state.get("momentum_metrics")),
        fundamentals=_parse(state.get("fundamentals_metrics")),
        sentiment=_parse(state.get("sentiment_metrics")),
    )
    result["company_name"] = (valuation_data or {}).get("company_name") or state.get(TICKER, "")
    result["ticker"] = state.get(TICKER, "")
    state[SCORE_BREAKDOWN] = result["category_scores"]
    logger.info(
        "Composite score computed: %.1f (%s)",
        result["overall_score"], result["grade"],
    )
    return result


compute_composite_score_tool = FunctionTool(compute_composite_score)
