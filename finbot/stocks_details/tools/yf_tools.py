"""
Extra tools for the stocks_details agent.

Provides:
  - get_price_snapshot  — summary stats for a timeframe (NOT raw arrays)
  - log_answer          — records Q&A pairs to session state
  + re-exports all shared yfinance tools from stock_health
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import diskcache
import yfinance as yf
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from finbot.config import config
from finbot.stock_health.tools.yf_tools import (
    fundamental_score_inputs_tool,
    news_headlines_tool,
    returns_metrics_tool,
    risk_metrics_tool,
    stock_info_tool,
    technical_indicators_tool,
)

logger = logging.getLogger(__name__)

_cache = diskcache.Cache("./.finbot_cache")
_PRICE_TTL = config.price_cache_ttl


def _sf(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Price snapshot — summary for a timeframe, never raw arrays
# ---------------------------------------------------------------------------

def _sync_get_price_snapshot(ticker: str, period: str = "1mo") -> dict:
    key = f"get_price_snapshot:{ticker}:{period}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df.empty:
            return {"error": f"No price data for {ticker!r} period={period!r}"}

        close = df["Close"]
        volume = df["Volume"]
        start = _sf(close.iloc[0])
        end = _sf(close.iloc[-1])

        result = {
            "ticker": ticker,
            "period": period,
            "start_date": df.index[0].date().isoformat(),
            "end_date": df.index[-1].date().isoformat(),
            "num_trading_days": len(df),
            "start_price": start,
            "end_price": end,
            "period_high": _sf(close.max()),
            "period_low": _sf(close.min()),
            "period_return_pct": _sf((end / start - 1) * 100) if start else None,
            "avg_daily_volume": _sf(volume.mean()),
            "total_volume": int(volume.sum()),
        }
        _cache.set(key, result, expire=_PRICE_TTL)
        return result
    except Exception as exc:
        logger.error("get_price_snapshot(%r, %r) failed: %s", ticker, period, exc)
        return {"error": str(exc)}


async def get_price_snapshot(ticker: str, period: str = "1mo") -> dict:
    """
    Return a price and volume summary for a specific timeframe.

    period options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd'

    Use this for ALL price or volume questions. Never return raw price arrays.
    """
    return await asyncio.to_thread(_sync_get_price_snapshot, ticker, period)


price_snapshot_tool = FunctionTool(get_price_snapshot)


# ---------------------------------------------------------------------------
# Q&A logger — writes each exchange to session state
# ---------------------------------------------------------------------------

def log_answer(question: str, answer: str, tool_context: ToolContext) -> str:
    """Record a question and answer pair in shared session state under 'qa_history'."""
    history = tool_context.state.get("qa_history") or []
    history.append({"question": question, "answer": answer})
    tool_context.state["qa_history"] = history
    return "Recorded."


log_answer_tool = FunctionTool(log_answer)


__all__ = [
    "stock_info_tool",
    "returns_metrics_tool",
    "risk_metrics_tool",
    "technical_indicators_tool",
    "news_headlines_tool",
    "fundamental_score_inputs_tool",
    "price_snapshot_tool",
    "log_answer_tool",
]
