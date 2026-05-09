"""
yfinance tool layer for the finbot stock-health agent.

All math lives here. LLM prompts receive pre-computed values only.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# pandas_ta 0.4.x requires numba, which doesn't build on Python 3.14.
# Stub it before the import so pandas_ta falls back to pure Python.
if "numba" not in sys.modules:
    from unittest.mock import MagicMock as _MagicMock
    _numba = _MagicMock()
    _numba.njit = lambda f=None, **kw: (lambda g: g) if f is None else f
    _numba.prange = range
    sys.modules["numba"] = _numba

import diskcache
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from scipy import stats
from google.adk.tools import FunctionTool

from finbot.config import config

logger = logging.getLogger(__name__)

_cache = diskcache.Cache("./.finbot_cache")

_PRICE_TTL = config.price_cache_ttl
_FUND_TTL = config.fundamentals_cache_ttl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sf(val: Any) -> float | None:
    """Return val as a Python float, or None if missing/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if f != f else f  # NaN check: NaN != NaN
    except (TypeError, ValueError):
        return None


def _fetch_history(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data for {ticker!r} period={period!r}")
    return df


def _period_return(closes: pd.Series, n_days: int) -> float | None:
    """(last_close / close_n_days_ago) - 1, or None if insufficient data."""
    if len(closes) < n_days + 1:
        return None
    start = float(closes.iloc[-(n_days + 1)])
    return float(closes.iloc[-1] / start - 1) if start != 0 else None


# ---------------------------------------------------------------------------
# Tool 1 — Fundamental snapshot
# ---------------------------------------------------------------------------

def _sync_get_stock_info(ticker: str) -> dict:
    key = f"get_stock_info:{ticker}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        info = yf.Ticker(ticker).info
        result = {
            "company_name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("currency"),
            "current_price": _sf(info.get("currentPrice") or info.get("regularMarketPrice")),
            "market_cap": _sf(info.get("marketCap")),
            "shares_outstanding": _sf(info.get("sharesOutstanding")),
            "trailing_pe": _sf(info.get("trailingPE")),
            "forward_pe": _sf(info.get("forwardPE")),
            "peg_ratio": _sf(info.get("pegRatio")),
            "price_to_book": _sf(info.get("priceToBook")),
            "price_to_sales": _sf(info.get("priceToSalesTrailing12Months")),
            "ev_to_ebitda": _sf(info.get("enterpriseToEbitda")),
            "dividend_yield": _sf(info.get("dividendYield")),
            "payout_ratio": _sf(info.get("payoutRatio")),
            "trailing_eps": _sf(info.get("trailingEps")),
            "forward_eps": _sf(info.get("forwardEps")),
            "earnings_growth": _sf(info.get("earningsGrowth")),
            "total_revenue": _sf(info.get("totalRevenue")),
            "revenue_growth": _sf(info.get("revenueGrowth")),
            "ebitda": _sf(info.get("ebitda")),
            "total_debt": _sf(info.get("totalDebt")),
            "total_cash": _sf(info.get("totalCash")),
            "gross_margins": _sf(info.get("grossMargins")),
            "operating_margins": _sf(info.get("operatingMargins")),
            "profit_margins": _sf(info.get("profitMargins")),
            "return_on_equity": _sf(info.get("returnOnEquity")),
            "return_on_assets": _sf(info.get("returnOnAssets")),
            "debt_to_equity": _sf(info.get("debtToEquity")),
            "current_ratio": _sf(info.get("currentRatio")),
            "quick_ratio": _sf(info.get("quickRatio")),
            "free_cash_flow": _sf(info.get("freeCashflow")),
            "beta": _sf(info.get("beta")),
            "fifty_two_week_high": _sf(info.get("fiftyTwoWeekHigh")),
            "fifty_two_week_low": _sf(info.get("fiftyTwoWeekLow")),
            "analyst_recommendation": info.get("recommendationKey"),
            "analyst_target_mean_price": _sf(info.get("targetMeanPrice")),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "short_percent_of_float": _sf(info.get("shortPercentOfFloat")),
        }
        _cache.set(key, result, expire=_FUND_TTL)
        return result
    except Exception as exc:
        logger.error("get_stock_info(%r) failed: %s", ticker, exc)
        return {"error": str(exc)}


async def get_stock_info(ticker: str) -> dict:
    """Return key statistics and fundamental data for a stock ticker."""
    return await asyncio.to_thread(_sync_get_stock_info, ticker)


# ---------------------------------------------------------------------------
# Tool 2 — Price history
# ---------------------------------------------------------------------------

def _sync_get_price_history(ticker: str, period: str = "2y") -> dict:
    key = f"get_price_history:{ticker}:{period}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        df = _fetch_history(ticker, period)
        result = {
            "dates": [d.isoformat() for d in df.index],
            "closes": [float(v) for v in df["Close"]],
            "volumes": [int(v) for v in df["Volume"]],
            "highs": [float(v) for v in df["High"]],
            "lows": [float(v) for v in df["Low"]],
        }
        _cache.set(key, result, expire=_PRICE_TTL)
        return result
    except Exception as exc:
        logger.error("get_price_history(%r, %r) failed: %s", ticker, period, exc)
        return {"error": str(exc)}


async def get_price_history(ticker: str, period: str = "2y") -> dict:
    """Return OHLCV time-series arrays for a ticker."""
    return await asyncio.to_thread(_sync_get_price_history, ticker, period)


# ---------------------------------------------------------------------------
# Tool 3 — Returns metrics
# ---------------------------------------------------------------------------

def _sync_get_returns_metrics(ticker: str, benchmark_ticker: str = "^GSPC") -> dict:
    key = f"get_returns_metrics:{ticker}:{benchmark_ticker}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        df = _fetch_history(ticker, "5y")
        closes = df["Close"]

        # YTD: find the first close of the current calendar year
        now_year = datetime.now(timezone.utc).year
        ytd_mask = df.index.year == now_year
        ytd_return: float | None = None
        if ytd_mask.any():
            ytd_start = float(closes[ytd_mask].iloc[0])
            if ytd_start != 0:
                ytd_return = float(closes.iloc[-1] / ytd_start - 1)

        # CAGR helper: use ~252 trading days per year
        end_price = float(closes.iloc[-1])

        def _cagr(n_years: int) -> float | None:
            n_days = int(n_years * 252)
            if len(closes) < n_days + 1:
                return None
            start = float(closes.iloc[-(n_days + 1)])
            if start <= 0:
                return None
            return float((end_price / start) ** (1.0 / n_years) - 1)

        # Alpha / beta vs benchmark using 1-year of daily returns
        bench_df = yf.Ticker(benchmark_ticker).history(period="1y", auto_adjust=True)
        stock_df_1y = _fetch_history(ticker, "1y")

        alpha_1y: float | None = None
        beta_computed_1y: float | None = None
        if not bench_df.empty and not stock_df_1y.empty:
            stock_ret = stock_df_1y["Close"].pct_change().dropna()
            bench_ret = bench_df["Close"].pct_change().dropna()
            common = stock_ret.index.intersection(bench_ret.index)
            if len(common) > 30:
                s = stock_ret.loc[common].to_numpy()
                b = bench_ret.loc[common].to_numpy()
                slope, intercept, *_ = stats.linregress(b, s)
                beta_computed_1y = float(slope)
                alpha_1y = float(intercept * 252)  # annualized

        # 52-week high/low vs current price
        window = closes.iloc[-252:] if len(closes) >= 252 else closes
        high_52 = float(window.max())
        low_52 = float(window.min())
        current = float(closes.iloc[-1])
        high_pct = (current / high_52 - 1) if high_52 != 0 else None
        low_pct = (current / low_52 - 1) if low_52 != 0 else None

        result = {
            "return_1d": _period_return(closes, 1),
            "return_1w": _period_return(closes, 5),
            "return_1m": _period_return(closes, 21),
            "return_3m": _period_return(closes, 63),
            "return_6m": _period_return(closes, 126),
            "return_ytd": ytd_return,
            "return_1y": _period_return(closes, 252),
            "return_3y": _period_return(closes, 756),
            "return_5y": _period_return(closes, 1260),
            "cagr_3y": _cagr(3),
            "cagr_5y": _cagr(5),
            "alpha_1y": alpha_1y,
            "beta_computed_1y": beta_computed_1y,
            "fiftytwo_week_high_pct": high_pct,
            "fiftytwo_week_low_pct": low_pct,
        }
        _cache.set(key, result, expire=_PRICE_TTL)
        return result
    except Exception as exc:
        logger.error("get_returns_metrics(%r) failed: %s", ticker, exc)
        return {"error": str(exc)}


async def get_returns_metrics(ticker: str, benchmark_ticker: str = "^GSPC") -> dict:
    """Compute period returns, CAGR, and alpha/beta versus a benchmark."""
    return await asyncio.to_thread(_sync_get_returns_metrics, ticker, benchmark_ticker)


# ---------------------------------------------------------------------------
# Tool 4 — Risk metrics
# ---------------------------------------------------------------------------

def _sync_get_risk_metrics(ticker: str) -> dict:
    key = f"get_risk_metrics:{ticker}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        df = _fetch_history(ticker, "2y")
        daily_ret = df["Close"].pct_change().dropna()

        annual_vol = float(daily_ret.std() * np.sqrt(252))
        annual_ret = float((1 + daily_ret.mean()) ** 252 - 1)
        rfr = config.risk_free_rate

        sharpe: float | None = None
        if annual_vol != 0:
            sharpe = float((annual_ret - rfr) / annual_vol)

        downside = daily_ret[daily_ret < 0]
        sortino: float | None = None
        if len(downside) > 1:
            down_vol = float(downside.std() * np.sqrt(252))
            if down_vol != 0:
                sortino = float((annual_ret - rfr) / down_vol)

        roll_max = df["Close"].cummax()
        drawdown = (df["Close"] / roll_max) - 1
        max_dd = float(drawdown.min())

        calmar: float | None = None
        if max_dd != 0:
            calmar = float(annual_ret / abs(max_dd))

        var_95 = float(np.percentile(daily_ret, 5))
        var_99 = float(np.percentile(daily_ret, 1))

        result = {
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "var_95": var_95,
            "var_99": var_99,
        }
        _cache.set(key, result, expire=_PRICE_TTL)
        return result
    except Exception as exc:
        logger.error("get_risk_metrics(%r) failed: %s", ticker, exc)
        return {"error": str(exc)}


async def get_risk_metrics(ticker: str) -> dict:
    """Compute volatility and risk-adjusted return metrics using 2y of data."""
    return await asyncio.to_thread(_sync_get_risk_metrics, ticker)


# ---------------------------------------------------------------------------
# Tool 5 — Technical indicators
# ---------------------------------------------------------------------------

def _first_col(df: pd.DataFrame, prefix: str) -> str | None:
    matches = [c for c in df.columns if c.startswith(prefix)]
    return matches[0] if matches else None


def _sync_get_technical_indicators(ticker: str) -> dict:
    key = f"get_technical_indicators:{ticker}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        df = _fetch_history(ticker, "1y")
        close = df["Close"]

        # RSI
        rsi_series = ta.rsi(close, length=14)
        rsi_val = _sf(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None
        if rsi_val is None:
            rsi_signal = "neutral"
        elif rsi_val > 70:
            rsi_signal = "overbought"
        elif rsi_val < 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"

        # MACD
        macd_df = ta.macd(close)
        macd_line = macd_signal_val = macd_hist = None
        macd_crossover = "neutral"
        if macd_df is not None and not macd_df.empty:
            if col := _first_col(macd_df, "MACD_"):
                macd_line = _sf(macd_df[col].iloc[-1])
            if col := _first_col(macd_df, "MACDs_"):
                macd_signal_val = _sf(macd_df[col].iloc[-1])
            if col := _first_col(macd_df, "MACDh_"):
                macd_hist = _sf(macd_df[col].iloc[-1])
            if macd_line is not None and macd_signal_val is not None:
                if macd_line > macd_signal_val:
                    macd_crossover = "bullish"
                elif macd_line < macd_signal_val:
                    macd_crossover = "bearish"

        # Bollinger Bands (20, 2)
        bb_df = ta.bbands(close, length=20, std=2)
        bb_upper = bb_lower = bb_pos = None
        if bb_df is not None and not bb_df.empty:
            if col := _first_col(bb_df, "BBU_"):
                bb_upper = _sf(bb_df[col].iloc[-1])
            if col := _first_col(bb_df, "BBL_"):
                bb_lower = _sf(bb_df[col].iloc[-1])
            cur = _sf(close.iloc[-1])
            if bb_upper is not None and bb_lower is not None and cur is not None:
                band_width = bb_upper - bb_lower
                if band_width != 0:
                    bb_pos = float((cur - bb_lower) / band_width)

        # SMAs
        sma50_s = ta.sma(close, length=50)
        sma200_s = ta.sma(close, length=200)
        sma_50 = _sf(sma50_s.iloc[-1]) if sma50_s is not None and not sma50_s.empty else None
        sma_200 = _sf(sma200_s.iloc[-1]) if sma200_s is not None and not sma200_s.empty else None

        cur_price = _sf(close.iloc[-1])
        price_vs_sma50 = float(cur_price / sma_50 - 1) if cur_price and sma_50 else None
        price_vs_sma200 = float(cur_price / sma_200 - 1) if cur_price and sma_200 else None

        golden_cross = bool(sma_50 > sma_200) if sma_50 is not None and sma_200 is not None else False
        death_cross = bool(sma_50 < sma_200) if sma_50 is not None and sma_200 is not None else False

        # Rate of change (10-day); pandas_ta returns as percentage, convert to decimal
        roc_s = ta.roc(close, length=10)
        roc_10: float | None = None
        if roc_s is not None and not roc_s.empty:
            raw = _sf(roc_s.iloc[-1])
            if raw is not None:
                roc_10 = raw / 100.0

        result = {
            "rsi_14": rsi_val,
            "rsi_signal": rsi_signal,
            "macd_line": macd_line,
            "macd_signal": macd_signal_val,
            "macd_histogram": macd_hist,
            "macd_crossover": macd_crossover,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "price_vs_sma50_pct": price_vs_sma50,
            "price_vs_sma200_pct": price_vs_sma200,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_position": bb_pos,
            "roc_10": roc_10,
        }
        _cache.set(key, result, expire=_PRICE_TTL)
        return result
    except Exception as exc:
        logger.error("get_technical_indicators(%r) failed: %s", ticker, exc)
        return {"error": str(exc)}


async def get_technical_indicators(ticker: str) -> dict:
    """Return RSI, MACD, Bollinger Bands, SMA, and ROC indicators."""
    return await asyncio.to_thread(_sync_get_technical_indicators, ticker)


# ---------------------------------------------------------------------------
# Tool 6 — News headlines
# ---------------------------------------------------------------------------

def _sync_get_news_headlines(ticker: str, max_items: int = 5) -> list[dict]:
    key = f"get_news_headlines:{ticker}:{max_items}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        raw_news = yf.Ticker(ticker).news or []
        result: list[dict] = []
        for item in raw_news[:max_items]:
            # yfinance ≥0.2.40 wraps data under a "content" dict
            content = item.get("content") or {}
            title = content.get("title") or item.get("title", "")
            provider = content.get("provider") or {}
            publisher = provider.get("displayName") or item.get("publisher", "")
            url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
            link = url_obj.get("url") or item.get("link", "")
            pub_time = content.get("pubDate") or ""
            if not pub_time:
                raw_ts = item.get("providerPublishTime")
                if raw_ts:
                    pub_time = datetime.fromtimestamp(raw_ts, tz=timezone.utc).isoformat()
            result.append({"title": title, "publisher": publisher, "link": link, "publish_time": pub_time})
        _cache.set(key, result, expire=_FUND_TTL)
        return result
    except Exception as exc:
        logger.error("get_news_headlines(%r) failed: %s", ticker, exc)
        return [{"error": str(exc)}]


async def get_news_headlines(ticker: str, max_items: int = 5) -> list[dict]:
    """Fetch recent news headlines for a ticker from yfinance."""
    return await asyncio.to_thread(_sync_get_news_headlines, ticker, max_items)


# ---------------------------------------------------------------------------
# Tool 7 — Fundamental score inputs
# ---------------------------------------------------------------------------

def _score_roe(val: float | None) -> int | None:
    if val is None:
        return None
    if val < 0.05: return 1
    if val < 0.10: return 4
    if val < 0.15: return 6
    if val < 0.20: return 8
    return 10


def _score_net_margin(val: float | None) -> int | None:
    if val is None:
        return None
    if val < 0.00: return 1
    if val < 0.05: return 4
    if val < 0.10: return 6
    if val < 0.20: return 8
    return 10


def _score_de(val: float | None) -> int | None:
    """Score D/E ratio. yfinance returns debtToEquity as ratio * 100."""
    if val is None:
        return None
    v = val / 100.0  # normalise to ratio form (e.g. 150 → 1.5)
    if v > 3: return 1
    if v > 2: return 3
    if v > 1: return 6
    if v > 0.5: return 8
    return 10


def _score_growth(val: float | None) -> int | None:
    if val is None:
        return None
    if val < 0.00: return 2
    if val < 0.05: return 5
    if val < 0.15: return 7
    if val < 0.30: return 9
    return 10


def _sync_get_fundamental_score_inputs(ticker: str) -> dict:
    key = f"get_fundamental_score_inputs:{ticker}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        info = yf.Ticker(ticker).info
        roe = _sf(info.get("returnOnEquity"))
        net_margin = _sf(info.get("profitMargins"))
        de = _sf(info.get("debtToEquity"))
        rev_growth = _sf(info.get("revenueGrowth"))
        eps_growth = _sf(info.get("earningsGrowth"))

        result = {
            "roe": {
                "value": roe,
                "thresholds": [0.05, 0.10, 0.15, 0.20],
                "score": _score_roe(roe),
            },
            "net_margin": {
                "value": net_margin,
                "thresholds": [0.00, 0.05, 0.10, 0.20],
                "score": _score_net_margin(net_margin),
            },
            "debt_to_equity": {
                "value": de,
                "thresholds": [0.5, 1.0, 2.0, 3.0],
                "score": _score_de(de),
            },
            "revenue_growth": {
                "value": rev_growth,
                "thresholds": [0.00, 0.05, 0.15, 0.30],
                "score": _score_growth(rev_growth),
            },
            "eps_growth": {
                "value": eps_growth,
                "thresholds": [0.00, 0.05, 0.15, 0.30],
                "score": _score_growth(eps_growth),
            },
        }
        _cache.set(key, result, expire=_FUND_TTL)
        return result
    except Exception as exc:
        logger.error("get_fundamental_score_inputs(%r) failed: %s", ticker, exc)
        return {"error": str(exc)}


async def get_fundamental_score_inputs(ticker: str) -> dict:
    """Return pre-scored fundamental KPIs for composite scoring."""
    return await asyncio.to_thread(_sync_get_fundamental_score_inputs, ticker)


# ---------------------------------------------------------------------------
# ADK tool wrappers
# ---------------------------------------------------------------------------

stock_info_tool = FunctionTool(get_stock_info)
price_history_tool = FunctionTool(get_price_history)
returns_metrics_tool = FunctionTool(get_returns_metrics)
risk_metrics_tool = FunctionTool(get_risk_metrics)
technical_indicators_tool = FunctionTool(get_technical_indicators)
news_headlines_tool = FunctionTool(get_news_headlines)
fundamental_score_inputs_tool = FunctionTool(get_fundamental_score_inputs)
