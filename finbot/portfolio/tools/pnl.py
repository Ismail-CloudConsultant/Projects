import asyncio

from google.adk.tools import FunctionTool


def _sync_calculate_pnl(holdings: list[dict]) -> dict:
    import yfinance as yf

    tickers = list({h["ticker"] for h in holdings})
    prices = {}
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        prices[ticker] = info.get("currentPrice") or info.get("regularMarketPrice", 0.0)

    results = []
    total_cost = 0.0
    total_value = 0.0
    for h in holdings:
        ticker = h["ticker"]
        shares = float(h["shares"])
        cost_per_share = float(h["cost_basis"])
        current_price = float(prices.get(ticker, 0.0))
        cost = shares * cost_per_share
        value = shares * current_price
        pnl = value - cost
        pct = (pnl / cost * 100) if cost else 0.0
        results.append(
            {
                "ticker": ticker,
                "shares": shares,
                "cost_basis": cost_per_share,
                "current_price": round(current_price, 4),
                "cost": round(cost, 2),
                "market_value": round(value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pct, 2),
            }
        )
        total_cost += cost
        total_value += value

    total_pnl = total_value - total_cost
    return {
        "holdings": results,
        "total_cost": round(total_cost, 2),
        "total_market_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round((total_pnl / total_cost * 100) if total_cost else 0.0, 2),
    }


async def calculate_pnl(holdings: list[dict]) -> dict:
    """Calculate current profit/loss for a list of holdings using live prices.

    Each holding dict must have: ticker, shares, cost_basis (per share).
    """
    return await asyncio.to_thread(_sync_calculate_pnl, holdings)


pnl_tool = FunctionTool(calculate_pnl)
