from google.adk.tools import FunctionTool


def search_company_news(ticker: str, max_results: int = 5) -> dict:
    """Search for recent news headlines for a ticker to assess market sentiment."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    news = t.news or []
    items = [
        {
            "title": n.get("content", {}).get("title", ""),
            "publisher": n.get("content", {}).get("provider", {}).get("displayName", ""),
            "published": n.get("content", {}).get("pubDate", ""),
        }
        for n in news[:max_results]
    ]
    return {"ticker": ticker, "news": items}


search_news_tool = FunctionTool(search_company_news)
