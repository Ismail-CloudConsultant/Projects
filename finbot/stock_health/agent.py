"""
Stock Health Agent — multi-agent pipeline for equity analysis.
"""

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from finbot.config import config
from finbot.shared.callbacks import disclaimer_callback
from finbot.shared.state_keys import (
    COMPOSITE_SCORE,
    SCORE_BREAKDOWN,
    TICKER,
)
from finbot.stock_health.scoring import compute_composite_score_tool
from finbot.stock_health.tools.search_tools import search_news_tool
from finbot.stock_health.tools.yf_tools import (
    fundamental_score_inputs_tool,
    news_headlines_tool,
    price_history_tool,
    returns_metrics_tool,
    risk_metrics_tool,
    stock_info_tool,
    technical_indicators_tool,
)


# ---------------------------------------------------------------------------
# Utility tool — writes ticker to session state
# ---------------------------------------------------------------------------

def save_ticker(ticker: str, tool_context: ToolContext) -> str:
    """Save the stock ticker symbol to session state for use by analyst sub-agents."""
    clean = ticker.upper().strip()
    tool_context.state[TICKER] = clean
    return f"Ticker saved: {clean}"


save_ticker_tool = FunctionTool(save_ticker)


# ---------------------------------------------------------------------------
# Metric analyst sub-agents (run in parallel)
# ---------------------------------------------------------------------------

_MODEL = config.model_analyst

returns_agent = LlmAgent(
    name="returns_analyst",
    model=_MODEL,
    description="Retrieves price returns and benchmark performance metrics.",
    instruction=(
        "You are the returns analyst. The stock ticker to analyze is: {ticker}.\n"
        "Call get_stock_info and get_returns_metrics for this ticker.\n"
        "Output ONLY a valid JSON object containing the raw metric values from "
        "get_returns_metrics plus 'current_price' from get_stock_info. "
        "No commentary, no markdown — just the JSON object."
    ),
    tools=[stock_info_tool, returns_metrics_tool],
    output_key="returns_metrics",
)

risk_agent = LlmAgent(
    name="risk_analyst",
    model=_MODEL,
    description="Retrieves volatility, drawdown, and risk-adjusted return metrics.",
    instruction=(
        "You are the risk analyst. The stock ticker to analyze is: {ticker}.\n"
        "Call get_risk_metrics for this ticker.\n"
        "Output ONLY a valid JSON object with the raw values from get_risk_metrics. "
        "No commentary, no markdown — just the JSON object."
    ),
    tools=[risk_metrics_tool, price_history_tool],
    output_key="risk_metrics",
)

valuation_agent = LlmAgent(
    name="valuation_analyst",
    model=_MODEL,
    description="Retrieves valuation ratios and fundamental pricing metrics.",
    instruction=(
        "You are the valuation analyst. The stock ticker to analyze is: {ticker}.\n"
        "Call get_stock_info for this ticker.\n"
        "Output ONLY a valid JSON object with these fields extracted from the result: "
        "company_name, trailing_pe, forward_pe, peg_ratio, price_to_book, price_to_sales, "
        "ev_to_ebitda, dividend_yield, current_price, sector. "
        "No commentary, no markdown — just the JSON object."
    ),
    tools=[stock_info_tool],
    output_key="valuation_metrics",
)

momentum_agent = LlmAgent(
    name="momentum_analyst",
    model=_MODEL,
    description="Retrieves technical indicators and price momentum signals.",
    instruction=(
        "You are the momentum analyst. The stock ticker to analyze is: {ticker}.\n"
        "Call get_technical_indicators for this ticker.\n"
        "Output ONLY a valid JSON object with all fields from get_technical_indicators. "
        "No commentary, no markdown — just the JSON object."
    ),
    tools=[technical_indicators_tool],
    output_key="momentum_metrics",
)

fundamentals_agent = LlmAgent(
    name="fundamentals_analyst",
    model=_MODEL,
    description="Retrieves pre-scored fundamental health metrics.",
    instruction=(
        "You are the fundamentals analyst. The stock ticker to analyze is: {ticker}.\n"
        "Call get_fundamental_score_inputs for this ticker.\n"
        "Output ONLY a valid JSON object with all fields from get_fundamental_score_inputs. "
        "No commentary, no markdown — just the JSON object."
    ),
    tools=[fundamental_score_inputs_tool],
    output_key="fundamentals_metrics",
)

sentiment_agent = LlmAgent(
    name="sentiment_analyst",
    model=_MODEL,
    description="Retrieves analyst ratings and recent news sentiment.",
    instruction=(
        "You are the sentiment analyst. The stock ticker to analyze is: {ticker}.\n"
        "Call get_stock_info and get_news_headlines (max_items=5) for this ticker.\n"
        "Output ONLY a valid JSON object with these fields: "
        "analyst_recommendation, analyst_target_mean_price, analyst_count, "
        "short_percent_of_float, current_price, and a 'news' array where each "
        "element has 'title' and 'publish_time'. "
        "No commentary, no markdown — just the JSON object."
    ),
    tools=[stock_info_tool, news_headlines_tool],
    output_key="sentiment_metrics",
)


# ---------------------------------------------------------------------------
# Parallel gatherer
# ---------------------------------------------------------------------------

metric_gatherer = ParallelAgent(
    name="metric_gatherer",
    description="Runs all six analyst agents in parallel to gather stock metrics.",
    sub_agents=[
        returns_agent,
        risk_agent,
        valuation_agent,
        momentum_agent,
        fundamentals_agent,
        sentiment_agent,
    ],
)


# ---------------------------------------------------------------------------
# Scoring agent
# ---------------------------------------------------------------------------

scoring_agent = LlmAgent(
    name="composite_scorer",
    model=_MODEL,
    description="Computes the composite health score from gathered metrics.",
    instruction=(
        "You are the composite scorer. "
        "Call compute_composite_score — it takes no arguments and reads all "
        "category metrics from session state automatically.\n\n"
        "After receiving the result, format your response EXACTLY as follows "
        "(fill in the real values from the tool result):\n\n"
        "<company_name> (<ticker>)\n\n"
        "Overall Score: <overall_score>/100 (Grade: <grade>)\n\n"
        "Category Breakdown:\n\n"
        "Valuation (Score: <score>/10)\n"
        "Key Signals: <comma-separated signals for this category>\n\n"
        "Risk (Score: <score>/10)\n"
        "Key Signals: <comma-separated signals for this category>\n\n"
        "Momentum (Score: <score>/10)\n"
        "Key Signals: <comma-separated signals for this category>\n\n"
        "Fundamentals (Score: <score>/10)\n"
        "Key Signals: <comma-separated signals for this category>\n\n"
        "Returns (Score: <score>/10)\n"
        "Key Signals: <comma-separated signals for this category>\n\n"
        "Sentiment (Score: <score>/10)\n"
        "Key Signals: <comma-separated signals for this category>\n\n"
        "Top 3 Strengths:\n"
        "- <strength 1>\n"
        "- <strength 2>\n"
        "- <strength 3>\n\n"
        "Top 3 Concerns:\n"
        "- <concern 1>\n"
        "- <concern 2>\n"
        "- <concern 3>\n\n"
        "Output nothing else — no introductory sentence, no closing remarks."
    ),
    tools=[compute_composite_score_tool],
    output_key=COMPOSITE_SCORE,
)


# ---------------------------------------------------------------------------
# Sequential pipeline
# ---------------------------------------------------------------------------

analysis_pipeline = SequentialAgent(
    name="analysis_pipeline",
    description="Runs metric gathering and scoring in sequence.",
    sub_agents=[metric_gatherer, scoring_agent],
)


# ---------------------------------------------------------------------------
# Top-level Stock Health Agent
# ---------------------------------------------------------------------------

stock_health_agent = LlmAgent(
    name="stock_health_agent",
    model=_MODEL,
    description=(
        "Analyzes a stock's financial health and produces a composite score across "
        "valuation, fundamentals, risk, momentum, returns, and sentiment dimensions."
    ),
    instruction=(
        "You are the Stock Health Agent. When the user asks to analyze a stock:\n"
        "1. Extract the ticker symbol from their message.\n"
        "2. Call save_ticker with the ticker to save it to session state.\n"
        "3. Transfer to analysis_pipeline to run the full analysis.\n"
        "4. After the pipeline completes, present the analysis — paste the full formatted "
        "output produced by the composite_scorer exactly as-is, without any wrapper sentence.\n\n"
        "Scores are for informational purposes only and should not be considered "
        "financial advice."
    ),
    tools=[save_ticker_tool],
    sub_agents=[analysis_pipeline],
    after_model_callback=disclaimer_callback,
)

# Export for adk web
root_agent = stock_health_agent
