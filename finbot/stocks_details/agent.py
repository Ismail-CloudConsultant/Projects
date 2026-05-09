"""
Stock Details Agent — answers direct metric questions about individual stocks.
"""

from google.adk.agents import LlmAgent

from finbot.config import config
from finbot.shared.callbacks import disclaimer_callback
from finbot.stocks_details.prompts import STOCK_DETAILS_INSTRUCTION
from finbot.stocks_details.tools.search_tools import search_news_tool
from finbot.stocks_details.tools.yf_tools import (
    fundamental_score_inputs_tool,
    log_answer_tool,
    news_headlines_tool,
    price_snapshot_tool,
    returns_metrics_tool,
    risk_metrics_tool,
    stock_info_tool,
    technical_indicators_tool,
)

stock_details_agent = LlmAgent(
    name="stock_details_agent",
    model=config.model_analyst,
    description=(
        "Answers direct questions about a stock's specific metrics, indicators, "
        "and financial data. Records each Q&A pair in session state."
    ),
    instruction=STOCK_DETAILS_INSTRUCTION,
    tools=[
        stock_info_tool,
        price_snapshot_tool,
        returns_metrics_tool,
        risk_metrics_tool,
        technical_indicators_tool,
        news_headlines_tool,
        fundamental_score_inputs_tool,
        search_news_tool,
        log_answer_tool,
    ],
    after_model_callback=disclaimer_callback,
)

root_agent = stock_details_agent
