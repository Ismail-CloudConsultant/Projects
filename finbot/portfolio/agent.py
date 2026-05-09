from google.adk.agents import LlmAgent

from finbot.config import config
from finbot.portfolio.prompts import PORTFOLIO_INSTRUCTION
from finbot.portfolio.tools.pnl import pnl_tool
from finbot.shared.callbacks import confirmation_callback, disclaimer_callback

portfolio_agent = LlmAgent(
    name="portfolio_agent",
    model=config.model_analyst,
    description=(
        "Tracks the user's stock holdings and calculates profit/loss. "
        "Use when the user wants to add, remove, or review their portfolio positions."
    ),
    instruction=PORTFOLIO_INSTRUCTION,
    tools=[pnl_tool],
    after_model_callback=disclaimer_callback,
    before_tool_callback=confirmation_callback,
)
