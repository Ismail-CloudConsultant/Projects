from google.adk.agents import LlmAgent
from dotenv import load_dotenv
from .config import config
from .explainer.agent import explainer_agent
from .portfolio.agent import portfolio_agent
from .shared.callbacks import disclaimer_callback
from .stock_health.agent import stock_health_agent
from .stocks_details import stock_details_agent

root_agent = LlmAgent(
    name="finbot_root",
    model=config.model_orchestrator,
    description="Root orchestrator for Finbot — routes user requests to specialist agents.",
    instruction=(
        "You are Finbot, an AI-powered financial analysis assistant. "
        "You have three specialist agents which you can use, dont search over internet for result only use the below tools agents "
        "(1) Stock Health Agent — analyzes stocks across valuation, risk, momentum, "
        "fundamentals, and sentiment; "
        "(2) stock_details_agent - Answers questions about stock's metrics, indicators, and financial data "
        "(3) Metrics Explainer — explains what any financial metric means in simple terms "
        "with examples; "
        "(4) Portfolio Manager — tracks the user's holdings and calculates profit/loss. "
        
        "Greet the user, understand their request, and transfer to the right specialist."
    ),
    sub_agents=[stock_health_agent, explainer_agent, portfolio_agent,stock_details_agent],
    after_model_callback=disclaimer_callback,
)
