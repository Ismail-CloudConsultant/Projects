from google.adk.agents import LlmAgent

from finbot.config import config
from finbot.explainer.prompts import EXPLAINER_INSTRUCTION
from finbot.explainer.tools.rag import rag_explain_tool

explainer_agent = LlmAgent(
    name="explainer_agent",
    model=config.model_analyst,
    description=(
        "Explains any financial metric or concept in plain language with real-world examples. "
        "Use when the user asks what a term or ratio means."
    ),
    instruction=EXPLAINER_INSTRUCTION,
    tools=[rag_explain_tool],
)
