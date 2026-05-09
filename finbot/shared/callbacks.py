from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

_DISCLAIMER = (
    "\n\n⚠️ This is not financial advice -  ismail's POC project for Google - ADK"
)
_TRIGGER_WORDS = {"buy", "sell", "invest", "recommend"}
_DANGEROUS_SQL = {"DELETE", "DROP"}
_CONFIRM_TOKEN = "CONFIRM DELETE"


def disclaimer_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    if not (llm_response.content and llm_response.content.parts):
        return None

    full_text = "".join(
        p.text
        for p in llm_response.content.parts
        if hasattr(p, "text") and p.text
    )
    if not any(word in full_text.lower() for word in _TRIGGER_WORDS):
        return None

    new_parts = list(llm_response.content.parts) + [types.Part(text=_DISCLAIMER)]
    return LlmResponse(
        content=types.Content(role="model", parts=new_parts)
    )


def confirmation_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Optional[dict]:
    sql = args.get("query", "") or args.get("sql", "")
    sql_upper = sql.upper()
    if any(kw in sql_upper for kw in _DANGEROUS_SQL):
        return {
            "error": (
                f"Destructive SQL detected ({', '.join(_DANGEROUS_SQL & set(sql_upper.split()))})."
                f' To proceed, re-send your request with the exact phrase "{_CONFIRM_TOKEN}".'
            )
        }
    return None
