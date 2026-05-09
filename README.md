# Finbot — AI-Powered Financial Advisor

Multi-agent financial analysis assistant built with Google Agent Development Kit (ADK).

## Agents

| Agent | Role |
|---|---|
| `finbot_root` | Orchestrator — routes to specialists |
| `stock_health` | Scores stocks across valuation, fundamentals, risk, momentum, sentiment |
| `explainer` | Explains financial metrics in plain language |
| `portfolio` | Tracks holdings and calculates P&L |

## Quick Start

```bash
cp .env.example .env
# fill in your GCP credentials
uv run adk web finbot
```

## Development

```bash
uv run pytest tests/
uv run ruff check finbot/
```
