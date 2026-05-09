import os
from dataclasses import dataclass, field


@dataclass
class Config:
    model_orchestrator: str = field(
        default_factory=lambda: os.environ.get("MODEL_ORCHESTRATOR", "gemini-2.5-pro")
    )
    model_analyst: str = field(
        default_factory=lambda: os.environ.get("MODEL_ANALYST", "gemini-2.5-flash")
    )

    risk_free_rate: float = field(
        default_factory=lambda: float(os.environ.get("RISK_FREE_RATE", "0.045"))
    )
    benchmark_ticker: str = field(
        default_factory=lambda: os.environ.get("BENCHMARK_TICKER", "^GSPC")
    )

    bigquery_dataset: str = field(
        default_factory=lambda: os.environ.get("BIGQUERY_DATASET", "finbot_portfolio")
    )
    rag_corpus_name: str = field(
        default_factory=lambda: os.environ.get("RAG_CORPUS_NAME", "")
    )

    # Scoring weights — must sum to 1.0
    weight_valuation: float = 0.25
    weight_fundamentals: float = 0.25
    weight_risk: float = 0.15
    weight_momentum: float = 0.15
    weight_sentiment: float = 0.10
    weight_returns: float = 0.10

    # yfinance cache TTLs in seconds
    price_cache_ttl: int = 900
    fundamentals_cache_ttl: int = 86_400


config = Config()
