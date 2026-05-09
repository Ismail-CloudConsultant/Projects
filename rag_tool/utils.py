import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Errors that won't resolve on retry — raise immediately.
_NON_RETRIABLE = (ValueError, TypeError, FileNotFoundError, AttributeError)


def retry_with_backoff(fn: Callable[[], T], *, max_retries: int = 3, base_delay: float = 1.0) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except _NON_RETRIABLE:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2**attempt)
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs.",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
