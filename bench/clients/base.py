"""Provider-neutral client interface.

A client receives a state and one question in the canonical question format and returns a `CallResult`
whose `answer` is in the canonical answer format:

    choice: {"type": "choice", "choice": key, "probabilities": {key: p}, "confidence": float | None}
    score:  {"type": "score", "probabilities": {"0": p, "1": p, ...}, "score": float | None,
             "confidence": float | None}
    binary: {"type": "binary", "p": float}   # probability that the answer is yes

Canonical question format (built by bench.questions.to_request):

    {"type": "choice", "instructions": str, "criteria": {key: description}}
    {"type": "score",  "instructions": str, "criteria": [level descriptions, ordered]}
    {"type": "binary", "instructions": str, "criteria": {"true": str, "false": str}}

Providers translate to and from their own wire formats inside their client.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from bench.config import MAX_RETRIES


class ProviderError(RuntimeError):
    """Non-retryable failure, or retries exhausted. Recorded as an error on the run record."""


@dataclass
class CallResult:
    answer: dict  # canonical answer, keyed by displayed option keys
    raw_response: dict
    returned_model: str | None
    latency_ms: float  # client-observed, successful attempt only
    total_elapsed_ms: float  # including retries and backoff
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
    errors: list[str] = field(default_factory=list)


class Client(Protocol):
    model: str

    def ask(self, state: Any, question: dict) -> CallResult: ...

    def close(self) -> None: ...


def backoff_delay(attempt: int, retry_after: str | None = None) -> float:
    """Exponential backoff with jitter, capped at 30 s; honors a numeric retry-after header."""
    delay = min(30.0, 2**attempt) + random.uniform(0, 0.5)
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except ValueError:
            pass
    return delay


def sleep_before_retry(attempt: int, retry_after: str | None = None, sleep=time.sleep) -> None:
    if attempt < MAX_RETRIES:
        sleep(backoff_delay(attempt, retry_after))
