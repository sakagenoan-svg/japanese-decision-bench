"""Deterministic offline client returning canonical answers. For tests and pipeline checks only."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from bench.clients.base import CallResult


def _rng_floats(seed: str, n: int) -> list[float]:
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    return [(h[i] + 1) / 257 for i in range(n)]


class MockClient:
    def __init__(self, model: str = "mock-0"):
        self.model = model

    def close(self) -> None:
        pass

    def ask(self, state: Any, question: dict) -> CallResult:
        seed = json.dumps([state, question], ensure_ascii=False, sort_keys=True)
        qtype = question["type"]
        if qtype == "choice":
            keys = list(question["criteria"])
            w = _rng_floats(seed, len(keys))
            probs = {k: v / sum(w) for k, v in zip(keys, w, strict=True)}
            best = max(probs, key=probs.get)
            answer = {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}
        elif qtype == "score":
            w = _rng_floats(seed, len(question["criteria"]))
            probs = {str(i): v / sum(w) for i, v in enumerate(w)}
            answer = {"type": "score", "probabilities": probs, "score": None, "confidence": max(probs.values())}
        else:
            answer = {"type": "binary", "p": _rng_floats(seed, 1)[0]}
        raw = {"model": self.model, "answer": answer, "usage": {"input_tokens": 100, "output_tokens": 0}}
        return CallResult(answer=answer, raw_response=raw, returned_model=self.model, latency_ms=1.0,
                          total_elapsed_ms=1.0, retry_count=0, input_tokens=100, output_tokens=0)
