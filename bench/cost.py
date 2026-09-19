"""Cost estimate from *public* list prices only (provider `price` entries).

Never put contract prices, discounts or credit terms in a price entry. Prices must be re-checked on the run
date and `checked_at` updated.
"""

from __future__ import annotations

import json
from pathlib import Path


def estimate(model: str, input_tokens: int, output_tokens: int, n_requests: int) -> dict:
    from bench.config import TRANSLATOR_MODEL, TRANSLATOR_PRICE
    from bench.providers import price_of

    price = price_of(model)
    if price is None and model == TRANSLATOR_MODEL:
        price = {"provider": "anthropic", **TRANSLATOR_PRICE}
    if price is None:
        return {"model": model, "estimated_cost": None, "note": "no public price recorded"}
    cost = input_tokens / 1e6 * price["input_per_mtok"] + output_tokens / 1e6 * price["output_per_mtok"]
    return {
        "provider": price["provider"],
        "model": model,
        "public_price_checked_at": price["checked_at"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "requests": n_requests,
        "estimated_cost": cost,
        "estimated_cost_per_1000": cost / n_requests * 1000 if n_requests else None,
    }


def estimate_records(records: list[dict]) -> list[dict]:
    by_model: dict[str, dict] = {}
    for r in records:
        if r.get("error") is not None:
            continue
        agg = by_model.setdefault(r["requested_model"], {"in": 0, "out": 0, "n": 0})
        agg["in"] += r.get("input_tokens") or 0
        agg["out"] += r.get("output_tokens") or 0
        agg["n"] += 1
    return [estimate(m, a["in"], a["out"], a["n"]) for m, a in sorted(by_model.items())]


def write_estimate(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimate_records(records), indent=2), encoding="utf-8", newline="\n")
