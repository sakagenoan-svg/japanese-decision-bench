"""Metric definitions (frozen by protocol-v1). Pure Python, no numpy, so they are easy to audit.

All functions take plain lists. Record-level helpers at the bottom turn run records into metric inputs.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from statistics import mean

from bench.config import (
    BINARY_DECISION_THRESHOLD,
    BINARY_HIGH_CERTAINTY,
    BOOTSTRAP_ITERATIONS,
    HIGH_CERTAINTY_THRESHOLD,
    LOG_LOSS_EPS,
    SEED,
)

# ---------------------------------------------------------------- classification


def accuracy(gold: Sequence, pred: Sequence) -> float:
    _check(gold, pred)
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def macro_f1(gold: Sequence, pred: Sequence, labels: Sequence) -> float:
    """Unweighted mean of per-label F1 over `labels` (the full gold label set, fixed in config)."""
    _check(gold, pred)
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return mean(f1s)


def log_loss(gold: Sequence, probs: Sequence[dict]) -> float:
    """Multiclass log loss; p(gold) clipped to [eps, 1-eps] (eps fixed in config)."""
    _check(gold, probs)
    total = 0.0
    for g, p in zip(gold, probs, strict=True):
        pg = min(max(p.get(g, 0.0), LOG_LOSS_EPS), 1 - LOG_LOSS_EPS)
        total -= math.log(pg)
    return total / len(gold)


def confusion(gold: Sequence, pred: Sequence, labels: Sequence) -> dict:
    m = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred, strict=True):
        m[g][p] += 1
    return m


# ---------------------------------------------------------------- ordinal (T2)


def mae(gold: Sequence[float], score: Sequence[float]) -> float:
    _check(gold, score)
    return mean(abs(g - s) for g, s in zip(gold, score, strict=True))


# ---------------------------------------------------------------- binary probability (T5)


def auroc(gold: Sequence[bool], p: Sequence[float]) -> float | None:
    """Mann-Whitney AUROC with ties counted as 0.5. None if only one class present."""
    _check(gold, p)
    pos = [s for g, s in zip(gold, p, strict=True) if g]
    neg = [s for g, s in zip(gold, p, strict=True) if not g]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def brier(gold: Sequence[bool], p: Sequence[float]) -> float:
    _check(gold, p)
    return mean((float(g) - s) ** 2 for g, s in zip(gold, p, strict=True))


# ---------------------------------------------------------------- certainty


def certainty_of(answer: dict) -> float:
    """Pre-registered certainty signal.

    choice/score: provider `confidence` if the provider returns one, else max probability
    (e.g. self-reported by the LLM adapter). binary: distance-based, mapped so that p<=0.1 or p>=0.9 is >= 0.9.
    """
    if answer["type"] == "binary":
        return max(answer["p"], 1 - answer["p"])
    if answer.get("confidence") is not None:
        return answer["confidence"]
    return max(answer["probabilities"].values())


def is_high_certainty(answer: dict) -> bool:
    if answer["type"] == "binary":
        lo, hi = BINARY_HIGH_CERTAINTY
        return answer["p"] <= lo or answer["p"] >= hi
    return certainty_of(answer) >= HIGH_CERTAINTY_THRESHOLD


def coverage_accuracy(correct: Sequence[bool], high: Sequence[bool]) -> dict:
    _check(correct, high)
    covered = [c for c, h in zip(correct, high, strict=True) if h]
    return {
        "n": len(correct),
        "covered": len(covered),
        "coverage": len(covered) / len(correct) if correct else None,
        "accuracy_covered": (sum(covered) / len(covered)) if covered else None,
        "accuracy_all": sum(correct) / len(correct) if correct else None,
    }


def coverage_curve(correct: Sequence[bool], certainty: Sequence[float],
                   thresholds: Sequence[float] = tuple(i / 20 for i in range(21))) -> list[dict]:
    out = []
    for t in thresholds:
        covered = [c for c, s in zip(correct, certainty, strict=True) if s >= t]
        out.append({"threshold": t, "coverage": len(covered) / len(correct),
                    "accuracy": sum(covered) / len(covered) if covered else None})
    return out


# ---------------------------------------------------------------- agreement


def cohen_kappa(a: Sequence, b: Sequence, labels: Sequence, weights: str | None = None) -> float | None:
    """Cohen's kappa. weights=None (nominal) or "linear" (ordinal; labels must be ordered)."""
    _check(a, b)
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        obs[idx[x]][idx[y]] += 1 / n
    ra = [sum(row) for row in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        if weights == "linear":
            return abs(i - j) / (k - 1)
        raise ValueError(weights)

    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    pe = sum(w(i, j) * ra[i] * cb[j] for i in range(k) for j in range(k))
    if pe == 0:
        return None
    return 1 - po / pe


def percent_agreement(a: Sequence, b: Sequence) -> float:
    return accuracy(a, b)


# ---------------------------------------------------------------- latency / uncertainty


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy default 'linear')."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap_ci(units: Sequence, stat: Callable[[list], float | None], iterations: int = BOOTSTRAP_ITERATIONS,
                 alpha: float = 0.05, seed: int = SEED) -> tuple[float, float] | None:
    """Percentile bootstrap over `units` (resample units with replacement, recompute `stat`).

    For paired comparisons pass units that carry both arms (e.g. (A, B) tuples per example); for T3 pass one
    unit per *base* example so variants are resampled together.
    """
    rng = random.Random(seed)
    units = list(units)
    if not units:
        return None
    vals = []
    for _ in range(iterations):
        sample = [units[rng.randrange(len(units))] for _ in units]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return percentile(vals, 100 * alpha / 2), percentile(vals, 100 * (1 - alpha / 2))


# ---------------------------------------------------------------- record-level task metrics


def ok_records(records: Sequence[dict]) -> list[dict]:
    return [r for r in records if r.get("error") is None and r.get("answer") is not None]


def choice_metrics(records: Sequence[dict], labels: Sequence) -> dict:
    rs = ok_records(records)
    gold = [r["gold"] for r in rs]
    pred = [r["answer"]["prediction"] for r in rs]
    probs = [r["answer"]["probabilities"] for r in rs]
    correct = [g == p for g, p in zip(gold, pred, strict=True)]
    return {
        "n": len(rs),
        "n_error": len(records) - len(rs),
        "accuracy": accuracy(gold, pred) if rs else None,
        "macro_f1": macro_f1(gold, pred, labels) if rs else None,
        "log_loss": log_loss(gold, probs) if rs else None,
        "confusion": confusion(gold, pred, labels) if rs else None,
        "high_certainty": coverage_accuracy(correct, [is_high_certainty(r["answer"]) for r in rs]) if rs else None,
    }


def score_metrics(records: Sequence[dict]) -> dict:
    rs = ok_records(records)
    gold = [r["gold"] for r in rs]
    correct = [r["answer"]["prediction"] == r["gold"] for r in rs]
    return {
        "n": len(rs),
        "n_error": len(records) - len(rs),
        "mae": mae(gold, [r["answer"]["score"] for r in rs]) if rs else None,
        "argmax_accuracy": (sum(correct) / len(rs)) if rs else None,
        "high_certainty": coverage_accuracy(correct, [is_high_certainty(r["answer"]) for r in rs]) if rs else None,
    }


def binary_metrics(records: Sequence[dict]) -> dict:
    rs = ok_records(records)
    gold = [bool(r["gold"]) for r in rs]
    p = [r["answer"]["p"] for r in rs]
    correct = [(s >= BINARY_DECISION_THRESHOLD) == g for g, s in zip(gold, p, strict=True)]
    return {
        "n": len(rs),
        "n_error": len(records) - len(rs),
        "auroc": auroc(gold, p) if rs else None,
        "brier": brier(gold, p) if rs else None,
        "accuracy_at_0.5": (sum(correct) / len(rs)) if rs else None,
        "high_certainty": coverage_accuracy(correct, [is_high_certainty(r["answer"]) for r in rs]) if rs else None,
    }


def flip_metrics(variant_records: Sequence[dict], original_records: Sequence[dict],
                 examples_by_id: dict[str, dict]) -> dict:
    """T3. Compares each variant with the T1 run of its base example under the same condition/model.

    flip          prediction(variant) != prediction(original)
    gold_retention among bases predicted correctly on the original, share of their variants still correct
    confidence_delta  certainty(variant) - certainty(original)
    probability_delta p_variant(gold) - p_original(gold)
    """
    originals = {r["example_id"]: r for r in ok_records(original_records)}
    pairs = []
    for r in ok_records(variant_records):
        base_id = examples_by_id[r["example_id"]]["variant_of"]
        o = originals.get(base_id)
        if o is None:
            continue
        pairs.append((examples_by_id[r["example_id"]]["variant_type"], base_id, o, r))

    def summarize(ps: list) -> dict:
        if not ps:
            return {"n": 0}
        flips = [v["answer"]["prediction"] != o["answer"]["prediction"] for _, _, o, v in ps]
        kept = [v["answer"]["prediction"] == v["gold"] for _, _, o, v in ps
                if o["answer"]["prediction"] == o["gold"]]
        return {
            "n": len(ps),
            "flip_rate": sum(flips) / len(ps),
            "gold_retention": (sum(kept) / len(kept)) if kept else None,
            "confidence_delta": mean(certainty_of(v["answer"]) - certainty_of(o["answer"]) for _, _, o, v in ps),
            "probability_delta": mean(
                v["answer"]["probabilities"][v["gold"]] - o["answer"]["probabilities"][o["gold"]]
                for _, _, o, v in ps
            ),
        }

    by_type = defaultdict(list)
    for p in pairs:
        by_type[p[0]].append(p)
    result = summarize(pairs)
    result["by_variant_type"] = {t: summarize(ps) for t, ps in sorted(by_type.items())}
    by_base = defaultdict(list)
    for p in pairs:
        by_base[p[1]].append(p)

    def flip_rate_of(units: list) -> float | None:
        flat = [p for unit in units for p in unit]
        return (sum(v["answer"]["prediction"] != o["answer"]["prediction"] for _, _, o, v in flat) / len(flat)
                if flat else None)

    result["flip_rate_ci95"] = bootstrap_ci(list(by_base.values()), flip_rate_of)
    return result


def noise_metrics(records: Sequence[dict], examples_by_id: dict[str, dict]) -> dict:
    """T5 by noise level, plus paired probability delta vs the clean version of the same base."""
    rs = ok_records(records)
    by_level = defaultdict(list)
    for r in rs:
        by_level[examples_by_id[r["example_id"]]["variant_type"]].append(r)
    out = {"overall": binary_metrics(records), "by_level": {}}
    clean = {r["example_id"]: r for r in by_level.get("clean", [])}
    for level, lrs in sorted(by_level.items()):
        m = binary_metrics(lrs)
        if level != "clean":
            deltas = []
            for r in lrs:
                c = clean.get(examples_by_id[r["example_id"]]["variant_of"])
                if c is not None:
                    deltas.append(r["answer"]["p"] - c["answer"]["p"])
            m["probability_delta_vs_clean"] = mean(deltas) if deltas else None
            m["abs_probability_delta_vs_clean"] = mean(abs(d) for d in deltas) if deltas else None
        out["by_level"][level] = m
    return out


def latency_summary(records: Sequence[dict]) -> dict:
    lat = [r["latency_ms"] for r in ok_records(records) if r.get("latency_ms") is not None]
    return {"n": len(lat), "p50_ms": percentile(lat, 50), "p95_ms": percentile(lat, 95)}


def _check(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch {len(a)} != {len(b)}")
