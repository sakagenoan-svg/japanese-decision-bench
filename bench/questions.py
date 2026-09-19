"""Question files -> canonical questions, and canonical answers -> gold-label space.

Question files keep display text (`key`, `description`) separate from the gold `label`, so JA questions
can use Japanese option names while every result is scored in the same label space.
See bench/clients/base.py for the canonical question / answer formats.
"""

from __future__ import annotations

from typing import Any


class AnswerError(ValueError):
    """Provider answer does not fit the question schema."""


def to_request(question: dict) -> dict:
    """Question file -> canonical question sent to a client."""
    qtype = question["type"]
    if qtype == "choice":
        return {
            "type": "choice",
            "instructions": question["instructions"],
            "criteria": {opt["key"]: opt["description"] for opt in question["options"]},
        }
    if qtype == "score":
        return {
            "type": "score",
            "instructions": question["instructions"],
            "criteria": [lvl["description"] for lvl in question["levels"]],
        }
    if qtype == "binary":
        return {
            "type": "binary",
            "instructions": question["instructions"],
            "criteria": dict(question["criteria"]),
        }
    raise ValueError(f"unknown question type {qtype!r}")


def normalize_answer(question: dict, answer: dict) -> dict:
    """Canonical answer (keys as displayed) -> label-space answer used by metrics.

    Output shapes:
      choice: {"type", "prediction": label, "probabilities": {label: p}, "confidence"}
      score:  {"type", "prediction": level (argmax), "score": expected level, "probabilities": {level: p},
               "confidence"}
      binary: {"type", "p": probability of yes}
    """
    qtype = question["type"]
    if answer.get("type") != qtype:
        raise AnswerError(f"answer type {answer.get('type')!r} != question type {qtype!r}")

    if qtype == "choice":
        key_to_label = {opt["key"]: opt["label"] for opt in question["options"]}
        probs = answer.get("probabilities") or {}
        if set(probs) != set(key_to_label):
            raise AnswerError(f"choice probabilities keys {sorted(probs)} != options {sorted(key_to_label)}")
        if answer["choice"] not in key_to_label:
            raise AnswerError(f"choice {answer['choice']!r} not an option")
        return {
            "type": "choice",
            "prediction": key_to_label[answer["choice"]],
            "probabilities": {key_to_label[k]: float(v) for k, v in probs.items()},
            "confidence": _opt_float(answer.get("confidence")),
        }

    if qtype == "score":
        values = [lvl["value"] for lvl in question["levels"]]
        probs = answer.get("probabilities") or {}
        if set(probs) != {str(i) for i in range(len(values))}:
            raise AnswerError(f"score probabilities keys {sorted(probs)} != level indices")
        by_value = {values[int(i)]: float(p) for i, p in probs.items()}
        prediction = max(values, key=lambda v: (by_value[v], -v))  # ties -> lower level, deterministic
        # Expected level is always recomputed from the distribution so every provider is scored the same way.
        expected = sum(v * p for v, p in by_value.items()) / (sum(by_value.values()) or 1.0)
        return {
            "type": "score",
            "prediction": prediction,
            "score": expected,
            "probabilities": by_value,
            "confidence": _opt_float(answer.get("confidence")),
        }

    if qtype == "binary":
        p = float(answer["p"])
        if not 0.0 <= p <= 1.0:
            raise AnswerError(f"p {p} outside [0, 1]")
        return {"type": "binary", "p": p}

    raise ValueError(f"unknown question type {qtype!r}")


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)
