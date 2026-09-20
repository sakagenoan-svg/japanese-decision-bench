"""Baseline adapter for a general-purpose LLM (default configuration: Anthropic Messages API).

Shows the model the same state, instructions and options as any other provider, and asks for a JSON answer
with *self-reported* probabilities via structured outputs. Self-reported numbers are not calibrated
probabilities; certainty-type metrics are therefore reported per provider and not compared as one mechanism
(see DESIGN.md "Comparability").

Prediction rules shared with every provider (DESIGN.md):
  choice -> the returned `choice`
  score  -> argmax of probabilities; expected level = probability-weighted mean
  binary -> p(yes)
"""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from bench.clients.base import CallResult, ProviderError, sleep_before_retry
from bench.config import MAX_RETRIES, PROVIDERS, REQUEST_TIMEOUT_S
from bench.credentials import env_var_for, get_credential

SYSTEM_PROMPT = {
    "ja": (
        "あなたは分類器です。ユーザーが与える「状態」を読み、「質問」に答えてください。"
        "各選択肢について、それが正解である確率を 0〜1 で見積もり、合計が 1 になるようにしてください。"
        "指定された JSON 形式だけを出力してください。"
    ),
    "en": (
        "You are a classifier. Read the STATE provided by the user and answer the QUESTION. "
        "Estimate, for every option, the probability (0-1) that it is correct, summing to 1. "
        "Output only the requested JSON."
    ),
}

LABELS = {
    "ja": {"state": "状態", "question": "質問", "options": "選択肢", "levels": "段階",
           "yes": "「はい」の意味", "no": "「いいえ」の意味",
           "p_yes": "答えが「はい」である確率を p_yes に入れてください。"},
    "en": {"state": "STATE", "question": "QUESTION", "options": "OPTIONS", "levels": "LEVELS",
           "yes": "Yes means", "no": "No means", "p_yes": "Put the probability that the answer is yes in p_yes."},
}


def render_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def build_prompt(state: Any, question: dict, lang: str) -> tuple[str, dict]:
    """Canonical question -> (user prompt, JSON schema)."""
    L = LABELS[lang]
    parts = [f"## {L['state']}\n{render_state(state)}", f"## {L['question']}\n{render_state(question['instructions'])}"]
    qtype = question["type"]
    if qtype == "choice":
        keys = list(question["criteria"])
        opts = "\n".join(f"- {k}: {d}" if d else f"- {k}" for k, d in question["criteria"].items())
        parts.append(f"## {L['options']}\n{opts}")
        schema = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": keys},
                "probabilities": _prob_object(keys),
            },
            "required": ["choice", "probabilities"],
            "additionalProperties": False,
        }
    elif qtype == "score":
        idx = [str(i) for i in range(len(question["criteria"]))]
        lv = "\n".join(f"- {i}: {d}" for i, d in zip(idx, question["criteria"], strict=True))
        parts.append(f"## {L['levels']}\n{lv}")
        schema = {
            "type": "object",
            "properties": {"probabilities": _prob_object(idx)},
            "required": ["probabilities"],
            "additionalProperties": False,
        }
    elif qtype == "binary":
        crit = question.get("criteria") or {}
        if crit:
            parts.append(f"- {L['yes']}: {crit.get('true', '')}\n- {L['no']}: {crit.get('false', '')}")
        parts.append(L["p_yes"])
        schema = {
            "type": "object",
            "properties": {"p_yes": {"type": "number"}},
            "required": ["p_yes"],
            "additionalProperties": False,
        }
    else:
        raise ValueError(qtype)
    return "\n\n".join(parts), schema


def _prob_object(keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {k: {"type": "number"} for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


def _normalize(probs: dict[str, float]) -> dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in probs.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {k: 1.0 / len(clipped) for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def to_canonical_answer(question: dict, parsed: dict) -> dict:
    """LLM JSON -> canonical answer (bench/clients/base.py)."""
    qtype = question["type"]
    if qtype == "choice":
        return {"type": "choice", "choice": parsed["choice"],
                "probabilities": _normalize(parsed["probabilities"]), "confidence": None}
    if qtype == "score":
        probs = _normalize(parsed["probabilities"])
        return {"type": "score", "probabilities": probs, "score": sum(int(k) * p for k, p in probs.items()),
                "confidence": None}
    if qtype == "binary":
        return {"type": "binary", "p": min(1.0, max(0.0, float(parsed["p_yes"])))}
    raise ValueError(qtype)


class LLMAdapter:
    def __init__(self, model: str, lang: str, client: anthropic.Anthropic | None = None, sleep=time.sleep):
        cfg = PROVIDERS["baseline"]
        if client is None:
            api_key = get_credential(cfg["credential"])
            if not api_key:
                raise ProviderError(f"{env_var_for(cfg['credential'])} is not set")
            # The key is passed explicitly so no other credential source is picked up.
            # Retries are counted by this harness, not the SDK.
            client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT_S)
        self.client = client
        self.model = model
        self.lang = lang
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]
        self._sleep = sleep

    def close(self) -> None:
        pass

    def _sampling(self) -> dict:
        """Sampling settings for the request body.

        The protocol fixes temperature at 0. The Anthropic Python SDK v1 dropped `temperature` (and `top_p`
        / `top_k`) from the generated `messages.create()` signature, while the API still accepts it for this
        model, so it travels in `extra_body`. This is a transport detail: the frozen setting is unchanged.
        Only temperature is sent; no other sampling parameter is introduced.
        """
        if self.temperature is None:
            return {}
        return {"extra_body": {"temperature": self.temperature}}

    def ask(self, state: Any, question: dict) -> CallResult:
        prompt, schema = build_prompt(state, question, self.lang)
        errors: list[str] = []
        t_start = time.perf_counter()
        for attempt in range(MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT[self.lang],
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                    **self._sampling(),
                )
            except anthropic.RateLimitError:
                errors.append(f"attempt {attempt}: 429")
                sleep_before_retry(attempt, None, self._sleep)
                continue
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    errors.append(f"attempt {attempt}: HTTP {e.status_code}")
                    sleep_before_retry(attempt, None, self._sleep)
                    continue
                raise ProviderError(f"HTTP {e.status_code}: {str(e)[:500]}") from e
            except anthropic.APIConnectionError as e:  # includes APITimeoutError
                errors.append(f"attempt {attempt}: {type(e).__name__}")
                sleep_before_retry(attempt, None, self._sleep)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp.stop_reason != "end_turn":
                raise ProviderError(f"stop_reason={resp.stop_reason}")
            text = next((b.text for b in resp.content if b.type == "text"), None)
            if text is None:
                raise ProviderError("response has no text block")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise ProviderError(f"response is not JSON: {text[:200]!r}") from e
            return CallResult(
                answer=to_canonical_answer(question, parsed),
                raw_response=resp.model_dump(mode="json"),
                returned_model=resp.model,
                latency_ms=latency_ms,
                total_elapsed_ms=(time.perf_counter() - t_start) * 1000,
                retry_count=attempt,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                errors=errors,
            )
        raise ProviderError(f"gave up after {MAX_RETRIES + 1} attempts: {errors}")
