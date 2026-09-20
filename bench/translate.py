"""Pre-generate the frozen English translations used by condition C.

    uv run python -m bench.translate            # translates examples missing from data/translations/en.jsonl

Translations are produced once, committed, and then frozen with the protocol; they are audited against
the Japanese source but never post-edited, and never generated during a benchmark run (PROTOCOL.md 2.1).
Translation cost/latency is recorded separately from benchmark latency.

The translator is instructed to translate faithfully, preserving tone, sarcasm, politeness level, emoji and
slang register, and not to explain or normalize them. For T4 the context/target structure is preserved.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time

from bench.config import TASKS, TRANSLATIONS_PATH, TRANSLATOR_CREDENTIAL, TRANSLATOR_MODEL
from bench.credentials import env_var_for, get_credential
from bench.dataset import load_task, load_translations, source_state, write_jsonl

SYSTEM = (
    "You translate Japanese messages from a peer-to-peer goods exchange service into natural English. "
    "Translate faithfully: keep the speaker's tone, politeness level, sarcasm, indirectness, emoji, kaomoji, "
    "slang register and line breaks. Do not explain, soften, clarify or add anything. If the input is a JSON "
    "object, return the same JSON structure with only the text values translated. Output only the translation."
)


def strip_code_fence(text: str) -> str:
    """Unwrap ```json ... ``` if the translator wrapped structured output in a fence."""
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return body.rsplit("```", 1)[0].strip()


def translate_one(client, state) -> tuple[object, dict]:
    is_json = not isinstance(state, str)
    text = json.dumps(state, ensure_ascii=False, indent=2) if is_json else state
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=TRANSLATOR_MODEL,
        max_tokens=4096,
        system=SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={resp.stop_reason}")
    out = "".join(b.text for b in resp.content if b.type == "text").strip()
    translation = json.loads(strip_code_fence(out)) if is_json else out
    if is_json and set(translation) != set(state):
        raise RuntimeError("translated JSON changed structure")
    meta = {"latency_ms": round(latency_ms, 1), "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens, "returned_model": resp.model}
    return translation, meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    api_key = get_credential(TRANSLATOR_CREDENTIAL)
    if not api_key:
        sys.exit(f"{env_var_for(TRANSLATOR_CREDENTIAL)} is not set")
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    existing = load_translations()
    rows = list(existing.values())
    todo = [ex for task in TASKS for ex in load_task(task) if ex["id"] not in existing]
    if args.limit:
        todo = todo[: args.limit]
    for i, ex in enumerate(todo, 1):
        state = source_state(ex)
        translation, meta = translate_one(client, state)
        rows.append({
            "id": ex["id"],
            "source": state,
            "translation": translation,
            "translator_model": TRANSLATOR_MODEL,
            "translated_at": dt.datetime.now(dt.UTC).isoformat(),
            "review_status": "pending",
            **meta,
        })
        if i % 10 == 0:
            write_jsonl(TRANSLATIONS_PATH, rows)
            print(f"  {i}/{len(todo)}")
    write_jsonl(TRANSLATIONS_PATH, rows)
    print(f"wrote {TRANSLATIONS_PATH} ({len(rows)} rows, {len(todo)} new)")
    print("next: `uv run python -m bench.validate` checks the frozen translations "
          "(rows, pinned snapshot, source, T4 structure, non-empty, digest)")


if __name__ == "__main__":
    main()
