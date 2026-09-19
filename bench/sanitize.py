"""Sanitize raw run records before they can become public.

    uv run python -m bench.sanitize            # results/raw/*.jsonl -> results/sanitized/ (gated)

Two layers:
1. raw_response is reduced to an allow-list of fields (answer(s), content, usage, model ...).
2. Everything is then scrubbed recursively: secret-like keys are dropped and secret-like strings
   (API keys, bearer tokens, local paths, emails) are redacted.

Publication gate: records from a provider that is not explicitly listed as publishable
(bench.config.publication_allowed, fail-closed) are written to .private/unpublished-results/sanitized/
instead of results/sanitized/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from bench.config import RAW_DIR, SANITIZED_DIR, UNPUBLISHED_DIR, publication_allowed
from bench.dataset import read_jsonl, write_jsonl

# Fields of a provider response that may be kept. Everything else (ids, headers, metadata) is dropped.
RAW_RESPONSE_ALLOW = {"model", "answer", "answers", "type", "role", "content", "stop_reason", "usage"}
USAGE_ALLOW = {"input_tokens", "output_tokens"}

DENY_KEY = re.compile(
    r"(authorization|api[-_]?key|secret|token(?!s$)|cookie|password|request[-_]?id|trace|span[-_]?id|"
    r"account|organization|org[-_]?id|user[-_]?id|workspace|^id$|headers|metadata|path$|file$)",
    re.IGNORECASE,
)
# keys that match DENY_KEY by accident but are part of the benchmark record
KEEP_KEY = {"input_tokens", "output_tokens", "example_id", "run_id", "base_id"}

REDACTIONS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "[REDACTED_KEY]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9_]{32,}\b"), "[REDACTED_TOKEN]"),  # hyphenated ids such as run ids are kept
    (re.compile(r"[A-Za-z]:\\(?:[^\\\s\"']+\\)*[^\\\s\"']*"), "[REDACTED_PATH]"),
    (re.compile(r"(?<![\w.])/(?:home|Users|root|tmp|var)/[^\s\"']+"), "[REDACTED_PATH]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[REDACTED_EMAIL]"),
]


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k in KEEP_KEY or not DENY_KEY.search(k)}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        for pattern, repl in REDACTIONS:
            value = pattern.sub(repl, value)
        return value
    return value


def reduce_raw_response(provider: str, raw: dict | None) -> dict | None:
    if raw is None:
        return None
    out = {k: v for k, v in raw.items() if k in RAW_RESPONSE_ALLOW}
    if isinstance(out.get("usage"), dict):
        out["usage"] = {k: v for k, v in out["usage"].items() if k in USAGE_ALLOW}
    return out


def sanitize_record(record: dict) -> dict:
    rec = dict(record)
    rec["raw_response"] = reduce_raw_response(rec.get("provider", ""), rec.get("raw_response"))
    return scrub(rec)


def sanitize_file(src: Path, public_dir: Path = SANITIZED_DIR, private_dir: Path = UNPUBLISHED_DIR / "sanitized",
                  ) -> tuple[Path, bool]:
    rows = [sanitize_record(r) for r in read_jsonl(src)]
    providers = {r["provider"] for r in rows}
    public = bool(rows) and all(publication_allowed(p) for p in providers) and "mock" not in providers
    dest = (public_dir if public else private_dir) / src.name
    write_jsonl(dest, rows)
    return dest, public


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=RAW_DIR)
    args = ap.parse_args(argv)
    files = sorted(p for p in args.src.glob("*.jsonl") if not p.name.startswith("smoke-"))
    if not files:
        print(f"no raw result files in {args.src}")
    for f in files:
        dest, public = sanitize_file(f)
        print(f"{f.name} -> {dest} [{'PUBLIC' if public else 'PRIVATE (publication not allowed)'}]")


if __name__ == "__main__":
    main()
