"""Run the benchmark against one provider under one condition.

    uv run python -m bench.run --provider baseline --condition A
    uv run python -m bench.run --provider baseline --condition A --smoke  # 1 example per task
    uv run python -m bench.run --provider mock --condition A            # offline pipeline check

Providers: built-ins in bench/config.py plus optional local plugins (bench/providers.py).

Output: results/raw/<provider>-<condition>.jsonl (git-ignored). Re-running resumes: examples already
answered by the same requested model are skipped. Use --fresh to start a new file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import uuid
from pathlib import Path

from bench import providers
from bench.clients.base import ProviderError
from bench.config import CONDITIONS, PRIVATE_DIR, PROTOCOL_VERSION, RAW_DIR, ROOT, TASKS
from bench.dataset import load_all, load_question, load_translations, read_jsonl, state_for, write_jsonl
from bench.questions import AnswerError, normalize_answer, to_request


def git_revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "data", "bench"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def output_path(provider: str, condition: str, smoke: bool) -> Path:
    return RAW_DIR / f"{'smoke-' if smoke else ''}{provider}-{condition}.jsonl"


def run(provider: str, condition: str, tasks: list[str], limit: int | None, smoke: bool, fresh: bool,
        client=None, out: Path | None = None) -> Path:
    cfg = providers.get(provider)
    out = out or output_path(provider, condition, smoke)
    translations = load_translations()
    data = load_all()
    revision = git_revision()
    if revision.endswith("-dirty") and not smoke and provider != "mock":
        print("WARNING: data/ or bench/ has uncommitted changes; results will not map to a protocol commit.",
              file=sys.stderr)

    done: set[tuple[str, str]] = set()
    existing: list[dict] = []
    if out.exists() and not fresh:
        existing = read_jsonl(out)
        done = {(r["example_id"], r["requested_model"]) for r in existing if r.get("error") is None}
    client = client or providers.make_client(provider, condition)
    run_id = f"{provider}-{condition}-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    cond = CONDITIONS[condition]

    # Failed records from a previous invocation are dropped and retried, so each example appears once.
    records = [r for r in existing if r.get("error") is None]
    n_new = n_err = 0
    try:
        for task in tasks:
            question = load_question(task, cond["question_lang"])
            sq = to_request(question)
            examples = data[task][:1] if smoke else data[task]
            if limit is not None:
                examples = examples[:limit]
            for ex in examples:
                if (ex["id"], cfg["model"]) in done:
                    continue
                record = {
                    "run_id": run_id,
                    "protocol_version": PROTOCOL_VERSION,
                    "git_revision": revision,
                    "example_id": ex["id"],
                    "task": task,
                    "condition": condition,
                    "state_lang": cond["state_lang"],
                    "question_lang": cond["question_lang"],
                    "provider": cfg["provider"],
                    "route": cfg["route"],
                    "requested_model": cfg["model"],
                    "returned_model": None,
                    "started_at": dt.datetime.now(dt.UTC).isoformat(),
                    "latency_ms": None,
                    "total_elapsed_ms": None,
                    "retry_count": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "gold": ex["label"],
                    "answer": None,
                    "raw_answer": None,
                    "raw_response": None,
                    "error": None,
                }
                try:
                    state = state_for(ex, condition, translations)
                    res = client.ask(state, sq)
                    record.update(
                        returned_model=res.returned_model,
                        latency_ms=round(res.latency_ms, 1),
                        total_elapsed_ms=round(res.total_elapsed_ms, 1),
                        retry_count=res.retry_count,
                        input_tokens=res.input_tokens,
                        output_tokens=res.output_tokens,
                        raw_answer=res.answer,
                        raw_response=res.raw_response,
                    )
                    record["answer"] = normalize_answer(question, res.answer)
                except (ProviderError, AnswerError, KeyError, ValueError, TypeError) as e:
                    record["error"] = f"{type(e).__name__}: {e}"
                    n_err += 1
                    print(f"  ERROR {ex['id']}: {record['error']}", file=sys.stderr)
                records.append(record)
                n_new += 1
                if record["returned_model"] and record["returned_model"] != cfg["model"]:
                    print(f"  NOTE {ex['id']}: returned_model={record['returned_model']} "
                          f"!= requested {cfg['model']}", file=sys.stderr)
                if n_new % 10 == 0:
                    write_jsonl(out, records)
                    print(f"  {n_new} done ({n_err} errors)")
    finally:
        write_jsonl(out, records)
        client.close()
    print(f"wrote {out} (+{n_new} records, {n_err} errors)")
    if smoke and provider != "mock":
        _save_private_fixture(records[-1])
    return out


def _save_private_fixture(record: dict) -> None:
    """Real provider responses are kept as private test fixtures (tests use them only if present)."""
    from bench.sanitize import sanitize_record

    fx = PRIVATE_DIR / "fixtures" / f"{record['provider']}-{record['condition']}.json"
    fx.parent.mkdir(parents=True, exist_ok=True)
    import json

    fx.write_text(json.dumps(sanitize_record(record), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved private fixture {fx}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=sorted(providers.registry()))
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma-separated task names")
    ap.add_argument("--limit", type=int, default=None, help="max examples per task")
    ap.add_argument("--smoke", action="store_true", help="one example per task, separate output file")
    ap.add_argument("--fresh", action="store_true", help="overwrite instead of resuming")
    args = ap.parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise SystemExit(f"unknown tasks: {sorted(unknown)}")
    run(args.provider, args.condition, tasks, args.limit, args.smoke, args.fresh)


if __name__ == "__main__":
    main()
