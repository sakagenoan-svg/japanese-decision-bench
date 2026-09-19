"""Dataset validation (runs in CI).

    uv run python -m bench.validate
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bench.config import DATA_DIR, QUESTIONS_DIR, SOURCES, TASKS
from bench.dataset import load_task, read_jsonl

REQUIRED = ["id", "task", "state", "context", "label", "label_b", "agreed", "source", "variant_of",
            "variant_type", "note"]

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\d)(?:0\d{1,4}[-‐ー−\s]?\d{1,4}[-‐ー−\s]?\d{3,4}|\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3,4}"
                        r"[\s-]?\d{3,4})(?!\d)"),
    "api_key": re.compile(r"(?:sk-[A-Za-z0-9_\-]{16,}|(?i:api[_-]?key)\s*[:=]\s*\S{8,}|Bearer\s+[A-Za-z0-9._\-]{16,})"),
    "uuid": re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    "long_hex_id": re.compile(r"\b[0-9a-fA-F]{24,}\b"),
}


def _texts(example: dict) -> list[str]:
    out = [example["state"]]
    for turn in example.get("context") or []:
        out.append(turn["text"])
    return out


def validate(data_dir: Path = DATA_DIR, questions_dir: Path = QUESTIONS_DIR) -> list[str]:
    errors: list[str] = []
    all_ids: Counter = Counter()
    by_task: dict[str, list[dict]] = {}

    for task, spec in TASKS.items():
        path = data_dir / spec["file"]
        if not path.exists():
            errors.append(f"{task}: missing file {path.name}")
            continue
        try:
            rows = load_task(task, data_dir)
        except ValueError as e:
            errors.append(str(e))
            continue
        by_task[task] = rows
        if len(rows) != spec["expected_count"]:
            errors.append(f"{task}: expected {spec['expected_count']} examples, found {len(rows)}")
        for r in rows:
            rid = r.get("id", "<no id>")
            all_ids[rid] += 1
            missing = [k for k in REQUIRED if k not in r]
            if missing:
                errors.append(f"{rid}: missing fields {missing}")
                continue
            if r["task"] != task:
                errors.append(f"{rid}: task {r['task']!r} in {path.name}")
            if r["label"] not in spec["labels"] or isinstance(r["label"], bool) != isinstance(spec["labels"][0], bool):
                errors.append(f"{rid}: invalid label {r['label']!r}")
            if r["label_b"] is not None and r["label_b"] not in spec["labels"]:
                errors.append(f"{rid}: invalid label_b {r['label_b']!r}")
            if r["label_b"] is not None and r["agreed"] != (r["label"] == r["label_b"]):
                errors.append(f"{rid}: agreed flag inconsistent")
            if r["source"] not in SOURCES:
                errors.append(f"{rid}: invalid source {r['source']!r}")
            if not isinstance(r["state"], str) or not r["state"].strip():
                errors.append(f"{rid}: empty state")
            if r["context"] is not None:
                if not isinstance(r["context"], list) or not all(
                        isinstance(t, dict) and set(t) == {"speaker", "text"} for t in r["context"]):
                    errors.append(f"{rid}: context must be a list of {{speaker, text}}")
            for text in _texts(r):
                for name, pat in PII_PATTERNS.items():
                    if pat.search(text):
                        errors.append(f"{rid}: looks like {name}: {pat.search(text).group(0)!r}")

    for rid, n in all_ids.items():
        if n > 1:
            errors.append(f"duplicate id {rid} ({n}x)")

    errors += _check_t1(by_task.get("t1_moderation", []))
    errors += _check_variants(by_task, "t3_surface_variants")
    errors += _check_t5(by_task.get("t5_noise", []))
    errors += _check_questions(questions_dir)
    return errors


def _check_t1(rows: list[dict]) -> list[str]:
    spec = TASKS["t1_moderation"]
    counts = Counter(r["label"] for r in rows)
    return [f"t1_moderation: label {lab!r} has {counts[lab]} < {spec['min_per_label']}"
            for lab in spec["labels"] if rows and counts[lab] < spec["min_per_label"]]


def _check_variants(by_task: dict, task: str) -> list[str]:
    spec = TASKS[task]
    rows = by_task.get(task, [])
    base = {r["id"]: r for r in by_task.get(spec["base_task"], [])}
    errors = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["source"] != "derived":
            errors.append(f"{r['id']}: T3 variants must have source 'derived'")
        b = base.get(r["variant_of"])
        if b is None:
            errors.append(f"{r['id']}: variant_of {r['variant_of']!r} not found in {spec['base_task']}")
            continue
        if r["label"] != b["label"]:
            errors.append(f"{r['id']}: label {r['label']!r} differs from base {b['label']!r}")
        if r["variant_type"] not in spec["variant_types"]:
            errors.append(f"{r['id']}: invalid variant_type {r['variant_type']!r}")
        groups[r["variant_of"]].append(r)
    if rows and len(groups) != spec["expected_bases"]:
        errors.append(f"{task}: expected {spec['expected_bases']} bases, found {len(groups)}")
    for base_id, vs in groups.items():
        types = sorted(v["variant_type"] for v in vs)
        if types != sorted(spec["variant_types"]):
            errors.append(f"{task}: base {base_id} has variants {types}")
        if not base[base_id].get("t3_base"):
            errors.append(f"{task}: base {base_id} is not marked t3_base in T1 (bases must be pre-selected)")
    marked = [i for i, r in base.items() if r.get("t3_base")]
    if rows and sorted(marked) != sorted(groups):
        errors.append(f"{task}: T1 t3_base marks {sorted(marked)} != variant bases {sorted(groups)}")
    return errors


def _check_t5(rows: list[dict]) -> list[str]:
    spec = TASKS["t5_noise"]
    errors = []
    clean = {r["id"]: r for r in rows if r["variant_type"] == "clean"}
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["variant_type"] not in spec["variant_types"]:
            errors.append(f"{r['id']}: invalid noise level {r['variant_type']!r}")
            continue
        if r["variant_type"] == "clean":
            if r["variant_of"] is not None:
                errors.append(f"{r['id']}: clean example must have variant_of null")
            if r["source"] == "derived":
                errors.append(f"{r['id']}: clean example cannot be derived")
            groups[r["id"]].append("clean")
        else:
            c = clean.get(r["variant_of"])
            if c is None:
                errors.append(f"{r['id']}: variant_of {r['variant_of']!r} is not a clean T5 example")
                continue
            if r["source"] != "derived":
                errors.append(f"{r['id']}: noisy variants must be 'derived'")
            if r["label"] != c["label"]:
                errors.append(f"{r['id']}: label differs from clean base")
            if c["state"] not in r["state"]:
                errors.append(f"{r['id']}: noisy variant must contain the clean text verbatim")
            groups[r["variant_of"]].append(r["variant_type"])
    if rows and len(groups) != spec["expected_bases"]:
        errors.append(f"t5_noise: expected {spec['expected_bases']} bases, found {len(groups)}")
    for base_id, levels in groups.items():
        if sorted(levels) != sorted(spec["variant_types"]):
            errors.append(f"t5_noise: base {base_id} has levels {sorted(levels)}")
    return errors


def _check_questions(questions_dir: Path) -> list[str]:
    import json

    errors = []
    for spec in TASKS.values():
        for lang in ("ja", "en"):
            path = questions_dir / f"{spec['question']}.{lang}.json"
            if not path.exists():
                errors.append(f"missing question file {path.name}")
                continue
            q = json.loads(path.read_text(encoding="utf-8"))
            if q["type"] != spec["type"]:
                errors.append(f"{path.name}: type {q['type']} != {spec['type']}")
            if q["type"] == "choice":
                labels = [o["label"] for o in q["options"]]
                keys = [o["key"] for o in q["options"]]
                if sorted(labels) != sorted(spec["labels"]):
                    errors.append(f"{path.name}: option labels {labels} != {spec['labels']}")
                if len(set(keys)) != len(keys):
                    errors.append(f"{path.name}: duplicate option keys")
            elif q["type"] == "score":
                if [lvl["value"] for lvl in q["levels"]] != list(range(len(q["levels"]))):
                    errors.append(f"{path.name}: level values must be 0..n-1 in order")
                if [lvl["value"] for lvl in q["levels"]] != spec["labels"]:
                    errors.append(f"{path.name}: levels do not match labels {spec['labels']}")
            elif q["type"] == "binary":
                if set(q.get("criteria", {})) != {"true", "false"}:
                    errors.append(f"{path.name}: binary criteria must have true/false")
    return errors


def validate_translations(path: Path, data_dir: Path = DATA_DIR) -> list[str]:
    if not path.exists():
        return []
    ids = {r["id"] for task in TASKS for r in load_task(task, data_dir)}
    errors = []
    for row in read_jsonl(path):
        for k in ("id", "source", "translation", "translator_model", "translated_at"):
            if k not in row:
                errors.append(f"translation {row.get('id')}: missing {k}")
        if row.get("id") not in ids:
            errors.append(f"translation {row.get('id')}: unknown example id")
    return errors


def main() -> None:
    from bench.config import TRANSLATIONS_PATH

    errors = validate() + validate_translations(TRANSLATIONS_PATH)
    if errors:
        print(f"{len(errors)} validation error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    counts = {t: len(load_task(t)) for t in TASKS}
    print(f"dataset OK: {counts} total={sum(counts.values())}")


if __name__ == "__main__":
    main()
