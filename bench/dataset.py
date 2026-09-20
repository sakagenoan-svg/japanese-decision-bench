"""Loading examples, questions and frozen translations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bench.config import CONDITIONS, DATA_DIR, QUESTIONS_DIR, TASKS, TRANSLATIONS_PATH


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_task(task: str, data_dir: Path = DATA_DIR) -> list[dict]:
    return read_jsonl(data_dir / TASKS[task]["file"])


def load_all(data_dir: Path = DATA_DIR) -> dict[str, list[dict]]:
    return {task: load_task(task, data_dir) for task in TASKS}


def load_question(task: str, lang: str, questions_dir: Path = QUESTIONS_DIR) -> dict:
    name = TASKS[task]["question"]
    return json.loads((questions_dir / f"{name}.{lang}.json").read_text(encoding="utf-8"))


def load_translations(path: Path = TRANSLATIONS_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def translation_digest(rows: list[dict]) -> str:
    """Hash of the frozen translation content only: id + source + translation.

    Review metadata is deliberately excluded, so the digest answers one question: has any translation text
    changed since it was generated? The protocol pins the expected value in `bench.config`.
    """
    payload = [{"id": r["id"], "source": r["source"], "translation": r["translation"]}
               for r in sorted(rows, key=lambda r: r["id"])]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def source_state(example: dict) -> Any:
    """The Japanese state exactly as sent under conditions A/B.

    Examples with `context` (T4) are sent as a JSON object so the model sees speaker structure.
    """
    if example.get("context") is not None:
        return {"context": example["context"], "target": example["state"]}
    return example["state"]


def state_for(example: dict, condition: str, translations: dict[str, dict]) -> Any:
    lang = CONDITIONS[condition]["state_lang"]
    if lang == "ja":
        return source_state(example)
    tr = translations.get(example["id"])
    if tr is None:
        raise KeyError(
            f"No frozen translation for {example['id']}. Run `python -m bench.translate` before condition C."
        )
    return tr["translation"]
