"""Scan git-tracked files (or all files that would be committed) for secrets.

    uv run python scripts/secret_scan.py            # tracked + staged files
    uv run python scripts/secret_scan.py --history  # every blob in git history

If `.private/forbidden-terms.txt` exists (one term per line, never committed), those terms are also flagged.
Use it for names that must not appear in the public repository.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATTERNS = {
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "generic_sk_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9._\-]{24,}"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "assigned_secret": re.compile(r"(?i)(api[_-]?key|secret|token|password)[ \t]*[=:][ \t]*['\"]?[A-Za-z0-9_\-]{16,}"),
    "windows_user_path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+"),
    "email": re.compile(r"[\w.+-]+@(?!example\.com|users\.noreply\.github\.com|anthropic\.com)[\w-]+\.[\w.-]+"),
}
ALLOW_FILES = {"scripts/secret_scan.py", "tests/test_sanitize.py", "bench/sanitize.py",
               "bench/validate.py"}
FORBIDDEN_TERMS = ROOT / ".private" / "forbidden-terms.txt"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
                          encoding="utf-8").stdout


def candidate_files() -> list[str]:
    tracked = set(git("ls-files").splitlines())
    staged = set(git("diff", "--cached", "--name-only").splitlines())
    return sorted(p for p in tracked | staged if (ROOT / p).is_file())


def load_terms() -> list[str]:
    if not FORBIDDEN_TERMS.exists():
        return []
    return [t.strip() for t in FORBIDDEN_TERMS.read_text(encoding="utf-8").splitlines()
            if t.strip() and not t.startswith("#")]


def scan_text(name: str, text: str, terms: list[str]) -> list[str]:
    hits = []
    if name not in ALLOW_FILES:
        for label, pat in PATTERNS.items():
            for m in pat.finditer(text):
                hits.append(f"{name}: {label}: {m.group(0)[:40]!r}")
    for term in terms:
        if term.lower() in text.lower():
            hits.append(f"{name}: forbidden term {term!r}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true", help="scan every blob reachable in git history")
    args = ap.parse_args()
    terms = load_terms()
    hits: list[str] = []
    if args.history:
        for line in git("rev-list", "--all", "--objects").splitlines():
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            sha, path = parts
            if git("cat-file", "-t", sha).strip() != "blob":
                continue
            text = subprocess.run(["git", "cat-file", "-p", sha], cwd=ROOT, capture_output=True).stdout.decode(
                "utf-8", errors="replace")
            hits += scan_text(path, text, terms)
            if path.startswith(".private/") or path in {".env", "handoff.md"}:
                hits.append(f"{path}: private file present in history")
    else:
        for p in candidate_files():
            if p.startswith(".private/") or p in {".env", "handoff.md"}:
                hits.append(f"{p}: private file is tracked")
                continue
            hits += scan_text(p, (ROOT / p).read_text(encoding="utf-8", errors="replace"), terms)
    if hits:
        print(f"{len(hits)} potential secret(s):")
        for h in sorted(set(hits)):
            print("  -", h)
        sys.exit(1)
    print(f"secret scan OK ({'history' if args.history else 'working tree'}, {len(terms)} private terms)")


if __name__ == "__main__":
    main()
