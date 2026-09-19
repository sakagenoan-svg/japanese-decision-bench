"""Generic credential interface.

Code asks for a credential by *logical* name (e.g. "baseline"); this module maps it to an environment variable
and reads it from the current process environment only. Values are never logged.

Default mapping: see DEFAULT_ENV. A local, untracked JSON file can remap logical names to other variable names
({"baseline": "MY_VAR"}); its path is `DECISION_BENCH_CREDENTIALS_MAP` or `.private/credentials.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from bench.config import PRIVATE_DIR

DEFAULT_ENV = {"baseline": "DECISION_BENCH_BASELINE_API_KEY"}


def _overrides() -> dict[str, str]:
    path = Path(os.environ.get("DECISION_BENCH_CREDENTIALS_MAP", PRIVATE_DIR / "credentials.json"))
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def env_var_for(name: str) -> str:
    mapping = {**DEFAULT_ENV, **_overrides()}
    if name not in mapping:
        raise KeyError(f"no environment variable configured for credential {name!r}")
    return mapping[name]


def get_credential(name: str) -> str | None:
    return os.environ.get(env_var_for(name)) or None
