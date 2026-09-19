"""Protocol constants. Everything in this file is frozen by the `protocol-v1` tag.

Changing any value here after `protocol-v1` requires a new protocol tag and a CHANGELOG entry
(see PROTOCOL.md "Change policy").
"""

from __future__ import annotations

import json
from pathlib import Path

PROTOCOL_VERSION = "protocol-v1"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
QUESTIONS_DIR = DATA_DIR / "questions"
TRANSLATIONS_PATH = DATA_DIR / "translations" / "en.jsonl"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
SANITIZED_DIR = RESULTS_DIR / "sanitized"
PRIVATE_DIR = ROOT / ".private"
LEGAL_STATUS_PATH = PRIVATE_DIR / "legal" / "legal-status.json"
UNPUBLISHED_DIR = PRIVATE_DIR / "unpublished-results"

SEED = 20260920

# ---------------------------------------------------------------- tasks

TASKS: dict[str, dict] = {
    "t1_moderation": {
        "file": "t1_moderation.jsonl",
        "question": "t1",
        "type": "choice",
        "labels": ["ok", "harassment", "scam", "resale_spam"],
        "expected_count": 30,
        "min_per_label": 5,
    },
    "t2_politeness_anger": {
        "file": "t2_politeness_anger.jsonl",
        "question": "t2",
        "type": "score",
        "labels": [0, 1, 2],
        "expected_count": 30,
    },
    "t3_surface_variants": {
        "file": "t3_surface_variants.jsonl",
        "question": "t1",  # T3 re-asks the T1 question on surface variants
        "type": "choice",
        "labels": ["ok", "harassment", "scam", "resale_spam"],
        "expected_count": 40,
        "base_task": "t1_moderation",
        "expected_bases": 10,
        "variant_types": ["hiragana", "katakana_halfwidth", "emoji_kaomoji", "internet_slang"],
    },
    "t4_ellipsis": {
        "file": "t4_ellipsis.jsonl",
        "question": "t4",
        "type": "choice",
        "labels": ["accept", "reject", "question", "other"],
        "expected_count": 30,
    },
    "t5_noise": {
        "file": "t5_noise.jsonl",
        "question": "t5",
        "type": "binary",
        "labels": [True, False],
        "expected_count": 30,
        "expected_bases": 10,
        "variant_types": ["clean", "medium", "heavy"],
    },
}

SOURCES = {"handwritten", "llm_assisted", "derived"}

# ---------------------------------------------------------------- conditions

# A: JA state + JA questions / B: JA state + EN questions / C: machine-translated EN state + EN questions.
# C measures a translation *pipeline*, not a pure language effect.
CONDITIONS = {
    "A": {"state_lang": "ja", "question_lang": "ja"},
    "B": {"state_lang": "ja", "question_lang": "en"},
    "C": {"state_lang": "en", "question_lang": "en"},
}

# ---------------------------------------------------------------- providers

# Built-in providers. Additional providers can be added as private plugins (see bench/providers.py).
# Pinned model IDs only: moving aliases (`-latest`, `-preview`) are rejected by bench.run.
PROVIDERS = {
    # General-purpose LLM baseline. Credential: logical name "baseline" (see bench/credentials.py).
    "baseline": {
        "name": "baseline",
        "provider": "anthropic",
        "route": "direct",
        "model": "claude-haiku-4-5-20251001",
        "conditions": ["A"],
        "credential": "baseline",  # passed explicitly as api_key
        # thinking is off unless requested; temperature 0 for determinism
        "temperature": 0.0,
        "max_tokens": 512,
        # public list price, USD per 1M tokens
        "price": {"input_per_mtok": 1.0, "output_per_mtok": 5.0, "checked_at": "2026-09-20"},
    },
    # Offline stand-in for tests / pipeline checks. Never reported.
    "mock": {
        "name": "mock",
        "provider": "mock",
        "route": "local",
        "model": "mock-0",
        "conditions": ["A", "B", "C"],
        "credential": None,
    },
}

FORBIDDEN_MODEL_SUFFIXES = ("-latest", "-preview")

# Translator for condition C. Translations are generated once and frozen in data/translations/en.jsonl.
TRANSLATOR_MODEL = "claude-sonnet-5"
TRANSLATOR_CREDENTIAL = "baseline"
TRANSLATOR_PRICE = {"input_per_mtok": 2.0, "output_per_mtok": 10.0, "checked_at": "2026-09-20"}

# ---------------------------------------------------------------- metric definitions

HIGH_CERTAINTY_THRESHOLD = 0.9  # representative, pre-registered; not claimed optimal
BINARY_HIGH_CERTAINTY = (0.1, 0.9)  # p <= 0.1 or p >= 0.9
BINARY_DECISION_THRESHOLD = 0.5
LOG_LOSS_EPS = 1e-6
BOOTSTRAP_ITERATIONS = 2000

# ---------------------------------------------------------------- run behaviour

MAX_RETRIES = 5
REQUEST_TIMEOUT_S = 30.0


def load_legal_status() -> dict:
    """Read the private publication status. A missing file (e.g. a public clone) means nothing is publishable."""
    if not LEGAL_STATUS_PATH.exists():
        return {}
    return json.loads(LEGAL_STATUS_PATH.read_text(encoding="utf-8"))


def publication_allowed(provider: str) -> bool:
    """Whether results from `provider` may leave .private/. Fail-closed: only providers the maintainer lists
    explicitly in `publishable_providers` (after checking the provider's terms) are publishable."""
    return provider in load_legal_status().get("publishable_providers", [])
