"""Provider registry: built-in providers plus optional local plugins.

A plugin is a Python module in the plugin directory (default: `.private/providers/`, override with
`BENCH_PROVIDER_PLUGINS`) that defines:

    PROVIDER = {"name": ..., "provider": ..., "route": ..., "model": <pinned id>, "conditions": [...],
                "price": {"input_per_mtok": ..., "output_per_mtok": ..., "checked_at": ...}}
    def make_client(condition: str, lang: str) -> bench.clients.base.Client

Plugins are never required: the public repository runs with the built-in providers only.
"""

from __future__ import annotations

import importlib.util
import os
from functools import cache
from pathlib import Path
from types import ModuleType

from bench.config import CONDITIONS, FORBIDDEN_MODEL_SUFFIXES, PRIVATE_DIR, PROVIDERS


def plugin_dir() -> Path:
    return Path(os.environ.get("BENCH_PROVIDER_PLUGINS", PRIVATE_DIR / "providers"))


@cache
def _plugins() -> dict[str, ModuleType]:
    found: dict[str, ModuleType] = {}
    d = plugin_dir()
    if not d.is_dir():
        return found
    for path in sorted(d.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"bench_plugin_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "PROVIDER") and hasattr(mod, "make_client"):
            found[mod.PROVIDER["name"]] = mod
    return found


def registry() -> dict[str, dict]:
    reg = dict(PROVIDERS)
    for name, mod in _plugins().items():
        if name in reg:
            raise SystemExit(f"plugin provider {name!r} shadows a built-in provider")
        reg[name] = mod.PROVIDER
    return reg


def get(name: str) -> dict:
    reg = registry()
    if name not in reg:
        raise SystemExit(f"unknown provider {name!r}; available: {sorted(reg)}")
    return reg[name]


def make_client(name: str, condition: str):
    cfg = get(name)
    if cfg["model"].endswith(FORBIDDEN_MODEL_SUFFIXES):
        raise SystemExit(f"refusing alias model id {cfg['model']!r}; pin a versioned id")
    if condition not in cfg["conditions"]:
        raise SystemExit(f"{name} is not run under condition {condition} (protocol: {cfg['conditions']})")
    lang = CONDITIONS[condition]["question_lang"]
    if name == "baseline":
        from bench.clients.adapter import LLMAdapter

        return LLMAdapter(model=cfg["model"], lang=lang)
    if name == "mock":
        from bench.clients.mock import MockClient

        return MockClient(model=cfg["model"])
    return _plugins()[name].make_client(condition, lang)


def price_of(model: str) -> dict | None:
    for cfg in registry().values():
        if cfg["model"] == model and "price" in cfg:
            return {"provider": cfg["provider"], **cfg["price"]}
    return None
