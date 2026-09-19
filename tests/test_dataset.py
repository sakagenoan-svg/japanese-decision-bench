import json
import shutil

import pytest

from bench.config import DATA_DIR, QUESTIONS_DIR, TASKS
from bench.dataset import load_all, load_question, read_jsonl, source_state, state_for, write_jsonl
from bench.validate import validate


def test_shipped_dataset_validates():
    assert validate() == []


def test_counts_match_protocol():
    data = load_all()
    assert {t: len(rows) for t, rows in data.items()} == {t: s["expected_count"] for t, s in TASKS.items()}
    assert sum(len(r) for r in data.values()) == 160


def test_t1_label_balance():
    counts = {}
    for r in load_all()["t1_moderation"]:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    assert set(counts) == set(TASKS["t1_moderation"]["labels"])
    assert min(counts.values()) >= 5


def test_t3_variants_reference_preselected_bases_with_same_label():
    data = load_all()
    t1 = {r["id"]: r for r in data["t1_moderation"]}
    bases = {r["variant_of"] for r in data["t3_surface_variants"]}
    assert len(bases) == 10
    assert bases == {i for i, r in t1.items() if r["t3_base"]}
    for v in data["t3_surface_variants"]:
        assert v["label"] == t1[v["variant_of"]]["label"]
        assert v["source"] == "derived"


def test_halfwidth_variants_are_halfwidth_katakana():
    for v in load_all()["t3_surface_variants"]:
        if v["variant_type"] == "katakana_halfwidth":
            assert not any("ぁ" <= c <= "ヿ" for c in v["state"]), v["id"]  # no full-width kana


def test_t5_noisy_variants_contain_clean_text():
    rows = load_all()["t5_noise"]
    clean = {r["id"]: r for r in rows if r["variant_type"] == "clean"}
    for r in rows:
        if r["variant_type"] != "clean":
            assert clean[r["variant_of"]]["state"] in r["state"]
            assert len(r["state"]) > len(clean[r["variant_of"]]["state"])


def test_t4_state_is_structured_json():
    ex = load_all()["t4_ellipsis"][0]
    s = source_state(ex)
    assert set(s) == {"context", "target"} and s["target"] == ex["state"]


def test_condition_c_requires_frozen_translation():
    ex = load_all()["t1_moderation"][0]
    assert state_for(ex, "A", {}) == ex["state"]
    with pytest.raises(KeyError):
        state_for(ex, "C", {})
    assert state_for(ex, "C", {ex["id"]: {"translation": "hi"}}) == "hi"


def test_questions_exist_for_both_languages():
    for task in TASKS:
        for lang in ("ja", "en"):
            assert load_question(task, lang)["type"] == TASKS[task]["type"]


def _copy_data(tmp_path):
    d = tmp_path / "data"
    shutil.copytree(DATA_DIR, d, ignore=shutil.ignore_patterns("translations"))
    return d


def test_validate_catches_duplicate_ids_and_bad_labels(tmp_path):
    d = _copy_data(tmp_path)
    rows = read_jsonl(d / "t1_moderation.jsonl")
    rows[1]["id"] = rows[0]["id"]
    rows[2]["label"] = "spam"
    write_jsonl(d / "t1_moderation.jsonl", rows)
    errors = validate(d, QUESTIONS_DIR)
    assert any("duplicate id" in e for e in errors)
    assert any("invalid label 'spam'" in e for e in errors)


def test_validate_catches_pii(tmp_path):
    d = _copy_data(tmp_path)
    rows = read_jsonl(d / "t2_politeness_anger.jsonl")
    rows[0]["state"] = "連絡先は test.user@example.com か 090-1234-5678 です"
    write_jsonl(d / "t2_politeness_anger.jsonl", rows)
    errors = validate(d, QUESTIONS_DIR)
    assert any("email" in e for e in errors)
    assert any("phone" in e for e in errors)


def test_validate_catches_broken_variant_reference(tmp_path):
    d = _copy_data(tmp_path)
    rows = read_jsonl(d / "t3_surface_variants.jsonl")
    rows[0]["variant_of"] = "t1-999"
    write_jsonl(d / "t3_surface_variants.jsonl", rows)
    assert any("t1-999" in e for e in validate(d, QUESTIONS_DIR))


def test_dataset_build_is_reproducible(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_dataset", DATA_DIR.parent / "scripts" / "build_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    built = mod.build()
    for name, rows in built.items():
        assert rows == read_jsonl(DATA_DIR / name), f"{name} differs from scripts/build_dataset.py output"
    json.dumps(built, ensure_ascii=False)
