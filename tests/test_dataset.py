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


def test_validate_translations_checks_row_fields(tmp_path):
    from bench.validate import validate_translations

    ex = load_all()["t1_moderation"][0]
    row = {"id": ex["id"], "source": ex["state"], "translation": "hi",
           "translator_model": "m", "translated_at": "2026-01-01T00:00:00+00:00"}
    path = tmp_path / "en.jsonl"
    write_jsonl(path, [row])
    assert any("missing review_status" in e for e in validate_translations(path))
    write_jsonl(path, [{**row, "review_status": "done"}])
    errors = validate_translations(path)
    assert any("invalid review_status" in e for e in errors)
    assert any("is not the pinned snapshot" in e for e in errors)
    assert any("missing from the frozen translations" in e for e in errors)


def test_shipped_translations_pass_the_protocol_gate():
    """The frozen condition C artifact: 160 rows, pinned snapshot, digest unchanged since generation."""
    from bench.config import TRANSLATIONS_PATH
    from bench.validate import validate_translations

    assert validate_translations(TRANSLATIONS_PATH) == []


def test_validate_translations_detects_an_edited_or_regenerated_translation(tmp_path):
    from bench.config import TRANSLATIONS_PATH
    from bench.validate import validate_translations

    if not TRANSLATIONS_PATH.exists():
        pytest.skip("no frozen translations in this checkout")
    rows = read_jsonl(TRANSLATIONS_PATH)
    path = tmp_path / "en.jsonl"
    edited = [{**r, "translation": r["translation"] + "!"} if isinstance(r["translation"], str) and r is rows[0]
              else r for r in rows]
    write_jsonl(path, edited)
    assert any("digest" in e for e in validate_translations(path))
    write_jsonl(path, [{**r, "review_status": "reviewed"} for r in rows])
    assert validate_translations(path) == []  # review metadata is not part of the digest


def test_validate_translations_detects_empty_and_structural_damage(tmp_path):
    from bench.config import TRANSLATIONS_PATH
    from bench.validate import validate_translations

    if not TRANSLATIONS_PATH.exists():
        pytest.skip("no frozen translations in this checkout")
    rows = read_jsonl(TRANSLATIONS_PATH)
    path = tmp_path / "en.jsonl"

    def replace(example_id, translation):
        write_jsonl(path, [{**r, "translation": translation} if r["id"] == example_id else r for r in rows])
        return validate_translations(path)

    assert any("empty translation" in e for e in replace("t1-001", "   "))
    t4 = next(r for r in rows if r["id"].startswith("t4"))
    broken = {**t4["translation"], "context": t4["translation"]["context"][:-1]}
    assert any("context turn count differs" in e for e in replace(t4["id"], broken))
    flipped = {**t4["translation"],
               "context": [{**turn, "speaker": "Z"} for turn in t4["translation"]["context"]]}
    assert any("speaker differs from the source" in e for e in replace(t4["id"], flipped))


def _fake_client(text):
    from types import SimpleNamespace

    block = SimpleNamespace(type="text", text=text)
    resp = SimpleNamespace(stop_reason="end_turn", content=[block], model="m",
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))


def test_translate_one_handles_plain_text_and_fenced_json():
    from bench.translate import translate_one

    assert translate_one(_fake_client("Hello there"), "こんにちは")[0] == "Hello there"
    state = {"context": [{"speaker": "A", "text": "六時？"}], "target": "それで"}
    fenced = '```json\n{"context": [{"speaker": "A", "text": "Six?"}], "target": "That works"}\n```'
    out, meta = translate_one(_fake_client(fenced), state)
    assert out["target"] == "That works" and out["context"][0]["speaker"] == "A"
    assert meta["returned_model"] == "m"
    with pytest.raises(RuntimeError, match="changed structure"):
        translate_one(_fake_client('{"target": "only target"}'), state)
