import json
import textwrap
from types import SimpleNamespace

import pytest

from bench.clients.adapter import LLMAdapter, build_prompt, to_canonical_answer
from bench.clients.base import ProviderError, backoff_delay
from bench.dataset import load_question
from bench.questions import AnswerError, normalize_answer, to_request


def test_questions_map_back_to_label_space():
    q = load_question("t1_moderation", "ja")
    sq = to_request(q)
    assert list(sq["criteria"]) == ["問題なし", "嫌がらせ", "詐欺", "転売スパム"]
    ans = {"type": "choice", "choice": "嫌がらせ",
           "probabilities": {"問題なし": 0.1, "嫌がらせ": 0.7, "詐欺": 0.1, "転売スパム": 0.1}, "confidence": 0.6}
    norm = normalize_answer(q, ans)
    assert norm["prediction"] == "harassment"
    assert norm["probabilities"]["harassment"] == 0.7


def test_score_prediction_is_argmax_and_expected_level_is_recomputed():
    q = load_question("t2_politeness_anger", "en")
    ans = {"type": "score", "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65}, "score": 99.0, "confidence": 0.78}
    norm = normalize_answer(q, ans)
    assert norm["prediction"] == 2
    assert norm["score"] == pytest.approx(0.3 + 1.3)
    # ties go to the lower level
    tie = normalize_answer(q, {"type": "score", "probabilities": {"0": 0.4, "1": 0.4, "2": 0.2}})
    assert tie["prediction"] == 0


def test_normalize_rejects_mismatched_answers():
    q = load_question("t4_ellipsis", "en")
    with pytest.raises(AnswerError):
        normalize_answer(q, {"type": "choice", "choice": "accept", "probabilities": {"accept": 1.0}})
    with pytest.raises(AnswerError):
        normalize_answer(q, {"type": "binary", "p": 0.3})
    with pytest.raises(AnswerError):
        normalize_answer(load_question("t5_noise", "en"), {"type": "binary", "p": 1.5})


def test_backoff_honors_retry_after():
    assert backoff_delay(0, "10") >= 10
    assert backoff_delay(10) <= 30.5


def test_adapter_prompt_and_schema():
    sq = to_request(load_question("t1_moderation", "ja"))
    prompt, schema = build_prompt("テスト", sq, "ja")
    assert "テスト" in prompt and "嫌がらせ" in prompt
    assert schema["properties"]["choice"]["enum"] == list(sq["criteria"])
    assert schema["properties"]["probabilities"]["required"] == list(sq["criteria"])
    _, sschema = build_prompt({"context": [], "target": "x"}, to_request(load_question("t2_politeness_anger", "en")),
                              "en")
    assert sschema["properties"]["probabilities"]["required"] == ["0", "1", "2"]


def test_adapter_normalizes_self_reported_probabilities():
    sq = {"type": "score", "criteria": ["a", "b", "c"]}
    ans = to_canonical_answer(sq, {"probabilities": {"0": 0, "1": 1, "2": 3}})
    assert ans["probabilities"] == {"0": 0.0, "1": 0.25, "2": 0.75}
    assert ans["score"] == pytest.approx(1.75)
    assert ans["confidence"] is None
    assert to_canonical_answer({"type": "binary"}, {"p_yes": 1.4})["p"] == 1.0


def _fake_client(content, capture=None):
    class FakeMessages:
        def create(self, **kw):
            if capture is not None:
                capture.update(kw)
            return SimpleNamespace(
                stop_reason="end_turn", model="claude-haiku-4-5-20251001", content=content,
                usage=SimpleNamespace(input_tokens=50, output_tokens=8),
                model_dump=lambda mode="json": {"id": "msg_x", "model": "claude-haiku-4-5-20251001"},
            )

    return SimpleNamespace(messages=FakeMessages())


def test_llm_adapter_with_fake_client():
    captured = {}
    content = [SimpleNamespace(type="text", text=json.dumps({"p_yes": 0.2}))]
    adapter = LLMAdapter(model="claude-haiku-4-5-20251001", lang="en", client=_fake_client(content, captured))
    res = adapter.ask("state", {"type": "binary", "instructions": "Refund?", "criteria": {"true": "y", "false": "n"}})
    assert res.answer == {"type": "binary", "p": 0.2}
    assert captured["temperature"] == 0.0
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in captured


def test_llm_adapter_empty_or_non_json_is_a_provider_error():
    adapter = LLMAdapter(model="m", lang="en", client=_fake_client([]))
    with pytest.raises(ProviderError, match="no text block"):
        adapter.ask("s", {"type": "binary", "instructions": "?"})
    adapter = LLMAdapter(model="m", lang="en", client=_fake_client([SimpleNamespace(type="text", text="oops")]))
    with pytest.raises(ProviderError, match="not JSON"):
        adapter.ask("s", {"type": "binary", "instructions": "?"})


def test_plugin_providers_are_discovered(tmp_path, monkeypatch):
    (tmp_path / "demo.py").write_text(textwrap.dedent('''
        from bench.clients.mock import MockClient
        PROVIDER = {"name": "demo", "provider": "demo-co", "route": "direct", "model": "demo-2.0",
                    "conditions": ["A"],
                    "price": {"input_per_mtok": 1.0, "output_per_mtok": 0.0, "checked_at": "2026-01-01"}}
        def make_client(condition, lang):
            return MockClient(model="demo-2.0")
    '''), encoding="utf-8")
    from bench import providers

    monkeypatch.setenv("BENCH_PROVIDER_PLUGINS", str(tmp_path))
    providers._plugins.cache_clear()
    try:
        assert "demo" in providers.registry()
        assert providers.make_client("demo", "A").model == "demo-2.0"
        assert providers.price_of("demo-2.0")["provider"] == "demo-co"
        with pytest.raises(SystemExit):
            providers.make_client("demo", "B")
    finally:
        providers._plugins.cache_clear()


def test_llm_adapter_passes_baseline_credential_explicitly(monkeypatch, tmp_path):
    import anthropic

    captured = {}

    class FakeAnthropic:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setenv("DECISION_BENCH_CREDENTIALS_MAP", str(tmp_path / "none.json"))
    monkeypatch.setenv("DECISION_BENCH_BASELINE_API_KEY", "baseline-test-key")
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    LLMAdapter(model="m", lang="en")
    assert captured["api_key"] == "baseline-test-key"
    assert captured["max_retries"] == 0


def test_credentials_map_can_rename_variables(monkeypatch, tmp_path):
    from bench import credentials

    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"baseline": "MY_LOCAL_VAR"}), encoding="utf-8")
    monkeypatch.setenv("DECISION_BENCH_CREDENTIALS_MAP", str(mapping))
    monkeypatch.setenv("MY_LOCAL_VAR", "local-key")
    assert credentials.env_var_for("baseline") == "MY_LOCAL_VAR"
    assert credentials.get_credential("baseline") == "local-key"
    monkeypatch.delenv("MY_LOCAL_VAR")
    assert credentials.get_credential("baseline") is None
