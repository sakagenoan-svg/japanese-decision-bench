import json

from bench import config
from bench.dataset import read_jsonl, write_jsonl
from bench.sanitize import sanitize_file, sanitize_record


def _record(provider="example"):
    return {
        "run_id": "example-A-20260921T000000Z-abc123",
        "example_id": "t1-001",
        "provider": provider,
        "requested_model": "example-model-1.0",
        "input_tokens": 10,
        "output_tokens": 0,
        "answer": {"type": "binary", "p": 0.5},
        "error": None,
        "raw_response": {
            "model": "example-model-1.0",
            "answer": {"type": "binary", "p": 0.5},
            "usage": {"input_tokens": 10, "output_tokens": 0, "account_id": "acct_123"},
            "request_id": "req_abcdef",
            "trace_id": "tr_1",
            "metadata": {"org": "x"},
        },
        "headers": {"Authorization": "Bearer sk-live-abcdefghijklmnopqrstuvwxyz"},
        "debug": "loaded from C:\\Users\\someone\\secret\\file.json and /home/someone/x; key sk-abcdefghijklmnopqrstuv",
    }


def test_raw_response_is_reduced_to_allow_list():
    r = sanitize_record(_record())
    assert set(r["raw_response"]) == {"model", "answer", "usage"}
    assert r["raw_response"]["usage"] == {"input_tokens": 10, "output_tokens": 0}


def test_secret_keys_and_strings_are_removed():
    r = sanitize_record(_record())
    dumped = json.dumps(r)
    assert "headers" not in r
    for needle in ("sk-live", "sk-abc", "acct_123", "req_abcdef", "someone", "tr_1"):
        assert needle not in dumped, needle
    assert r["example_id"] == "t1-001" and r["run_id"].startswith("example-A")
    assert r["input_tokens"] == 10


def test_anthropic_message_id_is_dropped():
    rec = _record("anthropic")
    rec["raw_response"] = {"id": "msg_01ABC", "model": "claude-haiku-4-5-20251001", "content": [{"type": "text",
                           "text": "{}"}], "usage": {"input_tokens": 1, "output_tokens": 2,
                                                     "service_tier": "standard"}}
    r = sanitize_record(rec)
    assert "id" not in r["raw_response"]
    assert r["raw_response"]["usage"] == {"input_tokens": 1, "output_tokens": 2}


def test_results_stay_private_without_permission(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEGAL_STATUS_PATH", tmp_path / "missing.json")
    src = tmp_path / "example-A.jsonl"
    write_jsonl(src, [_record()])
    dest, public = sanitize_file(src, public_dir=tmp_path / "public", private_dir=tmp_path / "private")
    assert not public
    assert dest.parent == tmp_path / "private"
    assert not (tmp_path / "public").exists()


def test_results_public_only_when_provider_listed(tmp_path, monkeypatch):
    status = tmp_path / "legal-status.json"
    status.write_text(json.dumps({"publishable_providers": ["example"]}), encoding="utf-8")
    monkeypatch.setattr(config, "LEGAL_STATUS_PATH", status)
    src = tmp_path / "example-A.jsonl"
    write_jsonl(src, [_record()])
    dest, public = sanitize_file(src, public_dir=tmp_path / "public", private_dir=tmp_path / "private")
    assert public and dest.parent == tmp_path / "public"
    assert read_jsonl(dest)[0]["raw_response"]["model"] == "example-model-1.0"


def test_providers_need_explicit_opt_in(tmp_path, monkeypatch):
    status = tmp_path / "legal-status.json"
    status.write_text(json.dumps({"benchmark_publication": "allowed"}), encoding="utf-8")
    monkeypatch.setattr(config, "LEGAL_STATUS_PATH", status)
    assert not config.publication_allowed("anthropic")  # other keys never imply publication
    status.write_text(json.dumps({"publishable_providers": ["anthropic"]}), encoding="utf-8")
    assert config.publication_allowed("anthropic")
    assert not config.publication_allowed("example")
