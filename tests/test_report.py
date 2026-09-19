from bench.clients.mock import MockClient
from bench.cost import estimate, estimate_records
from bench.report import build_report
from bench.run import run


def test_report_with_no_results_says_so(tmp_path):
    src = tmp_path / "sanitized"
    src.mkdir()
    path = build_report(src, tmp_path / "out")
    text = path.read_text(encoding="utf-8")
    assert "No benchmark results are published" in text
    assert not (tmp_path / "out" / "tables").exists()


def test_end_to_end_mock_run_and_report(tmp_path):
    raw = tmp_path / "raw"
    for cond in ("A", "B"):
        run("mock", cond, ["t1_moderation", "t2_politeness_anger", "t3_surface_variants", "t4_ellipsis", "t5_noise"],
            limit=None, smoke=False, fresh=True, client=MockClient(), out=raw / f"mock-{cond}.jsonl")
    # mock results are excluded unless explicitly included
    assert "No benchmark results" in build_report(raw, tmp_path / "public").read_text(encoding="utf-8")

    out = tmp_path / "private"
    text = build_report(raw, out, include_mock=True).read_text(encoding="utf-8")
    assert "| mock | mock-0 | A |" in text and "| mock | mock-0 | B |" in text
    assert "B-A" in text  # paired condition delta
    figs = sorted(p.name for p in (out / "figures").iterdir())
    assert len(figs) <= 4
    assert (out / "tables" / "metrics.json").exists()
    assert (out / "costs" / "public-price-estimate.json").exists()


def test_run_resumes_without_duplicates(tmp_path):
    out = tmp_path / "mock-A.jsonl"
    run("mock", "A", ["t1_moderation"], limit=5, smoke=False, fresh=True, client=MockClient(), out=out)
    run("mock", "A", ["t1_moderation"], limit=10, smoke=False, fresh=False, client=MockClient(), out=out)
    from bench.dataset import read_jsonl

    ids = [r["example_id"] for r in read_jsonl(out)]
    assert len(ids) == len(set(ids)) == 10


def test_cost_estimate_uses_public_prices():
    e = estimate("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=0, n_requests=1000)
    assert e["estimated_cost"] == 1.0 and e["estimated_cost_per_1000"] == 1.0
    h = estimate("claude-haiku-4-5-20251001", 1_000_000, 1_000_000, 10)
    assert h["estimated_cost"] == 6.0
    assert estimate("unknown-model", 1, 1, 1)["estimated_cost"] is None
    recs = [{"requested_model": "claude-haiku-4-5-20251001", "input_tokens": 100, "output_tokens": 0, "error": None},
            {"requested_model": "claude-haiku-4-5-20251001", "input_tokens": 100, "output_tokens": 0, "error": "boom"}]
    assert estimate_records(recs)[0]["requests"] == 1


def test_alias_models_are_refused(monkeypatch):
    import pytest

    from bench import config, providers

    monkeypatch.setitem(config.PROVIDERS["mock"], "model", "some-model-latest")
    with pytest.raises(SystemExit, match="alias"):
        providers.make_client("mock", "A")
