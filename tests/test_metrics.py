import math

import pytest

from bench import metrics as M


def test_accuracy_and_macro_f1():
    gold = ["a", "a", "b", "c"]
    pred = ["a", "b", "b", "a"]
    assert M.accuracy(gold, pred) == 0.5
    # a: tp1 fp1 fn1 -> 0.5 ; b: tp1 fp1 fn0 -> 2/3 ; c: tp0 fn1 -> 0
    assert M.macro_f1(gold, pred, ["a", "b", "c"]) == pytest.approx((0.5 + 2 / 3 + 0) / 3)


def test_log_loss_clips():
    assert M.log_loss(["a"], [{"a": 1.0}]) == pytest.approx(-math.log(1 - 1e-6))
    assert M.log_loss(["a"], [{"a": 0.0}]) == pytest.approx(-math.log(1e-6))
    assert M.log_loss(["a", "b"], [{"a": 0.5, "b": 0.5}, {"a": 0.25, "b": 0.75}]) == pytest.approx(
        (-math.log(0.5) - math.log(0.75)) / 2)


def test_mae():
    assert M.mae([0, 1, 2], [0.5, 1.0, 1.0]) == pytest.approx(0.5)


def test_auroc_and_brier():
    assert M.auroc([True, False], [0.9, 0.1]) == 1.0
    assert M.auroc([True, False], [0.1, 0.9]) == 0.0
    assert M.auroc([True, False], [0.5, 0.5]) == 0.5
    assert M.auroc([True, True], [0.5, 0.6]) is None
    assert M.auroc([True, True, False, False], [0.8, 0.4, 0.6, 0.2]) == pytest.approx(0.75)
    assert M.brier([True, False], [0.8, 0.4]) == pytest.approx((0.04 + 0.16) / 2)


def test_cohen_kappa_known_values():
    # perfect agreement
    assert M.cohen_kappa(["x", "y"], ["x", "y"], ["x", "y"]) == pytest.approx(1.0)
    # textbook 2x2: [[20,5],[10,15]] -> po=0.7 pe=0.5 -> 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert M.cohen_kappa(a, b, ["y", "n"]) == pytest.approx(0.4)


def test_weighted_kappa_penalizes_far_disagreement_more():
    near = M.cohen_kappa([0, 1, 2, 0], [0, 1, 1, 0], [0, 1, 2], weights="linear")
    far = M.cohen_kappa([0, 1, 2, 0], [0, 1, 0, 0], [0, 1, 2], weights="linear")
    assert near > far


def test_percentile_matches_linear_interpolation():
    assert M.percentile([1, 2, 3, 4], 50) == 2.5
    assert M.percentile([10], 95) == 10
    assert M.percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
    assert M.percentile([], 50) is None


def test_high_certainty_rules():
    assert M.is_high_certainty({"type": "choice", "confidence": 0.9, "probabilities": {"a": 0.5}})
    assert not M.is_high_certainty({"type": "choice", "confidence": 0.89, "probabilities": {"a": 0.99}})
    # no provider confidence -> max probability (LLM adapter)
    assert M.is_high_certainty({"type": "choice", "confidence": None, "probabilities": {"a": 0.95, "b": 0.05}})
    assert M.is_high_certainty({"type": "binary", "p": 0.1})
    assert M.is_high_certainty({"type": "binary", "p": 0.9})
    assert not M.is_high_certainty({"type": "binary", "p": 0.5})


def test_coverage_accuracy():
    r = M.coverage_accuracy([True, False, True, True], [True, True, False, False])
    assert r["coverage"] == 0.5 and r["accuracy_covered"] == 0.5 and r["accuracy_all"] == 0.75


def _rec(ex_id, task, gold, pred, probs, conf=0.5):
    return {"example_id": ex_id, "task": task, "gold": gold, "error": None, "latency_ms": 100.0,
            "answer": {"type": "choice", "prediction": pred, "probabilities": probs, "confidence": conf}}


def test_flip_metrics():
    examples = {
        "t3-1": {"variant_of": "t1-1", "variant_type": "hiragana"},
        "t3-2": {"variant_of": "t1-1", "variant_type": "internet_slang"},
        "t3-3": {"variant_of": "t1-2", "variant_type": "hiragana"},
    }
    originals = [_rec("t1-1", "t1", "ok", "ok", {"ok": 0.8, "scam_risk": 0.2}, 0.8),
                 _rec("t1-2", "t1", "scam_risk", "ok", {"ok": 0.6, "scam_risk": 0.4}, 0.6)]
    variants = [_rec("t3-1", "t3", "ok", "ok", {"ok": 0.7, "scam_risk": 0.3}, 0.7),
                _rec("t3-2", "t3", "ok", "scam_risk", {"ok": 0.4, "scam_risk": 0.6}, 0.6),
                _rec("t3-3", "t3", "scam_risk", "ok", {"ok": 0.5, "scam_risk": 0.5}, 0.5)]
    m = M.flip_metrics(variants, originals, examples)
    assert m["n"] == 3
    assert m["flip_rate"] == pytest.approx(1 / 3)
    assert m["gold_retention"] == pytest.approx(0.5)  # only t1-1 was correct; 1 of its 2 variants stays correct
    assert m["by_variant_type"]["internet_slang"]["flip_rate"] == 1.0
    assert m["by_variant_type"]["hiragana"]["flip_rate"] == 0.0
    assert m["probability_delta"] == pytest.approx(((0.7 - 0.8) + (0.4 - 0.8) + (0.5 - 0.4)) / 3)
    assert m["confidence_delta"] == pytest.approx(((0.7 - 0.8) + (0.6 - 0.8) + (0.5 - 0.6)) / 3)
    lo, hi = m["flip_rate_ci95"]
    assert 0 <= lo <= hi <= 1


def test_errors_are_excluded_and_counted():
    recs = [_rec("a", "t1", "ok", "ok", {"ok": 1.0}), {"example_id": "b", "gold": "ok", "error": "x", "answer": None}]
    m = M.choice_metrics(recs, ["ok"])
    assert m["n"] == 1 and m["n_error"] == 1 and m["accuracy"] == 1.0


def test_bootstrap_ci_is_deterministic():
    units = list(range(20))
    stat = lambda us: sum(us) / len(us)  # noqa: E731
    assert M.bootstrap_ci(units, stat) == M.bootstrap_ci(units, stat)
    lo, hi = M.bootstrap_ci(units, stat)
    assert lo < 9.5 < hi


def test_translation_collision_counts_identical_translations_per_variant_type():
    from bench.metrics import translation_collision

    variants = [
        {"id": "t3-001", "variant_of": "t1-001", "variant_type": "hiragana"},
        {"id": "t3-002", "variant_of": "t1-001", "variant_type": "katakana_halfwidth"},
        {"id": "t3-003", "variant_of": "t1-001", "variant_type": "emoji_kaomoji"},
    ]
    translations = {
        "t1-001": {"translation": "Shipped today."},
        "t3-001": {"translation": "Shipped today."},          # collided with the base
        "t3-002": {"translation": "Shipped it out today."},   # still distinct
        "t3-003": {"translation": "Shipped today. 📦"},
    }
    out = translation_collision(variants, translations)
    assert out["n"] == 3 and out["overall"] == 1 / 3
    assert out["by_variant_type"]["hiragana"]["collision"] == 1.0
    assert out["by_variant_type"]["katakana_halfwidth"]["collision"] == 0.0


def test_translation_collision_skips_examples_without_a_translation():
    from bench.metrics import translation_collision

    variants = [{"id": "t3-001", "variant_of": "t1-001", "variant_type": "hiragana"}]
    assert translation_collision(variants, {})["overall"] is None
    assert translation_collision(variants, {})["n"] == 0
