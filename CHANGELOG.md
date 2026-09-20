# Changelog

## After protocol-v1

- **Baseline runner compatibility fix (no protocol change).** The first baseline run under condition A failed
  on all 160 examples with `TypeError: Messages.create() got an unexpected keyword argument 'temperature'`:
  the Anthropic Python SDK v1 dropped `temperature` (and `top_p` / `top_k`) from the generated
  `messages.create()` signature, while the API still accepts it for the pinned model. The failure happened
  before any request was built — **0 HTTP requests were sent and no prediction was produced** (latency, token
  counts and `returned_model` are null on every record). The adapter now sends the same value through
  `extra_body`, so the frozen sampling setting (`temperature = 0.0`) is unchanged and no other sampling
  parameter was introduced. This is a transport fix, not a protocol change: `protocol-v1` and its tag are
  untouched, and the runs already completed for the model under test are unaffected (different client).
- The fake Anthropic client in the tests accepted `**kwargs`, which is why the mismatch reached a run. It now
  mirrors the SDK signature with explicit keywords only, so passing a dropped parameter fails in unit tests.
- Environment for the benchmark execution: Anthropic SDK 1.7.0 (as locked in `uv.lock`).

## protocol-v1

The benchmark design is frozen at this tag. No benchmark prediction had been produced or inspected when it was
created: `results/raw/` was empty and no provider had been run outside a synthetic connectivity check that uses
no dataset example.

Frozen by this tag:

- **Dataset**: 160 examples. T1 `ok` 8 / `harassment` 8 / `scam_risk` 7 / `resale_spam` 7; T2 ten per level;
  T3 40 (10 bases × 4 variant types); T4 `accept` 8 / `reject` 8 / `question` 7 / `other` 7; T5 15 true / 15
  false over 10 bases × clean, medium, heavy. Gold labels reviewed by the maintainer (Japanese, 100/100).
- **Questions**: Japanese and English definitions for T1, T2, T4, T5.
- **Condition coverage**: A/B/C for the decision model under test; **condition A only for the baseline**,
  which is an external reference point in the native Japanese condition rather than a second factorial arm.
  The baseline is also the condition C translator, so evaluating it under C would measure a self-pipeline.
- **Model snapshots**: baseline `claude-haiku-4-5-20251001`, condition C translator
  `claude-haiku-4-5-20251001`. Moving aliases (`-latest`, `-preview`) are rejected by the provider loader.
- **Translations**: 160 frozen rows, digest
  `9da276b1a060936430b41f2ed4db4d61d6c40dce46a3aed8d7d0f20e773d346b`, generated once and never post-edited.
  Translation collision on the frozen artifact: 0/40 exact, 1/40 ignoring case, punctuation and emoji
  (`t3-023`). These come from the translation artifact, not from any prediction.
- **Metrics**: accuracy, macro-F1, log loss, MAE, flip rate (overall and by variant type), gold retention,
  AUROC, Brier, coverage/accuracy at the certainty threshold, kappa, translation collision.
- **Thresholds and constants**: high certainty 0.9, binary uncertainty band 0.1–0.9, decision threshold 0.5,
  log-loss epsilon 1e-6, bootstrap 2000, seed 20260920 (bootstrap CI and T5 noise selection).
- **Retry and timeout**: 5 retries, exponential backoff with jitter, `Retry-After` honored, SDK-side retries
  disabled, 30 s request timeout.
- **Exclusion rules**: API errors are counted as `n_error`, never dropped; no example is removed after results
  are seen; the primary sample set includes every example, including translation-audit findings.
- **T3 interpretation**: conditions A and B measure model surface-form robustness; condition C measures
  translation-pipeline surface-form robustness and is not read as the same quantity.

Changing any of the above after results are seen requires `protocol-v2` and an entry here.

## Unreleased

- Dataset v0.1 draft: T1–T5, 160 evaluated inputs (100 authored, 60 derived).
- JA / EN question definitions for all tasks.
- Harness: provider-neutral client interface, provider registry with local plugins, LLM baseline adapter,
  offline mock, resumable runner, validation, metrics, sanitization, and a report generator. No API calls in CI.
- Protocol draft (not yet locked).
- Condition C translation is pinned to the same model snapshot as the baseline provider
  (settings fix made before protocol lock, and before any benchmark prediction was inspected).
- Frozen translations carry a `review_status`; validation rejects a translation without one.
- Condition C redefined before protocol lock, and before any benchmark prediction was inspected: the
  translations are **frozen raw machine translation — machine-translated once with a pinned snapshot, frozen,
  structurally validated, and AI-audited for translation drift; not human post-edited**. This replaces the
  earlier design of "machine translation with a human-reviewed and corrected translation". Hand-correcting
  them would measure `machine translation -> human post-edit -> decision model` and hide the errors a real
  translation pipeline produces, so known failures are kept and reported as pipeline behaviour.
- Bilingual human review of the translations removed from the protocol, because it was not something this
  project can honestly claim: the maintainer approved keeping the frozen raw translations after reading the
  audit findings, but did not independently review English pragmatics. No "160/160 human reviewed" claim is
  made anywhere, and `review_status` is documented as a legacy field that gates nothing.
- Translation phase gated by validation instead: row count, pinned translator snapshot, source consistency,
  T4 structure, no empty translations, and a recorded SHA-256 over `id + source + translation` that fails if
  any translation was edited or regenerated.
- T3 under condition C documented as translation-pipeline surface-form robustness, with translation collision
  and translation drift recorded per variant type.
