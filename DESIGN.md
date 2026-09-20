# japanese-decision-bench — Design

This document describes what the benchmark measures and why. [PROTOCOL.md](./PROTOCOL.md) covers the run
procedure and the freeze rules. [ANNOTATION.md](./ANNOTATION.md) covers the labeling rules.

## 1. Scope

The benchmark measures **decisions** on short Japanese texts: single-label classification, ordinal scoring, and
yes/no probability. It does not measure text generation, factual knowledge, or reasoning chains.

Success means reporting reproducibly how behavior changes on Japanese-specific phenomena. Any outcome counts as a
valid result: large differences, small differences, or none.

## 2. Research questions

**RQ1.** How does classification and scoring behave on indirect expressions, sarcasm, polite anger and omitted
subjects (T1, T2, T4)?

**RQ2.** How do results change across three conditions, **within one provider**? The baseline is evaluated
only under condition A, as an external reference point in the native Japanese condition; A/B/C is not run as a
factorial comparison across providers (PROTOCOL.md §2.2).

| Condition | State | Questions / criteria | Interpretation |
|---|---|---|---|
| A | Japanese | Japanese | Direct Japanese use |
| B | Japanese | English | Effect of question/criteria language, with the state held fixed |
| C | Machine-translated English | English | Translation *pipeline*, not a pure language effect: translation errors and normalization are part of the treatment |

Condition C is `raw machine translation -> English questions -> decision model`. The frozen translations are
**machine-translated once, structurally validated, and AI-audited for translation drift; not human
post-edited**. Hand-correcting them would measure `machine translation -> human post-edit -> decision model`,
a different pipeline that hides the translation errors real users would hit. A curated-translation variant, if
it is ever run, is **exploratory**, reported separately, and never overwrites the raw translations.

The audit compared all 160 translations against their Japanese source and is recorded with the run. It is an
AI-assisted audit, not a bilingual human review: the maintainer reviewed the audit findings and approved
retaining the frozen raw translations, but did not independently verify English pragmatics. Nothing in this
benchmark describes the translations as human-reviewed or human-validated.

**RQ3.** How much do predictions change under surface-form variation (T3)?

**RQ4.** If only high-certainty answers are processed automatically, how do accuracy and coverage trade off?

## 3. Tasks

All texts are synthetic and set in a fictional peer-to-peer goods exchange service.

### T1 — Indirect / sarcastic moderation (Choice, 30)

Labels: `ok` (8), `harassment` (8), `scam_risk` (7), `resale_spam` (7). Items that would need more than one label are
avoided. Hard cases include sarcasm phrased as praise, insults in honorific language, veiled threats, and
platform-specific scams (for example, asking for a receipt rating before the item arrives). Non-sarcastic uses
of the same surface phrases act as controls (e.g. a genuine 「さすがですね」).

### T2 — Polite anger (Score, 30)

Levels: 0 neutral, 1 dissatisfied but calm, 2 strongly angry. There are 10 items per level. Level 2 is mostly
written in keigo, so politeness and anger point in opposite directions.

### T3 — Surface-form robustness (Choice, 40)

Ten T1 items (3 `ok`, 3 `harassment`, 2 `scam_risk`, 2 `resale_spam`) were marked `t3_base` **before any model
output existed**. They were chosen for comparatively unambiguous gold labels. Each has four variants:

| `variant_type` | Construction |
|---|---|
| `hiragana` | hand-written; kanji/katakana → hiragana, content unchanged |
| `katakana_halfwidth` | mechanical: hiragana variant → katakana → half-width katakana (full-width punctuation such as ！？（） is left as is) |
| `emoji_kaomoji` | hand-written; emoji/kaomoji added, wording unchanged |
| `internet_slang` | hand-written; rewritten in casual net slang with the same intent |

Variants keep the base's gold label. The originals are not re-queried. T3 compares each variant with the T1
prediction for its base under the same model and condition.

Emoji and slang can shift pragmatics as well as spelling. The construct is therefore called *surface-form*
robustness, and the per-variant breakdown is always reported.

**T3 under condition C measures something different.** Japanese surface variation passes through machine
translation before the model sees it, so the English inputs may carry less of the variation than the Japanese
ones did (writing-system variants have no English counterpart at all), and the translator may also mistranslate
one variant and not another. A condition C flip therefore combines three sources — the decision model,
translation normalization, and translation errors — and is reported as *translation-pipeline* surface-form
robustness. It is not compared with conditions A and B as if it were a property of the model. Two diagnostics
are recorded alongside the flip rate, per `variant_type`:

| Diagnostic | Definition | Source |
|---|---|---|
| Translation collision | share of variants whose English translation is identical to their base's | `bench/metrics.py` (`translation_collision`), computed from the frozen translations |
| Translation drift | share of variants the translation audit flagged as changing meaning, polarity, register or speaker | the translation audit record, not the harness |

### T4 — Ellipsis / context (Choice, 30)

State is a JSON object `{"context": [{"speaker", "text"}...], "target": "..."}`. `target` is the reply by the
participant who did not speak last. Labels: `accept` (8), `reject` (8), `question` (7), `other` (7). Typical
difficulties are one-word replies (「ぜひ」), trailing-off refusals (「ちょっとそれは…」), and idioms
(「お気持ちだけいただいておきます」).

### T5 — Long-text noise (Binary, 30)

Question: *Does the customer request a refund?* There are 10 clean support messages (5 yes, 5 no). The no-side
includes hard negatives: an exchange request that says the money can stay, and a question about a past refund.
Each clean message also has two derived versions:

| Level | Construction | Approx. length |
|---|---|---|
| `clean` | the authored message | 38–70 chars |
| `medium` | + opener, one unrelated aside, closing | ~165–195 chars |
| `heavy` | + two openers, four asides, signature block, two quoted earlier support threads | ~580–610 chars |

Distractors are chosen with a fixed seed from pools that never mention money, refunds, cancellation or returns,
so the label cannot change. The clean text is always contained verbatim, and validation checks this.

## 4. Dataset construction

| Source | Meaning | Count in v0.1 draft |
|---|---|---|
| `handwritten` | written by the maintainer | 0 |
| `llm_assisted` | drafted with an LLM; gold label assigned by the maintainer | 100 (T1, T2, T4, T5-clean) |
| `derived` | produced from another sample by a documented rule | 60 (T3, T5 medium/heavy) |

`scripts/build_dataset.py` holds every authored text and generates the JSONL files. The tests check that the
shipped data equals the script output. Validation (`bench/validate.py`, run in CI) checks the following: ids,
required fields, label sets, T1 label balance, T3 base pre-selection and completeness, T5 level completeness
and verbatim containment, and patterns for e-mail addresses, phone numbers, API keys and long identifiers.

**Known risk.** Because authored items were LLM-drafted, the phrasing may be more "textbook" than real user
text. LLM baselines may also find such text systematically easier. This is reported as a limitation. Replacing
or supplementing items with handwritten ones before the protocol lock is encouraged. Any such replacement is
recorded through `source`.

## 5. Questions

Each task has one JA and one EN question file in `data/questions/`. For Choice questions, the file separates the
displayed option name (`key`, e.g. 「嫌がらせ」 or `harassment`) from the gold `label`. JA questions can
therefore use Japanese option names while every result is scored in one label space. The EN file is a
translation of the JA file with the same option semantics.

Decision models can read instructions literally, so the instructions say explicitly what to judge (for example,
"judge sarcasm by what the writer actually means"). The criteria describe boundary cases rather than giving bare
labels.

## 6. Providers and adapters

All providers implement one interface (`bench/clients/base.py`). They receive a canonical question
(`choice` / `score` / `binary` with instructions and criteria) and return a canonical answer (probabilities and,
where the provider has one, a native confidence value). Each provider maps to and from its own wire format.

| Provider | Model | Route | Settings |
|---|---|---|---|
| Baseline (general-purpose LLM) | `claude-haiku-4-5-20251001` | direct, official SDK | temperature 0, no extended thinking, JSON-schema structured output |
| Mock | `mock-0` | local | deterministic pseudo-random answers, never reported |
| Local plugins | pinned ID declared by the plugin | declared by the plugin | declared by the plugin |

The baseline adapter (`bench/clients/adapter.py`) shows the same input, instructions and options, and asks
for one probability per option (or `p_yes`). Probabilities are clipped at 0 and renormalized.

**Comparability.** A dedicated decision model may return probabilities and a native confidence from its own
calibration procedure. An LLM returns *self-reported* numbers in text. Accuracy-type metrics are comparable
across providers. Certainty and calibration-type metrics are reported per provider and are **not** interpreted
as the same mechanism.

Prediction rules, identical across providers:

- Choice → the returned choice
- Score → argmax of the level probabilities (ties go to the lower level). The expected level is recomputed as
  the probability-weighted mean for every provider.
- Binary → `p(yes)`. The decision is `p ≥ 0.5`.

## 7. Metrics

| Task | Metric | Definition |
|---|---|---|
| T1, T4 | Accuracy | share of `prediction == gold` |
| | Macro-F1 | unweighted mean of per-label F1 over the full label set |
| | Log loss | mean `−ln p(gold)`, with `p` clipped to [1e−6, 1−1e−6] |
| T2 | MAE (primary) | mean \|expected level − gold\| |
| | Argmax accuracy | share of `argmax == gold` |
| T3 | Flip Rate (primary) | share of variants with `prediction(variant) ≠ prediction(original)` |
| | Flip Rate by variant type | same, per `variant_type` (n = 10 each) |
| | Gold retention | among bases predicted correctly on the original, share of their variants still correct |
| | Confidence delta | mean `certainty(variant) − certainty(original)` |
| | Probability delta | mean `p_variant(gold) − p_original(gold)` |
| | Translation collision (condition C only) | share of variants whose English translation equals their base's |
| T5 | AUROC (primary) | Mann–Whitney, ties count ½ |
| | Brier score | mean `(p − y)²` |
| | Probability delta by level | mean `p(level) − p(clean)` for the same base, plus its absolute value |
| All | Latency | client-observed end-to-end time of the successful attempt; p50 / p95 |
| All | Cost | tokens × public list price on the check date |

### High-certainty processing (RQ4)

- Choice / Score: *certainty* = the provider's native `confidence` if it returns one, else max probability
  (e.g. the LLM adapter). An answer is high-certainty if certainty ≥ **0.9**.
- Binary: an answer is high-certainty if `p ≤ 0.1` or `p ≥ 0.9`.
- Reported: coverage (share of high-certainty answers) and accuracy on the covered subset. A certainty-threshold
  curve (0 to 1, step 0.05) is also drawn.
- 0.9 is a pre-registered representative threshold. It is not claimed to be optimal.

### Calibration

With n = 30 per task, a 10-bin ECE is too noisy to be a headline metric. The report uses Brier score, the
certainty distribution, and the accuracy–coverage curve. ECE may appear in an appendix only.

### Statistical uncertainty

95% percentile bootstrap intervals (2,000 resamples, fixed seed):

- A/B/C comparisons are **paired**: examples are resampled and both conditions are kept for each one.
- T3 resamples **bases** (all four variants together), not variant rows.

Bootstrap intervals are secondary. If they are not available, that does not delay a release.

## 8. Latency and cost

Latency is measured around the single HTTP call that succeeded. Retries and backoff are recorded separately
(`retry_count`, `total_elapsed_ms`). Requests are sent sequentially. Different providers use different network
routes, so the report calls this *observed API latency*, never model inference speed.

Cost is estimated from public list prices only (`bench/cost.py`), with the date each price was checked. The
one-time translation cost for condition C is reported separately from per-request benchmark cost.

## 9. Figures

At most four figures:

1. primary metric by task
2. conditions A / B / C
3. T3 flip rate by variant type
4. certainty threshold vs. coverage / accuracy

## 10. Out of scope for v0.1

Repeated runs to measure nondeterminism, local/open-weight models, a second additional LLM, and a
normalization-layer experiment. Any of these added after results are seen will be labeled *exploratory*.
