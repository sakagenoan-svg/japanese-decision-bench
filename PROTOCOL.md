# Run protocol

Status: **locked as `protocol-v1`.** The items in §1 are frozen as of that tag, which was created before any
benchmark prediction was produced or inspected. Changing a frozen item requires `protocol-v2` and a
[CHANGELOG](./CHANGELOG.md) entry; tags are never moved or overwritten.

## 1. What is frozen by `protocol-v1`

| Item | Where |
|---|---|
| Dataset and gold labels | `data/*.jsonl`, `scripts/build_dataset.py` |
| Task definitions, label sets, T3 bases, T5 levels | `bench/config.py` (`TASKS`), `data/t1_moderation.jsonl` (`t3_base`) |
| Japanese and English questions | `data/questions/*.json` |
| Frozen EN translations (condition C) | `data/translations/en.jsonl`, digest in `bench/config.py` |
| Model IDs, provider routes, sampling / thinking settings | `bench/config.py` (`PROVIDERS`) |
| LLM adapter prompts and output schema | `bench/clients/adapter.py` |
| Metric definitions, thresholds, clipping, seed | `bench/metrics.py`, `bench/config.py` |
| Prediction rules and exclusion rules | this file and [DESIGN.md](./DESIGN.md) |

## 2. Order of operations

1. Human review of gold labels (see ANNOTATION.md). Dataset, label and wording fixes are allowed up to this
   point, and until step 7, as long as no benchmark predictions have been looked at.
2. Generate the condition C translations once: `uv run python -m bench.translate`.
3. AI-assisted audit of `data/translations/en.jsonl` against the Japanese source, then confirm the
   translation phase gates in §2.1. The translations are never edited.
4. `uv run python -m bench.validate`, `uv run ruff check .` and `uv run pytest` pass on a clean checkout.
5. Final review of questions and criteria (`data/questions/`).
6. Review `git diff` against the previous commit, then commit.
7. `git tag protocol-v1`.
8. Smoke test for each provider and condition (`--smoke`, 1 example per task). Smoke results are never reported.
9. Full runs for every provider and condition declared in its configuration. Each output file records
   `git_revision`; a `-dirty` revision means the run is not a protocol run. Then `uv run python -m
   bench.sanitize` and `uv run python -m bench.report`.

### 2.1 Condition C translations

Condition C is the pipeline `raw machine translation -> English questions -> decision model`. Its translations
are **frozen raw machine translation**: machine-translated once with a pinned model snapshot, frozen,
structurally validated, and AI-audited for translation drift; **not human post-edited**.

**No bilingual human review of the translations is part of this protocol.** The maintainer reviewed the audit
findings and approved retaining the frozen raw translations, but did not independently perform a bilingual
translation review. The translations are never described as human-reviewed or human-validated, and no
"160/160 human reviewed" claim is made. (`review_status` in the translation file is a legacy field from an
earlier design; it gates nothing and every row stays `pending`, because the frozen file is not rewritten.)

The translation phase is complete when all of the following hold:

| # | Gate | Checked by |
|---|---|---|
| 1 | 160 rows present, one per dataset example, no duplicates | `bench.validate` |
| 2 | Fixed translator snapshot recorded on every row | `bench.validate` (`TRANSLATOR_MODEL`) |
| 3 | Each row's `source` matches the current dataset | `bench.validate` |
| 4 | Structured (T4) translations keep the source's keys, turn count and speakers | `bench.validate` |
| 5 | Zero empty translations | `bench.validate` |
| 6 | Frozen translation digest recorded | `bench.config.TRANSLATIONS_DIGEST` |
| 7 | Translation text unchanged since generation (digest matches) | `bench.validate` |
| 8 | AI semantic audit completed over all 160 | audit record, kept with the run |
| 9 | Known translation failures documented | audit record + this file |
| 10 | No human post-edit | digest (7) |
| 11 | No regeneration | digest (7) + `translated_at` |

Interpretation rules, fixed before any prediction was inspected:

- Every translation is included in the primary condition C metrics. Rows the audit flagged are **not**
  excluded — that would remove exactly the cases the pipeline gets wrong. A flagged-subset or clean-subset
  figure may be produced afterwards as a clearly separated secondary analysis.
- Translation failures are attributed to the translation pipeline, not to the decision model under test, and
  are described that way in the report.
- Hand-corrected translations are a different pipeline. If one is ever evaluated it is **exploratory**, reported
  separately, and must not overwrite `data/translations/en.jsonl`.

### 2.2 Condition coverage per provider

Each provider declares the conditions it is evaluated under (`conditions` in its configuration). They are not
all evaluated under all three.

**The baseline is evaluated only under condition A.** Conditions B and C are within-provider experimental
conditions for the decision model under test, and are not evaluated for the baseline in protocol-v1. The
baseline therefore serves as an external reference point under the native Japanese condition, not as a second
A/B/C factorial comparison.

The RQ2 A/B/C comparison is a within-provider comparison, read only within the provider it was run for.

One reason this matters for condition C: the baseline model is also the translator that produced the frozen
English. Evaluating the baseline under condition C would measure a self-pipeline (its own translation feeding
its own decision), which is not part of the primary comparison. If it is ever wanted, it is an exploratory
analysis after protocol-v1, or a protocol-v2 change.

## 3. Exclusion and error rules

- Requests are retried on HTTP 429 / 5xx / 529 and on connection errors, up to 5 retries with exponential backoff
  (the `retry-after` header is honored).
- Non-retryable errors (e.g. 4xx validation errors) and answers that do not fit the question schema are recorded
  with `error` set. **They are counted as errors and are not silently dropped.** Each task reports `n_error`.
- Re-running `bench.run` resumes. Only examples without a successful record are sent again. Every successful
  answer is kept; the harness never re-asks an example to get a different answer.
- No example is removed after results are seen. The primary sample set is fixed at `protocol-v1` and covers
  every example, including the ones the translation audit flagged. Changing an exclusion rule after seeing
  results requires `protocol-v2`, recorded in the changelog; it is never done silently.
- If the API returns a `returned_model` different from the requested pinned ID, the run is kept and the
  mismatch is reported.

## 4. Change policy

After `protocol-v1`:

- Any change to a frozen item creates `protocol-v2`, with the reason recorded in [CHANGELOG.md](./CHANGELOG.md).
  Tags are never moved or overwritten.
- Analyses added after looking at results (other thresholds, extra models, normalization experiments, subsets)
  are labeled **exploratory** in the report and in any article.
- Commercial considerations never change labels, protocol, criteria, or reported results (see
  [DISCLOSURE.md](./DISCLOSURE.md)).

## 5. Publication

Raw API output is kept in `results/raw/` (git-ignored). Only the output of `bench.sanitize` can be published. That
step reduces each provider response to an allow-list of fields and removes identifiers and secrets. Results for a
provider are published only when the maintainer has confirmed that the provider's terms allow it (fail-closed).
Until then they stay private, and the public report states that no results are published.
