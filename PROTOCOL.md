# Run protocol

Status: **draft, not locked.** The protocol becomes binding when the `protocol-v1` git tag is created. That
happens before the first full (non-smoke) benchmark run.

## 1. What is frozen by `protocol-v1`

| Item | Where |
|---|---|
| Dataset and gold labels | `data/*.jsonl`, `scripts/build_dataset.py` |
| Task definitions, label sets, T3 bases, T5 levels | `bench/config.py` (`TASKS`), `data/t1_moderation.jsonl` (`t3_base`) |
| Japanese and English questions | `data/questions/*.json` |
| Frozen EN translations (condition C) | `data/translations/en.jsonl` |
| Model IDs, provider routes, sampling / thinking settings | `bench/config.py` (`PROVIDERS`) |
| LLM adapter prompts and output schema | `bench/clients/adapter.py` |
| Metric definitions, thresholds, clipping, seed | `bench/metrics.py`, `bench/config.py` |
| Prediction rules and exclusion rules | this file and [DESIGN.md](./DESIGN.md) |

## 2. Order of operations

1. Human review of gold labels (see ANNOTATION.md). Dataset, label and wording fixes are allowed up to this
   point, and until step 7, as long as no benchmark predictions have been looked at.
2. Generate the condition C translations once: `uv run python -m bench.translate`.
3. Human review of `data/translations/en.jsonl`.
4. `uv run python -m bench.validate`, `uv run ruff check .` and `uv run pytest` pass on a clean checkout.
5. Final review of questions and criteria (`data/questions/`).
6. Review `git diff` against the previous commit, then commit.
7. `git tag protocol-v1`.
8. Smoke test for each provider and condition (`--smoke`, 1 example per task). Smoke results are never reported.
9. Full runs for every provider and condition declared in its configuration. Each output file records
   `git_revision`; a `-dirty` revision means the run is not a protocol run. Then `uv run python -m
   bench.sanitize` and `uv run python -m bench.report`.

## 3. Exclusion and error rules

- Requests are retried on HTTP 429 / 5xx / 529 and on connection errors, up to 5 retries with exponential backoff
  (the `retry-after` header is honored).
- Non-retryable errors (e.g. 4xx validation errors) and answers that do not fit the question schema are recorded
  with `error` set. **They are counted as errors and are not silently dropped.** Each task reports `n_error`.
- Re-running `bench.run` resumes. Only examples without a successful record are sent again. Every successful
  answer is kept; the harness never re-asks an example to get a different answer.
- No example is removed after results are seen.
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
