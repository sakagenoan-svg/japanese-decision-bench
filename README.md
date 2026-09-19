# japanese-decision-bench

A small, reproducible benchmark for structured AI decision models on Japanese-specific language challenges.

> **Status:** Work in progress — the dataset and protocol are currently being finalized.

日本語での詳しい解説は、公開後に Zenn 記事として追加する予定です。

## Overview

Many applications use AI models to make *decisions*: pick a category, rate severity, or answer yes/no with a
probability. They do not generate prose. Japanese makes several of these decisions harder: indirectness,
sarcasm, politeness that masks dissatisfaction, omitted subjects, and a very wide range of surface forms for
the same sentence.

**japanese-decision-bench** is a small synthetic evaluation suite for these cases. The tasks are posed as typed
questions (Choice, Score, Binary), so any decision model or general-purpose LLM can be evaluated through a thin
adapter. The goal is not to rank models. The goal is to find where decisions change, and whether the language
of the questions, translation, or surface form affects the result.

## Why Japanese?

These are common in Japanese text and can make a decision task harder:

- indirect expressions and sarcasm (「さすが、発送が早いですね（3週間待ちました）」)
- polite wording that hides dissatisfaction or anger
- omitted subjects and objects, where the context supplies the meaning
- hiragana / katakana / half-width forms of the same sentence
- emoji, kaomoji and internet slang
- long, noisy messages where the relevant sentence is surrounded by unrelated text

The benchmark covers **classification, scoring, binary decisions, and confidence-based routing**. It does not
evaluate generation quality.

## Research Questions

- **RQ1** How do decisions behave on ambiguous Japanese expressions such as sarcasm, indirectness, and polite
  dissatisfaction?
- **RQ2** Does changing the language of the questions/criteria, or translating the input into English, change
  the result?
- **RQ3** How stable are predictions under Japanese surface-form variations?
- **RQ4** How useful is high-certainty filtering for automated processing?

See [DESIGN.md](./DESIGN.md) for the full design and [PROTOCOL.md](./PROTOCOL.md) for the run protocol.

## Benchmark Tasks

| Task | Focus | Type | Size |
| ---- | ----- | ---- | ---: |
| T1 | Indirect / sarcastic moderation (`ok` / `harassment` / `scam` / `resale_spam`) | Choice | 30 |
| T2 | Polite anger intensity (0 neutral / 1 calm dissatisfaction / 2 strong anger) | Score | 30 |
| T3 | Surface-form robustness | Choice | 40 derived variants |
| T4 | Ellipsis / context dependency (`accept` / `reject` / `question` / `other`) | Choice | 30 |
| T5 | Long-text noise ("Does the customer request a refund?") | Binary | 30 (10 × 3 noise levels) |

That makes 160 evaluated inputs. T3 reuses the T1 predictions for its 10 originals.

## Surface-form Robustness

Ten T1 messages were **selected before any model run** and rewritten four ways while keeping the intended
meaning as close as possible:

| Variant | Example of the change |
| --- | --- |
| `hiragana` | kanji and katakana written in hiragana |
| `katakana_halfwidth` | the hiragana form converted mechanically to half-width katakana |
| `emoji_kaomoji` | emoji and kaomoji added |
| `internet_slang` | the same message rewritten in casual net slang |

The main metric is the **Flip Rate**:

```text
flip = prediction(variant) != prediction(original)
```

It is reported overall and per variant type, along with gold retention and changes in probability and
certainty. Emoji and slang can also change pragmatics, not only spelling. For that reason this is called
*surface-form* robustness, not orthography robustness.

## Evaluation Conditions

| Condition | Input | Questions / Criteria |
| --------- | ----- | -------------------- |
| A | Japanese | Japanese |
| B | Japanese | English |
| C | Machine-translated English | English |

Condition C is treated as a translation preprocessing pipeline, not as a pure language-isolation experiment.
Translations are generated once, reviewed, frozen in `data/translations/`, and never regenerated during a run.

## Providers

The runner talks to models through a small provider-neutral interface (`bench/clients/base.py`):

- **Baseline provider**: a general-purpose comparison model reached through a thin adapter (pinned model
  snapshot, temperature 0, structured JSON output with self-reported probabilities).
- **Mock provider** for offline tests and pipeline checks.
- **Local provider plugins**: any other model can be added without changing the harness (see
  [`bench/providers.py`](./bench/providers.py)).

Every run records the requested model ID, the model ID the API reports back, and the date. Moving aliases such as
`*-latest` are rejected. Self-reported LLM probabilities and a decision model's native probabilities are
different mechanisms, so certainty-type metrics are reported per provider.

## Dataset

- Synthetic messages set in **a fictional peer-to-peer goods exchange service**.
- No real users, no real names, no real transaction IDs, no real customer messages.
- Each sample records its origin in `source`:
  - `handwritten`: written by the maintainer
  - `llm_assisted`: drafted with an LLM, then labeled by the maintainer
  - `derived`: generated from another sample (T3 variants, T5 noise levels)
- In the current draft, all authored samples are `llm_assisted`. All T3 and T5-noise samples are `derived`.
- Gold labels come from a single annotator (the maintainer) following [ANNOTATION.md](./ANNOTATION.md). A second,
  blind annotation pass is planned.
- `scripts/build_dataset.py` rebuilds every data file deterministically, and the tests check that the shipped
  files match it.

See [data/README.md](./data/README.md) for the schema.

## Protocol

The dataset, labels, questions, model settings, metrics, and exclusion rules are frozen before the first full
benchmark run. Any experiment added after inspecting results is marked as exploratory. Changes after the freeze
require a new protocol version and a [CHANGELOG](./CHANGELOG.md) entry.

The protocol is **not locked yet**. A link to the `protocol-v1` tag will appear here once it exists.

## Metrics

- Accuracy, Macro-F1, multiclass log loss (T1, T4)
- MAE and argmax accuracy (T2)
- Flip Rate overall and by variant type, gold retention, probability and certainty deltas (T3)
- AUROC, Brier Score, and probability shift by noise level (T5)
- Accuracy and coverage in the high-certainty subset (pre-registered threshold 0.9)
- p50 / p95 observed API latency
- Cost estimated from public list prices

Exact definitions are in [DESIGN.md](./DESIGN.md#7-metrics).

## Repository Structure

```text
japanese-decision-bench/
├── README.md, DESIGN.md, PROTOCOL.md, ANNOTATION.md, DISCLOSURE.md, CHANGELOG.md
├── data/
│   ├── t1_moderation.jsonl … t5_noise.jsonl
│   └── questions/          # JA / EN question definitions per task
├── bench/
│   ├── clients/            # provider interface, baseline adapter, offline mock
│   ├── providers.py        # provider registry (built-ins + local plugins)
│   ├── credentials.py      # logical credential names -> environment variables
│   ├── run.py              # run one provider × condition
│   ├── validate.py         # dataset checks (CI)
│   ├── metrics.py
│   ├── sanitize.py         # strips IDs/secrets from raw API output before publication
│   └── report.py           # tables, figures, summary from saved results (no API calls)
├── scripts/                # dataset build, secret scan
├── tests/
└── results/                # sanitized results and reports only
```

## Quick Start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m bench.validate      # dataset checks
uv run pytest                        # tests, fully offline
uv run python -m bench.report        # rebuild the report from results/sanitized (no API calls)
uv run python -m bench.run --provider mock --condition A   # offline pipeline check
```

Runs against real models call paid APIs. Copy `.env.example` to `.env` or export the variables:

| Variable | Used for |
| --- | --- |
| `DECISION_BENCH_BASELINE_API_KEY` | baseline provider and the one-time translation for condition C |

Credentials are requested by logical name through `bench/credentials.py` and read from the process environment
only.

```bash
uv run python -m bench.run --provider baseline --condition A --smoke   # 1 example per task
uv run python -m bench.run --provider baseline --condition A
```

Raw responses go to `results/raw/` (git-ignored). `python -m bench.sanitize` removes request IDs, account
metadata and similar fields before anything can be published.

## Results

Results will be added when the applicable publication requirements are satisfied and the protocol-locked
evaluation is complete.

## Roadmap

- [x] Initial benchmark design
- [x] Dataset v0.1 draft (160 inputs)
- [x] Japanese / English question definitions
- [x] Evaluation harness (offline-tested)
- [ ] Gold-label review of the v0.1 draft
- [ ] Frozen English translations for condition C
- [ ] Protocol lock (`protocol-v1`)
- [ ] Full benchmark run
- [ ] Result report
- [ ] Second annotator pass
- [ ] Japanese Zenn article

## Limitations

- Small benchmark (30 items per task, 10 bases for T3/T5); expect wide confidence intervals.
- Synthetic data from a single fictional setting. It does not represent all Japanese users or domains.
- Labels for sarcasm and tone are subjective, and there is currently a single annotator.
- Authored examples were drafted with LLM assistance. Examples written this way may be easier or harder for
  models than naturally occurring text, including for LLM baselines.
- Model behavior can change between versions. Results apply only to the recorded model IDs and dates.
- Condition C measures a translation pipeline, not a pure language effect.
- API latency includes network and provider-route effects. It is not model inference speed.

## License

- Source code: MIT ([LICENSE](./LICENSE))
- Dataset (`data/`): CC BY 4.0 ([LICENSE-DATA](./LICENSE-DATA))

If you use the dataset or benchmark methodology, please link back to this repository.

Maintained by [@sakagenoan-svg](https://github.com/sakagenoan-svg).
