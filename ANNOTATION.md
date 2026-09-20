# Annotation guide

The gold label is what a careful native Japanese reader would conclude the writer *means*. This can differ from
what the words literally say. Label each message on its own, using only the text (and the `context` for T4).

## General rules

- Assign exactly one label. If a message plausibly needs two labels, rewrite or drop it **before** the protocol
  lock. Never do this after seeing results.
- Do not use knowledge of any model's output. Annotators never see predictions, model names, or which items are
  variants of each other.
- `note` explains the phenomenon (e.g. 皮肉, 婉曲な断り). It is not shown to models.

## T1 — moderation

| Label | Use when | Not when |
|---|---|---|
| `ok` | normal contact, thanks, questions, legitimate complaints, polite declines | the message is sarcastic or demeaning in effect |
| `harassment` | mocks, insults, belittles or intimidates the other party, **including** praise that is sarcastic in context, keigo insults, and veiled threats | a complaint that is stated plainly without attacking the person |
| `scam_risk` | tries to move payment or communication off-platform, requests prepayment, credentials, card details, or a receipt rating before delivery | a genuine off-platform question with no deceptive element (not present in v0.1) |

`scam_risk` does not assert that fraud is established: it marks an exchange whose *risk* of fraud is high.
Moving a conversation off-platform is labelled on that risk, not on a finding that the writer is a fraudster.
| `resale_spam` | bulk or for-profit resale promotion, buy-ups for resale, markups on limited goods | an individual selling one item they own |

Sarcasm cue examples: praise that contradicts a stated fact (「さすが、発送が早いですね（3週間待ちました）」),
exaggerated gratitude after a problem, 「勉強になりました」 used as a dig.

## T2 — anger intensity

Rate how dissatisfied the writer actually is. Politeness is not a signal by itself.

- **0** no dissatisfaction (information, thanks, neutral questions, apologies).
- **1** dissatisfaction stated calmly, with room for resolution (「少し残念でした」「ご確認いただけますか」).
- **2** strong indignation: ultimatums, threats to report, severing the relationship, cold sarcasm, or repeated
  complaints, even when every sentence is in keigo.

## T4 — response to a proposal

Consider the last proposal or request from the *other* party in `context`. The author of `target` is the
participant who did not speak last.

- **accept**: agrees, even implicitly (「それでお願いします」「助かります！」「乗ります」).
- **reject**: declines, including trailing-off or idiomatic refusals (「ちょっとそれは…」「お気持ちだけ」).
- **question**: asks for information needed before deciding.
- **other**: defers (「考えさせてください」), changes the topic, or only greets.

## T5 — refund request

`true` only if the customer asks for money they paid to be returned, fully or partly, including via cancellation
("取引を取り消して代金を返して"). Exchanges, resends, reports, thanks, and questions *about* past refunds are
`false`.

## Agreement

When a second annotator is available:

- They label the authored items (T1, T2, T4, T5-clean) blind to the maintainer's labels, model outputs, model
  names and variant relationships. Their labels go into `label_b`, and `agreed` is filled in.
- Reported: raw percent agreement and Cohen's κ (T1, T4, T5), and linearly weighted κ (T2).
- Disagreements are adjudicated before the protocol lock, and the adjudication is recorded in `note`.

A maintainer re-labeling their own items later is reported as *intra-rater stability*, never as a second
annotator. Until a second annotator exists, the benchmark is described as a **single-annotator benchmark**.
