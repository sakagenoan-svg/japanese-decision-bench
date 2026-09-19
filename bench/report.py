"""Build tables, figures and summary.md from run records. No API calls.

    uv run python -m bench.report                      # results/sanitized -> results/ (public)
    uv run python -m bench.report --source raw         # results/raw -> .private/unpublished-results/report

With no result files the public report states that no results are published yet.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from bench import metrics as M
from bench.config import (
    PROTOCOL_VERSION,
    RAW_DIR,
    RESULTS_DIR,
    ROOT,
    SANITIZED_DIR,
    TASKS,
    UNPUBLISHED_DIR,
)
from bench.cost import estimate_records
from bench.dataset import load_all, read_jsonl

PRIMARY = {  # (metric key, display name) per task, higher is better unless noted
    "t1_moderation": ("accuracy", "T1 accuracy"),
    "t2_politeness_anger": ("argmax_accuracy", "T2 argmax accuracy"),
    "t4_ellipsis": ("accuracy", "T4 accuracy"),
    "t5_noise": ("auroc", "T5 AUROC"),
}


def load_records(src: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(src.glob("*.jsonl")):
        if f.name.startswith("smoke-"):
            continue
        rows.extend(read_jsonl(f))
    return rows


def group_runs(records: list[dict], include_mock: bool = False) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r["provider"] == "mock" and not include_mock:
            continue
        groups[(r["provider"], r["requested_model"], r["condition"])].append(r)
    return dict(sorted(groups.items()))


def compute(records: list[dict], examples_by_id: dict[str, dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    out: dict = {"n_records": len(records),
                 "protocol_versions": sorted({r.get("protocol_version", "?") for r in records}),
                 "returned_models": sorted({r["returned_model"] for r in records if r.get("returned_model")}),
                 "tasks": {}}
    for task, rs in by_task.items():
        ttype = TASKS[task]["type"]
        if task == "t3_surface_variants":
            out["tasks"][task] = {
                **M.choice_metrics(rs, TASKS[task]["labels"]),
                "flip": M.flip_metrics(rs, by_task.get("t1_moderation", []), examples_by_id),
            }
        elif task == "t5_noise":
            out["tasks"][task] = M.noise_metrics(rs, examples_by_id)
        elif ttype == "choice":
            out["tasks"][task] = M.choice_metrics(rs, TASKS[task]["labels"])
        elif ttype == "score":
            out["tasks"][task] = M.score_metrics(rs)
    out["latency"] = M.latency_summary(records)
    out["cost"] = estimate_records(records)
    return out


def primary_value(run: dict, task: str) -> float | None:
    t = run["tasks"].get(task)
    if t is None:
        return None
    key = PRIMARY[task][0]
    return t["overall"][key] if task == "t5_noise" else t.get(key)


def paired_condition_deltas(groups: dict[tuple, list[dict]]) -> list[dict]:
    """Paired bootstrap CI for accuracy(B) - accuracy(A) and accuracy(C) - accuracy(A), per provider/model."""
    out = []
    by_model = defaultdict(dict)
    for (prov, model, cond), rs in groups.items():
        by_model[(prov, model)][cond] = {r["example_id"]: r for r in M.ok_records(rs)}
    for (prov, model), conds in by_model.items():
        if "A" not in conds:
            continue
        for other in ("B", "C"):
            if other not in conds:
                continue
            for task in ("t1_moderation", "t4_ellipsis"):
                units = [(conds["A"][i], conds[other][i]) for i in conds["A"]
                         if i in conds[other] and conds["A"][i]["task"] == task]
                if not units:
                    continue

                def diff(us):
                    return (sum(b["answer"]["prediction"] == b["gold"] for _, b in us)
                            - sum(a["answer"]["prediction"] == a["gold"] for a, _ in us)) / len(us)

                out.append({"provider": prov, "model": model, "task": task, "comparison": f"{other}-A",
                            "n": len(units), "accuracy_delta": diff(units), "ci95": M.bootstrap_ci(units, diff)})
    return out


def fmt(v, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_tables(runs: dict[tuple, dict], deltas: list[dict], out_dir: Path) -> None:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    serial = {"|".join(k): v for k, v in runs.items()}
    (tables / "metrics.json").write_text(json.dumps({"runs": serial, "condition_deltas": deltas}, indent=2,
                                                    ensure_ascii=False, default=str), encoding="utf-8", newline="\n")
    with (tables / "primary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["provider", "model", "condition", *[PRIMARY[t][1] for t in PRIMARY], "T2 MAE",
                    "T3 flip rate", "p50 ms", "p95 ms"])
        for (prov, model, cond), run in runs.items():
            t2 = run["tasks"].get("t2_politeness_anger", {})
            t3 = run["tasks"].get("t3_surface_variants", {}).get("flip", {})
            w.writerow([prov, model, cond, *[fmt(primary_value(run, t)) for t in PRIMARY], fmt(t2.get("mae")),
                        fmt(t3.get("flip_rate")), fmt(run["latency"]["p50_ms"], 0),
                        fmt(run["latency"]["p95_ms"], 0)])


def write_figures(runs: dict[tuple, dict], groups: dict[tuple, list[dict]], out_dir: Path) -> list[str]:
    """At most four figures (see DESIGN.md)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    names = {k: f"{k[1]} / {k[2]}" for k in runs}

    # 01 performance by task
    tasks = list(PRIMARY)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(1, len(runs))
    for i, (k, run) in enumerate(runs.items()):
        vals = [primary_value(run, t) or 0 for t in tasks]
        ax.bar([x + i * width for x in range(len(tasks))], vals, width, label=names[k])
    ax.set_xticks([x + width * (len(runs) - 1) / 2 for x in range(len(tasks))], [PRIMARY[t][1] for t in tasks])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Primary metric by task")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_performance_by_task.png", dpi=150)
    plt.close(fig)
    written.append("01_performance_by_task.png")

    # 02 conditions A/B/C per model (only models run under >1 condition)
    by_model = defaultdict(dict)
    for (_prov, model, cond), run in runs.items():
        by_model[model][cond] = run
    multi = {m: c for m, c in by_model.items() if len(c) > 1}
    if multi:
        fig, ax = plt.subplots(figsize=(8, 4))
        for model, conds in multi.items():
            for cond, run in sorted(conds.items()):
                ax.plot([PRIMARY[t][1] for t in tasks], [primary_value(run, t) for t in tasks], marker="o",
                        label=f"{model} / {cond}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.set_title("Conditions A / B / C")
        fig.tight_layout()
        fig.savefig(fig_dir / "02_conditions.png", dpi=150)
        plt.close(fig)
        written.append("02_conditions.png")

    # 03 T3 flip rate by variant type
    vtypes = TASKS["t3_surface_variants"]["variant_types"]
    flips = {k: run["tasks"]["t3_surface_variants"]["flip"] for k, run in runs.items()
             if "t3_surface_variants" in run["tasks"]}
    if flips:
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.8 / len(flips)
        for i, (k, fl) in enumerate(flips.items()):
            vals = [fl["by_variant_type"].get(v, {}).get("flip_rate") or 0 for v in vtypes]
            ax.bar([x + i * width for x in range(len(vtypes))], vals, width, label=names[k])
        ax.set_xticks([x + width * (len(flips) - 1) / 2 for x in range(len(vtypes))], vtypes)
        ax.set_ylim(0, 1)
        ax.set_ylabel("flip rate")
        ax.legend(fontsize=7)
        ax.set_title("T3 flip rate by variant type (n=10 per type)")
        fig.tight_layout()
        fig.savefig(fig_dir / "03_t3_flip_rate.png", dpi=150)
        plt.close(fig)
        written.append("03_t3_flip_rate.png")

    # 04 certainty threshold vs accuracy / coverage (T1 + T4, choice tasks)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    for k, rs in groups.items():
        ok = [r for r in M.ok_records(rs) if r["task"] in ("t1_moderation", "t4_ellipsis")]
        if not ok:
            continue
        curve = M.coverage_curve([r["answer"]["prediction"] == r["gold"] for r in ok],
                                 [M.certainty_of(r["answer"]) for r in ok])
        a1.plot([c["threshold"] for c in curve], [c["coverage"] for c in curve], label=names.get(k, str(k)))
        a2.plot([c["threshold"] for c in curve], [c["accuracy"] for c in curve], label=names.get(k, str(k)))
    for a, t in ((a1, "coverage"), (a2, "accuracy on covered")):
        a.set_xlabel("certainty threshold")
        a.set_title(t)
        a.set_ylim(0, 1.02)
        a.axvline(0.9, color="grey", lw=0.8, ls="--")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_certainty_threshold.png", dpi=150)
    plt.close(fig)
    written.append("04_certainty_threshold.png")
    return written


def write_summary(runs: dict[tuple, dict], deltas: list[dict], figures: list[str], out_dir: Path,
                  source: str) -> Path:
    lines = ["# Results summary", ""]
    if not runs:
        lines += ["No benchmark results are published in this repository yet.", "",
                  "Results will be added after the protocol is locked and the benchmark run is complete. "
                  "Publication of provider-specific benchmark results is subject to the applicable service "
                  "terms and permissions."]
    else:
        lines += [f"Generated by `python -m bench.report` from `{source}`. Protocol: `{PROTOCOL_VERSION}`.", "",
                  "Small synthetic benchmark (n=30 per task); read differences together with the CIs.", "",
                  "## Primary metrics", "",
                  "| provider | model | cond | " + " | ".join(PRIMARY[t][1] for t in PRIMARY)
                  + " | T2 MAE ↓ | T3 flip ↓ | p50 ms | p95 ms |",
                  "|---|---|---|" + "---|" * (len(PRIMARY) + 4)]
        for (prov, model, cond), run in runs.items():
            t2 = run["tasks"].get("t2_politeness_anger", {})
            t3 = run["tasks"].get("t3_surface_variants", {}).get("flip", {})
            lines.append(f"| {prov} | {model} | {cond} | "
                         + " | ".join(fmt(primary_value(run, t)) for t in PRIMARY)
                         + f" | {fmt(t2.get('mae'))} | {fmt(t3.get('flip_rate'))} | "
                         f"{fmt(run['latency']['p50_ms'], 0)} | {fmt(run['latency']['p95_ms'], 0)} |")
        lines += ["", "Latency is client-observed end-to-end API latency, not model inference speed.", ""]
        lines += ["## High-certainty subset (pre-registered threshold 0.9)", "",
                  "| provider | model | cond | task | coverage | accuracy (covered) | accuracy (all) |",
                  "|---|---|---|---|---|---|---|"]
        for (prov, model, cond), run in runs.items():
            for task, t in run["tasks"].items():
                hc = (t.get("overall") or t).get("high_certainty")
                if hc:
                    lines.append(f"| {prov} | {model} | {cond} | {task} | {fmt(hc['coverage'])} | "
                                 f"{fmt(hc['accuracy_covered'])} | {fmt(hc['accuracy_all'])} |")
        if deltas:
            lines += ["", "## Condition deltas (paired bootstrap, 95% CI)", "",
                      "| provider | model | task | comparison | n | Δ accuracy | 95% CI |",
                      "|---|---|---|---|---|---|---|"]
            for d in deltas:
                ci = f"{fmt(d['ci95'][0])} … {fmt(d['ci95'][1])}" if d["ci95"] else "—"
                lines.append(f"| {d['provider']} | {d['model']} | {d['task']} | {d['comparison']} | {d['n']} | "
                             f"{fmt(d['accuracy_delta'])} | {ci} |")
        if figures:
            lines += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figures]
    path = out_dir / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def build_report(src: Path, out_dir: Path, include_mock: bool = False, figures: bool = True) -> Path:
    examples_by_id = {r["id"]: r for rows in load_all().values() for r in rows}
    groups = group_runs(load_records(src), include_mock=include_mock)
    runs = {k: compute(rs, examples_by_id) for k, rs in groups.items()}
    deltas = paired_condition_deltas(groups)
    figs: list[str] = []
    if runs:
        write_tables(runs, deltas, out_dir)
        costs = [c for run in runs.values() for c in run["cost"]]
        (out_dir / "costs").mkdir(parents=True, exist_ok=True)
        (out_dir / "costs" / "public-price-estimate.json").write_text(json.dumps(costs, indent=2), encoding="utf-8",
                                                                     newline="\n")
        if figures:
            figs = write_figures(runs, groups, out_dir)
    try:
        shown = src.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = src.name  # never leak local absolute paths into the report
    return write_summary(runs, deltas, figs, out_dir, source=shown)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sanitized", "raw"], default="sanitized")
    ap.add_argument("--include-mock", action="store_true", help="for pipeline checks only")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    if args.source == "raw":
        src, out = RAW_DIR, UNPUBLISHED_DIR / "report"
    else:
        src, out = SANITIZED_DIR, RESULTS_DIR
    src.mkdir(parents=True, exist_ok=True)
    path = build_report(src, out, include_mock=args.include_mock, figures=not args.no_figures)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
