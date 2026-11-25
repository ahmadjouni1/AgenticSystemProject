"""Both success metrics for every evaluated setup (step 13 wrap-up; no API calls).

    uv run python scripts/metrics_summary.py      # prints Markdown, writes runs/metrics_summary.md

- **Strict success**: the run ended with its latest compile passing every check and a final
  message (the metric used since step 8).
- **Holding a passing build**: the run's best build passed every check (the trajectory check
  too, when declared), whatever the end state. For the Articraft baseline: a compile passed
  Articraft's own checks.

Runs cut off from outside (out of credit, a connection failure, stopped from outside, a crash)
are left out, as in eval_report.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from eval_report import holds_passing_build, load, strict_success, wilson  # noqa: E402

RUNS = ROOT / "runs"
OUT = RUNS / "metrics_summary.md"
EVALS = [  # (label, eval folder, what it is)
    ("eval_v1", "eval_v1", "our agent, benchmark_v1 (12 prompts x 3), prompt set v1"),
    ("Articraft baseline", "eval_articraft_v1", "Articraft in WSL, benchmark_v1 x 1, its own checks"),
    ("synth_v1", "eval_synth_v1", "4 closed-loop prompts, v2 with path-only synthesis"),
    ("synth_v2", "eval_synth_v2", "4 closed-loop prompts, v2 with function targets"),
    ("synth_v3", "eval_synth_v3", "wipers + pump, v2 with the skeleton main.py"),
]


def eval_records(folder: str) -> tuple[list[dict], int]:
    _benchmark, rows = load(RUNS / folder)
    counted = [r["record"] for r in rows if r["record"]]
    return counted, sum(1 for r in rows if r["interrupted"])


def step10b_b2() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((RUNS / "step10b" / "B2").glob("*/record.json"))]


def rate(k: int, n: int) -> str:
    low, high = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}; {low:.0%}-{high:.0%})" if n else "-"


def row(label: str, what: str, records: list[dict], cut: int = 0) -> str:
    n = len(records)
    strict = sum(strict_success(r) for r in records)
    held = sum(holds_passing_build(r) for r in records)
    cost = sum(float(r.get("cost_usd") or 0) for r in records)
    per = lambda k: f"${cost / k:.3f}" if k else "-"  # noqa: E731
    note = f" (+{cut} interrupted)" if cut else ""
    return (f"| {label} | {what} | {n}{note} | {rate(strict, n)} | {rate(held, n)} | "
            f"${cost / n:.3f} | {per(strict)} | {per(held)} |")


def main() -> int:
    lines = ["| Setup | What | Counted runs | Strict success (95% interval) | Holding a passing "
             "build (95% interval) | Cost / run | Cost / strict success | Cost / passing build |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, folder, what in EVALS[:2]:
        records, cut = eval_records(folder)
        lines.append(row(label, what, records, cut))
    lines.append(row("step 10b B2", "our agent, 4 step-10 prompts x 3 (incl. the pump), v1",
                     step10b_b2()))
    for label, folder, what in EVALS[2:]:
        records, cut = eval_records(folder)
        lines.append(row(label, what, records, cut))
    text = "\n".join(lines) + "\n"
    print(text)
    OUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
