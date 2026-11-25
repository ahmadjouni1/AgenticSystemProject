"""Step 13 re-test: the closed-loop prompts with the synthesize tool (prompt set v2) against the
same prompts without it (prompt set v1).

    uv run python scripts/synthesis_report.py      # prints Markdown, writes the report file

Without the tool: eval_v1 (scissor lift, wipers, pantograph) and step 10b's B2 runs (the
crank-slider pump). With the tool: eval_synth_v1. Runs cut off from outside (out of credit,
a connection failure, stopped from outside) are left out, as in eval_report.py.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from eval_report import interrupted  # noqa: E402

RUNS = ROOT / "runs"
PROMPTS = {"scissor_lift": "scissor lift", "windshield_wipers": "windshield wipers",
           "pantograph": "pantograph", "crank_slider_pump": "crank-slider pump"}
WITH_TOOL = RUNS / "eval_synth_v1"
OUT = WITH_TOOL / "synthesis_report.md"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_record(job: Path) -> dict | None:
    records = sorted(job.glob("*/record.json"), key=lambda p: p.parent.name)
    return read(records[-1]) if records else None


def eval_records(eval_dir: Path, prompt: str) -> list[dict]:
    out = []
    for job in sorted(eval_dir.glob(f"jobs/{prompt}-r*")):
        record = latest_record(job)
        if record is not None and not interrupted(record["status"], record):
            out.append(record)
    return out


def without_tool(prompt: str) -> tuple[str, list[dict]]:
    if prompt == "crank_slider_pump":
        return "step 10b B2", [read(p) for p in sorted((RUNS / "step10b" / "B2").glob(
            "*crank-slider-pump*/record.json"))]
    return "eval_v1", eval_records(RUNS / "eval_v1", prompt)


def synthesize_calls(record: dict) -> int:
    return int((record.get("tool_calls") or {}).get("synthesize", 0))


def row(label: str, runs: list[dict], tool: bool) -> str:
    if not runs:
        return f"| {label} | 0 runs | | | | | |"
    ok = sum(r["status"] == "success" for r in runs)
    cost = sum(r["cost_usd"] for r in runs)
    per_success = f"${cost / ok:.3f}" if ok else "-"
    minutes = statistics.mean(r["duration_s"] for r in runs) / 60
    reasoning = statistics.median(r.get("reasoning_before_first_compile", 0) for r in runs)
    use = (f"{sum(synthesize_calls(r) > 0 for r in runs)}/{len(runs)}" if tool else "-")
    return (f"| {label} | {ok}/{len(runs)} | ${cost / len(runs):.3f} | {per_success} | "
            f"{minutes:.1f} min | {reasoning / 1000:.0f}k | {use} |")


def per_run(prompt: str, record: dict) -> str:
    s = record.get("synthesis") or {}
    calls = s.get("calls") or []
    outcomes = ", ".join(f"{c.get('mechanism')}: {c.get('outcome')}" for c in calls) or "-"
    declared = s.get("final_declared_target_path")
    path = s.get("final_target_path") or {}
    check = ("-" if not path else
             f"{'passed' if path.get('passed') else 'FAILED'}, max {path.get('max_error_mm')} mm")
    return (f"| {PROMPTS[prompt]} | {record['status']} | ${record['cost_usd']:.3f} | "
            f"{record['duration_s'] / 60:.1f} min | {len(calls)} ({outcomes}) | "
            f"{s.get('solve_seconds', 0):.0f} s | "
            f"{'yes' if declared else 'no' if declared is False else 'unknown'} | {check} |")


def main() -> int:
    head = ("| Prompt | Success | Cost / run | Cost / success | Time | Reasoning before first "
            "compile (median) | Tool used |\n| --- | --- | --- | --- | --- | --- | --- |")
    lines = ["## Without the tool (prompt set v1) vs with it (v2)", "", head]
    all_without, all_with, details = [], [], []
    for prompt, name in PROMPTS.items():
        source, before = without_tool(prompt)
        after = eval_records(WITH_TOOL, prompt)
        all_without += before
        all_with += after
        lines.append(row(f"{name}, without ({source})", before, False))
        lines.append(row(f"{name}, **with**", after, True))
        details += [per_run(prompt, r) for r in after]
    lines.append(row("**all four, without**", all_without, False))
    lines.append(row("**all four, with**", all_with, True))
    lines += ["", "## Runs with the tool", "",
              "| Prompt | Status | Cost | Time | synthesize calls | Solve time | Final build "
              "declares target_path | Trajectory check |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |", *details]
    text = "\n".join(lines) + "\n"
    print(text)
    if WITH_TOOL.is_dir():
        OUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
