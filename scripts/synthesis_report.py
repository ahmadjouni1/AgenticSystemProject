"""Step 13 re-tests: the closed-loop prompts without the synthesize tool (prompt set v1), with
the first, mismatched tool (synth_v1) and with the fixed tool (synth_v2).

    uv run python scripts/synthesis_report.py      # prints Markdown, writes the report file

Without the tool: eval_v1 (scissor lift, wipers, pantograph) and step 10b's B2 runs (the
crank-slider pump). Runs cut off from outside (out of credit, a connection failure, stopped
from outside, crashed) are left out and counted apart, as in eval_report.py.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from eval_report import holds_passing_build, interrupted, strict_success  # noqa: E402

RUNS = ROOT / "runs"
PROMPTS = {"scissor_lift": "scissor lift", "windshield_wipers": "windshield wipers",
           "pantograph": "pantograph", "crank_slider_pump": "crank-slider pump"}
SETUPS = {"without": "no tool (v1)", "synth_v1": "synth_v1 (mismatched)",
          "synth_v2": "synth_v2 (fixed)", "synth_v3": "synth_v3 (skeleton)"}
OUT = RUNS / "eval_synth_v3" / "synthesis_report.md"
TOOL_PROMPTS = ("windshield_wipers", "crank_slider_pump")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_records(name: str, prompt: str) -> tuple[list[dict], int]:
    """(counted records, interrupted count) of one evaluation's runs of one prompt."""
    out, cut = [], 0
    for job in sorted((RUNS / f"eval_{name}").glob(f"jobs/{prompt}-r*")):
        records = sorted(job.glob("*/record.json"), key=lambda p: p.parent.name)
        if not records:
            continue
        record = read(records[-1])
        status = record["status"]
        if status == "running":  # its process is gone: a crash (storage.status_of)
            status = "crashed"
        if interrupted(status, record):
            cut += 1
        else:
            out.append(record)
    return out, cut


def records(setup: str, prompt: str) -> tuple[list[dict], int]:
    if setup != "without":
        return eval_records(setup, prompt)
    if prompt == "crank_slider_pump":
        return [read(p) for p in sorted((RUNS / "step10b" / "B2").glob(
            "*crank-slider-pump*/record.json"))], 0
    return eval_records("v1", prompt)


def synthesize_calls(record: dict) -> int:
    return int((record.get("tool_calls") or {}).get("synthesize", 0))


def path_check(record: dict) -> tuple[bool, bool]:
    """(the final build declares target_path, and its check passed)."""
    s = record.get("synthesis") or {}
    path = s.get("final_target_path")
    return bool(s.get("final_declared_target_path")), bool(path and path.get("passed"))


def row(label: str, runs: list[dict], cut: int, tool: bool) -> str:
    note = f" (+{cut} interrupted)" if cut else ""
    if not runs:
        return f"| {label} | 0 runs{note} | | | | | | | |"
    ok = sum(strict_success(r) for r in runs)
    cost = sum(r["cost_usd"] for r in runs)
    per_success = f"${cost / ok:.3f}" if ok else "-"
    minutes = statistics.mean(r["duration_s"] for r in runs) / 60
    reasoning = statistics.median(r.get("reasoning_before_first_compile", 0) for r in runs)
    use = f"{sum(synthesize_calls(r) > 0 for r in runs)}/{len(runs)}" if tool else "-"
    checks = [path_check(r) for r in runs]
    passed = (f"{sum(p for _d, p in checks)}/{len(runs)} (declared {sum(d for d, _p in checks)})"
              if tool else "-")
    kept = sum(holds_passing_build(r) for r in runs)
    return (f"| {label} | {ok}/{len(runs)}{note} | ${cost / len(runs):.3f} | {per_success} | "
            f"{minutes:.1f} min | {reasoning / 1000:.0f}k | {use} | {passed} | {kept}/{len(runs)} |")


def per_run(setup: str, prompt: str, record: dict) -> str:
    s = record.get("synthesis") or {}
    calls = s.get("calls") or []
    outcomes = ", ".join(f"{c.get('mechanism')} {c.get('target', '')}: {c.get('outcome')}"
                         for c in calls) or "-"
    declared, passed = path_check(record)
    path = s.get("final_target_path") or {}
    check = ("-" if not path else f"{'passed' if passed else 'FAILED'}, max "
             f"{path.get('max_error_mm')} {path.get('unit', 'mm')}")
    return (f"| {SETUPS[setup]} | {PROMPTS[prompt]} | {record['status']} | "
            f"${record['cost_usd']:.3f} | {record['duration_s'] / 60:.1f} min | "
            f"{len(calls)} ({outcomes}) | {s.get('solve_seconds', 0):.0f} s | "
            f"{'yes' if declared else 'no'} | {check} |")


def main() -> int:
    head = ("| Prompt, setup | Strict success | Cost / run | Cost / strict success | Time | "
            "Reasoning before first compile (median) | Tool used | Trajectory check passed | "
            "Holding a passing build |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines = ["## Closed-loop prompts: no tool vs synth_v1, synth_v2 and synth_v3", "", head]
    totals = {setup: ([], 0) for setup in SETUPS}
    details = []
    for prompt, name in PROMPTS.items():
        for setup, label in SETUPS.items():
            found, cut = records(setup, prompt)
            totals[setup] = (totals[setup][0] + found, totals[setup][1] + cut)
            lines.append(row(f"{name}, {label}", found, cut, setup != "without"))
            if setup != "without":
                details += [per_run(setup, prompt, r) for r in found]
    for setup, label in SETUPS.items():
        found, cut = totals[setup]
        lines.append(row(f"**all four, {label}**", found, cut, setup != "without"))
    for setup, label in SETUPS.items():  # the two prompts the tool fits (synth_v3 runs only these)
        pairs = [records(setup, prompt) for prompt in TOOL_PROMPTS]
        lines.append(row(f"**wipers + pump, {label}**", [r for found, _ in pairs for r in found],
                         sum(cut for _, cut in pairs), setup != "without"))
    lines += ["", "## Runs with the tool", "",
              "| Setup | Prompt | Status | Cost | Time | synthesize calls | Solve time | "
              "Final build declares target_path | Trajectory check |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |", *details]
    text = "\n".join(lines) + "\n"
    print(text)
    if OUT.parent.is_dir():
        OUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
