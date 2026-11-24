"""Step 10 failure analysis: where each run's turns and minutes went (no API calls).

    uv run python scripts/step10_failures.py            # B and C; --variants A B C for all

Reads ``runs/step10/<variant>/*/`` (record.json, conversation.jsonl, result/compile_log.jsonl)
and prints, per run, its timeline as segments:
- ``planning``: from the start to the end of the turn that first calls compile;
- ``fix <kind>``: after a failed compile, up to and including the next compile call, charged
  to that compile's primary failure kind;
- ``verify``: after a passing compile (probes, renders, the final message, or more edits);
- ``unfinished <kind>``: after the last compile, when it failed, until the run ended.
Writes the same data to ``runs/step10/failures.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "step10"
SLOW_TURN_S = 200  # a model call this long is a "slow reasoning turn"


def _time(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def failures(entry: dict) -> list[dict]:
    return [s for s in entry.get("signals", []) if s.get("severity") == "failure"]


def describe(signal: dict) -> str:
    joints = ", ".join(signal.get("pose") or {})
    where = "/".join(signal.get("bodies") or [])
    return f"{signal['kind']}" + (f" [{where}]" if where else "") + (f" @{joints}" if joints else "")


def analyse(run_dir: Path) -> dict:
    record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    messages = [json.loads(line) for line in
                (run_dir / "conversation.jsonl").read_text(encoding="utf-8").splitlines()]
    log_file = run_dir / "result" / "compile_log.jsonl"
    log = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()] \
        if log_file.is_file() else []
    start = _time(messages[0]["time"])
    end = start + record["duration_s"]
    turn_end: dict[int, float] = {}
    for m in messages:
        if m["turn"] > 0:
            turn_end[m["turn"]] = max(turn_end.get(m["turn"], 0.0), _time(m["time"]))
    turns = record["per_turn"]
    last_turn = max(turn_end) if turn_end else 0

    def ends(turn: int) -> float:
        return turn_end.get(turn, end) if turn < last_turn else end

    compiles = []  # (turn, status, primary, log entry)
    calls = iter(log)
    for t in turns:
        if t.get("compile"):
            entry = next(calls, {})
            compiles.append((t["turn"], t["compile"]["status"], t["compile"]["primary"], entry))
    segments = []
    previous_turn, previous_time = 0, start
    label = "planning"
    for turn, status, primary, entry in compiles:
        segments.append({"label": label, "turns": turn - previous_turn,
                         "minutes": (ends(turn) - previous_time) / 60})
        previous_turn, previous_time = turn, ends(turn)
        label = "verify" if status == "success" else f"fix {primary}"
    tail = last_turn - previous_turn
    if tail > 0 or end - previous_time > 1:
        if not compiles:
            label = "planning (never compiled)"
        elif label.startswith("fix"):
            label = "unfinished " + label.removeprefix("fix ")
        segments.append({"label": label, "turns": tail, "minutes": (end - previous_time) / 60})
    compile_rows = []
    for turn, status, primary, entry in compiles:
        fails = failures(entry)
        compile_rows.append({"turn": turn, "minute": (ends(turn) - start) / 60, "status": status,
                             "primary": primary, "failures": len(fails),
                             "kinds": dict(Counter(s["kind"] for s in fails)),
                             "details": [describe(s) for s in fails][:4],
                             "messages": [s["message"][:200] for s in fails][:3]})
    slow = [t for t in turns if t.get("latency_s", 0) >= SLOW_TURN_S]
    return {
        "variant": run_dir.parent.name, "run": run_dir.name, "prompt": record["prompt"],
        "status": record["status"], "reason": record.get("reason"), "turns": record["turns"],
        "minutes": record["duration_s"] / 60, "cost": record["cost_usd"],
        "model_minutes": sum(t.get("latency_s", 0) for t in turns) / 60,
        "slow_turns": [{"turn": t["turn"], "seconds": round(t["latency_s"]),
                        "reasoning": t.get("reasoning_tokens", 0),
                        "output": t.get("output_tokens", 0), "tools": t.get("tools", [])}
                       for t in slow],
        "cut_off_turns": sum(1 for t in turns if not t.get("tools")
                             and t.get("output_tokens", 0) >= 15_000),
        "segments": segments, "compiles": compile_rows,
        "best": record.get("best"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="*", default=["B", "C"])
    args = parser.parse_args()
    runs = [analyse(p.parent) for v in args.variants
            for p in sorted((OUT / v).glob("*/record.json"))]
    (OUT / "failures.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    costs = [r["cost"] for r in runs]
    print(f"{len(runs)} runs; cost median ${statistics.median(costs):.3f}")
    for r in runs:
        print(f"\n## {r['variant']} {r['prompt'][:40]} {r['run'][:15]}: {r['status']}, "
              f"{r['turns']} turns, {r['minutes']:.1f} min (model {r['model_minutes']:.1f}), "
              f"${r['cost']:.3f}, best={r['best']}")
        for s in r["segments"]:
            print(f"  {s['label']:<32} {s['turns']:>3} turns {s['minutes']:5.1f} min")
        for c in r["compiles"]:
            print(f"  compile t{c['turn']} @{c['minute']:.1f}m {c['status']} {c['primary']} "
                  f"({c['failures']} failures) {c['details']}")
        for t in r["slow_turns"]:
            print(f"  slow turn {t['turn']}: {t['seconds']} s, {t['reasoning']:,} reasoning, "
                  f"tools {t['tools']}")


if __name__ == "__main__":
    main()
