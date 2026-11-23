"""Compare batches of agent runs: before/after tables from their record.json files.

    uv run python scripts/context_report.py runs/step9-baseline runs/step9-after

Per run: status, turns, compiles, input / cached / output tokens, peak input tokens (as the
provider reported them), peak context estimate, cost, time. Then a per-batch total and the
context-policy hit counts. Markdown on stdout.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def records(folder: Path) -> list[dict]:
    out = []
    for path in sorted(folder.glob("*/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_name"] = path.parent.name[16:]
        out.append(record)
    return out


def row(r: dict) -> list[str]:
    t = r["tokens"]
    peak_est = r.get("peak_context_tokens_est") or max(
        (x.get("context_tokens_est", 0) for x in r["per_turn"]), default=0)
    peak_in = r.get("peak_input_tokens") or max(
        (x.get("input_tokens", 0) for x in r["per_turn"]), default=0)
    c = r["compiles"]
    return [r["_name"][:38], r["status"], str(r["turns"]), f"{c['passed']}/{c['failed']}",
            f"{t['input_tokens']:,}", f"{t['cached_share']:.0%}", f"{t['output_tokens']:,}",
            f"{t['reasoning_tokens']:,}", f"{peak_in:,}", f"{peak_est:,}",
            f"${r['cost_usd']:.4f}", f"{r['duration_s']:.0f} s"]


HEAD = ["run", "status", "turns", "compiles ok/fail", "input tok", "cached", "output tok",
        "reasoning", "peak input", "peak ctx est", "cost", "time"]


def table(rows: list[list[str]], head: list[str] = HEAD) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _output_share(batch: list[dict]) -> float:
    """The share of cost from output tokens (reasoning included), per call, at list prices."""
    from llm.config import PRICES

    out_cost = total = 0.0
    for r in batch:
        log = Path(r["run_dir"]) / "llm_calls.jsonl"
        if not log.is_file():
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            call = json.loads(line)
            price = PRICES.get(call.get("model", ""))
            if call.get("status") != "ok" or price is None:
                continue
            u = call["usage"]
            uncached = u["input_tokens"] - u["cached_input_tokens"]
            inp = uncached * price.input + u["cached_input_tokens"] * price.cached_input
            out = u["output_tokens"] * price.output
            out_cost, total = out_cost + out, total + inp + out
    return out_cost / total if total else 0.0


def _peak_input(r: dict) -> int:
    return r.get("peak_input_tokens") or max((x.get("input_tokens", 0) for x in r["per_turn"]),
                                             default=0)


def totals(batch: list[dict]) -> dict:
    n = len(batch)
    inp = sum(r["tokens"]["input_tokens"] for r in batch)
    cached = sum(r["tokens"]["cached_input_tokens"] for r in batch)
    return {
        "runs": n, "success": sum(r["status"] == "success" for r in batch),
        "turns": sum(r["turns"] for r in batch) / n,
        "input": inp / n, "cached_share": cached / inp if inp else 0.0,
        "output": sum(r["tokens"]["output_tokens"] for r in batch) / n,
        "peak_input": max(_peak_input(r) for r in batch),
        "peak_input_mean": sum(_peak_input(r) for r in batch) / n,
        "output_cost_share": _output_share(batch),
        "cost": sum(r["cost_usd"] for r in batch), "cost_mean": sum(r["cost_usd"]
                                                                    for r in batch) / n,
        "time_mean": sum(r["duration_s"] for r in batch) / n,
    }


def main() -> None:
    folders = [Path(a) for a in sys.argv[1:]]
    summary = []
    for folder in folders:
        batch = records(folder)
        print(f"\n### {folder.name} ({len(batch)} runs)\n")
        print(table([row(r) for r in batch]))
        hits: Counter[str] = Counter()
        effort: Counter[str] = Counter()
        for r in batch:
            hits.update((r.get("context_policy") or {}).get("hits", {}))
            effort.update(r.get("effort_turns") or {})
        if hits:
            print(f"\ncontext policy hits (all runs): {dict(sorted(hits.items()))}")
        if effort:
            print(f"effort turns (all runs): {dict(effort)}")
        summary.append((folder.name, totals(batch)))
    print("\n### Per-run means\n")
    print(table([[name, f"{t['success']}/{t['runs']}", f"{t['turns']:.1f}",
                  f"{t['input']:,.0f}", f"{t['cached_share']:.1%}", f"{t['output']:,.0f}",
                  f"{t['peak_input_mean']:,.0f}", f"{t['peak_input']:,}",
                  f"${t['cost_mean']:.4f}", f"${t['cost']:.4f}",
                  f"{t['output_cost_share']:.0%}", f"{t['time_mean']:.0f} s"]
                 for name, t in summary],
                ["batch", "success", "turns", "input tok", "cached", "output tok",
                 "peak input (mean)", "peak input (max)", "cost / run", "cost total",
                 "output share of cost", "time / run"]))


if __name__ == "__main__":
    main()
