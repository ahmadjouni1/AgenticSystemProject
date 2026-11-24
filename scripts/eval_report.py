"""Step 12: the report of one evaluation (scripts/eval.py).

    uv run python scripts/eval_report.py --name v1             # report.md + BUILD_LOG.md
    uv run python scripts/eval_report.py --name v1 --no-log    # report.md only

Writes ``runs/eval_<name>/report.md`` and one render sheet per category in
``runs/eval_<name>/renders/`` (the iso view of every run's best build, labelled with the
prompt id and the end state). The key tables (success, cost and time, failures) also go to
BUILD_LOG.md, between ``<!-- eval_<name> -->`` markers: a second report replaces them.

Runs cut off by something outside the agent (``interrupted``: HTTP 402 out of credit, a
connection failure, a crash) are not counted in any table; they are listed with what they
cost, and a rerun of scripts/eval.py resumes them.

Success rates come with Wilson 95% confidence intervals. With 1 run per prompt (2-3 per
category) they are wide: the numbers are indicative only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import storage  # noqa: E402
from fileio import atomic_write_text  # noqa: E402

RUNS_ROOT = ROOT / "runs"
BUILD_LOG = ROOT / "BUILD_LOG.md"
CELL = (400, 300)
Z95 = 1.959964


# ---- statistics ---------------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for a success rate (sound for small n and 0 or n
    successes, unlike the normal approximation)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def median(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ---- loading ------------------------------------------------------------------------------------


def load(eval_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The benchmark snapshot, and one row per planned job (``record`` None if not run)."""
    benchmark = json.loads((eval_dir / "benchmark.json").read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in benchmark["prompts"]}
    rows = []
    jobs = eval_dir / "jobs"
    for folder in sorted(jobs.iterdir()) if jobs.is_dir() else []:
        match = re.fullmatch(r"(.+)-r(\d+)", folder.name)
        if not match or match.group(1) not in by_id:
            continue
        found = storage.find_runs(folder)
        run_dir = found[0] if found else None
        record = storage.read_record(run_dir) if run_dir else None
        status = storage.status_of(run_dir, record) if record else "not_started"
        why = interrupted(status, record)
        rows.append({**by_id[match.group(1)], "rep": int(match.group(2)), "run_dir": run_dir,
                     "record": None if why else record, "status": status,
                     "interrupted": why, "interrupted_record": record if why else None})
    order = {p["id"]: i for i, p in enumerate(benchmark["prompts"])}
    rows.sort(key=lambda r: (order[r["id"]], r["rep"]))
    return benchmark, rows


INTERRUPTIONS = ("HTTP 402", "Insufficient Balance", "APIConnectionError",
                 "failed after", "APITimeoutError")


def interrupted(status: str, record: dict[str, Any] | None) -> str | None:
    """Why a run was cut off by something outside the agent, or None."""
    if record is None:
        return None
    if status in ("crashed", "running"):
        return status
    reason = record.get("reason") or ""
    if status == "error":
        for mark in INTERRUPTIONS:
            if mark in reason:
                return "out of credit (HTTP 402)" if mark in INTERRUPTIONS[:2] else \
                    "connection failure"
    return None


def failure_kinds(record: dict[str, Any]) -> list[str]:
    """The primary failure kind of every failed compile of a run, in order."""
    return [t["compile"]["primary"] for t in record.get("per_turn", [])
            if t.get("compile") and t["compile"].get("status") != "success"
            and t["compile"].get("primary")]


# ---- tables -------------------------------------------------------------------------------------


def success_rows(benchmark: dict[str, Any], rows: list[dict[str, Any]]) -> list[list[str]]:
    out = []
    groups = [(f"{c} ({benchmark['categories'][c]})", [r for r in rows if r["category"] == c])
              for c in benchmark["categories"]]
    groups.append(("**overall**", [r for r in rows if r["record"]]))
    for label, group in groups:
        group = [r for r in group if r["record"]]
        n = len(group)
        k = sum(r["status"] == "success" for r in group)
        low, high = wilson(k, n)
        out.append([label, f"{k}/{n}", fmt(k / n if n else None, ".0%"),
                    f"{low:.0%}-{high:.0%}" if n else "-"])
    return out


def cost_rows(benchmark: dict[str, Any], rows: list[dict[str, Any]]) -> list[list[str]]:
    out = []
    groups = [(c, [r for r in rows if r["category"] == c and r["record"]])
              for c in benchmark["categories"]]
    groups.append(("**overall**", [r for r in rows if r["record"]]))
    for label, group in groups:
        records = [r["record"] for r in group]
        k = sum(r["status"] == "success" for r in group)
        total = sum(float(r.get("cost_usd") or 0) for r in records)
        out.append([label, str(len(records)),
                    f"${total / len(records):.3f}" if records else "-",
                    f"${total / k:.3f}" if k else "-",
                    fmt(mean(r.get("duration_s") for r in records), ".0f") + " s",
                    fmt(mean(r.get("turns") for r in records), ".1f"),
                    fmt(mean((r.get("compiles") or {}).get("failed") for r in records), ".1f"),
                    f"${total:.3f}"])
    return out


def per_run_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    out = []
    for r in rows:
        record = r["record"] or {}
        best = record.get("best") or {}
        kinds = failure_kinds(record) if record else []
        state = f"interrupted: {r['interrupted']}" if r["interrupted"] else r["status"]
        out.append([f"{r['id']}-r{r['rep']}", r["category"], state,
                    str(record.get("turns", "-")),
                    f"${record['cost_usd']:.3f}" if record.get("cost_usd") is not None
                    else "-",
                    fmt(record.get("duration_s"), ".0f") + " s" if record else "-",
                    str(len(record.get("sessions") or [None])) if record else "-",
                    (best.get("version") or "-") if best.get("passed_checks") else
                    ("last_failed" if best else "-"),
                    ", ".join(f"{k} {n}" for k, n in Counter(kinds).most_common()) or "-"])
    return out


def failure_sections(rows: list[dict[str, Any]]) -> list[str]:
    run = [r for r in rows if r["record"]]
    failed = [r for r in run if r["status"] != "success"]
    states = Counter(r["status"] for r in failed)
    parts = ["\n## Failure end states\n",
             table(["end state", "runs"], [[s, str(n)] for s, n in states.most_common()])
             if failed else "No failed runs."]
    every = Counter(k for r in run for k in set(failure_kinds(r["record"])))
    last = Counter(failure_kinds(r["record"])[-1] for r in failed
                   if failure_kinds(r["record"]))
    kinds = sorted(set(every) | set(last), key=lambda k: (-every[k], k))
    parts += ["\n## Failure kinds\n",
              "Runs with at least one failed compile of each kind (all runs), and the kind of "
              "the last failed compile of each failed run.\n",
              table(["kind", "runs with it", "last failure of a failed run"],
                    [[k, str(every[k]), str(last[k])] for k in kinds]) if kinds
              else "No failed compiles."]
    return parts


def quality_rows(benchmark: dict[str, Any], rows: list[dict[str, Any]]) -> list[list[str]]:
    out = []
    groups = [(c, [r for r in rows if r["category"] == c]) for c in benchmark["categories"]]
    groups.append(("**overall**", rows))
    for label, group in groups:
        q = [r["record"]["quality"] for r in group if r["status"] == "success"
             and (r["record"] or {}).get("quality") and "bodies" in r["record"]["quality"]]
        out.append([label, str(len(q)), fmt(mean(x["bodies"] for x in q), ".1f"),
                    fmt(mean(x["shapes"] for x in q), ".1f"),
                    fmt(mean(x["movable_joints"] for x in q), ".1f"),
                    fmt(mean(x["non_box_share"] for x in q), ".0%"),
                    fmt(mean(x["helper_count"] for x in q), ".1f"),
                    fmt(median(x["triangles"] for x in q), ",.0f")])
    return out


def key_tables(benchmark: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    done = [r for r in rows if r["record"]]
    reps = Counter((r["id"]) for r in done)
    low, high = min(reps.values(), default=0), max(reps.values(), default=0)
    per_prompt = str(low) if low == high else f"{low}-{high}"
    missing = len({r["id"] for r in rows} - set(reps))
    note = (f"{len(done)} counted runs, {per_prompt} per prompt"
            + (f" ({missing} prompts with none)" if missing else "")
            + ". With so few runs per category the "
            "intervals are wide: **indicative only**, not a measured difference.")
    return ["## Success (Wilson 95% intervals)\n", note + "\n",
            table(["category", "success", "rate", "95% interval"],
                  success_rows(benchmark, rows)),
            "\n## Cost, time and turns (per run, means)\n",
            table(["category", "runs", "cost / run", "cost / success", "time / run",
                   "turns", "failed compiles", "spent"], cost_rows(benchmark, rows)),
            *failure_sections(rows)]


def report(name: str, benchmark: dict[str, Any], rows: list[dict[str, Any]],
           sheets: dict[str, Path]) -> str:
    not_run = [f"{r['id']}-r{r['rep']}" for r in rows if not r["record"]]
    parts = [f"# Evaluation eval_{name} ({benchmark.get('name', '?')})\n",
             *key_tables(benchmark, rows),
             "\n## Quality of the successful builds (means; triangles: median)\n",
             table(["category", "builds", "bodies", "shapes", "movable joints",
                    "non-box shapes", "helpers", "triangles"],
                   quality_rows(benchmark, rows)),
             "\n## Per run\n",
             table(["run", "category", "end state", "turns", "cost", "time", "sessions",
                    "best build", "failed compile kinds"], per_run_rows(rows)),
             "\n## Render sheets (iso view of each run's best build)\n",
             "\n".join(f"- {c}: `{p.name}`" for c, p in sheets.items()) or "None."]
    cut = [r for r in rows if r["interrupted"]]
    if cut:
        spent = sum(float(r["interrupted_record"].get("cost_usd") or 0) for r in cut)
        parts.append("\n## Interrupted runs (not counted)\n")
        parts.append(table(["run", "why", "turns", "cost", "best build"], [
            [f"{r['id']}-r{r['rep']}", r["interrupted"],
             str(r["interrupted_record"].get("turns", "-")),
             f"${float(r['interrupted_record'].get('cost_usd') or 0):.3f}",
             ((r["interrupted_record"].get("best") or {}).get("version") or "-")]
            for r in cut]))
        parts.append(f"\n{len(cut)} runs, ${spent:.3f}; a rerun of scripts/eval.py "
                     "resumes them.")
    not_run = [n for n in not_run if not any(f"{r['id']}-r{r['rep']}" == n for r in cut)]
    if not_run:
        parts.append(f"\nNot run (yet): {', '.join(not_run)}.")
    return "\n".join(parts) + "\n"


# ---- render sheets ------------------------------------------------------------------------------


def _label(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 18), fill=(255, 255, 255))
    draw.text((6, 3), text, fill=(20, 20, 20))
    return image


def render_sheets(eval_dir: Path, benchmark: dict[str, Any],
                  rows: list[dict[str, Any]]) -> dict[str, Path]:
    folder = eval_dir / "renders"
    folder.mkdir(parents=True, exist_ok=True)
    sheets = {}
    for category in benchmark["categories"]:
        group = [r for r in rows if r["category"] == category and r["record"]]
        if not group:
            continue
        columns = min(3, len(group))
        rows_n = math.ceil(len(group) / columns)
        sheet = Image.new("RGB", (CELL[0] * columns, CELL[1] * rows_n), (255, 255, 255))
        for i, r in enumerate(group):
            iso = r["run_dir"] / "renders" / "final" / "iso.png"
            if iso.is_file():
                cell = Image.open(iso).convert("RGB").resize(CELL, Image.LANCZOS)
            else:
                cell = Image.new("RGB", CELL, (210, 210, 210))
            tag = f"{r['id']}-r{r['rep']}  {r['status']}" + ("" if iso.is_file()
                                                              else "  (no build)")
            sheet.paste(_label(cell, tag), ((i % columns) * CELL[0], (i // columns) * CELL[1]))
        path = folder / f"{category}.png"
        sheet.save(path)
        sheets[category] = path
    return sheets


# ---- BUILD_LOG ----------------------------------------------------------------------------------


def update_log(log: Path, name: str, text: str) -> None:
    """Put ``text`` between this evaluation's markers in the log: replaced if they exist,
    else appended."""
    start, end = f"<!-- eval_{name} -->", f"<!-- /eval_{name} -->"
    block = f"{start}\n{text.rstrip()}\n{end}\n"
    current = log.read_text(encoding="utf-8") if log.is_file() else ""
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end) + r"\n?", re.S)
    if pattern.search(current):
        new = pattern.sub(lambda _m: block, current)
    else:
        new = current.rstrip("\n") + "\n\n" + block
    atomic_write_text(log, new, newline="\n")


def log_text(name: str, benchmark: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    body = "\n".join(key_tables(benchmark, rows)).replace("\n## ", "\n#### ")
    body = body.replace("## Success", "#### Success", 1)
    return (f"### Evaluation eval_{name} ({benchmark.get('name', '?')})\n\n"
            f"From `scripts/eval_report.py --name {name}`; the full report is "
            f"`runs/eval_{name}/report.md`.\n\n{body}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--log", default=str(BUILD_LOG), help="the build log to update")
    parser.add_argument("--no-log", action="store_true", help="leave the build log alone")
    args = parser.parse_args(argv)
    eval_dir = Path(args.runs_root) / f"eval_{args.name}"
    if not (eval_dir / "benchmark.json").is_file():
        parser.error(f"{eval_dir} is not an evaluation folder (no benchmark.json)")
    benchmark, rows = load(eval_dir)
    sheets = render_sheets(eval_dir, benchmark, rows)
    text = report(args.name, benchmark, rows, sheets)
    atomic_write_text(eval_dir / "report.md", text, newline="\n")
    if not args.no_log:
        update_log(Path(args.log), args.name, log_text(args.name, benchmark, rows))
    print(text)
    print(f"report: {eval_dir / 'report.md'}\nrenders: {eval_dir / 'renders'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
