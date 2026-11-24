"""Step 10 report: variant tables from record.json, and every run's final renders in one folder.

    uv run python scripts/step10_report.py            # reads runs/step10/{A,B,C}

Writes ``runs/step10/report.md`` (also printed) and ``runs/step10/renders/``:
- ``<prompt>__<variant>-<n>.png``: the run's 4 views (front, side, top, iso);
- ``<prompt>__iso.png``: one sheet per prompt, a row per variant, a column per run, iso views.
A run without a build gets a grey "no build" cell.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "step10"
VARIANTS = {"A": "step 9 prompt + docs", "B": "step 10 prompt + docs",
            "C": "B + low effort before the first compile"}
PROMPTS = json.loads((ROOT / "prompts" / "eval_prompts.json").read_text(encoding="utf-8"))["step10"]
QUICKSTART = "docs/sdk/quickstart.md"
SHORT = {p: p.split("(")[0].strip().removeprefix("a ").removeprefix("pair of ") for p in PROMPTS}


def load() -> dict[str, list[dict]]:
    out = {}
    for variant in VARIANTS:
        records = []
        for path in sorted((OUT / variant).glob("*/record.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] == "running":
                continue  # unfinished
            record["_dir"] = path.parent
            records.append(record)
        out[variant] = records
    return out


def mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def reasoning_total(r: dict) -> int:
    return r["tokens"]["reasoning_tokens"]


def quality_rows(rs: list[dict]) -> list[dict]:
    return [r["quality"] for r in rs if r["status"] == "success" and r.get("quality")
            and "error" not in r["quality"]]


def summary(rs: list[dict]) -> dict:
    q = quality_rows(rs)
    planning = [r["reasoning_before_first_compile"] for r in rs]
    total = sum(reasoning_total(r) for r in rs)
    return {
        "runs": len(rs), "success": sum(r["status"] == "success" for r in rs),
        "turns": mean(r["turns"] for r in rs),
        "failed_compiles": mean(r["compiles"]["failed"] for r in rs),
        "reminders": mean(sum(r["reminders"].values()) for r in rs),
        "quickstart_rereads": sum(r["doc_reads"].get(QUICKSTART, 0) for r in rs),
        "page_reads": sum(n for r in rs for p, n in r["doc_reads"].items()
                          if p != QUICKSTART),
        "planning_reasoning": mean(planning),
        "planning_share": sum(planning) / total if total else float("nan"),
        "reasoning": mean(reasoning_total(r) for r in rs),
        "output": mean(r["tokens"]["output_tokens"] for r in rs),
        "first_input": mean(r["per_turn"][0].get("input_tokens", 0) for r in rs if r["per_turn"]),
        "cost": mean(r["cost_usd"] for r in rs), "cost_total": sum(r["cost_usd"] for r in rs),
        "time": mean(r["duration_s"] for r in rs),
        "q_n": len(q),
        "bodies": mean(x["bodies"] for x in q), "shapes": mean(x["shapes"] for x in q),
        "joints": mean(x["joints"] for x in q),
        "non_box": mean(x["non_box_share"] for x in q),
        "helpers": mean(x["helper_count"] for x in q),
        "helpers_all": sorted({h for x in q for h in x["helpers"]}),
        "triangles": statistics.median([x["triangles"] for x in q]) if q else float("nan"),
        "statuses": dict(Counter(r["status"] for r in rs)),
    }


def table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def f(value: float, spec: str) -> str:
    return "-" if math.isnan(value) else format(value, spec)


def by_prompt(records: list[dict], prompt: str) -> list[dict]:
    return [r for r in records if r["prompt"] == prompt]


def report(data: dict[str, list[dict]]) -> str:
    parts = ["## Per variant (all prompts)\n"]
    rows = []
    for v, rs in data.items():
        s = summary(rs)
        rows.append([f"{v}: {VARIANTS[v]}", f"{s['success']}/{s['runs']}", f(s["turns"], ".1f"),
                     f(s["failed_compiles"], ".1f"), f(s["reminders"], ".1f"),
                     str(s["quickstart_rereads"]), str(s["page_reads"]),
                     f(s["planning_reasoning"], ",.0f"), f(s["planning_share"], ".0%"),
                     f(s["reasoning"], ",.0f"), f(s["output"], ",.0f"),
                     f"${f(s['cost'], '.4f')}", f"${f(s['cost_total'], '.3f')}",
                     f(s["time"], ".0f") + " s"])
    parts.append(table(["variant", "success", "turns", "failed compiles", "reminders",
                        "quickstart re-reads", "doc page reads", "reasoning before 1st compile",
                        "share of reasoning", "reasoning / run", "output / run", "cost / run",
                        "cost total", "time / run"], rows))
    parts.append("\nQuality of the successful builds (means; triangles: median):\n")
    rows = []
    for v, rs in data.items():
        s = summary(rs)
        rows.append([v, str(s["q_n"]), f(s["bodies"], ".1f"), f(s["shapes"], ".1f"),
                     f(s["joints"], ".1f"), f(s["non_box"], ".0%"), f(s["helpers"], ".1f"),
                     f(s["triangles"], ",.0f"), ", ".join(s["helpers_all"]) or "-"])
    parts.append(table(["variant", "builds", "bodies", "shapes", "joints", "non-box shapes",
                        "helpers / run", "triangles", "helpers used (any run)"], rows))
    parts.append("\nEnd states: " + "; ".join(f"{v} {summary(rs)['statuses']}"
                                              for v, rs in data.items()))
    parts.append("\n## Per prompt\n")
    rows = []
    for p in PROMPTS:
        for v, rs in data.items():
            s = summary(by_prompt(rs, p))
            rows.append([SHORT[p], v, f"{s['success']}/{s['runs']}", f(s["turns"], ".1f"),
                         f(s["planning_reasoning"], ",.0f"), f"${f(s['cost'], '.4f')}",
                         f(s["time"], ".0f") + " s", f(s["bodies"], ".1f"),
                         f(s["shapes"], ".1f"), f(s["joints"], ".1f"), f(s["non_box"], ".0%"),
                         f(s["helpers"], ".1f"), f(s["triangles"], ",.0f")])
    parts.append(table(["prompt", "variant", "success", "turns", "reasoning before 1st compile",
                        "cost / run", "time / run", "bodies", "shapes", "joints",
                        "non-box", "helpers", "triangles"], rows))
    parts.append("\n## Per run\n")
    rows = []
    for v, rs in data.items():
        for r in rs:
            q = r.get("quality") or {}
            rows.append([v, SHORT[r["prompt"]], r["_dir"].name[:15], r["status"],
                         str(r["turns"]), f"{r['compiles']['passed']}/{r['compiles']['failed']}",
                         f"{r['reasoning_before_first_compile']:,}",
                         f"{reasoning_total(r):,}", f"${r['cost_usd']:.4f}",
                         f"{r['duration_s']:.0f} s", str(q.get("shapes", "-")),
                         f"{q['non_box_share']:.0%}" if "non_box_share" in q else "-",
                         ", ".join(q.get("helpers", [])) or "-"])
    parts.append(table(["variant", "prompt", "run", "status", "turns", "compiles ok/fail",
                        "reasoning before 1st compile", "reasoning", "cost", "time", "shapes",
                        "non-box", "helpers"], rows))
    return "\n".join(parts) + "\n"


# ---- renders --------------------------------------------------------------------------------

CELL = (480, 360)


def _label(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 18), fill=(255, 255, 255))
    draw.text((6, 3), text, fill=(20, 20, 20))
    return image


def renders(data: dict[str, list[dict]]) -> Path:
    folder = OUT / "renders"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for p in PROMPTS:
        name = SHORT[p].replace(" ", "-")
        n_cols = max((len(by_prompt(rs, p)) for rs in data.values()), default=0)
        sheet = Image.new("RGB", (CELL[0] * max(n_cols, 1), CELL[1] * len(data)), (255,) * 3)
        for row, (v, rs) in enumerate(data.items()):
            for col, r in enumerate(by_prompt(rs, p), 1):
                views = r["_dir"] / "renders" / "final" / "views.png"
                tag = f"{v}-{col}  {r['status']}  {r['turns']} turns  ${r['cost_usd']:.3f}"
                if views.is_file():
                    shutil.copyfile(views, folder / f"{name}__{v}-{col}.png")
                    cell = Image.open(r["_dir"] / "renders" / "final" / "iso.png")
                    cell = cell.convert("RGB").resize(CELL, Image.LANCZOS)
                else:
                    cell = Image.new("RGB", CELL, (200, 200, 200))
                    tag += "  (no build)"
                sheet.paste(_label(cell, tag), ((col - 1) * CELL[0], row * CELL[1]))
        sheet.save(folder / f"{name}__iso.png")
    return folder


def main() -> None:
    data = load()
    text = report(data)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"renders: {renders(data)}")


if __name__ == "__main__":
    main()
