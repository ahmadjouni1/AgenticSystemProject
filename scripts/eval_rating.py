"""Step 12: blind human rating of successful evaluation runs.

    uv run python scripts/eval_rating.py make --names v1 [other ...] --out runs/rating_v1
    (fill in runs/rating_v1/ratings.csv while looking at runs/rating_v1/sheets/)
    uv run python scripts/eval_rating.py score runs/rating_v1

``make`` takes every successful run of the given evaluations (``runs/eval_<name>/``), shuffles
them together (fixed seed) and gives each a neutral id (``item_07``). For each it renders a
sheet of 4 views with no labels, no body names, no variant or model: the rest pose (iso,
front) on top and every driven joint moved most of the way toward its farther limit
below. ``ratings.csv`` has one row per item, with the prompt and the expected motion
(prompts/benchmark_v1.json) and three columns to fill in:

- ``realistic``: 1 (not recognizable) to 5 (looks like the real object, right proportions);
- ``motion_correct``: yes / no (the parts that should move do, the right way, about the
  right axis; compare the two rows of the sheet);
- ``notes``: free text.

``key.json`` says which evaluation and run each item is: do not open it while rating.
``score`` checks the CSV and prints (and writes ``scores.md``) the mean rating and the share
of correct motion per evaluation and per category.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_report import load, table  # noqa: E402

from fileio import atomic_write_json, atomic_write_text  # noqa: E402

RUNS_ROOT = ROOT / "runs"
SEED = 12
FIELDS = ["item", "prompt", "expected_motion", "realistic", "motion_correct", "notes"]
MOVE_FRACTIONS = (0.7, 0.4, 0.15)  # of the way to the farther limit; smaller if a loop won't close


# ---- the sheet ----------------------------------------------------------------------------------


def moved_pose(build) -> dict[str, float] | None:
    """Every driven joint most of the way toward its farther limit (None: no driven joint,
    or no such pose closes the model's loops)."""
    from sdk.errors import LoopClosureError, PoseError

    drivers = build.loaded.drivers
    if not drivers:
        return None
    for fraction in MOVE_FRACTIONS:
        pose = {d["name"]: fraction * (d["upper"] if abs(d["upper"]) >= abs(d["lower"])
                                       else d["lower"]) for d in drivers}
        try:
            build.loaded.assembly.solve(pose)
        except (LoopClosureError, PoseError):
            continue
        return pose
    return None


def blind_sheet(version_dir: Path) -> Image.Image:
    """2 x 2 views without any text: rest pose on top, moved pose below."""
    from agent.tools.builds import Build, world_meshes
    from agent.tools.render import HEIGHT, WIDTH, build_front, rasterize, views
    from viewer.loader import load_version

    build = Build("", load_version(version_dir))
    directions = views(build_front(build))
    sheet = Image.new("RGB", (2 * WIDTH, 2 * HEIGHT), (255, 255, 255))
    pose = moved_pose(build)
    for row, values in enumerate(({}, pose)):
        if values is None:
            blank = Image.new("RGB", (2 * WIDTH, HEIGHT), (235, 235, 235))
            ImageDraw.Draw(blank).text((12, 12), "no moved pose (no driven joint, or the "
                                       "loops do not close)", fill=(60, 60, 60))
            sheet.paste(blank, (0, HEIGHT))
            continue
        meshes = world_meshes(build, build.loaded.assembly.solve(values).transforms)
        for col, view in enumerate(("iso", "front")):
            forward, right, _label = directions[view]
            image = rasterize(meshes, forward, "", right, legend=False)
            sheet.paste(image, (col * WIDTH, row * HEIGHT))
    return sheet


# ---- make ---------------------------------------------------------------------------------------


def successful_runs(runs_root: Path, names: list[str]) -> list[dict[str, Any]]:
    items = []
    for name in names:
        eval_dir = runs_root / f"eval_{name}"
        _benchmark, rows = load(eval_dir)
        for r in rows:
            if r["status"] != "success" or not (r["record"] or {}).get("best"):
                continue
            items.append({"eval": name, "job": f"{r['id']}-r{r['rep']}", "id": r["id"],
                          "category": r["category"], "prompt": r["prompt"],
                          "motion": r["motion"],
                          "version_dir": r["run_dir"] / r["record"]["best"]["path"],
                          "run": r["run_dir"].relative_to(runs_root).as_posix()})
    return items


def make(runs_root: Path, names: list[str], out: Path, seed: int = SEED) -> list[dict]:
    items = successful_runs(runs_root, names)
    random.Random(seed).shuffle(items)
    sheets = out / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    for number, item in enumerate(items, 1):
        name = f"item_{number:02d}"
        image = blind_sheet(item["version_dir"])
        image.save(sheets / f"{name}.png")
        key[name] = {k: item[k] for k in ("eval", "job", "id", "category", "run")}
        rows.append({"item": name, "prompt": item["prompt"],
                     "expected_motion": item["motion"], "realistic": "",
                     "motion_correct": "", "notes": ""})
    with (out / "ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(out / "key.json", key, indent=2)
    atomic_write_text(out / "README.md", (
        "# Blind rating\n\nLook at `sheets/item_NN.png` (top: rest pose; bottom: the driven "
        "joints moved) and fill in `ratings.csv`: `realistic` 1-5, `motion_correct` yes/no, "
        "`notes`. Do not open `key.json` until you are done. Then run\n"
        f"`uv run python scripts/eval_rating.py score {out.as_posix()}`.\n"), newline="\n")
    return rows


# ---- score --------------------------------------------------------------------------------------


def read_ratings(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The rated rows (realistic as int or None, motion as bool or None) and the problems."""
    rows, problems = [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            realistic = (row.get("realistic") or "").strip()
            motion = (row.get("motion_correct") or "").strip().lower()
            value = None
            if realistic:
                if realistic in {"1", "2", "3", "4", "5"}:
                    value = int(realistic)
                else:
                    problems.append(f"line {line}: realistic must be 1-5, got {realistic!r}")
            correct = None
            if motion:
                if motion in ("yes", "y", "no", "n"):
                    correct = motion.startswith("y")
                else:
                    problems.append(f"line {line}: motion_correct must be yes or no, got "
                                    f"{motion!r}")
            rows.append({"item": row["item"], "realistic": value, "motion_correct": correct,
                         "notes": row.get("notes", "")})
    return rows, problems


def score(folder: Path) -> str:
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    rows, problems = read_ratings(folder / "ratings.csv")
    unknown = [r["item"] for r in rows if r["item"] not in key]
    problems += [f"unknown item {item!r}" for item in unknown]
    rated = [{**r, **key[r["item"]]} for r in rows if r["item"] in key]

    def line(label: str, group: list[dict[str, Any]]) -> list[str]:
        marks = [r["realistic"] for r in group if r["realistic"] is not None]
        motion = [r["motion_correct"] for r in group if r["motion_correct"] is not None]
        return [label, str(len(group)), str(len(marks)),
                f"{sum(marks) / len(marks):.2f}" if marks else "-",
                f"{sum(motion)}/{len(motion)}" if motion else "-"]

    head = ["group", "items", "rated", "realistic (mean, 1-5)", "motion correct"]
    evals = sorted({r["eval"] for r in rated})
    categories = sorted({r["category"] for r in rated})
    parts = ["# Blind rating scores\n", "## Per evaluation\n",
             table(head, [line(f"eval_{e}", [r for r in rated if r["eval"] == e])
                          for e in evals] + [line("**all**", rated)]),
             "\n## Per evaluation and category\n",
             table(head, [line(f"eval_{e} / {c}", g) for e in evals for c in categories
                          if (g := [r for r in rated if r["eval"] == e
                                    and r["category"] == c])])]
    if problems:
        parts += ["\n## Problems in ratings.csv\n", "\n".join(f"- {p}" for p in problems)]
    text = "\n".join(parts) + "\n"
    atomic_write_text(folder / "scores.md", text, newline="\n")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make_parser = sub.add_parser("make", help="blind sheets and an empty ratings.csv")
    make_parser.add_argument("--names", nargs="+", required=True, help="evaluation names")
    make_parser.add_argument("--out", required=True, help="the rating folder")
    make_parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    make_parser.add_argument("--seed", type=int, default=SEED)
    score_parser = sub.add_parser("score", help="score a filled-in ratings.csv")
    score_parser.add_argument("folder")
    args = parser.parse_args(argv)
    if args.command == "make":
        out = Path(args.out)
        if (out / "ratings.csv").exists():
            parser.error(f"{out / 'ratings.csv'} exists; rate it or choose another --out")
        rows = make(Path(args.runs_root), args.names, out, args.seed)
        print(f"{len(rows)} items: {out / 'sheets'}; fill in {out / 'ratings.csv'}")
        return 0
    print(score(Path(args.folder)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
