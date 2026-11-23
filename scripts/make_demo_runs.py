"""Create demo runs for the viewer. Run from the replica folder:

    uv run python scripts/make_demo_runs.py
    uv run python -m viewer runs/four_bar
    uv run python -m viewer runs/drawer

- ``runs/four_bar``: the four-bar example, compiled to v0000 (a loop mechanism; the
  viewer asks the server to solve each pose).
- ``runs/drawer``: the drawer fixture whose back panel hits the cabinet lip near full
  extension. The compile fails its checks, so there is no version; the model is kept in
  ``result/last_failed`` with its signals, which the viewer opens.
- ``runs/desk_lamp``: a tree mechanism, posed entirely in the browser.

``runs/`` is ignored by git.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from check_fixtures import DRAWER_SOURCE  # noqa: E402
from compiler import compile  # noqa: E402

RUNS = {
    "four_bar": "from examples.four_bar import build\n\nobject_model = build()\n",
    "drawer": DRAWER_SOURCE.lstrip(),
    "desk_lamp": "from examples.desk_lamp import build, run_tests  # noqa: F401\n\n"
                 "object_model = build()\n",
}


def main() -> None:
    for name, source in RUNS.items():
        run_dir = Path("runs") / name
        shutil.rmtree(run_dir, ignore_errors=True)
        (run_dir / "workspace").mkdir(parents=True)
        (run_dir / "workspace" / "main.py").write_text(source, encoding="utf-8")
        result = compile(run_dir, timeout=120)
        where = f"v{result.version:04d}" if result.version is not None else "last_failed"
        print(f"{run_dir}: {result.status} -> result/{where}")
        print(f"    uv run python -m viewer {run_dir.as_posix()}")


if __name__ == "__main__":
    main()
