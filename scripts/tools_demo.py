"""Call the agent tools by hand on two runs. Run from replica/:

    uv run python scripts/tools_demo.py

- runs/tools_demo/laptop: write main.py, compile, render (lid at 70 deg), probe;
- runs/tools_demo/drawer: the drawer fixture (fails its checks), probe at 270 mm.
Prints each tool's JSON as the model would see it. ``runs/`` is git-ignored.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from agent.tools import ToolContext, run_tool, schema_size  # noqa: E402
from check_fixtures import DRAWER_SOURCE  # noqa: E402


def show(ctx: ToolContext, name: str, args: dict) -> None:
    out = run_tool(ctx, name, args)
    print(f"\n> {name}({json.dumps(args)})\n{out.text}")
    for image in out.images:
        print(f"  image: {image}")


def main() -> None:
    print(f"tool schemas: {schema_size()}")
    root = Path("runs") / "tools_demo"
    shutil.rmtree(root, ignore_errors=True)
    laptop = ToolContext(root / "laptop")
    show(laptop, "write", {"path": "main.py", "content":
                           "from examples.laptop import build\n\nobject_model = build()\n"})
    show(laptop, "compile", {})
    show(laptop, "render", {"pose": {"hinge": math.radians(70)}})
    show(laptop, "probe", {"pose": {"hinge": math.radians(70)}, "queries": [
        {"type": "point", "body": "lid", "point": [0, 0.2, 0]},
        {"type": "gap", "a": "base", "b": "lid"}]})
    drawer = ToolContext(root / "drawer")
    show(drawer, "write", {"path": "main.py", "content": DRAWER_SOURCE.lstrip()})
    show(drawer, "compile", {})
    show(drawer, "probe", {"pose": {"slide": 0.27}, "queries": [
        {"type": "collisions"}, {"type": "gap", "a": "cabinet", "b": "drawer"}]})


if __name__ == "__main__":
    main()
