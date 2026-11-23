"""Measure the cost of a fresh-process compile. Run from the replica folder:

    uv run python scripts/measure_cold_start.py

For each model it prints the median of 3 forced compiles:

- **wall**: the whole compile call;
- **startup**: wall minus the time spent inside phases, i.e. interpreter start, imports,
  and process spawn and teardown;
- the main.py and export phases.

It also times a bare interpreter, and the imports alone.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")

from compiler import compile  # noqa: E402
from compiler.runner import child_environment  # noqa: E402

MODELS = {
    "one box (sdk only)": "from sdk import Assembly, Box\na = Assembly('box')\n"
                          "a.body('b').add('s', Box((0.1, 0.1, 0.1)))\nobject_model = a\n",
    "laptop (trimesh primitives)": "from examples.laptop import build\nobject_model = build()\n",
    "door hinge (build123d)": "from examples.door_hinge import build\nobject_model = build()\n",
    "toaster (build123d + manifold3d)": "from examples.toaster import build\n"
                                        "object_model = build()\n",
}


def timed(command: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(command, env=child_environment(), check=True, capture_output=True)
    return time.perf_counter() - start


def main() -> None:
    python = [sys.executable, "-c"]
    print(f"bare interpreter:        {timed([*python, 'pass']):.2f} s")
    print(f"import sdk:              {timed([*python, 'import sdk']):.2f} s")
    print(f"import sdk + build123d:  {timed([*python, 'import sdk, build123d']):.2f} s")
    print(f"import sdk.export, manifold3d: "
          f"{timed([*python, 'import sdk.export, manifold3d']):.2f} s")
    with tempfile.TemporaryDirectory() as tmp:
        for label, source in MODELS.items():
            run_dir = Path(tmp) / label.split()[0]
            (run_dir / "workspace").mkdir(parents=True)
            (run_dir / "workspace" / "main.py").write_text(source)
            runs = [compile(run_dir, timeout=120, force=True) for _ in range(3)]
            assert all(r.ok for r in runs), runs[0].signals_text
            wall = statistics.median(r.duration for r in runs)
            startup = statistics.median(r.startup for r in runs)
            load = statistics.median(r.phases["loading main.py"] for r in runs)
            export = statistics.median(r.phases["exporting"] for r in runs)
            print(f"{label:34} wall {wall:5.2f} s  startup {startup:5.2f} s  "
                  f"main.py {load:5.2f} s  export {export:5.2f} s")


if __name__ == "__main__":
    main()
