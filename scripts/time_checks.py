"""Time the baseline checks on every example. Run from the replica folder:

    uv run python scripts/time_checks.py

Prints a table with the median of 3 runs. Building each example (imports and geometry) is
not counted; only ``run_tests(ctx)`` plus ``run_checks``.
"""

from __future__ import annotations

import importlib
import statistics
import sys
import time

sys.path.insert(0, ".")

from sdk.testing import TestContext, call_run_tests, run_checks  # noqa: E402

EXAMPLES = ["laptop", "four_bar", "gantry", "toaster", "door_hinge", "desk_lamp",
            "computer_mouse", "folding_chair"]


def main() -> None:
    print(f"| {'example':15} | tris | joints | mesh+dims+islands | rest | sweeps | combined "
          "| total |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    grand = 0.0
    for name in EXAMPLES:
        module = importlib.import_module(f"examples.{name}")
        runs = []
        for _ in range(3):
            model = module.build()
            start = time.perf_counter()
            ctx = TestContext(model)
            if hasattr(module, "run_tests"):
                call_run_tests(module.run_tests, ctx)
            report = run_checks(model, ctx)
            runs.append((time.perf_counter() - start, report))
        total, report = sorted(runs, key=lambda r: r[0])[1]
        t = report.timings
        triangles = sum(len(s.to_trimesh().faces) for b in model.bodies for s in b.shapes)
        static = t["mesh_health"] + t["dimensions"] + t["islands"]
        rest = t["overlap_at_rest"] + t["floating"]
        print(f"| {name:15} | {triangles} | {len(model.joints)} | {static:.3f} s | {rest:.3f} s | "
              f"{t['sweeps']:.3f} s | {t['combined']:.3f} s | {total:.3f} s |")
        grand += statistics.median(r[0] for r in runs)
    print(f"| **all 8** | | | | | | | **{grand:.2f} s** |")


if __name__ == "__main__":
    main()
