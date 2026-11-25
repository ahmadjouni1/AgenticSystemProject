"""Step 13: solve times and accuracy of the synthesis plug-ins on known answers (no network).

    uv run python scripts/synthesis_bench.py            # prints a Markdown table
    uv run python scripts/synthesis_bench.py --isolated # each solve in its own process

Each case's target comes from a known design, so the error at the known design is 0:
- spatial 3R arm (plug-in A): the example in solvers/nested/examples, with its endpoints;
- four-bar and slider-crank (plug-in B): 13 points over one crank turn, solved from a start
  8% off the known lengths (and 0.1 rad / 3 mm off the rest), and from the default start
  the plug-in makes from the target alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solvers.base import solve, solve_isolated  # noqa: E402
from solvers.planar import LINKAGES, sweep  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "solvers" / "nested" / "examples" / \
    "spatial_3r.json"
FOUR_BAR = {"crank_pivot_x": 0.0, "crank_pivot_y": 0.0, "crank_length": 0.03,
            "coupler_length": 0.09, "ground_length": 0.10, "ground_angle": 0.0,
            "coupler_angle": math.radians(50), "crank_phase": 0.0, "point_x": 0.05,
            "point_y": 0.03}
SLIDER_CRANK = {"crank_pivot_x": 0.0, "crank_pivot_y": 0.0, "crank_length": 0.025,
                "rod_length": 0.08, "rod_angle": math.radians(-20), "slider_angle": 0.0,
                "crank_phase": math.radians(20), "point_x": 0.04, "point_y": 0.015}


def perturbed(params: dict) -> dict:
    return {k: v * 1.08 if "length" in k else v + 0.1 if ("angle" in k or "phase" in k)
            else v + 0.003 for k, v in params.items()}


def linkage_target(name: str, params: dict, tolerance: float) -> dict:
    assembly, body = LINKAGES[name].build(params, 2 * math.pi)
    curve = sweep(assembly, body, (params["point_x"], params["point_y"]), 2 * math.pi)
    path = curve.at(np.linspace(0, 1, 13))
    return {"points": np.c_[path, np.zeros(13)].tolist(), "tolerance": tolerance}


def cases() -> list[tuple[str, str, dict, dict, dict | None]]:
    spec = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    arm_target = {"points": spec["waypoints"]["positions_m"],
                  "tolerance": spec["position_tolerance_m"]}
    arm = {"urdf": (EXAMPLE.parent / spec["urdf"]).read_text(encoding="utf-8"),
           "base_link": "base_link", "tip_link": "tool", "parameters": spec["design_parameters"],
           "q_start": spec["q_start"], "q_end": spec["q_end"],
           "max_step": spec["max_joint_step"]}
    known_arm = dict(zip([p["name"] for p in spec["design_parameters"]], (0.28, 0.42, 0.33),
                         strict=True))
    out = [("serial_arm", "spatial 3R, the example's start", arm_target, arm, known_arm),
           ("serial_arm", "spatial 3R, endpoints by IK", arm_target,
            {k: v for k, v in arm.items() if k not in ("q_start", "q_end", "max_step")}, None)]
    for name, known in (("four_bar", FOUR_BAR), ("slider_crank", SLIDER_CRANK)):
        target = linkage_target(name, known, 0.001)
        out.append((name, "8% off the known lengths", target,
                    {"initial": perturbed(known), "time_limit_s": 150}, known))
        out.append((name, "default start (target only)", target, {"time_limit_s": 150}, None))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--isolated", action="store_true", help="one process per solve")
    args = parser.parse_args()
    rows = ["| Mechanism | Start | Max error | RMS | Within 1-2 mm tol. | Lengths vs known | "
            "Search evals | Nested error | Seconds | Per pose |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, start, target, constraints, known in cases():
        started = time.perf_counter()
        runner = solve_isolated if args.isolated else solve
        kwargs = {"timeout_s": 300} if args.isolated else {}
        result = runner(name, target, constraints, **kwargs)
        seconds = time.perf_counter() - started
        d = result.details
        if known is None:
            lengths = "-"
        else:
            keys = [k for k in known if "length" in k]
            worst = max(abs(result.parameters[k] / known[k] - 1) for k in keys)
            lengths = f"within {worst * 100:.2g}%"
        pose = d.get("seconds_per_pose")
        rows.append(
            f"| {name} | {start} | {result.error['max'] * 1000:.3g} mm | "
            f"{result.error['rms'] * 1000:.3g} mm | {'yes' if result.within_tolerance else 'no'}"
            f" | {lengths} | {d['geometry_evaluations']} | "
            f"{d['nested_max_error'] * 1000:.3g} mm | {seconds:.0f} | "
            f"{'-' if pose is None else f'{pose * 1000:.1f} ms'} |")
        print(rows[-1], flush=True)
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
