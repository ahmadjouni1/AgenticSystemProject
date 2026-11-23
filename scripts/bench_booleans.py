"""Time boolean operations on typical inputs. Run from the replica folder:

    uv run python scripts/bench_booleans.py [engine]

``engine`` is "auto" (default: what boolean_* do), "manifold", or "occ" (OpenCASCADE for
every input, the step-1b behaviour). Each case is timed as the best of 3 runs.
"""

from __future__ import annotations

import math
import sys
import time

sys.path.insert(0, ".")

from sdk import (  # noqa: E402
    BoxGeometry,
    CylinderGeometry,
    ExtrudeWithHolesGeometry,
    LatheGeometry,
    LoftGeometry,
    RoundedBoxGeometry,
    rounded_rect_profile,
    wire_from_points,
)
from sdk import booleans  # noqa: E402


def cases():
    lathe = LatheGeometry([(0, 0), (0.04, 0), (0.04, 0.03), (0.02, 0.05), (0, 0.05)], segments=64)
    bore = CylinderGeometry(0.01, 0.2, radial_segments=32)
    frame = wire_from_points([(-0.2, 0, 0), (-0.2, 0, 0.4), (0.2, 0, 0.4), (0.2, 0, 0)],
                             radius=0.012, cap_ends=True, corner_radius=0.05)
    rings = [[(x * s, y * s, z) for x, y in rounded_rect_profile(0.1, 0.06, 0.01)]
             for s, z in ((1, 0), (1.3, 0.1), (0.8, 0.2))]
    loft = LoftGeometry(rings, interpolation="catmull_rom", samples_per_span=8)
    rounded = RoundedBoxGeometry((0.28, 0.16, 0.18), 0.02)
    plate = ExtrudeWithHolesGeometry(rounded_rect_profile(0.1, 0.06, 0.01),
                                     [[(0.01 * math.cos(a), 0.01 * math.sin(a))
                                       for a in (k * math.pi / 16 for k in range(32))]], 0.4)
    return [
        ("lathe - bore (mesh, mesh)", "difference", lathe, bore),
        ("tube frame + box (mesh, mesh)", "union", frame, BoxGeometry((0.1, 0.1, 0.1)).translate(0, 0, 0.4)),
        ("smooth loft - box (mesh, mesh)", "difference", loft, BoxGeometry((0.05, 0.2, 0.05))),
        ("rounded box - box (exact, mesh)", "difference", rounded,
         BoxGeometry((0.2, 0.03, 0.12)).translate(0, 0.035, 0.05)),
        ("rounded box - plate (exact, exact)", "difference", rounded, plate),
    ]


def main() -> None:
    engine = sys.argv[1] if len(sys.argv) > 1 else "auto"
    print(f"engine={engine}")
    for label, op, a, b in cases():
        triangles = len(a.faces) + len(b.faces)
        best = math.inf
        for _ in range(3):
            start = time.perf_counter()
            result = booleans.boolean(op, a, b, engine=engine)
            best = min(best, time.perf_counter() - start)
        ok = result.is_watertight()
        print(f"{label:38} {triangles:6d} tris  {best * 1000:8.1f} ms  watertight={ok}")


if __name__ == "__main__":
    main()
