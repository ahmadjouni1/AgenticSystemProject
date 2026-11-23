"""Desk lamp: a turned base, a curved tube arm, and a spun shade that tilts.

Helpers shown: ``LatheGeometry`` (base), ``LatheGeometry.from_shell_profiles`` (a
thin-walled shade), ``tube_from_spline_points`` (the arm), ``CylinderGeometry``.
"""

from __future__ import annotations

import math

from sdk import Assembly, CylinderGeometry, LatheGeometry, tube_from_spline_points

ARM_TIP = (0.18, 0.0, 0.36)  # end of the arm, in the arm's frame


def build() -> Assembly:
    lamp = Assembly("desk_lamp")

    base = lamp.body("base")
    base.add("foot", LatheGeometry(
        [(0.0, 0.0), (0.08, 0.0), (0.078, 0.012), (0.03, 0.02), (0.015, 0.03), (0.0, 0.03)],
        segments=64,
    ))

    arm = lamp.body("arm")
    arm.add("collar", CylinderGeometry(0.014, 0.02, radial_segments=32).translate(0, 0, 0.01))
    arm.add("tube", tube_from_spline_points(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.15), (0.05, 0.0, 0.30), ARM_TIP], radius=0.008,
    ))

    head = lamp.body("head")
    head.add("socket", CylinderGeometry(0.02, 0.03, radial_segments=32).translate(0, 0, -0.015))
    head.add("shade", LatheGeometry.from_shell_profiles(
        [(0.02, -0.02), (0.075, -0.11)],  # outer wall, top to rim
        [(0.017, -0.02), (0.072, -0.108)],  # inner wall
        segments=64,
    ))

    lamp.joint("swivel", "revolute", base.at((0, 0, 0.03)), arm.at(), axis=(0, 0, 1))
    lamp.joint("tilt", "revolute", arm.at(ARM_TIP), head.at(), axis=(0, 1, 0),
               limits=(-0.6, 0.6))
    return lamp


def main() -> None:
    lamp = build()
    pose = lamp.solve({"swivel": math.pi / 2, "tilt": 0.4})
    print(lamp, "\nshade origin:", pose.transforms["head"][:3, 3].round(3))


if __name__ == "__main__":
    main()
