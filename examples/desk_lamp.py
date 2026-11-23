"""Desk lamp: a turned base, a curved tube arm, and a spun shade that tilts.

Helpers shown: ``LatheGeometry`` (base), ``LatheGeometry.from_shell_profiles`` (a
thin-walled shade), ``tube_from_spline_points`` (the arm), ``CylinderGeometry`` (the
hinge barrel the head rolls on).
"""

from __future__ import annotations

import math

from sdk import Assembly, CylinderGeometry, LatheGeometry, tube_from_spline_points

ARM_TIP = (0.18, 0.0, 0.36)  # end of the arm, in the arm's frame
BARREL_R = 0.012


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
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.15), (0.05, 0.0, 0.30), (0.14, 0.0, 0.36), ARM_TIP],
        radius=0.008,  # the last stretch is level, so the tube meets the barrel side-on
    ))
    # The tilt hinge: a barrel on the arm, centered on the tilt axis (y through the tip).
    arm.add("barrel", CylinderGeometry(BARREL_R, 0.03, radial_segments=48)
            .rotate_x(math.pi / 2).translate(*ARM_TIP))

    # The head hangs below the barrel. Its stem's flat top is tangent to the barrel, so it
    # rolls on the barrel as the head tilts instead of passing through it.
    head = lamp.body("head")
    head.add("stem", CylinderGeometry(0.006, 0.02, radial_segments=32)
             .translate(0, 0, -BARREL_R - 0.01))
    socket_top = -BARREL_R - 0.02
    head.add("socket", CylinderGeometry(0.02, 0.03, radial_segments=32)
             .translate(0, 0, socket_top - 0.015))
    shade_top = socket_top - 0.03
    head.add("shade", LatheGeometry.from_shell_profiles(
        [(0.02, shade_top), (0.075, shade_top - 0.09)],  # outer wall, top to rim
        [(0.017, shade_top), (0.072, shade_top - 0.088)],  # inner wall
        segments=64,
    ))

    lamp.joint("swivel", "revolute", base.at((0, 0, 0.03)), arm.at(), axis=(0, 0, 1))
    lamp.joint("tilt", "revolute", arm.at(ARM_TIP), head.at(), axis=(0, 1, 0),
               limits=(-0.6, 0.4))
    return lamp


def run_tests(ctx) -> None:
    ctx.expect_contact("base", "arm")
    ctx.expect_contact("arm", "head")  # the stem rests on the barrel
    ctx.expect_contact("arm", "head", pose={"tilt": -0.6})  # ... at any tilt


def main() -> None:
    lamp = build()
    pose = lamp.solve({"swivel": math.pi / 2, "tilt": 0.4})
    print(lamp, "\nshade origin:", pose.transforms["head"][:3, 3].round(3))


if __name__ == "__main__":
    main()
