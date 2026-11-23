"""Computer mouse: a lofted shell, two clicking buttons, a scroll wheel, and a cable.

Helpers shown: ``section_loft`` with ``LoftSection`` rings (the shell, smoothed by
Catmull-Rom between sections), ``superellipse_profile`` (soft-rectangle sections and a
flat cable), ``boolean_difference`` (the wheel slot), ``ExtrudeGeometry`` with
``rounded_rect_profile`` (buttons),
``sweep_profile_along_spline`` (the cable), ``CylinderGeometry`` (wheel).
"""

from __future__ import annotations

import math

from sdk import (
    Assembly,
    CylinderGeometry,
    ExtrudeGeometry,
    LoftSection,
    SectionLoftSpec,
    rounded_rect_profile,
    section_loft,
    superellipse_profile,
    sweep_profile_along_spline,
)
from sdk import BoxGeometry, boolean_difference

# (y position, width, height) of the shell's cross sections, back to front.
SECTIONS = [(-0.06, 0.050, 0.022), (-0.03, 0.062, 0.036), (0.0, 0.064, 0.040),
            (0.03, 0.058, 0.034), (0.055, 0.042, 0.020)]
BUTTON_TOP = 0.040
WHEEL_R = 0.009
WHEEL_AT = (0.0, 0.022, 0.034)  # axle below the shell top, tyre above it
SLOT_WIDTH = 0.004


def _section(y: float, width: float, height: float) -> LoftSection:
    """A soft rectangle in the x-z plane at ``y``, resting on z = 0."""
    return LoftSection(tuple((x, y, z + height / 2)
                             for x, z in superellipse_profile(width, height, 3.0, segments=40)))


def build() -> Assembly:
    mouse = Assembly("computer_mouse")

    shell = mouse.body("shell")
    body = section_loft(SectionLoftSpec(
        sections=tuple(_section(*s) for s in SECTIONS),
        interpolation="catmull_rom", samples_per_span=4,
    ))
    # A slot through the shell for the scroll wheel: 1 mm wider than the wheel each side.
    slot = BoxGeometry((SLOT_WIDTH, 2 * (WHEEL_R + 0.0015), 0.04)).translate(*WHEEL_AT[:2], 0.04)
    shell.add("body", boolean_difference(body, slot))
    shell.add("cable", sweep_profile_along_spline(
        [(0.0, 0.055, 0.008), (0.0, 0.09, 0.01), (0.02, 0.13, 0.004)],
        profile=superellipse_profile(0.006, 0.003, segments=16),
    ))

    for name, x in (("left_button", -0.015), ("right_button", 0.015)):
        button = mouse.body(name)
        plate = ExtrudeGeometry(rounded_rect_profile(0.026, 0.045, 0.006), 0.003)
        button.add("plate", plate.translate(0, 0.0225, 0.0015))  # reaches forward from its hinge
        hinge = shell.at((x, 0.0, BUTTON_TOP))
        mouse.joint(f"{name}_click", "revolute", hinge, button.at(), axis=(1, 0, 0),
                    limits=(-0.05, 0.0))

    wheel = mouse.body("wheel")
    wheel.add("tyre", CylinderGeometry(WHEEL_R, 0.003, radial_segments=32).rotate_y(math.pi / 2))
    # The axle spans the slot exactly, so its ends bear on the slot walls.
    wheel.add("axle", CylinderGeometry(0.0015, SLOT_WIDTH, radial_segments=16).rotate_y(math.pi / 2))
    mouse.joint("scroll", "revolute", shell.at(WHEEL_AT), wheel.at(), axis=(1, 0, 0))
    return mouse


def run_tests(ctx) -> None:
    ctx.expect_contact("shell", "wheel")  # the axle bears on the slot walls
    for button in ("left_button", "right_button"):
        ctx.expect_contact("shell", button)


def main() -> None:
    mouse = build()
    pose = mouse.solve({"left_button_click": -0.04, "scroll": 2.0})
    print(mouse, "\nleft button joint:", pose.joint_values["left_button_click"])


if __name__ == "__main__":
    main()
