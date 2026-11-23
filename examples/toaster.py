"""Toaster: a rounded shell with two bread slots, a sliding lever, and a browning knob.

Helpers shown: ``RoundedBoxGeometry`` and ``boolean_difference`` (the slots are cut
out of the shell), ``BoxGeometry``, ``CapsuleGeometry`` (lever handle),
``CylinderGeometry``, ``TorusGeometry`` (knob grip ring), ``ConeGeometry`` (knob
pointer). The slots are plain meshes, so manifold3d makes the cuts (about 30 ms each on
the 14.7k-triangle shell).
"""

from __future__ import annotations

import math

from sdk import (
    Assembly,
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    RoundedBoxGeometry,
    TorusGeometry,
    boolean_difference,
)

SIZE = (0.28, 0.16, 0.18)  # x (width), y (depth), z (height)


def build() -> Assembly:
    toaster = Assembly("toaster")
    _, depth, height = SIZE

    body = toaster.body("body")
    shell = RoundedBoxGeometry(SIZE, 0.02).translate(0, 0, height / 2)
    for y in (-0.035, 0.035):
        slot = BoxGeometry((0.20, 0.03, 0.14)).translate(0, y, height - 0.05)
        shell = boolean_difference(shell, slot)
    body.add("shell", shell)
    for i, (x, y) in enumerate((sx * 0.11, sy * 0.06) for sx in (-1, 1) for sy in (-1, 1)):
        body.add(f"foot_{i}", CylinderGeometry(0.012, 0.006).translate(x, y, -0.003))

    lever = toaster.body("lever")
    lever.add("handle", CapsuleGeometry(0.008, 0.03).rotate_y(math.pi / 2))

    knob = toaster.body("knob")  # local z points out of the front face
    knob.add("grip", CylinderGeometry(0.016, 0.012, radial_segments=32).translate(0, 0, 0.006))
    knob.add("ring", TorusGeometry(0.016, 0.002).translate(0, 0, 0.012))
    knob.add("pointer", ConeGeometry(0.003, 0.01).rotate_y(math.pi / 2).translate(0.011, 0, 0.0125))

    front = -depth / 2
    toaster.joint("lever_slide", "prismatic", body.at((0.09, front - 0.008, 0.15)), lever.at(),
                  axis=(0, 0, 1), limits=(-0.10, 0.0))
    toaster.joint("browning", "revolute", body.at((-0.09, front, 0.06), rpy=(math.pi / 2, 0, 0)),
                  knob.at(), axis=(0, 0, 1), limits=(0.0, math.radians(270)))
    return toaster


def main() -> None:
    toaster = build()
    pose = toaster.solve({"lever_slide": -0.08, "browning": 1.0})
    print(toaster, "\nlever at z =", round(float(pose.transforms["lever"][2, 3]), 3))


if __name__ == "__main__":
    main()
