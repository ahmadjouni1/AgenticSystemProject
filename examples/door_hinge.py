"""Door hinge: two screw-hole leaves and three knuckles around one pin.

Helpers shown: ``ExtrudeWithHolesGeometry`` (leaves with screw holes),
``ExtrudeGeometry`` (a hex pin head), ``rounded_rect_profile``. The hinge axis comes
straight from geometry: ``body.at(knuckle_face)`` on a build123d cylinder gives a frame
whose z axis is the knuckle's axis, so the joint is simply "about z".

The leaves lie flat (fully open) at 0 and fold up to 170 degrees.
"""

from __future__ import annotations

import math

from build123d import Cylinder, GeomType, Pos, Rot

from sdk import Assembly, ExtrudeGeometry, ExtrudeWithHolesGeometry, rounded_rect_profile

LEAF = (0.04, 0.08)  # width (x) and height (y) of one leaf
THICK = 0.002
KNUCKLE_R = 0.004


def _circle(cx: float, cy: float, r: float, n: int = 24) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def _leaf(side: float) -> ExtrudeWithHolesGeometry:
    """A leaf on one side (-1 left, +1 right) of the pin line x = 0."""
    width, height = LEAF
    cx = side * (KNUCKLE_R + width / 2)
    outline = [(x + cx, y) for x, y in rounded_rect_profile(width, height, 0.004)]
    holes = [_circle(cx + side * 0.004, y, 0.0025) for y in (-0.025, 0.0, 0.025)]
    return ExtrudeWithHolesGeometry(outline, holes, THICK)


def _knuckle(y: float, length: float):
    """A build123d cylinder along y, centered on the pin line."""
    return Pos(0, y, 0) * Rot(90, 0, 0) * Cylinder(KNUCKLE_R, length)


def build() -> Assembly:
    hinge = Assembly("door_hinge")
    span = LEAF[1] / 2

    fixed = hinge.body("fixed_leaf")
    fixed.add("plate", _leaf(-1))
    top = fixed.add("knuckle_top", _knuckle(span - 0.013, 0.026)).geometry.shape
    fixed.add("knuckle_bottom", _knuckle(-span + 0.013, 0.026))
    hex_head = ExtrudeGeometry([(0.0035 * math.cos(a), 0.0035 * math.sin(a))
                                for a in (k * math.pi / 3 for k in range(6))], 0.003)
    fixed.add("pin_head", hex_head.rotate_x(-math.pi / 2).translate(0, span + 0.0015, 0))

    moving = hinge.body("moving_leaf")
    moving.add("plate", _leaf(+1))
    moving.add("knuckle_middle", _knuckle(0.0, 0.025))

    # The joint frame is the top knuckle's axis, read from its cylindrical face.
    knuckle_face = top.faces().filter_by(GeomType.CYLINDER)[0]
    pin = fixed.at(knuckle_face)
    hinge.joint("pin", "revolute", pin, moving.at(pin.frame), axis=(0, 0, 1),
                limits=(0.0, math.radians(170)))
    return hinge


def main() -> None:
    hinge = build()
    print(hinge, "\npin frame:", hinge.get_joint("pin").parent)


if __name__ == "__main__":
    main()
