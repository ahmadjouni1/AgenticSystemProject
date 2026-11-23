"""Laptop: a base and a lid on one hinge that opens from 0 to 135 degrees.

Authored closed: at 0 the lid lies flat on the base. The hinge line runs along the
base's back edge (+y), and the axis is -x so a positive angle lifts the lid's front edge.
"""

from __future__ import annotations

import math

from sdk import Assembly, Box

WIDTH = 0.32  # x, meters
DEPTH = 0.22  # y
BASE_THICKNESS = 0.015
LID_THICKNESS = 0.008
OPEN_LIMIT = math.radians(135.0)


def build() -> Assembly:
    laptop = Assembly("laptop")

    base = laptop.body("base")
    base.add("deck", Box((WIDTH, DEPTH, BASE_THICKNESS)), xyz=(0, 0, BASE_THICKNESS / 2))
    base.add("keyboard", Box((WIDTH - 0.04, DEPTH / 2, 0.001)),
             xyz=(0, 0.02, BASE_THICKNESS - 0.0005))  # inlaid in the deck's top face

    # The lid's frame sits on the hinge line; the lid reaches forward (-y) from it.
    lid = laptop.body("lid")
    lid.add("panel", Box((WIDTH, DEPTH, LID_THICKNESS)), xyz=(0, -DEPTH / 2, LID_THICKNESS / 2))
    lid.add("screen", Box((WIDTH - 0.02, DEPTH - 0.03, 0.001)),
            xyz=(0, -DEPTH / 2, 0.0005))  # inlaid in the panel's inner face

    laptop.joint(
        "hinge",
        "revolute",
        base.at((0, DEPTH / 2, BASE_THICKNESS)),
        lid.at(),
        axis=(-1, 0, 0),
        limits=(0.0, OPEN_LIMIT),
    )
    return laptop


def main() -> None:
    laptop = build()
    print(laptop)
    for degrees in (0, 45, 90, 135):
        lid = laptop.forward_kinematics({"hinge": math.radians(degrees)})["lid"]
        corner = lid[:3, :3] @ (WIDTH / 2, -DEPTH, LID_THICKNESS) + lid[:3, 3]
        print(f"hinge {degrees:>3} deg -> lid front corner {corner.round(4)}")


if __name__ == "__main__":
    main()
