"""Four-bar linkage: ground, crank, coupler, rocker, closed by one pin.

Four pivots, four joints, but only three fit in a tree. The pin between the coupler
and the rocker is left out of the tree: it is the loop closure. Driving the crank
leaves the coupler and rocker angles to the loop solver.

Authoring a ring the easy way:

1. Build the open chain ground -> crank -> coupler and the rocker's ground pivot,
   with every joint at its rest pose (0). The coupler's rest angle is a design choice.
2. Ask the assembly where the coupler tip is, expressed on the rocker:
   ``assembly.frame_in(coupler_tip, rocker)``. That frame is where the rocker must
   meet the coupler, so the rocker is simply drawn from its pivot to that point.
   No law-of-cosines setup is needed, and the rest pose is closed by construction.

With these numbers the rocker comes out 70 mm long, so the linkage is a Grashof
crank-rocker (shortest + longest <= sum of the other two): the crank turns all the
way round while the rocker swings back and forth.
"""

from __future__ import annotations

import math

import numpy as np

from sdk import Assembly, Box, Cylinder, wire_from_points

CRANK = 0.03  # meters
COUPLER = 0.09
GROUND = 0.10  # distance between the two ground pivots, along x
COUPLER_REST_ANGLE = math.radians(50.0)  # coupler direction relative to the crank at rest
THICK = 0.004
WIDTH = 0.010


def _bar(body, length: float, z: float) -> None:
    body.add("bar", Box((length, WIDTH, THICK)), xyz=(length / 2, 0, z))
    body.add("pivot", Cylinder(radius=WIDTH / 2, height=THICK), xyz=(0, 0, z))


def build(rocker_limits: tuple[float, float] | None = None) -> Assembly:
    """``rocker_limits`` (radians, relative to the rest pose; must contain 0)."""
    linkage = Assembly("four_bar")
    z = (0, 0, 1)

    ground = linkage.body("ground")
    ground.add("frame", Box((GROUND + 0.03, 0.02, THICK)), xyz=(GROUND / 2, -0.015, -THICK))
    crank = linkage.body("crank")
    _bar(crank, CRANK, 0.0)
    coupler = linkage.body("coupler")
    _bar(coupler, COUPLER, THICK)
    rocker = linkage.body("rocker")  # geometry added below, once we know where the pin is

    linkage.joint("crank_joint", "revolute", ground.at((0, 0, 0)), crank.at(), axis=z)
    linkage.joint(
        "coupler_joint", "revolute",
        crank.at((CRANK, 0, 0), rpy=(0, 0, COUPLER_REST_ANGLE)), coupler.at(), axis=z,
    )
    linkage.joint(
        "rocker_joint", "revolute", ground.at((GROUND, 0, 0)), rocker.at(), axis=z,
        limits=rocker_limits,
    )

    # Where the coupler tip sits, in rocker coordinates, with everything at rest.
    coupler_tip = coupler.at((COUPLER, 0, 0))
    pin_on_rocker = linkage.frame_in(coupler_tip, rocker)
    rocker.add(
        "bar", wire_from_points([(0, 0, 0), pin_on_rocker.frame.xyz], radius=THICK / 2,
                                cap_ends=True),
    )
    rocker.add("pivot", Cylinder(radius=WIDTH / 2, height=THICK))  # same layer as the crank

    linkage.joint("pin", "revolute", coupler_tip, pin_on_rocker, axis=z)
    linkage.articulation(root="ground", closures=["pin"])
    return linkage


def rocker_length(linkage: Assembly) -> float:
    """Pivot-to-pin length of the rocker, read back from the authored pin frame."""
    return float(np.linalg.norm(linkage.get_joint("pin").child.frame.xyz))


def main() -> None:
    linkage = build()
    print(f"rocker length {rocker_length(linkage) * 1000:.2f} mm")
    for step in range(0, 360, 45):
        pose = linkage.solve({"crank_joint": math.radians(step)})
        rocker = math.degrees(pose.joint_values["rocker_joint"])
        print(f"crank {step:>3} deg -> rocker joint {rocker:+7.2f} deg, residual {pose.residual:.1e}")


if __name__ == "__main__":
    main()
