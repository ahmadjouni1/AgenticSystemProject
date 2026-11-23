"""Gantry: a bridge on two rail carriages, a tool head, and a cable trolley.

Joints:

- ``x_drive`` drives the left carriage along x (the only motor on this axis).
- ``x_follow`` mimics it 1:1. A real dual-rail gantry drives both sides together
  so the bridge stays square.
- ``cable_trolley`` mimics it at 0.5. A loop of cable over a moving pulley travels
  half as far as the carriage it feeds.
- ``y_head`` is an ordinary slide for the tool head along the bridge.

The bridge is mounted on the left carriage (tree joint) and also bolted to the
right carriage (``bridge_right``, a fixed loop closure). With both carriages
mimicked, nothing on that loop is left free, so the solver only checks that the
mimic keeps the loop closed. A wrong multiplier raises ``LoopClosureError``.
"""

from __future__ import annotations

from sdk import Assembly, Box

TRAVEL = 0.9  # x travel, meters
RAIL_Y = 0.4  # rails at y = +-0.4
RAIL_TOP = 0.04
CARRIAGE_HEIGHT = 0.05
START_X = 0.1


def build(follow_multiplier: float = 1.0) -> Assembly:
    gantry = Assembly("gantry")

    base = gantry.body("base")
    for side, y in (("left", -RAIL_Y), ("right", RAIL_Y)):
        base.add(f"{side}_rail", Box((1.2, 0.04, RAIL_TOP)), xyz=(0.6, y, RAIL_TOP / 2))
    base.add("cable_guide", Box((0.7, 0.03, 0.01)), xyz=(0.35, -RAIL_Y - 0.1, 0.005))
    for end, x in (("front_frame", 0.02), ("back_frame", 1.18)):  # ties rails and guide
        base.add(end, Box((0.04, 2 * RAIL_Y + 0.14, 0.01)), xyz=(x, -0.05, 0.005))

    carriage = Box((0.12, 0.08, CARRIAGE_HEIGHT))
    left = gantry.body("left_carriage")
    left.add("block", carriage, xyz=(0, 0, CARRIAGE_HEIGHT / 2))
    right = gantry.body("right_carriage")
    right.add("block", carriage, xyz=(0, 0, CARRIAGE_HEIGHT / 2))

    # Bridge frame: its middle, on the carriage tops. It spans y = -0.44 .. 0.44.
    bridge = gantry.body("bridge")
    bridge.add("beam", Box((0.08, 2 * RAIL_Y + 0.08, 0.06)), xyz=(0, 0, 0.03))
    head = gantry.body("tool_head")
    head.add("block", Box((0.1, 0.1, 0.12)), xyz=(0.09, 0, 0.0))
    trolley = gantry.body("trolley")
    trolley.add("block", Box((0.06, 0.06, 0.03)), xyz=(0, 0, 0.015))

    x = (1, 0, 0)
    gantry.joint(
        "x_drive", "prismatic",
        base.at((START_X, -RAIL_Y, RAIL_TOP)), left.at(), axis=x, limits=(0.0, TRAVEL),
    )
    gantry.joint(
        "x_follow", "prismatic",
        base.at((START_X, RAIL_Y, RAIL_TOP)), right.at(), axis=x, limits=(0.0, TRAVEL),
        mimic=("x_drive", follow_multiplier, 0.0),
    )
    gantry.joint(
        "cable_trolley", "prismatic",
        base.at((START_X, -RAIL_Y - 0.1, 0.01)), trolley.at(), axis=x,
        limits=(0.0, TRAVEL / 2), mimic=("x_drive", 0.5, 0.0),
    )
    gantry.joint(
        "bridge_mount", "fixed", left.at((0, 0, CARRIAGE_HEIGHT)), bridge.at((0, -RAIL_Y, 0))
    )
    gantry.joint(
        "bridge_right", "fixed", right.at((0, 0, CARRIAGE_HEIGHT)), bridge.at((0, RAIL_Y, 0))
    )
    gantry.joint(
        "y_head", "prismatic",
        bridge.at((0, 0, 0.03)), head.at(), axis=(0, 1, 0), limits=(-0.30, 0.30),
    )
    gantry.articulation(root="base", closures=["bridge_right"])
    return gantry


def main() -> None:
    gantry = build()
    for drive in (0.0, 0.3, 0.9):
        pose = gantry.solve({"x_drive": drive, "y_head": 0.1})
        values = {k: round(v, 4) for k, v in pose.joint_values.items()}
        print(f"x_drive={drive}: {values}, residual {pose.residual:.1e}")


if __name__ == "__main__":
    main()
