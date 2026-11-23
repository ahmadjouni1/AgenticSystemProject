"""Broken (and fixed) objects for the check tests. Each builder returns an Assembly whose
defect sits at a known place, so the tests can check the finding's names, pose and size."""

from __future__ import annotations

import math

from sdk import Assembly, Box, Cylinder, wire_from_points

# ---- laptop variants --------------------------------------------------------------------------

DECK = (0.32, 0.22, 0.015)
LID_T = 0.008


def laptop(lid_sink: float = 0.0, hinge_offset: tuple[float, float] = (0.0, 0.0)) -> Assembly:
    """A laptop authored closed. ``lid_sink`` pushes the lid down into the deck (an overlap);
    ``hinge_offset`` (dy, dz) moves the hinge axis away from the lid's back edge (it then
    pulls the lid off the base as it opens)."""
    width, depth, thick = DECK
    m = Assembly("laptop")
    base = m.body("base")
    base.add("deck", Box(DECK), xyz=(0, 0, thick / 2))
    lid = m.body("lid")
    dy, dz = hinge_offset
    hinge = (0.0, depth / 2 + dy, thick + dz)
    # Lid panel center in the world at rest: on top of the deck (minus the sink).
    center = (0.0, 0.0, thick + LID_T / 2 - lid_sink)
    lid.add("panel", Box((width, depth, LID_T)),
            xyz=tuple(c - h for c, h in zip(center, hinge, strict=True)))
    m.joint("hinge", "revolute", base.at(hinge), lid.at(), axis=(-1, 0, 0),
            limits=(0.0, math.radians(135)))
    return m


# ---- a knob that floats ------------------------------------------------------------------------


def floating_knob(gap: float = 0.010) -> Assembly:
    """A control panel and a knob whose face stands ``gap`` off the panel."""
    m = Assembly("panel")
    panel = m.body("panel")
    panel.add("plate", Box((0.3, 0.02, 0.2)))
    knob = m.body("knob")  # local z points out of the panel (world -y)
    knob.add("grip", Cylinder(0.015, 0.02), xyz=(0, 0, gap + 0.01))
    m.joint("turn", "revolute", panel.at((0, -0.01, 0), rpy=(math.pi / 2, 0, 0)), knob.at(),
            axis=(0, 0, 1), limits=(0.0, math.radians(270)))
    return m


# ---- a door that hits its frame ------------------------------------------------------------------


def door_in_frame(limit_degrees: float = 120.0) -> Assembly:
    """A door hinged on its corner. Past 90 degrees its thickness swings into the post."""
    m = Assembly("doorway")
    frame = m.body("frame")
    frame.add("post", Box((0.04, 0.04, 2.0)), xyz=(-0.02, -0.02, 1.0))  # x, y in [-0.04, 0]
    frame.add("floor", Box((1.2, 1.2, 0.02)), xyz=(0.3, 0.3, -0.01))
    door = m.body("door")
    door.add("panel", Box((0.8, 0.04, 1.9)), xyz=(0.4, 0.02, 0.96))  # x in [0, 0.8], y in [0, 0.04]
    m.joint("hinge", "revolute", frame.at((0, 0, 0)), door.at(), axis=(0, 0, 1),
            limits=(0.0, math.radians(limit_degrees)))
    return m


# ---- a drawer that hits a lip when fully extended ----------------------------------------------

DRAWER_SOURCE = '''
from sdk import Assembly, Box


def build_object_model():
    cabinet = Assembly("cabinet")
    body = cabinet.body("cabinet")
    body.add("bottom", Box((0.44, 0.40, 0.02)), xyz=(0, -0.20, 0.01))
    body.add("left_side", Box((0.02, 0.40, 0.20)), xyz=(-0.23, -0.20, 0.10))
    body.add("right_side", Box((0.02, 0.40, 0.20)), xyz=(0.23, -0.20, 0.10))
    body.add("lip", Box((0.44, 0.02, 0.03)), xyz=(0, -0.01, 0.145))  # across the mouth, up high
    drawer = cabinet.body("drawer")
    drawer.add("front", Box((0.44, 0.02, 0.16)), xyz=(0, 0.01, 0.10))
    drawer.add("tray", Box((0.40, 0.30, 0.01)), xyz=(0, -0.15, 0.025))
    drawer.add("back", Box((0.40, 0.02, 0.14)), xyz=(0, -0.29, 0.09))  # tall enough to meet the lip
    cabinet.joint("slide", "prismatic", body.at((0, 0, 0)), drawer.at(), axis=(0, 1, 0),
                  limits=(0.0, 0.28))
    return cabinet


object_model = build_object_model()
'''


def drawer() -> Assembly:
    namespace: dict = {}
    exec(DRAWER_SOURCE, namespace)  # noqa: S102 - our own fixture source
    return namespace["object_model"]


# ---- a four-bar whose crank and rocker cross mid-cycle --------------------------------------------


def four_bar(a: float = 0.02, b: float = 0.04, d: float = 0.05, coupler_degrees: float = 160.0,
             thick: float = 0.004, width: float = 0.008) -> Assembly:
    """Crank and rocker in one layer, the coupler above them, the ground below. The rocker
    is drawn to the pin with frame_in. With the defaults the rocker sweeps through the
    crank's circle, so the two links cross for part of the cycle."""
    m = Assembly("crossing_four_bar")
    ground = m.body("ground")
    ground.add("frame", Box((d + 0.03, 0.02, thick)), xyz=(d / 2, -0.03, -thick))
    for name, x in (("post0", 0.0), ("post1", d)):
        ground.add(name, Cylinder(width / 2, thick), xyz=(x, 0, -thick))
        ground.add(f"{name}_web", Box((width, 0.03, thick)), xyz=(x, -0.015, -thick))
    crank = m.body("crank")
    crank.add("bar", Box((a, width, thick)), xyz=(a / 2, 0, 0))
    crank.add("pivot", Cylinder(width / 2, thick))
    coupler = m.body("coupler")
    coupler.add("bar", Box((b, width, thick)), xyz=(b / 2, 0, thick))
    coupler.add("pivot", Cylinder(width / 2, thick), xyz=(0, 0, thick))
    rocker = m.body("rocker")
    z = (0, 0, 1)
    m.joint("crank_joint", "revolute", ground.at(), crank.at(), axis=z)
    m.joint("coupler_joint", "revolute", crank.at((a, 0, 0), rpy=(0, 0, math.radians(coupler_degrees))),
            coupler.at(), axis=z)
    m.joint("rocker_joint", "revolute", ground.at((d, 0, 0)), rocker.at(), axis=z)
    tip = coupler.at((b, 0, 0))
    pin = m.frame_in(tip, rocker)
    rocker.add("bar", wire_from_points([(0, 0, 0), pin.frame.xyz], radius=thick / 2, cap_ends=True))
    rocker.add("pivot", Cylinder(width / 2, thick))
    m.joint("pin", "revolute", tip, pin, axis=z)
    m.articulation(root="ground", closures=["pin"])
    return m


# ---- one shape, judged at two scales ------------------------------------------------------------


def stacked_blocks(size: float, offset: float) -> Assembly:
    """A block on a base, ``size`` wide. ``offset`` > 0 leaves a gap, < 0 sinks the block in."""
    m = Assembly("blocks")
    base = m.body("base")
    base.add("slab", Box((size, size, size / 2)), xyz=(0, 0, size / 4))
    block = m.body("block")
    block.add("cube", Box((size / 2, size / 2, size / 2)), xyz=(0, 0, size / 4))
    m.joint("glue", "fixed", base.at((0, 0, size / 2 + offset)), block.at())
    return m


# ---- two arms that only collide when both swing ---------------------------------------------


def two_arms(spread: float = 0.5, swing_degrees: float = 60.0) -> Assembly:
    """Two arms on a plate, pivots ``spread`` apart, both pointing +y at rest. Each swings
    toward the other. Either one alone stays clear; both at full swing cross."""
    m = Assembly("two_arms")
    plate = m.body("plate")
    plate.add("slab", Box((spread + 0.4, 0.5, 0.01)), xyz=(spread / 2, 0.15, -0.005))
    limit = math.radians(swing_degrees)
    for name, x, axis in (("left_arm", 0.0, (0, 0, -1)), ("right_arm", spread, (0, 0, 1))):
        arm = m.body(name)
        arm.add("bar", Box((0.02, 0.3, 0.02)), xyz=(0, 0.15, 0.01))
        m.joint(f"{name}_swing", "revolute", plate.at((x, 0, 0)), arm.at(), axis=axis,
                limits=(0.0, limit))
    return m
