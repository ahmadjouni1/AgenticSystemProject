# Joints and loops

```text
model.joint(name, "revolute" | "prismatic" | "fixed", parent.at(point, rpy=None),
            child.at(point, rpy=None), axis=(x, y, z), limits=(lower, upper), mimic=None)
model.articulation(root="ground", closures=["pin"])
model.frame_in(attachment, body) -> attachment     # the same frame, on another body, at rest
```

## The joint frame

- `parent.at(p)` is a point in the parent's frame, `child.at(q)` the same point in the
  child's. At value 0 they coincide: **the pose you drew is the rest pose**, and **limits
  must contain 0**. Easiest: the child's origin on the joint (`child.at()`).
- `axis` is in the parent frame; positive values turn by the right-hand rule, or slide
  along it. Radians or meters; `limits=None` is unlimited.
- **Axis sign:** if the child reaches from the joint in direction `d` and should first move
  toward `n`, use `axis = d x n`. A lid reaching to -y that opens upward:
  `(0,-1,0) x (0,0,1) = (-1,0,0)`.
- Drawer or tray: prismatic along the pull-out direction, `(0, 0.8 * depth)`. Wheel, crank,
  knob: revolute on the axle, `None`. **Mimic** `mimic=("left_jaw", -1.0, 0.0)` mirrors
  another joint (it is not driven).

## Contact and clearance

- **Joined parts touch: gap 0.** The floating check fails a body that misses its parent by
  even 1 mm. Put the joint on a line both parts share (a door's back edge on the carcass
  front, a rail under a table top), or a pin inside a barrel with
  `ctx.allow_overlap(a, b, reason=..., shape_a="pin", shape_b="barrel")`.
- **Clearance (1-3 mm) only between parts that move past each other**: a drawer in its
  carcass, two doors side by side. Never at the joint. Crossing levers go on two layers.

## Recipe: a door on a carcass with real panel thickness

The hinge line is the door's back edge on the carcass front edge: the door swings out and
never enters the side panel.

```python
import math

from sdk import Assembly, Box, Cylinder

W, D, H, T = 0.40, 0.30, 0.50, 0.018  # outer size and panel thickness, m

model = Assembly("wall_cupboard")
carcass = model.body("carcass")
for name, x in (("left", T / 2 - W / 2), ("right", W / 2 - T / 2)):
    carcass.add(name, Box((T, D, H)), xyz=(x, 0, H / 2))
for name, z in (("bottom", T / 2), ("top", H - T / 2)):
    carcass.add(name, Box((W - 2 * T, D, T)), xyz=(0, 0, z))
door = model.body("door")  # origin on the hinge line
door.add("panel", Box((W, T, H)), xyz=(W / 2, -T / 2, H / 2))
door.add("knob", Cylinder(0.012, 0.02), xyz=(W - 0.04, -T - 0.01, H / 2), rpy=(math.pi / 2, 0, 0))
# reaches +x, opens toward -y: (1,0,0) x (0,-1,0) = (0,0,-1)
model.joint("door_hinge", "revolute", carcass.at((-W / 2, -D / 2, 0)), door.at(),
            axis=(0, 0, -1), limits=(0, math.radians(110)))
object_model = model
```

## Recipe: folding legs that nest

Each rail is a hinge axis touching the top's underside. Both fold inward; the narrow frame's
legs pass inside the wide one's. Rail radius > half the leg size: folded legs clear the top.

```python
import math

from sdk import Assembly, Box, Cylinder

TOP_Z, L, R, T = 0.74, 0.695, 0.015, 0.025  # top height, leg length, rail radius, leg size
model = Assembly("camp_bench")
top = model.body("top")
top.add("board", Box((1.2, 0.7, 0.03)), xyz=(0, 0, TOP_Z - 0.015))


def frame(name, half):
    body = model.body(name)  # origin on the rail axis
    body.add("rail", Cylinder(R, 2 * half + T), rpy=(math.pi / 2, 0, 0))
    for i, y in enumerate((-half, half)):
        body.add(f"leg_{i}", Box((T, T, L)), xyz=(0, y, -L / 2))
    return body


wide, narrow = frame("wide", 0.30), frame("narrow", 0.24)
z = TOP_Z - 0.03 - R  # rail top on the underside
model.joint("wide_fold", "revolute", top.at((-0.45, 0, z)), wide.at(), axis=(0, -1, 0),
            limits=(0, math.pi / 2))  # (0,0,-1) x (1,0,0)
model.joint("narrow_fold", "revolute", top.at((0.45, 0, z)), narrow.at(), axis=(0, 1, 0),
            limits=(0, math.pi / 2))  # (0,0,-1) x (-1,0,0)
object_model = model
```

## Closed loops: a four-bar with `frame_in`

N pivots in a ring are N joints, but only N - 1 fit in a tree: the last pin is a
**closure**, which the solver keeps closed. Author the chain at 0; `frame_in` gives where the
coupler tip sits on the rocker, so the rocker is drawn to it; then the pin and
`model.articulation(closures=[...])`. Drive the crank. Loops may hold prismatic joints too.

```python
import math

from sdk import Assembly, Box, ExtrudeGeometry, rounded_rect_profile

CRANK, COUPLER, GROUND, T, W = 0.03, 0.09, 0.10, 0.006, 0.012  # m


def link(length):  # a flat link with rounded ends, from its first pivot along +x
    return ExtrudeGeometry(rounded_rect_profile(length + W, W, W / 2), T).translate(length / 2, 0, 0)


model = Assembly("four_bar")
ground = model.body("ground")
ground.add("plate", Box((GROUND + 0.06, 0.05, T)), xyz=(GROUND / 2, -0.01, -T))
crank, coupler, rocker = model.body("crank"), model.body("coupler"), model.body("rocker")
crank.add("link", link(CRANK))
coupler.add("link", link(COUPLER), xyz=(0, 0, T))  # one layer up: passes over the others
z = (0, 0, 1)
model.joint("crank_joint", "revolute", ground.at((0, 0, 0)), crank.at(), axis=z)
model.joint("coupler_joint", "revolute", crank.at((CRANK, 0, 0), rpy=(0, 0, math.radians(50))),
            coupler.at(), axis=z)
model.joint("rocker_joint", "revolute", ground.at((GROUND, 0, 0)), rocker.at(), axis=z)
tip = coupler.at((COUPLER, 0, 0))
pin = model.frame_in(tip, rocker)  # the coupler tip, on the rocker
x, y, _ = pin.frame.xyz
rocker.add("link", link(math.hypot(x, y)), rpy=(0, 0, math.atan2(y, x)))
model.joint("pin", "revolute", tip, pin, axis=z)
model.articulation(root="ground", closures=["pin"])
object_model = model
```
