# Joints and loops

```text
model.joint(name, "revolute" | "prismatic" | "fixed", parent.at(point, rpy=None),
            child.at(point, rpy=None), axis=(x, y, z), limits=(lower, upper), mimic=None)
model.articulation(root="ground", closures=["pin"])
model.frame_in(attachment, body) -> attachment     # the same frame, on another body, at rest
```

## The joint frame

- `parent.at(p)` is a point (and optional `rpy`) in the parent's frame; `child.at(q)` the
  same physical point in the child's frame. At value 0 the two coincide: **the pose you drew
  is the rest pose**, and **limits must contain 0**.
- Easiest: give the child its own origin on the joint (`child.at()`), and draw its shapes
  around that origin as they sit at rest.
- `axis` is in the parent frame. Positive values turn by the right-hand rule, or slide along
  `axis`. Revolute limits are radians, prismatic meters; `limits=None` is unlimited.
- **Which axis sign?** If the child reaches from the hinge in direction `d` and should first
  move toward direction `n`, use `axis = d x n` (cross product). A lid reaching to -y that
  opens upward: `(0,-1,0) x (0,0,1) = (-1,0,0)`. Check it with `probe`.
- Every body except the root has exactly one tree joint to its parent. `"fixed"` joins two
  bodies rigidly (prefer one body with several shapes).

## Patterns

| Real part | Type | Frame on | Axis | Limits |
| --- | --- | --- | --- | --- |
| lid hinged at the back | revolute | back top edge | `d x n`, e.g. `(-1,0,0)` | `(0, 1.9)` |
| door on a side hinge | revolute | the vertical hinge line | `(0,0,1)` or `(0,0,-1)` | `(0, 1.7)` |
| drawer, tray, slide | prismatic | anywhere on the rail | the pull-out direction | `(0, 0.8 * depth)` |
| wheel, crank, knob | revolute | axle centre | axle direction | `None` |
| folding leg or arm | revolute | the pivot | `d x n` | the real fold angle |
| two levers on one pin | revolute | the pin centre | the pin direction | the real opening |

Making motion clean:
- **Keep hinged parts touching at the hinge.** Put the frame on an edge both parts share, or
  give the joint a barrel or pin that touches both bodies. The compiler fails a hinge whose
  gap grows as it turns.
- **Leave clearance** (1-3 mm) between parts that move past each other: a drawer inside its
  carcass, a door beside its neighbour, a tray on its rails.
- **Crossing levers** go on separate layers (one on top of the other, touching), or the pin
  passes through both with `ctx.allow_overlap(a, b, reason=..., shape_a="pin", shape_b=...)`.
- **Mimic:** `mimic=("left_jaw", -1.0, 0.0)` mirrors another joint (symmetric jaws, gears).
  A mimic joint is not driven; set the source joint instead.

## Closed loops: a four-bar with `frame_in`

A ring of N pivots has N joints but only N - 1 fit in a tree. Build the open chain, then add
the last pin as a **closure**: the solver moves the undriven joints so it stays closed.

1. Author the chain ground -> crank -> coupler, and the rocker's ground pivot, all at 0.
2. `model.frame_in(coupler.at(tip), rocker)` says where the coupler tip sits on the rocker
   at rest. Draw the rocker from its pivot to that point: the loop is closed by construction.
3. Add the closure pin and call `model.articulation(root=..., closures=["pin"])`.

Drive one joint (the crank); probe the others. A loop may contain prismatic joints too (a
slider in a guide): the same steps apply, with the closure a revolute pin.

```python
import math

from sdk import Assembly, Box, ExtrudeGeometry, rounded_rect_profile

CRANK, COUPLER, GROUND, T, W = 0.03, 0.09, 0.10, 0.006, 0.012  # m


def link(length):
    """A flat link with rounded ends, from its first pivot (origin) along +x."""
    return ExtrudeGeometry(rounded_rect_profile(length + W, W, W / 2), T).translate(length / 2, 0, 0)


model = Assembly("four_bar")
ground = model.body("ground")
ground.add("plate", Box((GROUND + 0.06, 0.05, T)), xyz=(GROUND / 2, -0.01, -T))
crank, coupler, rocker = model.body("crank"), model.body("coupler"), model.body("rocker")
crank.add("link", link(CRANK))
coupler.add("link", link(COUPLER), xyz=(0, 0, T))          # one layer up: passes over the others

z = (0, 0, 1)
model.joint("crank_joint", "revolute", ground.at((0, 0, 0)), crank.at(), axis=z)
model.joint("coupler_joint", "revolute", crank.at((CRANK, 0, 0), rpy=(0, 0, math.radians(50))),
            coupler.at(), axis=z)
model.joint("rocker_joint", "revolute", ground.at((GROUND, 0, 0)), rocker.at(), axis=z)

tip = coupler.at((COUPLER, 0, 0))
pin = model.frame_in(tip, rocker)                          # the coupler tip, on the rocker
x, y, _ = pin.frame.xyz
rocker.add("link", link(math.hypot(x, y)), rpy=(0, 0, math.atan2(y, x)))
model.joint("pin", "revolute", tip, pin, axis=z)
model.articulation(root="ground", closures=["pin"])

object_model = model
```
