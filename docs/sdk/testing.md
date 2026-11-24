# Testing: `run_tests(ctx)`

`main.py` may define `run_tests(ctx)`. The compiler calls it after building the model; a
failed expectation blocks the compile like any other check, with its name and the measured
value.

## Already checked for you (do not re-test)

Mesh health; floating bodies; overlap at rest; every joint swept over its range, and joints
moved together, without collisions; hinge gaps that grow as a joint turns; loops that stay
closed. Write tests only for what the design intends and the generic checks cannot know: the
lid rests on the base when closed, the drawer is still in its rails when fully open, a handle
clears the floor at every angle.

## API

Bodies are names (or `Body` objects). `pose` is `{joint: value}`; unset joints stay at 0.

| Call | Checks / does |
| --- | --- |
| `ctx.expect_contact(a, b, pose=None)` | bodies `a`, `b` touch (gap <= contact tolerance) |
| `ctx.expect_gap(a, b, min_gap=0.0, max_gap=None, pose=None)` | the closest gap is in range (m) |
| `ctx.expect_distance((a, pa), (b, pb), distance, tolerance=None, pose=None)` | two body-local points are `distance` apart |
| `ctx.expect_no_collision_at_poses([pose, ...], bodies=None)` | no collisions at these extra poses |
| `ctx.check(name, ok, details="")`, `ctx.fail(name, details)` | your own condition |
| `ctx.allow_overlap(a, b, reason, shape_a=None, shape_b=None)` | intended interpenetration (a pin in a bore); name the shapes to keep it narrow |
| `ctx.allow_isolated(body, reason)` | a body that may float (rare) |
| `ctx.sample_joint(joint, samples=5)` | `[(value, pose)]` over the range; `pose.transforms[body]` is a 4x4 matrix |
| `ctx.track_point(body, point, joint, samples=5)` | `[(value, world xyz)]` of a body-local point |
| `ctx.size` | the rest pose's bounding-box diagonal (m) |

Every `expect_*` takes an optional `name=` that labels it in the signals.

Tolerances scale with the object: contact means a gap <= `max(0.1 mm, 0.001 x size)`;
an overlap is a penetration deeper than `max(0.2 mm, 0.002 x size)`.

**Allowances need a concrete reason** and should be as narrow as possible: two named shapes,
not two whole bodies. Prefer fixing the geometry; an allowance that hides a real collision
is a defect.

## Example: a floor hatch on a pin hinge

```python
import math

from sdk import (Assembly, Box, Cylinder, ExtrudeWithHolesGeometry, rounded_rect_profile,
                 wire_from_points)

X = (0, math.pi / 2, 0)  # a cylinder's axis along x

hatch = Assembly("floor_hatch")
frame = hatch.body("frame")
frame.add("rim", ExtrudeWithHolesGeometry(rounded_rect_profile(0.70, 0.70, 0.02),
                                          [rounded_rect_profile(0.62, 0.62, 0.01)], 0.03),
          xyz=(0, 0, -0.015))
for side in (-1, 1):
    frame.add(f"lug_{side + 1}", Box((0.02, 0.02, 0.02)), xyz=(side * 0.34, 0.34, 0.01))
frame.add("pin", Cylinder(0.004, 0.70), xyz=(0, 0.34, 0.01), rpy=X)

lid = hatch.body("lid")  # its frame is on the pin axis
lid.add("barrel", Cylinder(0.01, 0.60), rpy=X)
lid.add("plate", Box((0.66, 0.66, 0.02)), xyz=(0, -0.34, 0))
lid.add("handle", wire_from_points([(-0.06, -0.60, 0.01), (-0.06, -0.60, 0.04),
                                    (0.06, -0.60, 0.04), (0.06, -0.60, 0.01)],
                                   radius=0.006, corner_radius=0.01, cap_ends=True))

hatch.joint("hinge", "revolute", frame.at((0, 0.34, 0.01)), lid.at(), axis=(-1, 0, 0),
            limits=(0.0, math.radians(110)))
object_model = hatch


def run_tests(ctx):
    ctx.allow_overlap("frame", "lid", reason="the hinge pin passes through the lid's barrel",
                      shape_a="pin", shape_b="barrel")
    ctx.expect_contact("frame", "lid", name="closed lid rests on the rim")
    for value, point in ctx.track_point("lid", (0, -0.60, 0.04), "hinge"):
        ctx.check(f"handle above the floor at {value:.2f} rad", point[2] > 0.0,
                  f"z = {point[2]:.3f} m")
```
