# SDK quickstart

`from sdk import ...`. Meters and radians; the world is Z-up. More pages: `shapes.md`,
`joints_and_loops.md`, `testing.md` (in `docs/sdk/`).

## Cheat sheet

```text
model = Assembly("name")                      # the object
body = model.body("name")                     # a rigid part; its frame is where you put it
body.add("shape", geometry, xyz=(x, y, z), rpy=(roll, pitch, yaw))   # place a shape on it
model.joint("name", "revolute" | "prismatic" | "fixed", parent.at(point, rpy=None),
            child.at(point, rpy=None), axis=(x, y, z), limits=(lower, upper), mimic=None)
model.articulation(root="base", closures=["pin"])  # only for closed loops
object_model = model                          # required
def run_tests(ctx): ...                       # optional extra checks (testing.md)
```

Geometry (signatures and examples in `shapes.md`):

| Need | Use |
| --- | --- |
| a part that really is a block | `Box((sx, sy, sz))`, centred |
| soft-edged block: housings, cushions, posts | `RoundedBoxGeometry((sx, sy, sz), radius)` |
| round part: knobs, legs, cups, shades | `LatheGeometry([(r, z), ...])` |
| plate or outline: brackets, jaws, panels, arches | `ExtrudeGeometry(profile_xy, thickness)` |
| plate with holes or a hollow section | `ExtrudeWithHolesGeometry(outer, [hole, ...], thickness)` |
| rod, rail, wire frame, bent handle | `wire_from_points(points, radius=r)` |
| smooth bent tube or cable | `tube_from_spline_points(points, radius=r)` |
| pin, axle, wheel | `Cylinder(radius, height)` (axis along local z) |
| holes, slots, unions | `boolean_difference(a, b)`, `boolean_union(a, b)` |

- Shapes on one body should touch each other. Turn a shape with `rpy`:
  `rpy=(0, math.pi / 2, 0)` puts a local z axis along x; `rpy=(math.pi / 2, 0, 0)` puts it
  along -y (so a 2D profile drawn in x, y stands up in x, z).
- `Box`, `Cylinder`, `Sphere` have no `.translate`. Mesh geometries (`BoxGeometry`,
  `CylinderGeometry` and every helper above) do: `.translate(x, y, z)`, `.rotate_z(a)`.

## Joints in one paragraph

`parent.at(...)` and `child.at(...)` are the same physical frame seen from each body; at joint
value 0 they coincide, so **0 is the pose you authored** and **limits must contain 0**. Put
the frame on the real hinge line or slide rail. `axis` is in the parent frame; positive
values turn by the right-hand rule or slide along it. Revolute limits are radians, prismatic
meters, `None` is unlimited (wheels). Every body but the root has exactly one joint to its
parent. `mimic=("source_joint", multiplier, offset)` makes a joint follow another. Loops
(four-bars, slider-cranks) and more patterns: `joints_and_loops.md`.

## A complete example: a mailbox with a hinged flap

```python
import math

from sdk import (Assembly, Box, Cylinder, ExtrudeGeometry, ExtrudeWithHolesGeometry,
                 LatheGeometry, RoundedBoxGeometry)

W, WALL_H, DEPTH, T = 0.18, 0.12, 0.46, 0.003  # width (x), straight side, depth (y), sheet
POST_H = 0.95


def arch(width, wall_h, inset=0.0, n=16):
    """Mailbox outline in the xz plane (x, z): straight sides and a half-round top."""
    r = width / 2 - inset
    pts = [(-r, inset), (r, inset)]
    pts += [(r * math.cos(a), wall_h + r * math.sin(a))
            for a in [math.pi * i / n for i in range(n + 1)]]
    return pts


UPRIGHT = (math.pi / 2, 0, 0)  # profile x -> x, profile y -> z, extrusion along -y

mailbox = Assembly("mailbox")

box = mailbox.body("box")  # the root: post, sheet-metal shell, back wall
box.add("post", RoundedBoxGeometry((0.09, 0.09, POST_H), 0.01), xyz=(0, 0, -POST_H / 2))
box.add("shell", ExtrudeWithHolesGeometry(arch(W, WALL_H), [arch(W, WALL_H, T)], DEPTH),
        rpy=UPRIGHT)                                           # a hollow tunnel, open at both ends
box.add("back", ExtrudeGeometry(arch(W, WALL_H), T), xyz=(0, DEPTH / 2 + T / 2, 0), rpy=UPRIGHT)

# The flap's frame is on its hinge line: the front bottom edge of the shell.
flap = mailbox.body("flap")
flap.add("panel", ExtrudeGeometry(arch(W, WALL_H), T), xyz=(0, -T / 2, 0), rpy=UPRIGHT)
flap.add("knob", LatheGeometry([(0, 0), (0.006, 0), (0.006, 0.006), (0.011, 0.014),
                                (0.011, 0.02), (0, 0.022)]),
         xyz=(0, -T, WALL_H + 0.03), rpy=UPRIGHT)               # lathe axis z -> out of the flap

# The flag pivots on the right side wall; at rest its arm points to the front.
flag = mailbox.body("flag")
flag.add("hub", Cylinder(0.012, 0.004), xyz=(0.002, 0, 0), rpy=(0, math.pi / 2, 0))
flag.add("arm", Box((0.004, 0.15, 0.012)), xyz=(0.002, -0.075, 0))
flag.add("plate", Box((0.004, 0.05, 0.035)), xyz=(0.002, -0.125, 0.0235))

mailbox.joint("flap_hinge", "revolute", box.at((0, -DEPTH / 2, 0)), flap.at(),
              axis=(1, 0, 0),                    # positive: the top of the flap falls forward
              limits=(0.0, math.radians(100)))
mailbox.joint("flag_pivot", "revolute", box.at((W / 2, 0.12, 0.06)), flag.at(),
              axis=(-1, 0, 0),                   # positive: the arm swings up
              limits=(0.0, math.pi / 2))

object_model = mailbox


def run_tests(ctx):
    ctx.expect_contact("box", "flap", pose={"flap_hinge": math.radians(100)})  # still hinged
    ctx.expect_contact("box", "flag", pose={"flag_pivot": math.pi / 2})
```

Why it works: parts are authored closed, so 0 is a real pose; the flap's hinge is on the edge
it shares with the shell, so it swings clear and stays in contact; the shell is a hollow 3 mm
sheet, and shaped parts use helpers (arched extrusion, lathed knob, rounded post).

The compiler checks mesh health, floating parts, overlap at rest, collisions over every
joint's range, hinges that come apart, loops that cannot close, and your `run_tests`.
