# SDK quickstart

`from sdk import ...`. Meters and radians everywhere; the world is Z-up.

## Building blocks

```python
model = Assembly("laptop")             # the object
base = model.body("base")              # a rigid body; its frame is where you put it
base.add("deck", Box((0.32, 0.22, 0.015)), xyz=(0, 0, 0.0075), rpy=(0, 0, 0))
```

- `body.add(name, geometry, xyz=..., rpy=...)` places a shape in the body's frame. Shape names
  are unique within a body. A body may have several shapes; they should touch each other.
- Geometry:
  - `Box((sx, sy, sz))`: full extents, centred on its origin.
  - `Cylinder(radius, height)`: axis along local z, centred. Turn it with `rpy`
    (`rpy=(0, math.pi / 2, 0)` puts the axis along x).
  - `Sphere(radius)`.
  - `RoundedBoxGeometry((sx, sy, sz), radius)`: a box with rounded edges.
  - `LatheGeometry([(r, z), ...])`: a profile revolved about z (knobs, shades, bottles).
  - `ExtrudeGeometry([(x, y), ...], height)`: a 2D outline extruded along z;
    `rounded_rect_profile(w, h, r)` gives a rounded outline.
  - `wire_from_points([(x, y, z), ...], radius=r, cap_ends=True)`: a rod through points.
  - `tube_from_spline_points([...], radius=r)`: a smooth bent tube.
  - `boolean_difference(a, b)`, `boolean_union(a, b)`: mesh booleans (cut a slot, a hole).
- Place every shape with `body.add(..., xyz=..., rpy=...)`. `Box`, `Cylinder` and `Sphere` have
  no `.translate`. Only mesh geometries (`BoxGeometry((sx, sy, sz))`, `CylinderGeometry(r, h)`,
  the helpers above) can be moved before a boolean: `.translate(x, y, z)`, `.rotate_z(angle)`.

## Joints

```python
model.joint(name, type, parent.at(point, rpy=None), child.at(point, rpy=None),
            axis=(x, y, z), limits=(lower, upper), mimic=None)
```

- `type`: `"revolute"` (hinge; limits in radians), `"prismatic"` (slide; limits in meters) or
  `"fixed"`.
- `parent.at(...)` is the joint frame on the parent, `child.at(...)` the same frame on the child.
  At joint value 0 the two frames coincide: that is the **rest pose**, exactly as you authored
  the geometry.
- `axis` is in the parent frame. A positive value turns the child about the axis by the
  right-hand rule, or slides it along the axis.
- **Limits must contain 0** (the rest pose is a valid pose): a lid authored closed opens with
  `limits=(0, 2.3)`, a drawer authored shut slides out with `limits=(0, 0.3)`. `None` means
  unlimited (wheels, cranks).
- **Every body except the root has exactly one tree joint** to its parent. The root is the body
  with no parent (the base, the cabinet, the ground).
- **Mimic:** `mimic=("source_joint", multiplier, offset)` makes a joint follow another (two
  gears, a symmetric gripper).
- **Loops** (four-bar linkages): build the open chain, add the last joint as usual, then
  `model.articulation(root="ground", closures=["pin"])`. The solver closes the loop. To place the
  closing pin, `model.frame_in(coupler.at((L, 0, 0)), rocker)` gives the coupler tip's frame on
  the rocker at rest.

## A complete example: a laptop

```python
import math

from sdk import Assembly, Box

WIDTH, DEPTH = 0.32, 0.22          # x, y
BASE_T, LID_T = 0.015, 0.008       # thicknesses

laptop = Assembly("laptop")

base = laptop.body("base")
base.add("deck", Box((WIDTH, DEPTH, BASE_T)), xyz=(0, 0, BASE_T / 2))
base.add("keyboard", Box((WIDTH - 0.04, DEPTH / 2, 0.001)),
         xyz=(0, 0.02, BASE_T - 0.0005))          # inlaid in the deck's top face

# The lid's frame sits on the hinge line; the lid reaches forward (-y) from it.
lid = laptop.body("lid")
lid.add("panel", Box((WIDTH, DEPTH, LID_T)), xyz=(0, -DEPTH / 2, LID_T / 2))
lid.add("screen", Box((WIDTH - 0.02, DEPTH - 0.03, 0.001)), xyz=(0, -DEPTH / 2, 0.0005))

laptop.joint(
    "hinge", "revolute",
    base.at((0, DEPTH / 2, BASE_T)),    # the back edge of the deck's top face
    lid.at(),                           # the lid's own origin, on its hinge line
    axis=(-1, 0, 0),                    # positive angle lifts the lid's front edge
    limits=(0.0, math.radians(135)),    # authored closed: 0 is shut
)

object_model = laptop


def run_tests(ctx):
    ctx.expect_contact("base", "lid")                         # closed: the lid rests on the deck
    ctx.expect_contact("base", "lid", pose={"hinge": math.radians(90)})  # still on its hinge
```

Why it works: the lid is authored closed (lying on the deck), so 0 is a real pose; the hinge
frame is on the shared edge, so the lid swings about it without passing through the base; the
limits stop at a realistic 135°.

## What the compiler checks

Mesh health; every body connected (no floating parts); no overlap at rest; each joint swept over
its range (and joints moved together) without collisions; hinged parts staying attached; loops
that close over the driver's range; and your `run_tests`. Blocking failures stop the export and
come back as `<compile_signals>` with a file, a line and a fix hint.
