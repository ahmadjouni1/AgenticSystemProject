# Shapes

Every helper returns geometry for `body.add(name, geometry, xyz=..., rpy=...)`. Profiles are
lists of 2D points `(x, y)`; paths are lists of 3D points. Sizes are meters. Pick the helper
that matches how the real part is made; a plain `Box` only for parts that are boxes.

## Primitives

- `Box((sx, sy, sz))`, `Cylinder(radius, height)` (axis z), `Sphere(radius)`: centred, placed
  only by `xyz`/`rpy` (no `.translate`).
- `BoxGeometry(size)`, `CylinderGeometry(radius, height)`: the same as meshes (movable, booleans).
- `RoundedBoxGeometry(size, radius)`: a box with every edge rounded: housings, cases, cushions.
- `ConeGeometry(radius, height)`: base at -h/2, tip at +h/2. `CapsuleGeometry(radius, length)`:
  a rod with round ends, `length + 2 * radius` tall. `TorusGeometry(radius, tube)`: a ring in xy.

```python
from sdk import CapsuleGeometry, ConeGeometry, RoundedBoxGeometry, TorusGeometry

housing = RoundedBoxGeometry((0.20, 0.12, 0.05), 0.008)
nozzle = ConeGeometry(0.01, 0.03)
grip = CapsuleGeometry(0.012, 0.10)
ring = TorusGeometry(0.03, 0.004)
```

## 2D profiles

- `rounded_rect_profile(width, height, radius)`: a rectangle with round corners.
- `superellipse_profile(width, height, exponent=2.6)`: between an ellipse (2) and a rectangle.
- `sample_catmull_rom_spline_2d(points, samples_per_segment=12, closed=False)`: a smooth curve
  through points (a jaw outline, a handle side).

## Solids from profiles

- `ExtrudeGeometry(profile, height)`: the outline extruded along z, centred
  (`ExtrudeGeometry.from_z0(...)` starts at z = 0). Plates, links, jaws, arched panels.
- `ExtrudeWithHolesGeometry(outer, [hole, ...], height)`: a plate with holes, or a hollow
  section (an outline with an inner outline as the hole). Holes must lie inside, apart.
- `LatheGeometry([(r, z), ...], segments=32, angle=2*pi, axis="z")`: a closed `(radius, z)`
  outline revolved: knobs, legs, feet, bottles. `LatheGeometry.from_shell_profiles(outer,
  inner)`: a thin-walled vessel (cup, shade, bowl) from the outer and inner wall curves.
- `LoftGeometry([ring, ...])`: a skin through 3D rings with the same point count.
  `section_loft([ring, ...])` resamples rings of different counts (a tapering handle).

```python
import math

from sdk import (ExtrudeGeometry, ExtrudeWithHolesGeometry, LatheGeometry, section_loft,
                 rounded_rect_profile, sample_catmull_rom_spline_2d, superellipse_profile)

plate = ExtrudeGeometry(rounded_rect_profile(0.12, 0.06, 0.01), 0.004)
circle = [(0.006 * math.cos(a / 8 * math.tau), 0.006 * math.sin(a / 8 * math.tau))
          for a in range(8)]
bracket = ExtrudeWithHolesGeometry(rounded_rect_profile(0.08, 0.04, 0.006),
                                   [[(x - 0.025, y) for x, y in circle],
                                    [(x + 0.025, y) for x, y in circle]], 0.005)
frame = ExtrudeWithHolesGeometry(rounded_rect_profile(0.3, 0.2, 0.01),
                                 [rounded_rect_profile(0.27, 0.17, 0.005)], 0.02)
knob = LatheGeometry([(0, 0), (0.012, 0), (0.014, 0.01), (0.009, 0.02), (0, 0.021)])
shade = LatheGeometry.from_shell_profiles([(0.03, 0), (0.08, 0.1)], [(0.028, 0), (0.078, 0.1)])
jaw = ExtrudeGeometry(sample_catmull_rom_spline_2d(
    [(0, 0), (0.06, 0.008), (0.09, 0.004), (0.06, -0.006)], closed=True), 0.008)
puck = ExtrudeGeometry(superellipse_profile(0.05, 0.03), 0.01)
handle = section_loft([[(x, y, 0) for x, y in rounded_rect_profile(0.03, 0.02, 0.005)],
                       [(x, y, 0.12) for x, y in rounded_rect_profile(0.02, 0.015, 0.004)]])
```

## Rods, tubes and sweeps

- `wire_from_points(points, radius=r, cap_ends=False, corner_radius=0.0)`: a round rod
  through straight segments; `corner_radius` bends the corners. Rails, frames, wire handles.
- `tube_from_spline_points(points, radius=r, cap_ends=True)`: a smooth tube through points:
  bent arms, hoses, cables, curved handles.
- `sweep_profile_along_spline(points, profile=p, cap_profile=True)`: any 2D profile along a
  smooth curve: a flat mug handle, a gasket, a rail with a rounded-rectangle section.
- `PipeGeometry(profile, path, cap=False)`: a profile swept along a polyline.

```python
from sdk import (rounded_rect_profile, sweep_profile_along_spline, tube_from_spline_points,
                 wire_from_points)

rail = wire_from_points([(0, 0, 0), (0, 0, 0.3), (0.4, 0, 0.3), (0.4, 0, 0)], radius=0.008,
                        cap_ends=True, corner_radius=0.03)
arm = tube_from_spline_points([(0, 0, 0), (0.05, 0, 0.12), (0.15, 0, 0.18)], radius=0.01)
loop = sweep_profile_along_spline([(0, 0, 0), (0.04, 0, 0.03), (0, 0, 0.07)],
                                  profile=rounded_rect_profile(0.012, 0.006, 0.002))
```

## Booleans and mesh edits

- `boolean_difference(a, b)`, `boolean_union(a, b)`, `boolean_intersection(a, b)`: closed
  meshes in, one mesh out (cut a slot, drill a hole, fuse a boss).
- Mesh geometries (not `Box`/`Cylinder`/`Sphere`) can be moved before a boolean, in place:
  `.translate(x, y, z)`, `.rotate_x(a)` / `.rotate_y(a)` / `.rotate_z(a)`,
  `.rotate(axis, angle, origin=(0, 0, 0))`, `.scale(sx, sy=None, sz=None)`; `.copy()` first
  to keep the original.

```python
import math

from sdk import BoxGeometry, CylinderGeometry, boolean_difference, boolean_union

block = BoxGeometry((0.06, 0.04, 0.02))
drilled = boolean_difference(block, CylinderGeometry(0.006, 0.03))
slotted = boolean_difference(BoxGeometry((0.06, 0.04, 0.02)),
                             BoxGeometry((0.04, 0.006, 0.03)).translate(0, 0.01, 0))
boss = boolean_union(BoxGeometry((0.06, 0.04, 0.02)),
                     CylinderGeometry(0.008, 0.02).rotate_y(math.pi / 2).translate(0, 0, 0.01))
```
