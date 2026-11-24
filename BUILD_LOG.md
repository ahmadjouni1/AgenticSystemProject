# Replica build log

We are rebuilding Articraft from scratch, one step at a time, to understand each part.
Every step records four things: what was built, how Articraft does it (section numbers
refer to `docs/ARTICRAFT_STUDY.md`), what we changed and why, and open ideas.

**Environment:** Windows 11, Python 3.13.7 through uv (`.python-version`, local `.venv`).
The dependencies are numpy 2.5.5, scipy 1.18.1, trimesh 5.2.2, build123d 0.13.0,
manifold3d 3.5.5 (added in step 1c) and pytest 9.2.0. There is no build backend:
`[tool.uv] package = false`, and tests import from the project root through pytest's
`pythonpath`.

```shell
uv sync
uv run pytest -q
uv run python -m examples.four_bar
```

---

## Step 1: Geometry SDK

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `sdk/transforms.py` | 100 | 4x4 math, URDF roll-pitch-yaw (`R = Rz·Ry·Rx`), Rodrigues rotation, rigid inverse |
| `sdk/geometry.py` | 143 | `Box`, `Cylinder`, `Sphere`, `Mesh`, and build123d shapes tessellated to trimesh at 0.5 mm |
| `sdk/frames.py` | 82 | `Frame` (xyz, rpy), `Attachment` (body + frame), and build123d `Location` conversion |
| `sdk/body.py` | 84 | `Body.add(name, shape, xyz, rpy)`, `Body.at(point, rpy)`, `Body.trimesh()` |
| `sdk/joint.py` | 139 | `Joint` (fixed, revolute or prismatic), axis, limits, `Mimic(source, multiplier, offset)` |
| `sdk/assembly.py` | 325 | `Assembly` authoring, validation, choice of tree and closures, `solve` with walking warm start |
| `sdk/kinematics.py` | 209 | tree propagation, mimic, closure residuals, `least_squares` loop solve, `Pose` |
| `examples/` | 232 | `laptop` (hinge 0–135°), `four_bar` (one pin closes the loop), `gantry` (two mimic carriages and a fixed loop closure) |
| `tests/` | 475 | 62 tests: validation, laptop corner positions, four-bar full turn, mimic, geometry |

**How it works.**
- **Placement rule.** A joint places its child with
  `T_child = T_parent @ F_parent @ motion(q) @ inv(F_child)`. The joint's axis is given in
  the parent attachment frame.
- **Tree and closures.** The tree is made of parent-to-child joints with exactly one root.
  You can name the loop-closure joints yourself (`articulation(closures=[...])`). Otherwise,
  any joint whose two bodies are already connected, in authored order, becomes a closure.
- **Solving a pose.** `solve()` does four things:
  1. fills in the joint values: the ones you drive, then rest values for the rest;
  2. applies mimic joints, sources first;
  3. places every body down the tree;
  4. solves the free joints on each loop with `scipy.least_squares` (trf, bounded by limits)
     until each closure joint can be satisfied again.

  A closure's residual is: the frames' offset, plus a check that the hinge axis maps onto
  itself (revolute), or that the offset is perpendicular to the axis and there is no rotation
  (prismatic and fixed).
- **Outputs.** `forward_kinematics(values)` returns `{body: 4x4}`. `solve(values)` returns a
  `Pose` with the transforms, every joint value (driven, mimic, solved, closure) and the
  residual.

**Verified:** 62 tests pass on Windows in about 5 s. On the laptop, the lid corner matches the
hand-derived formula at 0, 30, 45, 90, 120 and 135°; for example, (0.16, 0.118, 0.235) m at
90°. On the four-bar, the pin matches a circle-intersection solution to 1e-9 m at every 5°
over a full crank turn, and the rocker returns to 0. On the gantry, `x_follow == x_drive`,
`cable_trolley == 0.5·x_drive`, and the bridge closure residual is 0. A four-bar solve takes
about 1.7 ms per 1° step.

### How Articraft does it (study §1)

- **Joints.** `sdk/assembly.py` makes every joint a USD D6 joint (`Joint`, `JointDOF`,
  `JointAxis`). The DOFs you list are free and the rest are locked. Each DOF has
  `limits=(lower, upper)`, and that range must contain 0 (`JointDOF.__post_init__`).
- **Loops.** An `Articulation` lists the tree joints. Joints left out of it are closures.
  `_solve_closed_loops` runs `least_squares` (dogbox) in 5 homotopy steps from the zero pose.
  `LoopClosureError` is raised above 1e-6, and the message lists joints pinned at their
  limits.
- **Frames.** `body.at(...)` binds a point or a build123d feature (face, edge, axis). `rpy` is
  in radians, and `frames.py` converts build123d Locations.
- **No mimic joints.** Coupled motion has to be a real loop. `_synthesis.py` rejects URDF
  `<mimic>`.
- **Shapes.** A shape is a build123d shape tessellated at 1 mm, or `MeshGeometry` from about
  6k lines of `_mesh/*`.

### What we changed, and why

| Change | Why |
| --- | --- |
| Three explicit joint types (fixed, revolute, prismatic) with one axis, instead of D6 DOF lists | These cover the examples and match URDF, which step 2 will export. A D6 joint can come later as a type if it is ever needed. |
| ~~Limits only need `lower <= upper`. The rest value is 0 clamped into the limits~~ **Reversed in step 1b**, where Articraft's rule is restored | (original reason) Articraft's rule that 0 must be inside the limits forces awkward authoring, such as a lid that is only allowed to open from 20°. |
| **Mimic joints added:** `mimic=(source, multiplier, offset)`, sources may chain, cycles are rejected, and the mimic's own limits are checked | They are missing in Articraft (study §1). They are needed for synchronized dual drives, 2:1 cable trolleys, grippers and gears. |
| Mimic joints may sit on a loop. Then the loop solver moves the mimic's *source* | Without this the gantry's bolted bridge could not be verified. The solver confirms that the mimic keeps the loop closed (`test_gantry_wrong_mimic_breaks_the_loop`). |
| Closures are named (`closures=[...]`) or detected automatically | Naming one closure is shorter than listing every tree joint the way Articraft does, and auto-detection covers the usual case. |
| **Walking warm start:** each solve walks from the last solved pose (or rest) to the target in steps of at most 0.1 rad or 0.01 m. `warm_start=` overrides this | A bare warm start is not enough. A cold solve straight to crank 180° flipped the four-bar to its other branch (rocker 121° instead of 39°). Articraft walks from zero every time. We walk from the last pose, so the solver keeps whichever branch the mechanism is on, and small moves take one step. |
| Two error families: `ValidationError` (the model is wrong) and `PoseError` / `LoopClosureError` (the requested values are wrong) | Later, the compiler can map each family to its own feedback for the agent. |
| build123d `Location` is converted through its transformation matrix (`trsf.Value`), not its Euler angles | No Euler convention has to be guessed. Degrees exist only inside build123d. |
| Mesh primitives come from `trimesh.creation`. There is no in-house mesh library | It is about 140 lines instead of about 6k. Booleans, lofts and sweeps can come from build123d for now. |
| Names must be identifiers (`[A-Za-z_][A-Za-z0-9_]*`) | They will become URDF and USD names in step 2. |

### Open ideas

- **Attach at build123d features.** *Done in 1b.* `body.at()` cannot yet take a face, edge
  or axis the way Articraft can. That would save the manual frame arithmetic in the four-bar
  example.
- **A `frame_in(attachment, other_body)` helper.** *Done in 1b.* It would express a rest-pose
  frame on another body, so closures could be authored without the law-of-cosines setup in
  `four_bar.rest_angles()`.
- **Validate that the rest pose closes every loop.** *Done in 1b.* Right now an unclosed rest
  pose only shows up on the first solve.
- **Better solver residuals.** *Still open (LATER).* The residual mixes meters and radians with
  equal weight. Scale rotation rows by a body length, and use an analytic Jacobian instead of
  finite differences, if speed starts to matter.
- **Walking has limits.** It assumes the path between two poses stays reachable. A midpoint
  that cannot close falls back to one direct solve, which can still switch branch. A
  mechanism passing through a singular pose (a change-point four-bar) is not handled.
- **Mimic chains and limits.** *Range check done in 1b.* Mimic joints are always tree joints.
  Validate their limits against the source's range ahead of time, instead of only at solve
  time.
- **Missing structure checks.** *Empty bodies done in 1b.* Bodies with no shapes are allowed. A
  joint authored child-to-parent is not flipped automatically; it fails validation with a
  "two roots" or "two parents" error.
- **Next step candidates.** Export URDF and GLB from `Body.trimesh()` and `Pose` (step 2), and
  a `world_meshes(pose)` helper for collision checks (step 4).

---

## Step 1b: Step 1 review changes

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `sdk/mesh_geometry.py` | 326 | `MeshGeometry` (Articraft's mutable mesh: translate, rotate, scale, merge, copy), ear-clip `triangulate_polygon`, `skin` (rings into a closed solid), exact-solid tracking. `Mesh` is now an alias. |
| `sdk/primitives.py` | 115 | `BoxGeometry`, `CylinderGeometry`, `ConeGeometry`, `CapsuleGeometry`, `TorusGeometry`, `RoundedBoxGeometry` |
| `sdk/solids.py` | 389 | `ExtrudeGeometry`, `ExtrudeWithHolesGeometry`, `LatheGeometry` (plus `from_shell_profiles`), `LoftGeometry`, `LoftSection`, `SectionLoftSpec`, `section_loft` |
| `sdk/sweeps.py` | 295 | `PipeGeometry`, `SweepGeometry`, `wire_from_points`, `tube_from_spline_points`, `sweep_profile_along_spline` (parallel-transport frames, mitered corners) |
| `sdk/profiles.py` | 103 | `rounded_rect_profile`, `superellipse_profile`, `sample_catmull_rom_spline_2d` / `3d` |
| `sdk/booleans.py`, `sdk/occ.py` | 85 + 72 | `boolean_union` / `difference` / `intersection` on OpenCASCADE, and fast tessellation |
| `sdk/mesh.py` | 34 | Public operations module, like Articraft's `articraft.sdk.mesh` |
| `sdk/frames.py` | +54 | `body.at(face / edge / axis / plane / vertex / shape / mesh)` |
| `sdk/assembly.py` | +101 | `frame_in`, the rest-pose loop check, the empty-body check, mimic range checks |
| `examples/` | 5 new | `toaster`, `door_hinge`, `desk_lamp`, `computer_mouse`, `folding_chair`; `four_bar` rewritten with `frame_in` |
| `docs/SHAPE_HELPERS.md`, `scripts/` | | Ranked table of all 57 Articraft mesh helpers, and the two scripts that generate it |
| `tests/` | 122 tests (was 62) | Added `test_shape_helpers.py`, `test_attachments.py`, `test_example_objects.py`; validation tests extended |

**Verified:**
- 122 tests pass on Windows in about 5.5 s.
- Every ported helper produces a closed mesh with consistent winding and positive volume,
  inside the expected bounding box. Four volumes match closed-form values to 1e-9: a plate
  with a hole, a bored box, a lofted frustum, and a lathe.
- All 8 examples build, validate and pose, and every shape is watertight.

### The changes

1. **Limits: Articraft's rule is restored (a reversal).** Every limit range must contain 0,
   and 0 is the authored rest pose. `Joint.rest` and the clamping are gone; joints you do not
   drive sit at 0.
   - *Why we reversed it.* With clamping, the pose the geometry was authored in could be one
     the mechanism can never reach. Requiring 0 inside the range means the authored pose is
     always a real state, so later rest-pose checks, collision checks and the viewer can all
     start from it. The step 1 argument (a lid that may only open from 20°) is weaker than it
     looked: author that lid at 20° instead.
   - *No example needed changing*: the laptop is authored closed with limits (0, 135°).
   - *Mimic offsets.* With every source at 0, a mimic sits at its offset, not at 0. The range
     check (item 5) guarantees that value is inside the mimic's limits.
2. **Shape helpers.** We took the inventory and usage counts of all 57 public `_mesh`
   helpers, rated each one, and ported 26 names in 17 families: 1,419 replica lines against
   5,077. The full ranked table is in `docs/SHAPE_HELPERS.md`; in brief:
   - **Ported:**
     - primitives: `BoxGeometry`, `CylinderGeometry`, `RoundedBoxGeometry`,
       `CapsuleGeometry`, `ConeGeometry`, `TorusGeometry`;
     - profile solids: `ExtrudeGeometry`, `ExtrudeWithHolesGeometry`, `LatheGeometry`;
     - lofts: `LoftGeometry`, `section_loft` with `LoftSection` and `SectionLoftSpec`;
     - sweeps: `PipeGeometry`, `SweepGeometry`, `wire_from_points`,
       `tube_from_spline_points`, `sweep_profile_along_spline`;
     - booleans: `boolean_union`, `boolean_difference`, `boolean_intersection`;
     - profiles and splines: `rounded_rect_profile`, `superellipse_profile`, the Catmull-Rom
       samplers;
     - the `MeshGeometry` base.
   - **Skipped (27):**
     - `weld` and `smooth_difference`: they need libigl and manifold3d, and Articraft's own
       prompt warns against `weld`;
     - `SweepSection` (variable sections);
     - shell partitioning, mesh refinement and smoothing (trimesh already has them);
     - Bezier and arc samplers, `WirePath`, side-profile lofts;
     - `Sphere`, `Dome` and `ArcPipe`, which are already covered.
   - **Later (4):** the mesh-health report moves to the checks step (step 4).
   - **Names and arguments match Articraft.** An option whose behavior is not ported yet
     (`cap_style="round"`, `sections=`, a `path=` for `section_loft`, `spline="bezier"`) raises
     `NotImplementedError` naming that option. Nothing is silently ignored.
3. **`body.at(feature)`.** A build123d `Face`, `Edge`, `Axis`, `Plane`, `Location`, `Vertex`,
   any shape, or a mesh can be passed. The feature's natural axis becomes the frame's local z:
   - a flat face gives its center and outward normal;
   - a round face (cylinder, cone) gives its axis of symmetry;
   - a straight edge gives its midpoint and tangent;
   - a circular edge gives its center and normal.

   `door_hinge` takes its pin axis straight from a knuckle's cylindrical face.
4. **`frame_in(attachment, other_body)`.** It returns the same physical frame, expressed on
   another body, in the rest pose. `four_bar` no longer solves the triangle by hand. It
   builds the open chain, asks `frame_in(coupler_tip, rocker)` where the pin is, and draws the
   rocker to that point (as a `wire_from_points` rod). The rocker comes out 70.01 mm long,
   and the motion is unchanged.
5. **Validation**, now all inside `validate()`:
   - **The rest pose must close every loop.** The error gives the gap and the fix, for
     example: *"loop closure 'pin' is open in the rest pose (error 0.005; its two frames are
     5 mm apart). With every joint at 0, coupler.at(xyz=(0.105, 0.0, 0.0), …) and
     rocker.at(xyz=(0.1, 0.0, 0.0), …) must coincide. Build one end with
     assembly.frame_in(...) so the authored pose is closed"*.
   - **Bodies with no shapes are an error.**
   - **Mimic limits are checked against the source's whole range.** Negative multipliers and
     chains are handled, and a limited mimic of an unlimited source is rejected.

**Left out on purpose: ball (and other multi-axis) joints.** Articraft's joints are USD D6
joints: any subset of the 6 axes can be free, so 3 rotations make a ball joint and 2 make a
universal joint. The replica keeps three single-axis types (fixed, revolute, prismatic),
which cover every example and map one-to-one to URDF (step 2). A ball joint can be added
later as its own type. Until then it can be modeled as 3 revolutes through 2 small
intermediate bodies; the empty-body rule means those bodies need a token shape.

### How Articraft does it (study §1)

- **Zero inside the limits.** `JointDOF.__post_init__` rejects a range that does not
  contain 0. The reference state is "every DOF at 0".
- **Mesh library.** `sdk/_mesh/*` holds 57 public names. Booleans, lathe and extrude caps,
  and loft caps rely on manifold3d; `weld` and mesh health rely on libigl.
- **Features.** `frames.py::_as_joint_frame`, `_face_location` and `_edge_location` convert
  features to frames. The replica ports the same rules.
- **`RigidBodyAssembly.frame_in`** resolves the *whole* assembly, so every body must already
  be valid. The replica's version only walks the joints between the two bodies, so it also
  works half-way through authoring (the rocker has no shape yet when it is called).
- **Closed rest pose.** Articraft enforces it implicitly: `resolve()` runs `validate_state`
  on the reference state, which rejects a closure joint whose locked axes are off. The
  replica checks it explicitly and names the gap.
- **Empty bodies.** `RigidBody.validate` rejects a body without shapes. The replica now does
  the same.

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| Booleans run on OpenCASCADE (inside build123d), not manifold3d | manifold3d is not an approved dependency. Meshes become sewn polyhedral solids, which is exact but slow (about 1 ms per triangle). |
| A `MeshGeometry` remembers its exact build123d solid (`attach_brep` / `brep()`) while its vertices are unchanged. A hash of the vertex and face arrays detects edits, including in-place ones, and rigid transforms move the solid along | Without this, cutting two slots into a 14.7k-triangle rounded box took **19 s**. With it, the toaster's two slots take **0.11 s** and stay exact. Results of booleans keep their solid, so chained cuts stay fast. |
| Our own OpenCASCADE triangulation reader (`occ.tessellate`) | `Shape.tessellate` spent 12.6 s on a faceted result with 14k faces; reading OpenCASCADE's triangulations directly takes 1.6 s. |
| Our own ear-clipping triangulator for caps | trimesh's triangulation needs `mapbox_earcut` / `triangle` / `shapely`, which are not installed and not approved. It is 37 lines, and caps reuse the tube's own ring vertices, so every result is watertight. |
| Sweeps: miter-plane rings and parallel-transport frames, with the closed-loop twist spread evenly | The tube keeps its full thickness through sharp corners, and closed paths meet without a seam twist. Simpler than Articraft's 1,200-line `sweeps.py`. |
| `section_loft` resamples sections with different point counts | Articraft requires equal counts or a repair mode; resampling is simpler and handles the usual case. |

### Open ideas

- **LATER: solver residual scaling and an analytic Jacobian.** The residual weights meters
  and radians equally, and the Jacobian is finite-difference.
- **LATER: singular (change-point) linkages.** When a four-bar passes through a singular pose,
  the walking warm start can switch branches.
- **manifold3d** *(done in 1c)* would make booleans on mesh-native inputs (lathes, lofts,
  pipes) about 100× faster: milliseconds instead of about 1 ms per triangle. It needs your
  approval as a dependency.
- **`body.at(feature)` reads features in body coordinates** *(fixed in 1c)*. That is right
  when the shape was added without `xyz`/`rpy`. A `shape=` argument could compose the shape's
  placement.
- **`RoundedBoxGeometry` is heavy.** With Articraft's defaults (0.5 mm, 0.08 rad) a small box
  is 14.7k triangles. That is fine to look at, but heavy for collisions and export; consider a
  coarser angular tolerance once step 4 measures the cost.
- **Not ported yet:** `SweepSection` (variable profiles along a sweep), path-guided
  `section_loft`, round loft caps, Bezier tubes, and `weld`-style smooth blends. Each raises
  `NotImplementedError` today. Port them when a real example needs one.

---

## Step 1c: Finishing step 1

### manifold3d for mesh booleans

manifold3d 3.5.5 is approved and added. `sdk/booleans.py` now picks an engine per call:

- **OpenCASCADE**, when *both* inputs are exact: build123d shapes, or meshes that still
  carry their exact solid (`RoundedBoxGeometry`, `ExtrudeWithHolesGeometry`, an exact
  boolean result). The result stays exact.
- **manifold3d** otherwise, as in Articraft. It runs in float64 (`Mesh64`), and a build123d
  input is tessellated first. The result is a plain mesh.

`boolean(op, a, b, engine="occ" | "manifold")` forces one engine, for benchmarks.

Timings are the best of 3, from `uv run python scripts/bench_booleans.py occ` and then `auto`:

| Case | Triangles | Before (OpenCASCADE only) | After (auto) | Engine after |
| --- | --- | --- | --- | --- |
| lathe - bore (mesh, mesh) | 512 | 407.8 ms | **3.7 ms** | manifold3d |
| tube frame + box (mesh, mesh) | 648 | 354.3 ms | **3.0 ms** | manifold3d |
| smooth loft - box (mesh, mesh) | 960 | 498.6 ms | **3.5 ms** | manifold3d |
| rounded box - box (exact, mesh) | 14,696 | 83.5 ms | **30.2 ms** | manifold3d (no longer exact) |
| rounded box - plate (exact, exact) | 14,924 | 174.5 ms | 178.3 ms | OpenCASCADE (stays exact) |

Mesh-only booleans are about 110× faster. A mixed exact-and-mesh boolean is faster but gives
up exactness. Forcing manifold3d on two exact inputs would take 32 ms instead of 178 ms; we
keep OpenCASCADE there because its result stays exact and chains.

### `body.at(feature)` on shapes added with `xyz`/`rpy`

Step 1b read features in body coordinates, which was wrong for a shape added with a
placement. Now `Body.at` finds which of the body's shapes contains the feature and applies
that shape's placement:

- **build123d features:** a face, edge or vertex is matched to its shape by walking the
  shape's sub-shapes with `TopExp_Explorer` and comparing with `IsSame` (same underlying
  shape and location).
- **Meshes:** matched by identity.
- **`shape="name"`:** reads any input in a named shape's coordinates. This is needed for an
  `Axis`, `Plane` or `Location`, which belong to no shape, and when one build123d object was
  added twice at different placements (that case raises an error asking for `shape=`).
- **Unrelated features:** a feature from a shape that is not on the body is still read in
  body coordinates.

New tests use a box added at `xyz=(1, 0, 0)`, `rpy=(0, 0, π/2)`: its top face gives
(1, 0, 0.025) facing up; its +x face gives (1, 0.1, 0) facing +y; its top back edge gives
(0.95, 0, 0.025) along y.

### Solver: an already-closed loop is not touched

`kinematics.solve` now skips `least_squares` when the loop already closes at the starting
guess (error at most 1e-3 × tolerance). The export tests found the reason. In the gantry,
both carriages are mimics of `x_drive`, so the bridge loop closes for *any* `x_drive`.
Finite-difference noise used to nudge the undriven `x_drive` to 1e-10 instead of leaving
it at 0.

---

## Step 2: Export

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `sdk/export.py` | 427 | `export(assembly, out_dir, version=None, *, source=None, density=1000.0, densities=None, collision_max_faces=5000) -> ExportResult`, and `load_manifest(path)` |
| `tests/urdf_fk.py` | 113 | A URDF reader and forward kinematics built on the standard library and numpy only (no URDF parser dependency) |
| `tests/test_export.py` | 259 | 39 tests over all 8 examples |

Each call writes one new version folder:

```text
result/v0003/
  model.urdf                  links, tree joints, <mimic>, <inertial>
  model.glb                   one node per body, nested like the joint tree
  model.json                  manifest
  meshes/<body>.obj           visual mesh: every shape merged, in body coordinates
  meshes/<body>_collision.obj the visual mesh, or simplified if it has > 5000 triangles
  <source file>               snapshot of the source (main.py from step 3 on)
```

**How it works.**

1. **Refuse invalid input.** `export` calls `assembly.validate()` first, so an invalid
   assembly raises `ValidationError` before anything is written.
2. **Write atomically.** The version is built in a `.vNNNN-<random>.tmp` folder and renamed
   into place at the end. A version is either complete or absent. An existing version is
   never overwritten (an explicit `version=` that exists raises `FileExistsError`), and a
   failed export leaves no trace.
3. **Meshes.**
   - The visual OBJ is every shape merged, in body coordinates.
   - The collision OBJ is a copy of it, unless the body has more than 5,000 triangles. Then
     each large closed shape is reduced with manifold3d's `simplify`, whose tolerance doubles
     until the shape fits its share of the budget. An open shape gets its convex hull
     instead.
   - The toaster shell goes from 14.7k triangles to 4.9k or fewer, with bounds within 2 mm.
4. **Mass.**
   - The mass is volume × density: 1,000 kg/m³ by default, with `densities={body: rho}` to
     override it per body.
   - The center of mass and inertia come from trimesh's mass properties, taken about the
     center of mass in body axes. An open shape uses its convex hull.
   - A 0.2 × 0.1 × 0.05 m box at 2,000 kg/m³ gives exactly 2 kg and the textbook inertia
     tensor.
5. **URDF frames.** A URDF child link's frame sits on its joint frame, but our bodies keep
   their own frames (joint = `F_p` + `F_c`). So each link frame is the `F_c` of the joint
   that places the body (identity for the root):
   - visual, collision and inertial origins are `inverse(F_c)`;
   - a joint's origin is `inverse(F_c(parent)) @ F_p`.

   Joint types map as follows: fixed → fixed; a revolute with limits → revolute; a revolute
   without limits → continuous; prismatic → prismatic (an unlimited slide gets ±1000 m,
   noted in the manifest). URDF's required `effort` and `velocity` are placeholders (100 and
   2). Mimic joints get `<mimic joint multiplier offset>`.
6. **Loops.** URDF is a tree. Closure joints are left out of the URDF and listed in the
   manifest's `closures`, with both frames, so a loop-aware loader can add them back.
7. **GLB.**
   - There is one node per body, and each node's parent is its parent body in the joint
     tree. The node transforms are the rest pose.
   - To pose a body, set its node matrix to `F_p @ motion(q) @ inverse(F_c)`.
   - glTF is Y-up, so the bodies hang under a single root node, `z_up_to_y_up`, which turns
     our Z-up frame into Y-up.
8. **Manifest (`model.json`).**
   - Model: schema version, version number, units, up axis, file list.
   - Source: file name and SHA-256 of the snapshot.
   - Bodies: meshes, collision method, triangle counts, link frame, density, mass, center
     of mass, inertia, and shapes with their type and placement.
   - Joints: frames, axis, limits, mimic, a `tree` or `closure` role, and the URDF type.
   - Articulation: root, tree order and closures, plus separate `closures` and `mimics`
     lists.
   - Rest pose: joint values and body transforms.
   - Conventions used by the GLB and URDF.

**Verified:** 164 tests pass in about 6.6 s; 39 of them are new export tests. They check,
for all 8 examples:
- the URDF is valid XML, and every link, joint and mesh path exists;
- mimic tags are present and correct, and the closure joint is absent from the URDF;
- the test URDF reader gives the *same body poses as `assembly.solve()`* to 1e-9 at the
  rest pose and 1-2 more poses. This includes the four-bar, which gets the loop values the
  solver found, and the gantry, whose mimics the URDF reader computes itself;
- the GLB loads in trimesh with one node per body plus the root, a node tree equal to the
  joint tree, and bounds equal to the rest-pose meshes turned to Y-up (to 1e-5);
- the manifest round-trips through JSON, its closures and mimics match the assembly, its
  source hash matches the snapshot, and the rest pose rebuilt *from the manifest's frames
  alone* matches `solve()` to 1e-12;
- versions go 0, 1, 2, and an explicit version 7 works, after which the next automatic
  version is 8. An existing version raises, and v0000 is byte-for-byte unchanged afterwards;
- an invalid assembly or a missing source file writes nothing.

Exporting any example takes 0.02-0.15 s.

### How Articraft does it (study §2)

- **USDZ only.** `sdk/export.py::export_assembly` writes a USD stage with physics schemas:
  Revolute, Prismatic and Fixed joints, or a D6 joint with a per-axis `LimitAPI`. It packs
  the stage into a USDZ, validates it with `UsdValidation`, reads it back and audits it. A
  `model.json` (schema v2) sits next to it.
- **No URDF or GLB.** URDF exists only in the teammate's kinematic exporter, and MJCF only
  inside `simulate.py`.
- **Versions:** `result/usdz/NNNN.usdz` takes the next free number, but `model.json` is
  overwritten each time. A compile whose checks fail can still leave a USDZ behind.
- **Loops:** closure joints become USD joints with `excludeFromArticulation`, so USD can
  hold the loop. URDF cannot.
- **Mass:** resolved per shape from material densities (`mass.py`). There are no default
  densities without a material.

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| URDF + GLB + OBJ instead of USDZ | Every robotics tool reads URDF (ROS, PyBullet, MuJoCo import, Isaac); every viewer reads GLB. Both are easy to write and check, with no `usd-core` dependency. USDZ can come later if needed. |
| A complete version folder, built in a temporary folder then renamed, never overwritten | Articraft numbers the USDZ but overwrites `model.json`, so old versions lose their manifest. Ours keeps the URDF, GLB, meshes, manifest and source snapshot together for every version. |
| Closures listed in the manifest | URDF cannot hold a loop. The manifest keeps both closure frames, so nothing is lost. |
| Link frame = child attachment frame, meshes placed at `inverse(F_c)` | Our joints have two attachment frames and URDF has one origin. This mapping is exact, and the URDF forward-kinematics tests prove it. |
| GLB nodes nested like the joint tree, under a Z-up to Y-up root | A viewer poses a body by setting one node matrix. glTF's Y-up rule is honoured without changing our Z-up SDK. |
| Default density and per-body overrides | Materials are not ported yet; every URDF link needs an inertial. |
| Collision meshes reduced with manifold3d's `simplify` | Big meshes (a rounded shell is 14.7k triangles) are costly for physics. `simplify` keeps a closed mesh and moves surfaces less than its tolerance. |

### Open ideas

- **Ball and universal joints have no URDF type.** Export them as chains of revolutes
  through massless links, if they are added.
- **Materials and colors:** export a colour per shape in GLB and URDF `<material>`, and
  densities from materials, once materials are ported.
- **Overlapping shapes** in one body count their shared volume twice in the mass. A union
  (manifold3d) before computing mass properties would fix it.
- **Collision meshes** could be convex decompositions (V-HACD or CoACD), which physics
  engines handle best. That would need a new dependency.
- **USDZ export** as an extra output if a USD tool needs it; Articraft's exporter is the
  reference.
- **MJCF export** so MuJoCo can simulate loops natively, using equality constraints for the
  closures.
- **Loader:** read a version folder back into a kinematic model, for the viewer (step 5)
  and for diffs between versions.

---

## Step 3: Compiler

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `compiler/worker.py` | 194 | `python -m compiler.worker <run_dir> --timeout T`: runs the phases in a fresh interpreter and prints one JSON payload |
| `compiler/runner.py` | 335 | `compile(run_dir, timeout=300, *, force=False) -> CompileResult`: subprocess, process-tree kill, scrubbed environment, digest cache, log, repeat history |
| `compiler/feedback.py` | 348 | `CompileSignal`, `SignalKind`, classification of exceptions, `render_signals` |
| `tests/test_compiler.py` | 327 | 24 end-to-end tests with real subprocesses |
| `scripts/measure_cold_start.py` | 68 | Cold-start timings |

**The run folder:**

```text
run_dir/
  workspace/main.py         written by the agent: defines object_model, may define run_tests()
  result/v0000/ ...         exports, only from successful compiles
  result/compile_log.jsonl  one line per compile: stdout, stderr, traceback, phases, hang dump
  result/.compile-cache.json, .compile-history.json, .compile-progress.json (while running)
```

**The worker's phases:**
1. load `main.py` with `runpy` (the workspace is on `sys.path`, so helper modules import);
2. get `object_model`;
3. `validate()`;
4. `run_tests()`, if defined. It passes if it returns None or True. It fails if it raises
   `AssertionError`, returns False, or returns a non-empty list of messages;
5. baseline checks (a stub returning `[]`; step 4 fills it in);
6. `export(model, result, source=main.py)`.

Everything `main.py` prints is captured into the payload, and the payload is the last line
of stdout. Every phase change rewrites `result/.compile-progress.json`. Before importing
anything heavy, the worker arms `faulthandler.dump_traceback_later(timeout - margin)`, so
just before the deadline every thread's stack is written to `.compile-hang.txt`.

**The runner:**
- **Launch.** It launches `python -m compiler.worker` with `cwd = workspace`. The
  environment has every `*_API_KEY` / `*_TOKEN` variable removed, `PYTHONPATH` set to the
  replica, and `PYTHONDONTWRITEBYTECODE=1`.
- **Process group.** On Windows the worker gets `CREATE_NEW_PROCESS_GROUP`; on POSIX,
  `start_new_session=True`.
- **Timeout.** It kills the whole tree: `taskkill /PID <pid> /T /F` on Windows, `killpg(SIGKILL)`
  on POSIX. It then reads the progress file and hang dump. The timeout signal names the
  phase, the seconds spent in it, and the `main.py` line that was running.
- **No result.** If the worker printed no payload, the result is a `crash` signal with the
  exit code.
- **No leftover versions.** Any version folder (or temp export folder) created during a
  failed compile is deleted. That includes one exported by `main.py` itself before it
  failed.
- **Cache.** It is keyed by the workspace digest (SHA-256 over relative paths and bytes,
  caches ignored) *and* a digest of the replica's own `sdk/` and `compiler/` sources. A cached
  result is re-rendered with `cached=true`. A cached timeout is retried when the time limit
  changes, and a cached success is retried when its export folder is gone.
- **Repeat history.** Each real (non-cached) compile appends the hash of its failure set
  (kind, message, location). A streak of 2 adds "matches the previous compile"; a streak of
  3 or more adds "same failure N times in a row" plus a response rule to inspect before
  editing again.

**What the model sees:** only `CompileResult.signals_text`, for example:

```text
<compile_signals>
<summary>
status=error failures=1 warnings=0
Primary issue: [runtime_error] at main.py:10: ZeroDivisionError: float division by zero
</summary>

<failures>
Failures (blocking):
- FAILURE [runtime_error] ZeroDivisionError: float division by zero
  at main.py:10: lid_depth = 0.22 / HINGE_SPAN
  fix: Fix the exception at this line; no model checks run until main.py runs cleanly.
</failures>

<response_rules>
Suggested next steps:
- Fix this error first: nothing else is checked until main.py runs and defines object_model.
</response_rules>
</compile_signals>
```

**Signal kinds:** `missing_main`, `syntax_error`, `import_error`, `runtime_error`,
`object_model`, `validation`, `pose` (including `LoopClosureError`), `test_failure`,
`export_error`, `timeout`, `crash`. The location is chosen in this order:

- **Syntax errors:** the line Python reports.
- **Exceptions:** the deepest traceback frame *inside the workspace*. Only absolute paths
  count, so a compiled module like scipy's `_qhull.pyx`, which reports a relative name, is
  never taken for a workspace file.
- **Errors raised after `main.py` ran** (in `validate()` or the export): the first `main.py`
  line quoting an entity named in the message, such as `a.body("hollow")` for "bodies with
  no shapes: ['hollow']".
- **Timeouts:** the innermost workspace frame in the `faulthandler` dump.

Messages are cut to one line (at most 400 characters); the full text is in the log.
Validation hints are specialised for open rest loops, limits without 0, empty bodies, a
missing root, and duplicate names.

**Verified:** 187 tests pass in about 43 s; 24 of them are compiler tests (about 36 s, as
each spawns real processes). They show that:
- the laptop compiles to v0000, with `main.py` snapshotted next to it;
- each of 14 failure cases gives the right kind, the right `main.py` line, a hint, no
  traceback in the block, and no version folder;
- an endless loop that also started a grandchild process times out at 4 s with
  "timed out after 4s while loading main.py (3.2s in that phase)" at `main.py:9`, and the
  grandchild is dead (checked with `OpenProcess`/`GetExitCodeProcess`);
- fake `*_API_KEY` / `*_TOKEN` variables are invisible in `main.py`, while an ordinary
  variable is visible and `cwd` is the workspace;
- the cache works: an unchanged workspace is cached, one edit recompiles, `force=True`
  recompiles, and a changed toolchain drops the cache;
- 3 identical failures (with different digests) give the inspect hint, and a different
  failure resets the streak.

**Cold start** (median of 3 forced compiles on this machine, `scripts/measure_cold_start.py`):

| Model | Wall | Interpreter start + imports | main.py | Export |
| --- | --- | --- | --- | --- |
| one box (sdk only) | 1.11 s | 1.10 s | 0.00 s | 0.01 s |
| laptop | 1.12 s | 1.10 s | 0.00 s | 0.02 s |
| door hinge (imports build123d) | 3.84 s | 1.50 s | 2.19 s | 0.15 s |
| toaster (build123d + manifold3d) | 3.86 s | 1.49 s | 2.23 s | 0.13 s |

A bare interpreter starts in 0.04 s. Importing `sdk` (numpy, scipy, trimesh) costs about
1.1-1.4 s, and importing build123d (OpenCASCADE) about 2.2 s more. The model itself and
the export take milliseconds. **Anything using build123d is above 2 s**, so the warm-worker
idea is logged below; it is not built.

### How Articraft does it (study §3)

- **Worker.** `agent/workspace/local.py::_run_isolated_process` starts
  `python -m articraft.compiler.worker --raw <run_dir>` with `start_new_session=True` and a
  900 s timeout. On timeout it sends `SIGTERM` to the process group, then `SIGKILL`.
  `compiler/worker.py` runs `main.py` with `runpy`, then `run_tests()`, baseline checks and
  export, and prints one JSON payload.
- **Progress.** `_CompileTracker` writes `result/.compile-progress.json`, so a timeout names
  the phase.
- **Feedback.** `compiler/feedback.py` turns failures into typed `CompileSignal`s, renders
  `<compile_signals>` with per-kind response rules, and uses a SHA-1 failure signature for
  "matches the previous attempt" and "failure N in a row".
- **Cache.** The compile tool returns the cached result when `workspace_digest` is unchanged.
- **Environment hygiene only:** `child_environment()` strips `*_API_KEY`.
- **No Windows support:** `os.killpg`, `start_new_session` and `bash -lc` are POSIX-only.
- **Version leftovers:** the export runs before the test report is raised, so a compile
  whose checks fail can still leave a numbered USDZ.

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| Cross-platform process-tree kill (`CREATE_NEW_PROCESS_GROUP` + `taskkill /T /F`, or `killpg`) | Must run on Windows. Grandchildren must die, or they keep running and hold the output pipes open. |
| `faulthandler` stack dump just before the deadline | Articraft names the phase; we also name the `main.py` line that was running, which tells the agent *what* hung. |
| Export only when everything passed, plus a sweep that deletes any version created during a failed compile | No half-good versions, even if `main.py` exports by itself or the process dies right after an export. |
| Locations restricted to the workspace, with name search as a fallback | The agent should never be pointed at SDK internals. Errors raised in `validate()` still get a `main.py` line. |
| One-line messages, full text in `compile_log.jsonl` | Keeps the block compact. A 6-line qhull dump is noise for the model. |
| Cache keyed by workspace *and* toolchain digest | A manual check found a stale cached failure after the SDK was fixed. In an agent loop that would mislead the model. |
| `*_TOKEN` stripped as well as `*_API_KEY` | GitHub and Hugging Face style secrets are tokens. |
| `run_tests()` may return a list of failure messages | It allows several failures per compile. This is a stop-gap until step 4's `TestContext`. |
| The export fails clearly on an open or flat shape with no volume | Before, a qhull error from the mass calculation reached the agent. |

### Known risk: there is no real sandbox

As in Articraft, `main.py` runs **with the user's own permissions**. It can read and write
any file the user can, open network connections, start programs, and use all CPU and
memory up to the timeout. The only protections are:
- stripping `*_API_KEY` / `*_TOKEN` from the environment;
- a fresh process per compile;
- killing the process tree at the timeout.

Other secrets (files like `~/.aws/credentials` or `.env`, other environment variables) are
still reachable. Run the replica only on a machine and account you would let the model use.
Options for later: a Docker container with the workspace mounted and no network; a
low-privilege Windows user or AppContainer; a Windows Job Object with memory and CPU
limits; or at least a memory cap.

### Open ideas

- **Warm worker** (cold start is above 2 s with build123d). Keep one worker process per run,
  with `sdk`, build123d and manifold3d already imported. For each compile, fork it (POSIX), or
  on Windows keep a pool of pre-imported spare workers, used once each so every `main.py` still
  gets a clean interpreter. That would save about 1.1-3.7 s per compile. Articraft's test
  harness has a similar `WarmEnvironment`.
- **Faster imports:** import scipy (`least_squares`, qhull) lazily inside the functions that
  need it; that would cut some of the 1.1 s sdk import.
- **Windows Job Objects** instead of `taskkill /T`. They also catch a grandchild whose parent
  already exited (`taskkill /T` follows parent IDs), and can cap memory.
- **Faulthandler margin:** for very short timeouts (under about 2 s) the dump can come too
  close to the kill; the margin is `min(1 s, 20% of the timeout)`.
- **Warnings channel:** step 4's baseline checks will produce non-blocking `warning`
  signals; the renderer already has a section for them.

---

## Step 4: Checks

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `sdk/testing.py` | 908 | `CheckConfig`, `TestContext` (the `run_tests(ctx)` API), `run_checks` (baseline checks), and the collision scene on python-fcl and manifold3d |
| `compiler/worker.py`, `compiler/feedback.py` | (changed) | `run_tests(ctx)`, then baseline checks, then export only if nothing blocks; 9 new signal kinds; allowances shown with their reasons |
| `tests/check_fixtures.py`, `tests/test_checks.py` | 145 + 252 | Broken fixtures with known defects; 28 tests |
| `scripts/time_checks.py` | 51 | The timing table below |

New dependency: **python-fcl 0.7.0.11** (approved), used through trimesh's `CollisionManager`.

**The authored API.** `run_tests(ctx)` gets a `TestContext`. The old zero-argument
`run_tests()` still works.
- **Expectations:** `expect_contact`, `expect_gap(min, max)`, `expect_distance` (between two
  body-local points), `expect_no_collision_at_poses`, plus `check` and `fail`. They record
  findings instead of raising, and a failed expectation blocks the compile.
- **Sampling:** `sample_joint` and `track_point`.
- **Allowances:** `allow_overlap(a, b, reason, shape_a=, shape_b=)` and
  `allow_isolated(body, reason)`. A reason is required (blank is a `ValidationError`), and it
  appears in the report and in the `<compile_signals>` block as
  `(allowed: <reason>)`. An allowance that nothing used becomes a warning.

**Baseline checks** (`run_checks`, which always runs). The tolerances are
`max(floor, fraction × size)`, where size is the diagonal of the rest-pose bounding box.

| Check | Rule | Threshold (`CheckConfig` defaults) | Blocks? |
| --- | --- | --- | --- |
| model validity | `validate()` (step 1) | - | yes |
| mesh health | non-manifold edges; inconsistent or inside-out winding | - | yes |
| mesh health | degenerate triangles; tiny debris; open boundary | area < 1e-10 × size²; piece < 2% of the shape | warning |
| overlap at rest | two bodies interpenetrate | depth > max(0.2 mm, 0.002 × size) | yes, unless allowed |
| floating parts | the contact graph from the root does not reach a body | contact = gap ≤ max(0.1 mm, 0.001 × size) | yes, unless allowed |
| swept collision | each drivable joint (revolute **and** prismatic) is swept over its range, at 15° or 5% of size steps, 5-25 samples | depth > max(0.2 mm, 0.002 × size); 6 bisection steps | yes, unless allowed |
| loop reach | the loop cannot close for some driver value inside its limits | solver error > 1e-6 | yes |
| joint separation | a hinge's parent-child gap grows over its range | growth > max(0.5 mm, 0.005 × size) | yes |
| islands | shapes of one body do not touch | contact tolerance | warning |
| dimensions | a shape is larger than 50 m, or more than 100× the median shape | - | warning |

**The sweep:**
- **What gets swept.** Every drivable tree joint. Mimic joints follow their source. Of the
  joints on a loop, only the first one authored is swept; the loop solver poses the rest
  with the walking warm start. An unlimited revolute is swept over ±180°; an unlimited
  prismatic gets a warning asking for limits.
- **Refinement.** Coarse samples first. Where the collision state changes between two
  samples, bisect to locate it. Where bodies that were apart at rest come within 2% of size,
  add quarter-step samples.
- **Reporting.** Each colliding pair reports:
  - its **first contact**: a bisection on any interpenetration at all, from the rest pose
    outward;
  - the range where it is deeper than the tolerance;
  - the deepest penetration and where it happens.

**Collision pipeline:**
1. An AABB pre-check (numpy) keeps only shape pairs from different bodies whose world boxes
   overlap, at least one of which moved.
2. trimesh `CollisionManager.in_collision_internal` (fcl BVH) confirms which pairs touch.
3. manifold3d intersects each touching pair. The thinnest side of the intersection's box
   is the penetration in mm, and its volume is reported too. Faces that only touch give an
   empty intersection, so contact is not an overlap.
4. Gaps come from `fcl.distance` between bodies.

**Every finding becomes a `CompileSignal`.** It names the bodies and shapes, the pose (joint
values), and the amount (mm of penetration, gap, or gap growth). Its location is the
`main.py` line of the joint (for sweep, loop and separation findings) or of the first named
body. Blocking findings are failures, allowed findings are notes, and the rest are warnings.

**The drawer fixture, as the model sees it:**

```text
<compile_signals>
<summary>
status=error failures=1 warnings=0
Primary issue: [swept_collision] at main.py:15: 'cabinet'/lip hits 'drawer'/back while 'slide' moves: first contact at 260.0 mm, deeper than 1.34 mm at 261.5 mm to 280.0 mm (deepest 20.00 mm at 280.0 mm)
</summary>

<failures>
Failures (blocking):
- FAILURE [swept_collision] 'cabinet'/lip hits 'drawer'/back while 'slide' moves: first contact at 260.0 mm, deeper than 1.34 mm at 261.5 mm to 280.0 mm (deepest 20.00 mm at 280.0 mm)
  at main.py:15: cabinet.joint("slide", "prismatic", body.at((0, 0, 0)), drawer.at(), axis=(0, 1, 0),
  fix: Change the joint limits, axis or geometry so the parts clear each other over the whole range.
</failures>

<response_rules>
Suggested next steps:
- Check failures are design evidence: fix the geometry, joint origin, axis or limits. Use allow_* only for real, intended contact, with a concrete reason.
</response_rules>
</compile_signals>
```

**Timing** (median of 3, `scripts/time_checks.py`; building the example is not counted):

| Example | Triangles | Joints | Mesh, dims, islands | Rest checks | Sweeps | Total |
| --- | --- | --- | --- | --- | --- | --- |
| laptop | 48 | 1 | 0.003 s | 0.001 s | 0.002 s | 0.014 s |
| four_bar | 864 | 4 | 0.006 s | 0.003 s | 0.249 s | 0.275 s |
| gantry | 120 | 6 | 0.006 s | 0.002 s | 0.113 s | 0.135 s |
| toaster | 16,684 | 2 | 0.038 s | 0.003 s | 0.029 s | 0.273 s |
| door_hinge | 2,336 | 1 | 0.010 s | 0.002 s | 0.015 s | 0.102 s |
| desk_lamp | 2,460 | 2 | 0.007 s | 0.004 s | 0.099 s | 0.143 s |
| computer_mouse | 2,496 | 3 | 0.007 s | 0.002 s | 0.126 s | 0.170 s |
| folding_chair | 16,064 | 2 | 0.034 s | 0.001 s | 0.016 s | 0.227 s |
| **all 8** | | | | | | **1.34 s** (target < 2 s) |

The totals include building the fcl BVHs and manifold solids (about 0.2 s for the two
16k-triangle models). The four-bar's sweep is the slowest, because every sample is a loop
solve.

**Verified:** 217 tests pass in about 52 s. They show that:
- the **lid placed inside the base** is an overlap of 5.00 mm at rest (`base/deck`, `lid/panel`);
- the **floating knob** is reported 10.00 mm from the panel;
- the **door hits its frame** with first contact at 90.0°, and is clean when limited to 89°;
- the **drawer** (prismatic) first touches at 260.0 mm, the construction value, and is 20 mm deep at full extension;
- the **hinge on the wrong axis** pulls the lid off: the gap grows 128 mm;
- the **four-bar whose links cross** has its crank hit the rocker, with first contact near -12° of crank angle, posed by the loop solver, and no overlap at rest;
- a non-Grashof four-bar is a loop failure;
- `allow_overlap` with a reason passes and the reason is shown; a shape-level allowance does not cover other shapes;
- at **2 cm vs 2 m**, the same 0.5 mm gap is floating on the small part and contact on the large one, and the same 1 mm sink is an overlap on the small part and fine on the large one;
- **all 8 examples have no blocking findings**, and every allowance they declare is used;
- in compiler tests, the drawer `main.py` fails with the signal above and writes no version, and an allowed overlap compiles and shows `(allowed: …)`.

### The checks found real bugs, all fixed

**In the examples** (the swept and floating checks found every one):

| Example | Finding | Fix |
| --- | --- | --- |
| laptop | the screen sat 1 mm *below* the lid, inside the deck; the hinge barrel overlapped the lid by 4 mm | screen inlaid in the panel; barrel removed |
| four_bar | the rocker rod (r = 5 mm) overlapped the coupler by 3 mm | r = 2 mm, in the crank's layer |
| gantry | the tool head hit both carriages beyond ±314 mm | head travel ±300 mm; end frames make the base one piece |
| toaster | the lever floated 4 mm off the face | lever against the face |
| door_hinge | the leaves collided beyond 155° | limit 150° |
| desk_lamp | the shade hit the arm when tilted up past 8.6° | tilt range -0.6 to 0.1 rad; the tube in the socket is allowed, with a reason |
| computer_mouse | the wheel overlapped the shell and both buttons | a slimmer wheel; the shell slot is allowed |
| folding_chair | the rear frame floated (its pivot lay in the tube's filleted bend); the seat hit the back slat at -33° | taller rear frame, smaller fillets, frames side by side; fold limit -0.55 rad; slat on the legs |

**In the step 1 solver.** Sweeping a Grashof four-bar reported "cannot reach" at crank angles
it can reach. The cause: SciPy's `trf` sizes its first trust region from the norm of the
starting point, so a warm start like `[0, 4.4e-16]` got a 4e-16 trust region and stopped at
once (`ftol`). The fix in `kinematics._least_squares`:
- solve for the *offset* from the warm start, which always begins at exactly 0, so SciPy
  starts with a proper trust region;
- `x_scale=0.1` keeps that first step about 6° or 10 cm, so a rough guess does not jump to
  the other branch. The existing warm-start test caught that jump when only the offset fix
  was in place.

The loop-reach message now lists the unreachable intervals instead of first-to-last.

### How Articraft does it (study §4)

- **Authored checks:** `sdk/testing.py::TestContext` provides `expect_*` (distance, contact,
  gap, within, overlap, coaxial, ...), `sample_dof` and `track_point`, plus
  `allow_overlap(part_a, part_b, reason, shape_a, shape_b)`, `allow_isolated_part` and
  `allow_mesh_issues`.
- **Baseline checks:** `compiler/worker.py::_run_baseline_tests` runs them, using
  `_collision.py` (python-fcl BVH plus manifold intersection volume). Fixed absolute
  tolerances: contact 1e-6 m, overlap 5 mm and 5e-7 m³, hinge gap growth 3 mm.
- **Separation:** `fail_if_articulation_separates_child` samples revolute joints at 4 values.
  **Prismatic joints are exempt.** There is no swept collision check at all: collisions are
  checked only at rest.
- **Only three kinds block:** model validity, mesh health and missing mass. Overlap, floating
  parts and separation are diagnostics; the model is told to decide.

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| Overlap, floating parts, swept collision, separation and loop reach **block** unless allowed with a reason | Diagnostics the model may ignore end up as broken objects (a lid that sinks into its base, a door through its frame). An explicit allowance with a reason keeps intentional cases possible and visible. |
| Swept collision over every drivable joint, **prismatic included**, coarse then adaptive | Collisions that happen only in motion were invisible in Articraft. The drawer and the door are typical cases. |
| First contact found by bisection, separately from the "deeper than tolerance" range | The door's scaled tolerance (5.3 mm) is first exceeded at 97.7°, but the useful fact for a fix is first contact at 90.0°. |
| Loops are swept through their driver with the walking solver; an unreachable pose is a failure | Wrong link lengths or limits show up as a "cannot close" finding with the crank angles. |
| Scale-relative tolerances, `max(floor, fraction × size)`, all in one `CheckConfig` | Articraft's 5 mm overlap tolerance accepts half of a 2 cm part; for a 2 m part, 1e-6 m contact is absurdly strict. |
| Shape-level allowances, and unused allowances flagged | An allowance covers exactly what it says, and stale ones get cleaned up. |
| Penetration as the thinnest side of the manifold intersection, in mm | The model gets an amount it can act on ("20 mm at 280 mm"), not only a volume. |

### Open ideas

- **Faster loop sweeps:** step the solver along the sweep values directly (continuation)
  instead of repeating the full walk for each refinement point.
- **Sweeps of joint combinations:** joints are swept one at a time with the others at rest.
  Two joints that collide only together (a lid open *and* a drawer out) are missed; sample
  a few corner combinations of the limits.
- **Better depth:** the thinnest side of the intersection box overestimates the depth of a
  slanted wedge. A signed-distance query, or the smallest translation that separates the
  pair, would be exact.
- **Clearance requirements:** let `allow_*` and the sweeps take a minimum clearance (for
  example 2 mm between a drawer and its frame) instead of only "no penetration".
- **Mass check:** add missing or zero mass when a physics step comes (Articraft blocks on it
  with physics enabled).
- **Mesh repair hints:** say which source operation produced a bad mesh, as Articraft does
  with its mesh-health bounds.

### Step 4b: allowance audit and combined poses

**Allowance audit.** The 8 examples declared two allowances, and on inspection both hid
real defects. Both are now fixed in geometry, and **no example declares an allowance any
more**.

| Example | Body pair | Reason it gave | Verdict | Fix |
| --- | --- | --- | --- | --- |
| desk_lamp | `arm/tube` and `head/socket` | "the arm tube is pushed into the lamp socket" | defect: the socket belongs to the moving head, so it would rotate through the tube | a hinge barrel on the arm along the tilt axis; the head's stem top is tangent to it; the tube's last stretch is level. Tilt range back to -0.6 to 0.4 rad, checked clean |
| computer_mouse | `shell/body` and `wheel/tyre` | "the scroll wheel sits in a slot through the shell" | defect: there was no slot, so the wheel sat inside solid shell | `boolean_difference` cuts the slot; the wheel gets an axle whose ends bear on the slot walls |

The new `run_tests(ctx)` of both examples asserts the real contacts: the stem on the barrel
at rest and at -0.6 rad, and the axle on the slot walls.

**A false overlap.** The mouse fix showed a flaw in the depth measure: "overlap 2.89 mm" with
a volume of 2e-19 cm³. The axle's two ends touch the two slot walls, so the intersection is
two zero-volume discs 4 mm apart, and their joint box looked 2.9 mm thick. The depth is now
the maximum over the intersection's separate pieces (`Manifold.decompose`), ignoring pieces
smaller than 1e-15 m³.

**Combined poses** (`check: combined_collision`, signal kind `swept_collision`). After the
single-joint sweeps, the checker poses the model with:
- every driver at its range corners together, in Gray-code order (one joint changes per
  step, which keeps the walking solver on its branch). Above 64 corners, a seeded sample of
  64;
- plus 20 seeded random poses.

Only pairs not already reported by a single-joint sweep are reported, with the full pose,
for example `left_arm_swing=60.0 deg, right_arm_swing=60.0 deg`. Models with fewer than two
drivers skip this (the single sweep covered every pose). `max_corners`, `random_poses` and
`random_seed` are in `CheckConfig`.

New fixture: two arms that swing toward each other. Either arm alone is clear, and both at
full swing cross by 20 mm. The test also checks the Gray-code order and the 64-corner cap.

**Timing** with combined poses (median of 3):

| Example | Triangles | Joints | Mesh, dims, islands | Rest | Sweeps | Combined | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| laptop | 48 | 1 | 0.003 s | 0.001 s | 0.002 s | 0.000 s | 0.011 s |
| four_bar | 864 | 4 | 0.008 s | 0.002 s | 0.239 s | 0.000 s | 0.266 s |
| gantry | 120 | 6 | 0.006 s | 0.002 s | 0.110 s | 0.042 s | 0.174 s |
| toaster | 16,684 | 2 | 0.040 s | 0.003 s | 0.030 s | 0.038 s | 0.309 s |
| door_hinge | 2,336 | 1 | 0.010 s | 0.002 s | 0.015 s | 0.000 s | 0.104 s |
| desk_lamp | 3,164 | 2 | 0.018 s | 0.006 s | 0.109 s | 0.095 s | 0.287 s |
| computer_mouse | 2,598 | 3 | 0.011 s | 0.003 s | 0.261 s | 0.050 s | 0.376 s |
| folding_chair | 16,064 | 2 | 0.048 s | 0.002 s | 0.025 s | 0.000 s | 0.367 s |
| **all 8** | | | | | | | **1.89 s** (budget < 3 s; 1.9-2.3 s across runs) |

219 tests pass.

## Step 5: Viewer

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `viewer/loader.py` | 109 | `load_version(folder)`: rebuilds a kinematic model from `model.json` + `model.glb`. It gives the sliders and the mode ("tree" or "loop"), and `solve(values)` |
| `viewer/server.py` | 195 | stdlib `ThreadingHTTPServer` on 127.0.0.1: the page, vendored three.js, versions, the GLB, and `POST /pose` for loops |
| `viewer/__main__.py` | 43 | `python -m viewer <run>`: prints the URL, opens the browser, Ctrl+C stops |
| `viewer/static/kinematics.js` | 114 | Forward kinematics in JavaScript, a line-for-line port of the Python placement rule |
| `viewer/static/app.js`, `index.html` | 314 + 75 | The page: version picker, sliders, Play, Rest pose, signals list, red highlight |
| `viewer/static/vendor/three/` | (1.5 MB) | three.js r170 and three add-ons, unmodified, with its MIT `LICENSE` |
| `compiler/worker.py`, `compiler/feedback.py`, `sdk/export.py` | (changed) | Signals carry `bodies`, `pose` and `amount`; each version gets `signals.json`; a build that fails its checks is exported to `result/last_failed` |
| `tests/test_viewer.py`, `tests/js/pose_glb.mjs` | 202 + 43 | 16 tests |
| `scripts/make_demo_runs.py` | 50 | Compiles `runs/four_bar`, `runs/drawer` and `runs/desk_lamp` for the viewer (`runs/` is git-ignored) |

No new Python dependency. three.js is a vendored file, not a package.

**Usage.**

```shell
uv run python scripts/make_demo_runs.py
uv run python -m viewer runs/four_bar          # or runs/drawer, runs/desk_lamp
uv run python -m viewer runs/four_bar/result/v0000 --port 8000 --no-browser
```

The argument can be a run folder, its `result/` folder, or one version folder.

**The page.**
- **Versions.** The picker lists them newest first. If the latest compile failed its checks,
  "latest compile (failed checks)" comes first.
- **Sliders.** One per driven joint, in degrees or mm. An unlimited revolute gets ±180°, an
  unlimited prismatic ±1 m. Mimic joints have no slider; a note under the sliders says what
  they follow (`y_follow = 1 × y_head`).
- **Tree mechanisms.** The page computes every body transform in JavaScript from the
  manifest. A slider tick makes no server call.
- **Loop mechanisms.** The page POSTs the slider values. The server solves with the walking
  warm start. Only one request is in flight; ticks that arrive meanwhile collapse into one
  follow-up with the newest values. A pose that cannot close returns 409: the page keeps the
  last good pose and shows the reason in red ("loop closure 'pin' stays open ...").
- **Play.** Each driver in turn goes rest → upper → lower → rest, eased, 60 frames per leg.
  For a loop that means the crank, and the solver moves the rest. Clicking again stops.
- **Check overlay.** It lists the version's compile signals with severity colours. Clicking
  a signal that has a pose or bodies:
  - sets the sliders to the signal's full pose (a combined finding sets every joint);
  - paints its bodies red;
  - prints the message in the status bar.
  Moving a slider clears the highlight.

**Tree or loop.** A model is posed in the browser unless it has a loop closure that still
leaves unknowns once the drivers are set (`loop_unknowns`). The gantry has a closure, but its
mimic joints determine every joint, so it is a tree for the viewer. The four-bar is a loop.

**Failed compiles are viewable.** A compile that fails its checks still makes no version (step
3's rule), so before this step the model that failed was lost. Now:
- the worker exports it to `result/last_failed` (built in a staging folder and moved into
  place);
- the next compile removes it first.

The drawer fixture opens there:
- signal `swept_collision`, bodies `cabinet` and `drawer`;
- pose `slide = 260 mm` (first contact; the drawer's travel is 0-280 mm).

### Tests (16)

- **Loader = assembly** (8 examples). For each, the loaded model's body transforms equal
  the original `assembly.solve()` within 1e-9 at the rest pose, 30% and 85% of every
  slider's range. The four-bar includes its loop solve.
- **Modes:** the four-bar is a loop with one slider (`crank_joint`, ±180°); the gantry is a
  tree with `x_drive` and `y_head`.
- **Server:**
  - the page, `three.module.js` as `text/javascript` (Windows can map `.js` to
    `text/plain`, and browsers then refuse ES modules), the license, versions, the
    described version, and the GLB;
  - `static/../` is refused.
- **Four-bar pose endpoint:**
  - `crank_joint = 1.0` returns the same matrices as Python, plus the solved `rocker_joint`;
  - with the rocker limited to ±0.05 rad, `crank = π` returns 409 with the closure reason;
  - the next reachable pose still returns 200;
  - a non-number returns 400.
- **JavaScript = Python** (laptop and gantry, 3 poses each, under Node.js):
  - `kinematics.js` body transforms match `assembly.solve()` within 1e-9;
  - the GLB is parsed by the vendored `GLTFLoader` and posed exactly as the page does it
    (`localMatrices` into `node.matrix`);
  - three.js's world matrices then match `z_up_to_y_up @ T` within 1e-5 (float32).

  This covers the whole browser posing path except drawing. Node has no import map, so the
  test stages a temporary `node_modules/three` that points at the vendored file. The test is
  skipped when `node` is not installed (here: Node 24.13.1).
- **Failed compile:** the drawer compiles with status "error" and no `vNNNN`. The server
  lists only `last_failed`, and its signal has kind, bodies and pose.
- **CLI:** a missing folder exits with code 2.

**235 tests pass** (about 70 s).

**Not verified in this session: the page in a real browser.** I could not open a browser
here, so rendering, OrbitControls, the slider feel and the red highlight have not been
looked at. Everything up to the matrices three.js draws with is tested, and the CLI was
started on `runs/drawer` and answered over HTTP. A first manual look is the next check.

### How Articraft does it (study §5)

- **Files:** `viewer.py` (`load_viewer_run`, `kinematics_from_usdz`, `_solve_pose`,
  `_handler`) and `viewer.html`.
- **Server:** `articraft view <run>` runs a stdlib `ThreadingHTTPServer` on 127.0.0.1 and
  lists each numbered USDZ as a version.
- **Three.js** 0.185.1 is loaded from a CDN.
- **Every slider tick** POSTs to `/api/pose/N`. The server rebuilds kinematics from the USDZ
  (cached), runs `forward_kinematics`, and returns a 4x4 per body.
- **An unreachable pose** returns 409, and the page keeps the last good pose.
- **Compile results:** the study records none in the viewer; it shows geometry and sliders.

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| three.js vendored (r170, MIT license kept) instead of a CDN | Works offline and pins the exact code that is tested |
| Tree mechanisms posed in JavaScript; only loops call the server | No HTTP round trip and Python solve per tick for most models. Play runs at frame rate |
| One loop request in flight, newest values win | Dragging a slider fast does not queue up stale solves |
| The loader reads our own manifest + GLB (not USDZ) | It is what step 2 exports; the manifest has frames, axes, limits, mimics and the articulation tree |
| Compile signals in the viewer, click to jump to the pose, bodies in red | A finding such as "drawer hits cabinet at 260 mm" becomes something you can see |
| Failed builds kept in `result/last_failed` with their signals | The model with the defect is the one you most need to look at |
| Explicit MIME types | Windows registry mappings can break ES module loading |

### Open ideas

- **Contact markers:** draw the intersection volume (from the manifold check) at the signal
  pose, not only the two bodies in red.
- **Trajectory playback:** Play along a recorded path (study §13) instead of one joint at a
  time.
- **Version diff:** overlay two versions, or list the manifest changes between them.
- **Live reload:** poll `/api/versions` so a new compile appears without a page reload. That
  will matter when the agent loop (step 8) compiles in the background.
- **Loops in the browser:** a small Gauss-Newton loop solver in JavaScript would remove the
  server call for four-bars too. It needs the same walking warm start to stay on the branch.
- **Screenshots for the agent:** headless render of the signal pose, if image feedback
  (study §13) is added.

### Step 5b: bodies rendered black (found in a browser)

**Symptom.** Checked in a real browser: posing worked, but every body was solid black with no
shading.

**Cause.** The exported GLB has `POSITION` only, no `NORMAL` (trimesh's default). The page
did give each body a `MeshStandardMaterial`, and there were lights, but a lit material with
no normals cannot be shaded and renders black. The Node tests compared matrices, not
pixels, so they could not see it.

**Fix** (`viewer/static/display.js`, new, 80 lines; used by `app.js`):
- **Normals.** If a mesh has none, `toCreasedNormals` (three.js `BufferGeometryUtils`,
  already vendored) computes them with a 30° crease angle. Cylinders shade round, and
  boxes keep flat faces and sharp edges. Normals already present in a GLB are kept.
- **Colours.** A fixed 10-colour palette by body index (steel blue, orange, teal, green,
  gold, purple, khaki, light blue, light green, grey). It has no reds or pinks:
  red (`#e5383b`) is only the collision highlight.
- **Material.** `MeshStandardMaterial`, roughness 0.6, metalness 0.
- **Lights.** A hemisphere light (sky white, ground grey) plus a key and a fill
  directional light, so faces at different angles get different shades.
- **Outlines.** `EdgesGeometry` at 30°, drawn as thin dark lines (opacity 0.55) on each
  body. A polygon offset on the faces keeps the lines from z-fighting.
- **Legend.** Top right of the 3D view: body name → colour swatch. A highlighted body's
  row turns red and bold.

**Tests** (+2, under Node, for `four_bar` and `desk_lamp`). The real GLB is parsed with the
vendored `GLTFLoader` and styled by `display.js`, as the page does it. The test then checks:
- each body had no normals and now has unit normals (error below 1e-4) for every vertex;
- the material is lit (`MeshStandardMaterial`);
- each body has edge lines;
- the colours follow the palette by body index;
- highlighting turns a body red, and clearing it restores the body's own colour;
- the palette is distinct, with no colour within 25° of red hue. This check caught the
  first palette's brown (`#9c755f`, hue 22°), which is why the palette has khaki instead.

The staging folder now maps both `three` and `three/addons/*` (the page imports
`BufferGeometryUtils` as `three/addons/...`).

**237 tests pass.**

**Lesson.** Tests that compare numbers do not show what the page looks like. This bug was
caught by looking at the page in a browser.

**Open idea.** Write normals into the GLB at export, so other glTF viewers shade it too.
The viewer would keep its fallback.

## Step 6: LLM client

### Before step 6: allowances and combined poses, re-checked

All 8 examples were re-run through `run_tests(ctx)` + `run_checks`:

| Example | Allowances | Findings | Combined-pose sweep |
| --- | --- | --- | --- |
| laptop | 0 | 0 | skipped (1 driver) |
| four_bar | 0 | 0 | skipped (1 driver: the crank) |
| gantry | 0 | 0 | ran, 0.06 s |
| toaster | 0 | 0 | ran, 0.07 s |
| door_hinge | 0 | 0 | skipped (1 driver) |
| desk_lamp | 0 | 0 | ran, 0.12 s |
| computer_mouse | 0 | 0 | ran, 0.04 s |
| folding_chair | 0 | 0 | skipped (1 driver) |

The two allowances the examples used to have (desk lamp tube/socket, mouse wheel/shell) both
hid real defects. They were fixed in geometry in step 4b, so no example declares an
allowance. The combined-joint sweep (range corners in Gray-code order, at most 64, plus 20
seeded random poses) has been in `sdk/testing.py` since commit 30bc82a.

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `llm/messages.py` | 212 | **The** neutral message format (`System`, `User`, `Assistant`, `ToolResult`, `ToolCall`, `Tool`), its rules (`validate`), and conversion to and from Chat Completions |
| `llm/protocol.py` | 87 | `Model.query(messages, tools) -> Reply`, `Usage`, `LLMError` / `ProviderError` / `BudgetExceeded` |
| `llm/config.py` | 128 | Providers, roles, the price table (with its source and date), budget defaults, retry policy |
| `llm/accounting.py` | 124 | `cost()`, `Budget`, `CallLog` (`runs/<id>/llm_calls.jsonl`) and its summary |
| `llm/base.py` | 108 | What every model does around the provider call: validate, budget, cost, log, key redaction |
| `llm/openai_chat.py` | 148 | OpenAI and DeepSeek through the `openai` package; retry with backoff |
| `llm/scripted.py` | 61 | `ScriptedModel`: replays fixed replies, same protocol, no network |
| `tests/test_llm.py` | 480 | 36 tests, no network |
| `scripts/llm_smoke.py` | 114 | One real tool-call round trip per provider, printing tokens and cost |

New dependency: **openai 3.57.0** (approved). It brings httpx2, pydantic, anyio, jiter.

**The protocol.**

```python
class Model(Protocol):
    provider: str
    model: str
    def query(self, messages: list[Message], tools=()) -> Reply: ...

@dataclass
class Reply:
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage          # input, cached input, cache write, output, reasoning tokens
    cost: float | str     # USD, or "unknown"
    raw: dict             # the provider's response as JSON data
    provider, model, finish_reason, reasoning, latency_s, retries
    def message(self) -> Assistant   # append to continue the conversation
```

```python
model = get_model("builder", run_dir="runs/r1", budget=Budget(max_cost_usd=2, max_calls=50))
messages = [System("..."), User("Build a laptop.")]
reply = model.query(messages, tools)
messages.append(reply.message())
messages += [ToolResult(c.id, c.name, run(c)) for c in reply.tool_calls]
```

**Message format** (documented once, in `llm/messages.py`):

| Message | Fields | Meaning |
| --- | --- | --- |
| `System` | content | Only at the start, so the cache prefix stays stable |
| `User` | content | A user turn |
| `Assistant` | content, tool_calls, reasoning | A model turn; `reasoning` is DeepSeek's `reasoning_content` |
| `ToolResult` | call_id, name, content, is_error | Answers one tool call |

Other parts:
- **`ToolCall`** has `id`, `name`, `arguments` (the parsed dict) and `raw_arguments` (the
  model's exact string). If the string is not a JSON object, `error` says why.
- **`validate`** enforces three rules: system messages come first; every tool call is
  answered before the next turn; a result answers an open call.

**Providers.**
- **One adapter** (`OpenAIChatModel`, Chat Completions) serves both providers. DeepSeek is
  a `Provider` entry with its own `base_url` (`https://api.deepseek.com`) and
  `DEEPSEEK_API_KEY`.
- **Two differences**, both from the providers' docs:
  - **The output cap.** OpenAI uses `max_completion_tokens` (`max_tokens` is deprecated
    there). DeepSeek uses `max_tokens`.
  - **The reasoning echo.** DeepSeek's thinking mode returns `reasoning_content`. It is sent
    back on the assistant messages of the tool loop in progress (after the last user
    message), and older reasoning is dropped. OpenAI never gets it back.
- **Keys** come only from the provider's env var. A missing DeepSeek key is an error. It
  never falls back to `OPENAI_API_KEY`, which the SDK would otherwise do.

**Usage.** It is read from the Chat Completions `usage` object, and both providers fill the
same fields:
- `input_tokens` = `prompt_tokens`;
- `cached_input_tokens` = `prompt_tokens_details.cached_tokens` (DeepSeek also reports
  it as `prompt_cache_hit_tokens`);
- `cache_write_tokens` = `prompt_tokens_details.cache_write_tokens`;
- `output_tokens` = `completion_tokens`;
- `reasoning_tokens` = `completion_tokens_details.reasoning_tokens`.

Reasoning tokens are part of the output tokens and billed with them.

**Cost.**

```
cost = ( (input − cached − cache_write) × input_price
       + cached × cached_input_price
       + cache_write × cache_write_price      (input price if the model has none)
       + output × output_price ) / 1e6
       × off-peak factor                      (DeepSeek only)
```

- An unknown model logs a warning and costs `"unknown"`, never $0.
- An OpenAI call over 272K input tokens uses the long-context rate. If the table does not
  have that rate, the cost is `"unknown"` too.
- DeepSeek peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday. Outside them
  the rate is half. Chinese public holidays are also off-peak, but we don't have those
  dates, so a call on a holiday is costed at peak: over, never under.

**Price table** (`llm/config.py`). USD per 1M tokens, **checked 2026-11-23** on
developers.openai.com/api/docs/pricing (Standard tier) and
api-docs.deepseek.com/quick_start/pricing:

| Model | Input | Cached input | Output | Notes |
| --- | --- | --- | --- | --- |
| gpt-6.2-astra | 8.00 | 0.40 | 40.00 | cache write 10.00; above 272K input: 16.00 / 0.80 / 60.00 |
| gpt-6.2-sol | 2.00 | 0.10 | 10.00 | cache write 2.50; above 272K input: 4.00 / 0.20 / 15.00 |
| gpt-6.1-terra | 1.00 | 0.05 | 5.00 | long-context rate not recorded |
| gpt-6-luna | 0.10 | 0.01 | 0.50 | long-context rate not recorded |
| deepseek-flash | 0.30 | 0.006 | 1.20 | peak rates; off-peak half (0.15 / 0.003 / 0.60) |
| deepseek-v4-pro | 1.32 | 0.044 | 3.96 | peak rates; off-peak half |

**Roles** (one line each, in `llm/config.py`):
- `builder = STRONG = Role("openai", "gpt-6.2-astra")`. OpenAI's models page: "If you're
  not sure where to start, use GPT-6.2 Astra, our flagship model for complex reasoning and
  coding." Its model page lists `v1/chat/completions` and function calling as supported.
- `cheap = CHEAP = Role("deepseek", "deepseek-flash")`. DeepSeek's pricing page: "Use
  `deepseek-flash` as the model name". It is served by DeepSeek-V4.1-Flash and supports
  tool calls.

Switch with, for example, `ROLES["builder"] = CHEAP`.

**Budget.** One `Budget(max_cost_usd, max_calls)` per run, shared by all its models.
Defaults are $5 and 200 calls.
- **Before a call:** it is refused if the call cap is reached or the cost cap is already
  spent. The transport is not touched.
- **After a call:** if it took the total over the cost cap, `BudgetExceeded` is raised with
  the reply attached (it was paid for).
- **Unknown-cost calls** count toward the call cap and log a warning that the cost cap
  cannot count them.

**Reliability.**
- **Timeout:** each attempt has `timeout_s` (default 300 s; reasoning models are slow). The
  SDK's own retries are off.
- **Retries:** rate limits (429), 5xx and network errors (connection errors and timeouts),
  at most 4 retries.
- **Backoff:** 1, 2, 4, 8 s, times a jitter of 0.5-1.0, capped at 30 s. A server's
  `Retry-After` is honoured, with the same 30 s cap.
- **No retry on bad requests:** 400, 401, 403, 404 and 422 raise `ProviderError`
  immediately.

**Caching.** Both providers cache matching prompt prefixes automatically. The adapter keeps
the prefix byte-identical between calls:
- system messages first (enforced by `validate`);
- tool schemas converted once and reused while the tools are the same;
- tool-call arguments sent back exactly as the model wrote them, never re-serialized.

Each log record has:
- `prefix`, a 16-hex hash of the system prompt + tool schemas. If it changes inside a
  conversation, the cache cannot be reused;
- `usage.cached_share`.

`CallLog.summary()` gives the run's cached share.

**Logging.** Each call appends one line to `runs/<id>/llm_calls.jsonl`:

```json
{"time": "...", "role": "builder", "provider": "openai", "model": "gpt-6.2-astra",
 "prefix": "3f9c...", "messages": 4, "tools": 1, "status": "ok", "served_model": "...",
 "usage": {"input_tokens": 2000, "cached_input_tokens": 1500, "cache_write_tokens": 0,
           "output_tokens": 10, "reasoning_tokens": 0, "cached_share": 0.75},
 "cost": 0.005, "latency_s": 1.84, "retries": 1, "finish_reason": "stop",
 "tool_calls": [], "budget": {"calls": 1, "spent_usd": 0.005, ...}}
```

- Failed calls are logged as `"status": "error"`, with the HTTP status and a redacted
  message.
- Records hold no message text and no keys. Provider errors can quote a key ("Incorrect API
  key provided: sk-..."), so error text goes through `redact()`. That removes the
  configured key and anything shaped like `sk-...`.

### Tests (36, no network)

- **Messages:**
  - conversion both ways, including tool calls;
  - argument strings kept exactly;
  - reasoning echoed only for the current tool loop;
  - bad JSON arguments;
  - tool schemas both ways;
  - 5 rule violations.
- **Usage:** OpenAI and DeepSeek shapes.
- **Cost:**
  - cached tokens: $0.0592 vs $0.12 uncached on gpt-6.2-astra;
  - cache writes;
  - reasoning inside output;
  - long-context rate, and an unknown long-context rate;
  - DeepSeek peak, weekday off-peak and weekend;
  - an unknown model gives `"unknown"` plus a warning;
  - every role has a provider and a price.
- **Adapter** (a fake transport in place of `chat.completions.create`):
  - the OpenAI request shape (cap parameter, tools, `reasoning_effort`, timeout);
  - parsing of a tool-call reply;
  - DeepSeek's cap parameter and `reasoning_content` echo;
  - a byte-identical prefix across calls;
  - keys only from their own env var.
- **Retries:**
  - 429/500/503 retried with sleeps of 1, 2 and 4 s;
  - connection errors and timeouts retried;
  - `Retry-After` 7 s honoured and 999 s capped at 30 s;
  - exactly 5 attempts, then `ProviderError`;
  - 400/401/404/422 tried once;
  - a custom policy.
- **Budget:**
  - the call cap refuses before the call is made;
  - the cost cap raises after the call that crosses it, with that reply attached, and
    refuses after that;
  - one budget shared by two models;
  - unknown cost counts as a call.
- **Log:**
  - fields, the cached share, the same prefix for the same system prompt and tools;
  - error records;
  - a key quoted in a 401 message does not reach the exception or the file.
- **ScriptedModel:**
  - a full tool loop, costed like the named model;
  - running out of replies;
  - a tool that was not offered;
  - a bad conversation.
- **Through the real `openai` package**, with only HTTP mocked (`httpx2.MockTransport`):
  - a 429 and then a tool-call reply from the DeepSeek URL;
  - the request JSON the SDK actually sends;
  - the bearer header;
  - `reasoning_content` and usage parsed back;
  - no key in the log.

**273 tests pass.**

**Not verified here: the real APIs.** This machine has no OpenAI or DeepSeek key, so
`scripts/llm_smoke.py` has only been run without keys (both providers are skipped
cleanly). Two things only a real call can confirm:
- that DeepSeek accepts `reasoning_content` echoed in a thinking-mode tool loop. The thinking
  mode guide failed to load (HTTP 525); the API reference documents the field on replies;
- that `cache_write_tokens` is the field OpenAI uses for cache writes. That name came from a
  summary of the API reference page, not its verbatim text. If the field is absent, writes
  are simply billed as input.

The smoke script exercises the first point. Its `--cache` option exercises caching.

### Live smoke test (run by the user, 2026-11-23)

`scripts/llm_smoke.py`, with the user's keys from `replica/.env`. Log:
`runs/llm_smoke/20261123-165312/llm_calls.jsonl`.

| Provider / model | Result | Tokens (in / cached / out / reasoning) | Cost |
| --- | --- | --- | --- |
| deepseek / deepseek-v4-pro, tool call | OK: called `add(2, 3)` | 394 / 0 / 95 / 36 | $0.000448 |
| deepseek, answer after the tool result | OK: "5"; `reasoning_content` echoed and accepted | 502 / 384 / 17 / 8 | $0.000120 |
| deepseek, `--cache`, 1st long prompt | OK | 3,112 / 0 / 16 / 13 | $0.002086 |
| deepseek, `--cache`, 2nd long prompt | OK: **98.7% cached** | 3,112 / 3,072 / 21 / 18 | $0.000136 |
| openai / gpt-6.2-astra | **HTTP 400**: "Function tools with reasoning_effort are not supported for gpt-6.2-astra in /v1/chat/completions ... use /v1/responses" | - | $0 (logged as error) |

The DeepSeek costs are off-peak rates (16:53 UTC on a Monday). Both points that step 6 could
not verify offline now are:
- the reasoning echo in a tool loop works;
- DeepSeek reports cached tokens as expected.

OpenAI with tools needs the Responses API: a required item in the open ideas below. It is
not fixed now, as instructed.

### How Articraft does it (study §6)

- **Files:** `agent/protocols.py` (`Model.query`) and `agent/provider/` (OpenAI, Anthropic,
  Gemini, OpenRouter, plus a shared `_chat.py`).
- **Reply:** each adapter returns `{text, tool_calls, token_usage, cost, provider_content}`.
- **OpenAI:** the Responses API over WebSocket, with a `previous_response_id` chain.
- **Anthropic:** explicit 1-hour cache control.
- **Retries:** up to 4 attempts with exponential backoff (0.5-20 s), inside each adapter.
- **Cost:** hard-coded price tables. An unknown model is recorded as $0.
- **No cost cap.**

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| One adapter (Chat Completions) for two providers, instead of four adapters | DeepSeek speaks the same API. Fewer converters, fewer places for format bugs |
| Unknown model or rate → cost `"unknown"` plus a warning, never $0 | A $0 entry hides real spend |
| Per-run cost and call caps (`BudgetExceeded`), checked before every call | An agent loop can otherwise spend without limit |
| Retry policy in one place, shared by both providers; no retry on 4xx | Bad requests do not get better by repeating them |
| Prices with their source URL and check date; DeepSeek off-peak; OpenAI long-context and cache-write rates | Costs are only useful if they are right and you can see how old they are |
| Exact argument strings sent back; a prefix hash in every log record | Automatic caching needs a byte-identical prefix, and the hash shows when it breaks |
| Keys redacted from logged errors | A 401 message can quote the key |

### Open ideas

- **REQUIRED before any OpenAI evaluation: an OpenAI Responses API adapter** behind the same
  `Model.query` protocol. The live smoke test (below) showed that gpt-6.2-astra refuses
  function tools with reasoning on `/v1/chat/completions`. The adapter would also bring
  `previous_response_id`, which sends only new items each turn, as Articraft does.
- **`prompt_cache_key`:** OpenAI routes requests with the same key to the same cache. Check
  its docs before using it.
- **Strict tool schemas:** DeepSeek's beta `strict` mode (at `/beta`) and OpenAI Structured
  Outputs would reject malformed arguments at the source.
- **Streaming:** show progress on long reasoning calls. It is not needed for correctness.
- **Price freshness check:** warn when `PRICES_CHECKED` is more than, say, 60 days old.
- **Run storage (step 11)** will want to save the neutral messages; add `to_dict` /
  `from_dict` then.

## Step 7: Agent tools

### Before step 7: .env and development roles

**`.env` loader** (`llm/env.py`, 50 lines). It is a tiny loader instead of python-dotenv,
so there is no new dependency.
- **Format:** `NAME=value`, plus `export NAME=...`, quoted values, blank lines and `#`
  comments. Malformed lines are skipped silently: they are never echoed, since they may
  hold a secret.
- **No overriding:** a variable that is already set is never overridden; the real
  environment wins.
- **Where it runs:** `get_model()` and `scripts/llm_smoke.py` call `load_env()`.
- **Git:** `git check-ignore -v .env` prints `.gitignore:7:.env .env`. A test runs that
  check.
- **Never logged:** `load_env` returns only the names it set and nothing prints values.
  The call log holds no environment. The compiler already strips `*_API_KEY` / `*_TOKEN`
  from the environment of model-written code, so keys from `.env` do not reach it.

**Development roles** (`llm/config.py`):
- `builder` = `deepseek-v4-pro` and `cheap` = `deepseek-flash`. Both models appear on
  DeepSeek's pricing page (checked 2026-11-23).
- The default budget is **$1 and 100 calls per run**.
- The OpenAI rows stay in the price table, and `OPENAI_STRONG` (gpt-6.2-astra) stays defined
  for evaluation runs.
- `REFERENCE_MODELS` gives the smoke script a model for a provider that no role uses.

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `agent/tools/core.py` | 209 | `AgentTool`, `ToolContext`, `ToolError`, `ToolOutput`, path scoping, argument checking, output cap |
| `agent/tools/__init__.py` | 70 | The registry: fixed order, `schemas()`, `run_tool()`, `schema_size()` |
| `agent/tools/files.py` | 178 | `read`, `write`, `edit` |
| `agent/tools/compile_tool.py` | 24 | `compile` |
| `agent/tools/builds.py` | 130 | The newest build, loaded once and cached; pose checks; joint report |
| `agent/tools/render.py` | 144 | `render`: CPU rasterizer (numpy + Pillow) |
| `agent/tools/probe.py` | 151 | `probe`: positions, points, gaps and collisions at a pose |
| `viewer/loader.py` | (changed) | Splits each body's GLB mesh into closed pieces (see below) |
| `sdk/testing.py` | (changed) | `CollisionScene`, a public name for the check's collision scene |
| `docs/sdk/README.md` | 3 | Placeholder for the SDK docs (step 10) |
| `tests/test_agent_tools.py` | 323 | 26 tests, no LLM |
| `scripts/tools_demo.py` | 52 | Calls each tool by hand on the laptop and the drawer and prints the JSON |

Dependency: **pillow 13.0.0** is now declared. It was already installed through another
package; rendering uses it directly.

**The tool type.**

```python
@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str           # one line: what it does and when to use it
    parameters: dict           # JSON Schema
    run: Callable[[ToolContext, dict], dict]   # JSON-ready dict, or raise ToolError
    parallel_ok: bool = False

ctx = ToolContext(run_dir)                      # workspace = run_dir/workspace; compiler = compile
out = run_tool(ctx, "probe", {...})             # never raises
out.text                                        # JSON, at most 16,000 characters
out.images                                      # PNG paths (render), for models that see images
messages.append(out.message(call.id))           # -> llm.ToolResult
```

**The tools** (registry order, which is fixed for a stable cache prefix):

| Tool | Arguments | Returns | Parallel |
| --- | --- | --- | --- |
| `read` | `path`, `offset?`, `limit?` (400 lines default, 2,000 max) | `{path, lines, total_lines, text: "L12: ...", next_offset?}`; a folder gives `{path, entries}` | yes |
| `write` | `path`, `content` | `{path, created, bytes, lines}` | no |
| `edit` | `path`, `edits[{old_text, new_text}]` | `{path, replaced, bytes, lines}` | no |
| `compile` | none | `{status, version, cached, compile_signals}` | no |
| `render` | `pose?`, `views?` (front, side, top, iso) | `{build, images[{view, path}], joints, bodies{min_m, max_m, size_mm}, overall_size_mm, note?}` | no |
| `probe` | `pose?`, `queries[]` | `{build, joints, results[], note?}` | no |

**Errors** always have one shape, `{"error_kind", "message", "hint"}`. The kinds:
- `bad_arguments`: checked against the tool's schema before it runs;
- `path_refused`, `not_found`, `not_text`, `too_large`;
- `edit_failed`;
- `no_build`, `bad_pose`, `pose_unreachable`;
- `unknown_tool`;
- `internal_error`: a tool bug, with the file and line.

A failed compile is not a tool error: `compile` returns `status: "error"` and the signal
block.

**Rules.**
- **Paths.** A path is relative to the workspace, or `docs/sdk/...`, which is read-only.
  These are refused: absolute paths (`/etc/passwd`, `C:\...`, `C:main.py`, UNC paths), any
  `..`, and anything that resolves outside the allowed folder (symlinks included). Writing
  or editing under `docs/sdk/` is refused too.
- **No shell tool.** Articraft's `exec_command` is a full shell. Here the model gets file
  access through `read`/`write`/`edit`, and looks at the model through `render`/`probe`.
- **Output cap.** Each result is at most 16,000 characters, about 4k tokens. The longest
  string field is cut to its head (2/3) + tail (1/3), with a note: "[... N characters of
  text omitted (head and tail shown); read a smaller range with offset/limit ...]".
- **Schema size.** 3,054 characters as sent in Chat Completions JSON, **about 764 tokens**.
  That is an estimate at 4 characters per token, because no tokenizer is installed. Per
  tool, in characters: read 467, write 344, edit 546, compile 275, render 527, probe 883.

**`read`**
- Line-numbered (`L12: ...`).
- Long lines are cut at 1,000 characters.
- Gives `next_offset` when there is more to read.
- Lists a folder: hidden files and `__pycache__` are left out.
- A non-UTF-8 file gives `not_text`.

**`write`**
- Creates parent folders.
- Content is capped at 400,000 characters.
- Writes UTF-8 with LF line endings.

**`edit`**
- Edits apply in order, and each `old_text` must match exactly once.
- If any edit fails, nothing is written, and the error names it: `edit 2 of 2: old_text
  was not found in main.py; nothing was changed`.
- No match: the hint looks for a whitespace-insensitive match ("it matches at line 1 if
  whitespace is ignored; copy the exact indentation ...").
- Several matches: the error lists the lines ("matches 2 times (lines 4, 5)") and asks for
  more surrounding lines.

**`compile`**
- Calls `ctx.compiler(run_dir)` (step 3) and returns only status, version, cached and the
  `<compile_signals>` block.
- The compiler's own cache answers when nothing changed (`cached: true`).
- The result is kept in `ctx.last_compile`, so render and probe can say when the build is
  older than `main.py`.

**The newest build** (shared by render and probe):
- `result/last_failed` if it exists. It exists only when the latest compile failed its
  checks; every compile removes it first.
- Otherwise the latest `result/vNNNN`.
- A note says which one was used. Two cases:
  - "the latest compile failed its checks; this is the model it built";
  - "the latest compile failed before a model was built; this is the last good build,
    v0002, which is older than main.py".
- Builds are loaded once per folder (manifest mtime) with step 5's loader, and cached in
  the context.
- Poses: only driven joints may be set. Anything else is refused with the list of driven
  joints and their ranges ("joint 'rocker_joint' is moved by a mimic or loop and cannot be
  set").
- Out-of-limit values give `bad_pose`. A loop that cannot close gives `pose_unreachable`.

**`render`**
- **CPU only**, with numpy + Pillow, 640×480 per view.
- **Views:** orthographic front (looking +Y), side (−X), top (−Z) and iso.
- **Drawing:** flat Lambert shading with a light from the upper side of the camera;
  painter's algorithm (triangles drawn far to near); 2× supersampling.
- **Colours:** the viewer's palette, which has no reds. Each image has a legend and a
  view label.
- **Back-face culling on closed bodies.** Without it, the painter's algorithm drew hidden
  faces over visible ones on big two-triangle box faces: dark slivers on the laptop deck,
  and the lid's back showing through. Open meshes are still drawn two-sided.
- **Files:** PNGs go to `run_dir/renders/NNNN/<view>.png`. The JSON gives their paths and
  each body's world bounding box, so a text-only model can still use the result.
  `ToolOutput.images` carries the files for a model that accepts images (step 8).
- **Speed:** about 0.2 s for 4 views of the desk lamp (3,164 triangles).

**`probe`**
- **Queries:**
  - `position{body}`: origin in m and rpy in degrees, in the SDK's convention;
  - `point{body, point}`: a body-frame point in world coordinates (a tool tip, a handle);
  - `gap{a, b}`: `apart` / `touching` / `penetrating`, with `gap_mm`, and `depth_mm` if
    they overlap;
  - `collisions{}`: every interpenetrating pair, with depth, volume, and whether it is
    over the check tolerance.
- **Joints:** every moving joint's solved value is always returned, in rad and deg, or m.
- **Loops** go through `assembly.solve`, with the walking warm start.
- **Collisions** use the same scene as the checks: AABB pre-check, then fcl, then manifold
  depth. The tolerance is the check's `max(0.2 mm, 0.002 × size)`.

**A false collision, found by the probe.** On the four-bar at crank 60° (which passes every
check), the first probe reported `coupler`/`rocker` 5.0 mm deep with zero volume.
- **Cause:** the export merges a body's shapes into one GLB mesh. Where the shapes overlap
  (a bar and its pivot), the merged mesh is not watertight. So the depth fell back to fcl's
  contact estimate, which is unreliable for meshes.
- **Fix:** the loader splits each body's mesh into its connected pieces. Each piece is a
  closed solid again, as in the original model, so the manifold depth works. The probe now
  agrees with the checks: no collisions, and coupler/rocker `touching`.

The viewer's poses are unaffected; its tests still pass.

### Tests (26; 25 pass, 1 skipped on this machine)

- **Registry:**
  - the fixed order and identical schemas on every call;
  - no shell-like tool;
  - one-line descriptions under 170 characters;
  - only `read` parallel;
  - schema size under 1,000 estimated tokens. It is 764.
- **Arguments:**
  - a missing field, a wrong type, an unknown field, an unknown tool, and a non-object;
  - a tool bug becomes `internal_error`.
- **Output cap:** the head and tail are kept, the note is there, and the JSON is still
  valid.
- **Paths:** 7 escapes (`../`, `a/../../`, `docs/sdk/../../`, `/etc/passwd`, `C:\...`,
  `C:main.py`, UNC) are refused by read, write and edit alike.
  - The symlink escape test is skipped here: Windows without developer mode cannot create
    symlinks.
  - `docs/sdk` can be listed and read, but write and edit there are refused.
- **read / write:** line numbers, offset/limit, `next_offset`, folder listing, missing file,
  binary file, offset past the end, a capped 2,000-line read.
- **edit:**
  - several edits applied together;
  - no match, with the whitespace hint;
  - all or nothing: edit 1 is not applied when edit 2 fails;
  - multi-match with line numbers, empty old_text, missing file.
- **compile:**
  - a broken `main.py` (syntax error) gives `status: error` and a `<compile_signals>` block
    naming main.py;
  - the second call is `cached: true`;
  - the four-bar compiles to v0000.
- **render:**
  - the laptop with the lid at 1.2 rad gives 4 PNGs of 640×480, each with more than 50
    colours (shaded, not blank);
  - bounding boxes and the joint report;
  - the next render goes to `renders/0002`;
  - a bad view name is refused;
  - render and probe without a build give `no_build`.
- **probe:**
  - the four-bar at crank 60°: `rocker_joint` = −0.083705 rad (−4.796°), matching
    `assembly.solve()` within 1e-5 rad; the rocker origin and a coupler point match the
    solved transforms; no collisions;
  - **the drawer at 270 mm:** `last_failed` is used, with its note; `cabinet`/`drawer` are
    10.0 mm deep (tolerance 1.337 mm); the gap query agrees; at 100 mm there is no
    collision;
  - bad input: an unknown body, a missing field, a 2-element point, a loop member set
    directly, an unknown joint, empty queries;
  - an unreachable loop pose gives `pose_unreachable` ("loop closure 'pin' ..."), and a
    reachable one still works afterwards.

**300 tests pass, 1 skipped.**

Example probe result (four-bar, crank 60°):

```json
{"build": "v0000",
 "joints": {"crank_joint": {"rad": 1.047198, "deg": 60.0},
            "rocker_joint": {"rad": -0.083705, "deg": -4.796},
            "coupler_joint": {"rad": -1.412386, "deg": -80.924},
            "pin": {"rad": 0.281483, "deg": 16.128}},
 "results": [{"type": "position", "body": "rocker", "origin_m": [0.1, 0.0, 0.0],
              "rpy_deg": [0.0, -0.0, -4.796]},
             {"type": "point", "body": "coupler", "local_m": [0.05, 0.0, 0.0],
              "world_m": [0.058699, 0.050279, 0.0]},
             {"type": "collisions", "pairs": [], "tolerance_mm": 0.328}]}
```

### How Articraft does it (study §7)

- **Files:** `agent/tools/` has a registry, `_core.py` (`Tool`, `ToolContext`, path
  scoping), one module per tool, and `_exec.py`.
- **Tools:** `read` ‖, `view_image` ‖, `edit`, `write`, `exec_command` and `write_stdin`
  (a full login shell with sessions), `compile`, and `synthesize_trajectory`.
- **Errors** come back as a bare `{"error": str}`.
- **Previews:** the model renders them itself, by running Python through `exec_command`.

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| No shell tool | A shell gives model-written commands the user's full permissions. Every legitimate use here (inspect files, render, measure) has a narrow tool |
| `render` and `probe` as tools | The model can look at and measure the object without writing preview scripts. `probe` answers motion questions (where is the tip at 60°? do these collide at 270 mm?) without a compile |
| Structured errors `{error_kind, message, hint}` | The model can branch on the kind and act on the hint. A bare string is easy to ignore |
| Arguments checked against the schema before running | Bad calls fail with a clear message instead of a Python traceback |
| `edit` errors name the edit, the lines, and a whitespace-insensitive near miss | Most failed edits are whitespace or duplicate matches; the hint says which |
| Head + tail output cap with a note | The end of a file or log is often what matters; the note says how to read the rest |
| Fixed registry order | The tool schemas are part of the cached prompt prefix |

### Open ideas

- **Images to the model (step 8):** attach `ToolOutput.images` as image parts for models
  that accept them. DeepSeek models are text-only through Chat Completions here, so the
  bounding-box summary carries the information.
- **Probe sweeps:** a `sweep{joint, from, to, steps}` query giving the minimum gap or the
  first contact over a range. It is the checks' bisection, offered to the model.
- **Render highlights:** paint colliding bodies red at the render pose, like the viewer.
- **Exact token counts:** a tokenizer (it would need approval) would replace the
  4-characters-per-token estimate.
- **Parallel execution:** the loop (step 8) can run consecutive `read` calls concurrently.
  Probe and render share a cached scene, so they stay sequential.

## Step 8: Agent loop

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `prompts/system.md` | 68 | System prompt v0 (4,356 chars, about 1.1k tokens): role, the four quality rules, the workflow, the `main.py` contract, tool tips |
| `docs/sdk/quickstart.md` | 104 | SDK quickstart v0 (5,039 chars, about 1.3k tokens): building blocks, joints and limit rules, loops, a complete laptop example, what the compiler checks |
| `prompts/main_template.py` | 18 | Starter `workspace/main.py`: imports, comments for bodies and joints, `object_model`, an empty `run_tests` |
| `agent/loop.py` | 447 | `run(prompt, model, budget, max_turns=60) -> RunResult`: turns, the stop rule, end states, the run folder, `record.json` |
| `agent/__main__.py` | 86 | `python -m agent "<prompt>"`: one progress line per turn, the run folder at the end |
| `llm/messages.py`, `llm/config.py`, `llm/openai_chat.py`, `llm/scripted.py` | (changed) | `User(content, images)`; `Role.images` / `supports_images`; `ScriptedModel` `raw_arguments` |
| `tests/test_agent_loop.py` | 326 | 19 tests: ScriptedModel or a fake transport, the real compiler, no network |

The quickstart's laptop example is compiled by the tests: it is the "good `main.py`" of the
loop tests, so the docs cannot drift from the SDK.

**The loop.**

```python
budget = Budget(max_cost_usd=1.0, max_calls=100)
model = get_model("builder", run_dir=run_dir, budget=budget)
result = run("a laptop that opens and closes", model, budget, run_dir=run_dir)
result.status, result.reason, result.best, result.record
```

- **Messages:** `[System(system.md), System(quickstart.md), User(task)]`. Both system
  messages plus the tool schemas are the stable, cached prefix. The task message asks the
  model to plan bodies and joints first.
- **Each turn:**
  1. `model.query(messages, tools)`; the budget is checked inside;
  2. the reply is appended;
  3. its tool calls run in order, with consecutive `parallel_ok` calls (reads) on a
     4-thread pool, their results kept in order;
  4. the results are appended.
- **Images:** for a model whose role has `images=True`, rendered PNGs are added after the
  tool results as one `User` message with image parts (data URLs). That is gpt-6.2-astra
  only; DeepSeek is off, so it relies on the bounding-box summary.
- **Malformed tool arguments** come back as a `bad_arguments` tool error without running.
  If the reply was cut at the output limit (`finish_reason=length`), they come back as
  `reply_cut_off`, whose hint says to reason less and write large files in parts.

**Stop rule** (Articraft's freshness gate, plus our blocking checks). Success needs all
three:
- a reply with text and no tool calls;
- the latest compile passed every blocking check;
- `workspace_digest(workspace)` still equals that compile's digest.

Otherwise one reminder is added as a user message, and the loop goes on:
- `<compile_required>`, with the reason: "you have not compiled yet", "the latest compile
  failed (swept_collision: ...)", or "the workspace changed after the last compile";
- `<final_response_required>`, for an empty reply after a fresh passing compile.

**End states.** Every non-success reason ends with the best build: the latest `vNNNN`
(passed), else `result/last_failed` (built, failed its checks), else "no build was
produced".

| Status | When |
| --- | --- |
| `success` | the stop rule above |
| `failed_checks` | 3 finishes in a row refused while the latest compile failed |
| `budget_exceeded` | `BudgetExceeded` from the model: the call cap before a call, or the cost cap after the call that crossed it (that reply is kept in the conversation) |
| `max_turns` | 60 turns by default |
| `no_progress` | 3 empty replies in a row; or 5 compile failures in a row with the same failure signature (cached recompiles count); or 3 refused finishes with no failing build to blame |
| `error` | a model call failed after retries (`LLMError`), or a loop bug (traceback kept in `reason`) |

**Run folder** `runs/<UTC timestamp>-<slug>/`:
- `workspace/` (the starter `main.py` is copied in), `result/`, `renders/`;
- `llm_calls.jsonl` (step 6);
- `conversation.jsonl`: every message with its turn and time, including system prompts,
  tool calls with their exact argument strings, tool results, reminders and reasoning;
- `record.json`, rewritten atomically (temp file + rename) after every turn.

`record.json` holds:
- status, reason, prompt, model, start, updated, duration, turns;
- compiles (calls / passed / failed / cached / last status / last version);
- tool call counts, token totals, cost, the budget state;
- best build, final text;
- **`per_turn`**: context size estimate, input / cached / output / reasoning tokens,
  cached share, cost, latency, tools, compile result, reminder.

The context estimate counts what the request actually sends, at about 4 characters per
token: messages, the tool schemas, and the reasoning DeepSeek gets back.

**CLI.**

```
uv run python -m agent "a drawer that slides out of a cabinet" [--role builder|cheap]
                       [--max-turns 60] [--max-cost 1.0] [--max-calls 100] [--runs runs]

turn   9  compile  -> compile success v0000  | ctx~7.7k tok, cached 100%, $0.0467
success: finished after a passing, fresh compile
view it: uv run python -m viewer runs/20261123-170310-a-drawer-that-slides-out-of-a-cabinet
```

### Tests (19, no network)

- **Clean success:** write, compile, final text. Checked:
  - the record fields and the compile counts;
  - `per_turn` growing turn by turn;
  - the first request is `[System, System, User]` with the tools in registry order;
  - the conversation roles in order;
  - the run folder layout.
- **Compile failure, then a fix, then success:** limits `(0.5, 1.0)` give "must contain 0";
  an edit fixes it; the second compile passes.
- **Final text without compiling:** a `<compile_required>` reminder ("you have not compiled
  yet"), then success.
- **An edit after a good compile:** the finish is refused ("the workspace changed after the
  last compile"); a recompile, then success.
- **An empty reply after a passing compile:** `<final_response_required>`.
- **Call cap:** `budget_exceeded` with best build `v0000`, and `record.json` says so.
- **Cost cap:** a priced scripted model (deepseek-v4-pro) crosses $0.01; the paid reply is
  kept; "no build was produced".
- **5 identical failures** (the drawer fixture, compiled 6 times; 4 are cached): `no_progress`
  after the 5th, best build `result/last_failed`.
- **Finishing over failed checks**, 3 times: `failed_checks`.
- **Other end states:** 3 empty replies give `no_progress`; `max_turns`; a model error gives
  `error` with the record written.
- **Tool execution:**
  - three consecutive reads run in parallel and keep their order;
  - images attached only when `supports_images`;
  - broken JSON arguments come back as structured `bad_arguments`;
  - a cut-off reply comes back as `reply_cut_off`.
- **DeepSeek reasoning across a tool loop** (`OpenAIChatModel` with a fake transport):
  - request 2 carries `reasoning_content` "plan", and request 3 carries "plan" and "check";
  - the first 3 messages and the tool schemas are identical in every request;
  - the token totals and the cached share are in the record;
  - the context estimate includes the echoed reasoning.
- **CLI:** the progress lines, the success line, the viewer hint, and the run folder name.

**319 tests pass, 1 skipped.**

### Live runs (DeepSeek, builder = deepseek-v4-pro, $1 cap each, 2026-11-23 ~17:03 UTC, off-peak)

| Prompt | Status | Turns | Compiles (pass / fail) | Tokens in / cached / out (reasoning) | Cost | Time |
| --- | --- | --- | --- | --- | --- | --- |
| a laptop that opens and closes | success | 5 | 1 / 0 | 36,495 / 31,616 (87%) / 5,135 (4,109) | $0.0141 | 62 s |
| a drawer that slides out of a cabinet | success | 12 | 1 / 2 | 220,524 / 212,352 (96%) / 20,841 (17,588) | $0.0513 | 253 s |
| a four-bar linkage | success | 9 | 1 / 0 | 246,274 / 230,656 (94%) / 35,597 (32,603) | $0.0859 | 403 s |
| **all three** | 3 / 3 | 26 | 3 / 2 | 503,293 / 474,624 (94%) / 61,573 | **$0.151** | 12 min |

What happened:
- **Laptop.** One write, one compile, two renders and a probe, then a correct summary.
  **This run is not evidence of skill:** its `main.py` is 94% identical to the quickstart's
  laptop example (difflib ratio 0.94). Evaluation prompts must not match the docs example
  (step 12).
- **Drawer.** A 2-body cabinet and drawer, a prismatic slide 0-300 mm, a knob. Two failed
  compiles:
  - `AttributeError: 'Box' object has no attribute 'translate'`. The quickstart was
    ambiguous: it said "mesh geometries can be moved first: `.translate`". Fixed: it now
    says `Box`/`Cylinder`/`Sphere` are placed with `xyz`/`rpy`, and only mesh geometries
    have `.translate`;
  - `[floating] drawer ... 2.00 mm from cabinet`: the drawer did not touch its cabinet. The
    model closed the gap.
- **Four-bar.** Ground, crank, coupler, rocker. `pin` is a loop closure, and the crank is
  limited to ±π. It compiled first time, then 4 probe calls and 3 renders checked the
  motion.
  - **Turn 3 was cut off:** 31,646 reasoning tokens in 361 s reached the 32,000-token
    output cap, and the `write` arguments were truncated. The tool error then was a
    confusing "not valid JSON". The model recovered by writing again.
  - **Fixes:** a `reply_cut_off` error with an actionable hint, and a test.
- **The reasoning echo grows the context.** DeepSeek's thinking mode needs the reasoning of
  the current tool loop sent back. An agent run has no new user message, so the whole run
  is one loop.
  - In the four-bar run, the 31.6k reasoning tokens of turn 3 were re-sent on every later
    turn: input went from 4.3k to 36k tokens.
  - The cache made it cheap (98-100% cached), but it is the largest cost in context.
  - The old context estimate ignored it; it now counts it. This is step 9's first target.
- **Wasted reads.** The drawer and four-bar runs spent 2-3 turns reading `docs/sdk/README.md`
  and `docs/sdk/quickstart.md`, which are already in the context. Step 10: say so in the
  prompt.

The three fixes (the cut-off hint, the honest context estimate, the quickstart wording) came
after these runs.

### How Articraft does it (study §8)

- **File:** `agent/harness.py` (`Agent._run_created`, `_run_tool_calls`, reminders).
- **Turns:** up to 100. Each turn: compact if needed, query, append to memory and to
  `conversation.jsonl`, run the tool calls.
- **Success:** visible text with no tool calls, and a workspace SHA-256 equal to the last
  successful compile's. Otherwise a `<compile_required>` or `<final_response_required>`
  reminder.
- **Endings:** 3 empty replies end the run. There are no loop-level retries; a failed query
  ends with `status=error`.
- **No budgets:** no cost, time or repair budget.
- **"Compiled" is not "good":** its diagnostics never block, so a passing compile can still
  have overlaps or floating parts.

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| Success also needs the latest compile to pass the blocking checks (step 4), not only to have compiled | Otherwise "done" can mean a door through its frame |
| Explicit end states: `failed_checks`, `budget_exceeded`, `no_progress` (identical failures, refused finishes, empty replies) | A run that cannot converge stops early, and the record says why |
| The best build is always named: last passed, else last failed | A stopped run still leaves something to look at and to repair |
| Per-run cost and call budget (step 6) enforced in the loop | Articraft has none |
| `record.json` with per-turn context size, cached share and cost | Step 9 needs numbers to cut tokens |
| Consecutive reads run in parallel; images only for models that accept them | Speed; text-only models still get the bounding boxes |
| A reply cut off at the output limit is named as such | A truncated tool call otherwise looks like a JSON bug |

### Open ideas

- **Step 9 (context):**
  - drop or summarize old reasoning (the run is one long tool loop);
  - cut stale tool results (old file reads, superseded compile signals);
  - lower `reasoning_effort` for routine turns.
- **Step 10 (prompts):**
  - tell the model the quickstart is already in its context;
  - a stricter plan-first step;
  - a different quickstart example from any evaluation prompt.
- **Time budget:** the four-bar spent 361 s on one reply. A per-run wall-clock cap would
  bound runs.
- **Repair stages:** explicit plan → build → check → fix stages (study §8 idea). Today the
  prompt asks for them, but the loop does not enforce them.
- **REQUIRED before any OpenAI evaluation:** the Responses API adapter (see step 6).

## Step 9: Context and memory

Goal: fewer tokens per run without lowering success, measured on 4 prompts × 2 runs, before
and after. The step 8 follow-up fixes (quickstart wording, cut-off hint, context estimate) were
already in the step 8 commit (a6c4723); nothing was left uncommitted.

### What was built

| File | Lines | What it does |
| --- | --- | --- |
| `agent/context.py` | 311 | `ContextPolicy`: the sent view of the conversation, rules a-e, hit counts |
| `agent/lessons.py` | 78 | Cross-run lessons (`runs/lessons.md`), off by default |
| `agent/loop.py`, `agent/__main__.py` | (changed) | The policy applied before every query; checkpoints; `EffortPolicy`; wall-clock budget (`time_exceeded`); `--context`, `--effort`, `--max-time`, `--lessons` |
| `llm/*` | (changed) | Per-call `effort` / `max_output_tokens` overrides; empty assistant messages are not sent |
| `scripts/context_report.py` | 127 | Before/after tables from `record.json` files |
| `scripts/deepseek_reasoning_probe.py` | 84 | A live check of which reasoning send-back variants DeepSeek accepts |
| `tests/test_context.py` | 307 | 16 tests: each rule, pairing, prefix, saved history, effort, time, lessons |

**The policy** (`ContextPolicy.view(history, turns, turn)`, applied before every query). The
full history is unchanged and still saved to `conversation.jsonl`.

| Rule | What it does | When |
| --- | --- | --- |
| a. checkpoints + reasoning | A compile's signals are sent as a user message, `<checkpoint turn=N compile=error primary=validation version=none>` + the signal block, after the tool results. The compile tool result says "[in the <checkpoint> message ...]" | After every compile |
| b. pinned state | Per workspace file, only the latest snapshot (a `read` result or a `write`'s content) stays in full; older ones become "[older read of main.py, superseded at turn N]" / "[older content of main.py, ...]". Only the latest compile signals stay in full | At rebuilds |
| c. docs in context | `read docs/sdk/quickstart.md` returns "docs/sdk/quickstart.md is already in your context (the second system message) ..." | When the tool runs |
| d. old outputs | Tool results older than 6 turns: the first 5 lines + "[... shortened: output from turn N; call the tool again ...]". `probe` and `render` are kept in full, and so is the latest snapshot of a workspace file | At rebuilds |
| e. compaction | Above 40k estimated tokens, the middle of the conversation (after the task, before the last 4 turns, cut at an assistant message) goes to the cheap model (deepseek-flash), on the same budget. The summary has 4 headings: Tried / Failed / Works / Next. A `<checkpoint kind=summary turns=a-b>` holds it, plus the current main.py read from disk | When over the threshold |

Rule a and DeepSeek's reasoning:
- `to_chat` already sends reasoning only after the last user message.
- So a checkpoint user message starts a new turn, and the finished tool loop's reasoning is
  no longer sent.

**Cache-friendliness.**
- **The prefix** (system prompt, quickstart, task, tool schemas) never changes. Every test
  checks it.
- **Rewriting only at rebuilds.** Rules b, d and e are applied only when the view is
  rebuilt, at a checkpoint or a compaction. Between checkpoints the view only grows at the
  end, which a test checks.
- **The price:** each rebuild invalidates the cache from the first changed message on.
  Uncached input went from 3.5k to 7.8k tokens per run (cached share 97.8% → 88.9%).

**Output limits.**
- **`EffortPolicy`:** once the latest compile passed (verifying, probing, finishing), calls
  get `reasoning_effort="low"` and `max_tokens=16000`. Planning and fixing turns keep the
  model's default.
- **Wall clock:** a 15-minute budget per run, checked before every query. A new end state,
  `time_exceeded`.

**Cross-run lessons** (`--lessons`, **off** for all measurements):
- **What is learned:** after a run, the primary issue of each failed compile of kind
  runtime_error / validation / syntax / import is added to `runs/lessons.md`. For example:
  "AttributeError: 'Box' object has no attribute 'translate' (check the SDK quickstart for
  the right method or argument)". Duplicates are dropped and at most 15 are kept.
- **How it is used:** the next run appends them to the system prompt under "Pitfalls seen
  in earlier runs".
- **What is left out:** check failures (collisions, floating parts) are one object's design,
  not SDK pitfalls.

**DeepSeek's reasoning rule, verified live.** The thinking-mode guide, which holds the rule,
did not load (HTTP 525 earlier, and the related reasoning-model page now gives 404). The
multi-round and changelog pages do not state it. So `scripts/deepseek_reasoning_probe.py` asked
the API directly with deepseek-v4-pro: one `add(2, 3)` tool call, then 4 follow-up variants.

| Variant | Result |
| --- | --- |
| `echo`: reasoning kept on the tool-call message | 200, "2 + 3 = **5**" |
| `checkpoint_drop`: a user checkpoint follows, reasoning dropped (rule a) | 200, "2 + 3 = **5**" |
| `drop_in_loop`: reasoning dropped with no user message | 200, "2 + 3 = **5**" |
| `old_echo`: old reasoning still sent after a user message | 200, "2 + 3 = **5**" |

So rule a is accepted, and in fact every variant is; whether quality differs was not
measured. The live runs below also used rule a on every compile without an error from it.

### Measurement (DeepSeek builder = deepseek-v4-pro, $1 cap, 2026-11-23 17:35-18:10 UTC, off-peak)

Before: `runs/step9-baseline/` (the step 8 loop). After: `runs/step9-after/` (`--context v1
--effort adaptive`, 15-minute cap). 2 runs per prompt.

| Prompt | Success | Turns | Input tokens / run | Peak input | Cost / run | Time / run |
| --- | --- | --- | --- | --- | --- | --- |
| a drawer that slides out of a cabinet | 2/2 → 2/2 | 4.5 → 9.5 | 56,526 → 81,256 | 16,599 → 17,088 | $0.0278 → $0.0418 | 128 → 162 s |
| a four-bar linkage | 2/2 → 2/2 | 9.0 → 9.5 | 190,452 → 82,826 | 38,692 → 28,732 | $0.0706 → $0.0688 | 331 → 329 s |
| a desk lamp with two hinged arms | 2/2 → 1/2 | 8.5 → 5.5 | 304,555 → 61,994 | 46,000 → 21,410 | $0.0900 → $0.0789 | 423 → 416 s |
| a toolbox with a hinged lid and a sliding tray | 2/2 → 2/2 | 6.0 → 7.5 | 95,856 → 54,687 | 20,823 → 16,019 | $0.0352 → $0.0334 | 164 → 150 s |

| Batch | Success | Turns | Input tok / run | Cached | Output tok / run | Peak input (mean / max) | Cost / run | Cost total | Output share of cost | Time / run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before | 8/8 | 7.0 | 161,847 | 97.8% | 25,290 | 30,529 / 61,753 | $0.0559 | $0.447 | 90% | 262 s |
| after | 7/8 | 8.0 | **70,191 (−57%)** | 88.9% | 24,854 | **20,812 (−32%) / 35,295 (−43%)** | $0.0557 | $0.446 | 88% | 264 s |
| after, desk lamp re-run after the fix below | 2/2 | 10.0 | 131,235 | 87% | 42,094 | 28,816 / 29,114 | $0.0979 | $0.196 | - | 438 s |

**Policy hits** (all 8 after-runs):
- a_checkpoints 13, a_reasoning_dropped 39 (assistant messages whose reasoning was no
  longer sent);
- b_pinned_stubs 13;
- c_doc_notes 4: the quickstart was re-read in 4 of 8 runs;
- d_shortened 0: runs are 5-11 turns, so few outputs reach 6 turns old after rules a and b;
- e_compactions 0 (none over 40k);
- effort: 39 default turns, 25 low.

In the 2 re-runs: a 3, a_reasoning_dropped 9, b 2, c 2, **e_compactions 1**.

What the numbers say:
- **Context: cut as intended.** Input tokens per run −57%; peak input −32% on average and
  −43% at the maximum (61.7k → 35.3k). The biggest saving is rule a. Before, every
  reasoning block since the task was re-sent on every later turn (whole planning turns of
  20-30k reasoning tokens).
- **Cost: unchanged ($0.0559 → $0.0557 per run).**
  - 88-90% of the cost is output tokens, almost all of it reasoning, and 92% of the
    reasoning is in the turns before the first compile: 21.3k tokens per run before, 19.3k
    after. Planning and writing turns run at the default effort.
  - The input was already 98% cached at $0.022-0.044 per 1M tokens. The rebuilds doubled
    uncached input (3.5k → 7.8k tokens per run), which offsets the smaller context.
  - The low-effort routine turns saved nothing measurable: the 25 low turns averaged 118
    reasoning tokens, and those turns were already cheap.
- **Turns: +1 on average.** The drawer took 4.5 → 9.5 turns (1 extra failed compile, and 1
  run with 2 passing compiles). The desk lamp's average includes the errored run. With 2
  runs per prompt this is within noise; no clear effect on success or turns.
- **Time: unchanged (262 → 264 s).** It is dominated by reasoning latency (planning turns of
  200-350 s).
- **time_exceeded:** never fired; the longest run was 603 s.

**The one failure was a pre-existing bug, now fixed.** In a desk-lamp after-run:
- **Turn 3:** the model reasoned for 32,000 tokens (352 s), hit the output cap, and returned
  no text and no tool call.
- **Then:** the loop added a reminder, so that empty assistant message lost its reasoning on
  the next request. DeepSeek rejected it: HTTP 400 "Invalid assistant message: content,
  tool_calls or reasoning_content must be set".
- **Not the policy:** it is a `to_chat` issue and could have happened in the baseline too.
- **Fixes:**
  - `to_chat` skips an assistant message with no content, no tool calls and no echoed
    reasoning;
  - an empty cut-off reply now gets `<reply_cut_off>` ("plan in a few lines, then call write
    ...") instead of `<compile_required>`;
  - 2 regression tests.

Two re-runs of that prompt then succeeded. One of them compacted live:
- **When and why:** at turn 5, reasoning echoed inside the current tool loop (21k + 23k)
  pushed the estimate to 43.6k.
- **What it did:** deepseek-flash summarized turn 1 for $0.0004, and the request went out at
  24.3k.
- **The cost:** the rewrite broke the cache (12% cached that turn), which cost more than the
  summary saved.

### Tests (16 new, no network)

- **a:**
  - the checkpoint follows the compile result, which points to it;
  - no reasoning is sent from finished loops, and the current loop's reasoning still is;
  - hit counts;
  - the saved conversation keeps the checkpoint, the full signals and all 3 reasoning
    blocks.
- **b:**
  - after 2 reads, 1 write and 3 compiles, only the latest read and the latest signals stay
    whole, with "superseded at turn N" stubs;
  - nothing is stubbed between checkpoints;
  - the saved history still has the original write.
- **c:** the quickstart read gives the note; `docs/sdk/README.md` is still read; the path
  forms `./`, `\`.
- **d:**
  - a turn-2 compile result is cut to its first lines plus a stub;
  - a probe result and a turn-4 result are kept.
- **e:**
  - a scripted cheap model writes the summary, which holds the current main.py from disk;
  - the summary call is on the run's budget;
  - no summarizer means compaction is skipped.
- **Every view:** `validate()` (the tool-call/result pairing) and the same first 3 messages,
  byte-identical in `to_chat`. Between checkpoints, each view is a prefix of the next.
- **Record:** the policy config, hits, summaries, and both context sizes (sent vs history).
- **Effort:** the first 4 calls `{}`, then `{"effort": "low", "max_output_tokens": 16000}`
  once the compile passed.
- **`time_exceeded`:** with a stubbed clock.
- **Lessons:**
  - off: nothing is learned;
  - on: the `Box.translate` error is learned once, and shows up in the next run's system
    prompt.
- **No policy:** the full history is sent.
- **Regression:** an empty cut-off reply is never sent as an empty assistant message.

**336 tests pass, 1 skipped** (16 new here, plus 1 in `test_llm.py`).

### How Articraft does it (study §9)

- **Files:** `agent/compaction.py`, `prompts/compaction.md`, and the provider's
  `summarize_context` (Anthropic's native compaction beta).
- **History:** append-only and verbatim. Messages 0-2 (system prompt, quickstart, task) stay
  forever.
- **Trimming:** nothing is trimmed per turn. Only tool outputs are kept small (compile
  returns only its signals; exec output is capped at 10k tokens).
- **Compaction:** it starts above the window minus 16k (about 256k tokens) and keeps the
  newest 20k tokens or more. The rest is summarized with fixed headings, with tool outputs
  clipped to 2,000 characters; the checkpoint lists the modified files.
- **Docs:** read on demand. **Cross-run memory:** none.
- **Weak points:**
  - superseded main.py reads, old images and old compile reports are billed again every
    turn;
  - typical runs never reach 256k, so they never compact.

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| Compile signals as checkpoint user messages; finished loops' reasoning not re-sent | DeepSeek's thinking mode re-sends reasoning for the whole tool loop, which in an agent run is the whole run. This was the largest part of the context (−57% input) |
| Pinned state: the latest file snapshot and the latest compile signals only | Articraft's weak point: superseded reads and reports billed every turn |
| A note for re-reading a doc already in context | Seen in 4 of 8 runs |
| Rewrites only at checkpoints | Keeps the cached prefix as long as possible between checkpoints |
| Compaction at 40k (configurable), by the cheap model, with the current main.py from disk | Articraft's 256k threshold is never reached in practice. The file is copied, not summarized, so it cannot be misremembered |
| Wall-clock budget, `time_exceeded` | One planning turn can take 350 s |
| Cross-run lessons file, off by default | Articraft has no cross-run memory. It stays off until it can be measured on its own |

### Open ideas

- **The real cost lever is planning-turn reasoning:** 92% of reasoning comes before the first
  compile. Try `reasoning_effort` for the planning turn (low vs high), a lower output cap
  with the `<reply_cut_off>` guidance, or a brief explicit plan in the prompt (step 10).
  Measure success along with the cost.
- **Drop reasoning without waiting for a checkpoint:** the probe shows DeepSeek accepts
  reasoning dropped inside a loop (`drop_in_loop`). Sending only the latest reply's reasoning
  would have avoided the live compaction. Quality is not measured yet.
- **Fewer rebuilds:** batch the rule b and d rewrites into every 2nd or 3rd checkpoint, to
  win back the cache (uncached input doubled).
- **Compaction trigger:** count the reasoning that rule a is about to drop before deciding
  to compact.
- **More runs per prompt:** with 2 runs, the drawer's 4.5 → 9.5 turns cannot be told from
  noise. Use 5 or more for step 12.
- **REQUIRED before any OpenAI evaluation:** the Responses API adapter (step 6).

## Step 10: System prompt and SDK docs

Goals: (1) less planning reasoning (92% of reasoning came before the first compile in step 9);
(2) more realistic, detailed objects; (3) no wasted turns. Each is measured below.

Policy defaults are unchanged: context policy v1 and adaptive effort stay on.

### What was built

| File | What it is |
| --- | --- |
| `prompts/system.md` | System prompt v1: 69 lines, 4,359 characters, **1,103 tokens** (DeepSeek's count) |
| `prompts/v0/` | Step 9's system prompt and docs, frozen for comparison (variant A) |
| `docs/sdk/quickstart.md` | Cheat sheet plus a complete example, a mailbox with a hinged flap and a flag (replaces the laptop) |
| `docs/sdk/shapes.md` | Every ported shape helper: signature, one line of use, a small example |
| `docs/sdk/joints_and_loops.md` | Joint frames and the axis sign rule, motion patterns, clearance, mimic joints, closed loops: a four-bar with `frame_in` |
| `docs/sdk/testing.md` | The `run_tests(ctx)` API, what is already checked, tolerances, allowances; a floor hatch on a pin hinge |
| `docs/sdk/README.md` | Index of the pages |
| `prompts/eval_prompts.json` | Every evaluation prompt so far (steps 8, 9, 10) |
| `agent/quality.py` | Quality metrics of a build, and the final renders |
| `agent/loop.py` | Prompt sets (`PROMPT_SETS`: v0 and v1, each a system prompt, quickstart, docs folder and task message); `EffortPolicy.planning_effort`; new record fields |
| `agent/context.py` | Rules c and d extended to docs pages (below) |
| `agent/__main__.py` | `--prompt-set v0/v1` (default v1), `--planning-effort default/low/medium/high` |
| `scripts/step10_experiment.py`, `scripts/step10_report.py` | The A/B/C experiment, and its tables plus render folder |
| `scripts/doc_tokens.py` | Real token sizes of the prompt and docs (cheap model, output capped at 1 token) |
| `tests/test_docs.py`, `tests/test_quality.py` | 22 + 10 tests, plus 2 in `test_context.py`, 1 in `test_checks.py` and 3 in `test_export.py` |

**System prompt v1.** Its size is about the same as v0 (1,103 vs 1,102 tokens), with the content shifted:
- **Docs.** The quickstart is already in context, so it is never re-read. The 3 on-demand pages are listed with one line each.
- **Realism (rule 1, rewritten).** Real proportions, and the details that make an object recognizable: handles, knobs, hinges, rims, feet, rounded edges, real wall and panel thickness ("a cabinet is a hollow carcass of panels"). One line names the shape helper for each kind of real part (rounded box, lathe, extrude, extrude with holes, tube, wire). "Avoid one-box-per-part designs".
- **Plan step.** "Plan briefly, in your reply text, then write": at most about 15 lines (bodies with shapes and sizes, joints with type, parent -> child, pivot, axis, limits), then `write` in the same reply. "Do not deliberate at length before writing". The task message says the same.
- **Finish checklist.** The latest compile passed with no edits after it; every movable joint probed at both ends of its range; a short final message.
- **Moved out.** The `run_tests` API list moved to `testing.md`; the tool tips are shorter.

**Docs pages** (token counts by DeepSeek, `scripts/doc_tokens.py`):

| Page | Sent | Characters | Tokens |
| --- | --- | --- | --- |
| `prompts/system.md` (v1) | always | 4,359 | 1,103 |
| `docs/sdk/quickstart.md` | always (second system message) | 5,745 | 1,820 |
| `docs/sdk/shapes.md` | on demand | 5,685 | 1,923 |
| `docs/sdk/joints_and_loops.md` | on demand | 4,822 | 1,471 |
| `docs/sdk/testing.md` | on demand | 3,930 | 1,236 |
| `docs/sdk/README.md` | on demand | 419 | 111 |
| v0: `prompts/v0/system.md` + quickstart | always | 4,356 + 4,934 | 1,102 + 1,513 |

The code-heavy pages tokenize at about 3 characters per token, not 4. So the size test uses
2.9 characters per token, below every measured ratio (2.96–3.95); with 4, a page could have
grown to about 2.6k real tokens and still passed.

**Quality metrics** (`record.json` → `quality`, computed by code from the best build's
`model.json` and its `main.py`):
- bodies, shapes, joints (loop closures included) and movable joints;
- `non_box_share`: the share of shapes that are not `Box` or `BoxGeometry`;
- `helpers`: the distinct SDK shape helpers `main.py` refers to outside its imports. This is every geometry constructor or mesh function except the plain primitives (Box, Cylinder, Sphere and their mesh forms), so a `rounded_rect_profile` feeding an extrusion counts;
- triangles; the count of each shape type.

**Other new record fields:**
- `prompt_set`;
- `first_compile_turn`, `reasoning_before_first_compile` and `output_before_first_compile` (tokens up to and including the turn that first calls compile);
- `doc_reads` (path → count);
- `reminders` (kind → count);
- `final_renders`: the best build at rest, front/side/top/iso plus a 2x2 sheet, in `renders/final/`.

**Rules c and d for docs pages** (`agent/context.py`):
- **Before:** rule d shortened every tool result older than 6 turns, docs reads included. In the first smoke run the model then read `shapes.md` 3 times.
- **Now:** the latest read of each file or docs page stays whole, and reading a page again returns a note ("already in your context: your read of it at turn N, kept in full").
- **After a compaction:** a page read in the summarized part can be read again.

### Four SDK bugs, found while writing the docs, smoke-testing and running the experiment

1. **Mesh health crashed on any open shape** (`sdk/testing.py`). The debris check split each shape with trimesh's default `repair=True`, which fills holes with networkx, and networkx is not a dependency. So a wire frame drawn as a closed polyline without `closed_path` compiled to `import_error: No module named 'networkx'`, with the hint to fix the imports. It was found in the first live smoke run, where the model fought this error:
   - it wrote a one-box probe file to test whether the SDK worked at all;
   - the probe passed and became the run's "best build";
   - the run then hit `time_exceeded` (34 turns, $0.28).

   The fix is `repair=False`; the shape now gets the "is open" warning it should. A regression test fails on the old code with the same error.
2. **Export collapsed thin walls** (`sdk/export.py`). The collision-mesh simplifier doubled its tolerance until the mesh fit the triangle budget. On a 3 mm sheet-metal shell that erased the walls: "simplify produced an empty result", an `export_error` on a model that had passed every check. It was found writing the mailbox example. The fix stops at the last result that keeps the volume within 10%, or keeps the mesh unsimplified (over budget). On the test shell the old code kept 1.5% of the volume (32 triangles).
3. **The viewer loader** (`viewer/loader.py`, used by `render`, `probe` and the viewer) had the same `split()` repair. It crashed on builds with an open piece; fixed the same way, with a regression test.

4. **Thousands of one-triangle shapes in the loader** (`viewer/loader.py`), found in the
   experiment. Once bug 3 was fixed, the split no longer crashed, but a body mesh with
   zero-area triangles (a `RoundedBoxGeometry` writes about 15k triangles, a few of them
   degenerate) split into one-triangle fragments, and each one became a shape. One B cabinet
   build loaded as 35,095 shapes. The probe's collision scene compares shapes pairwise in
   Python, so one `probe` call took 13 minutes, and the run hit `time_exceeded` with a
   passing build. The fix keeps the closed pieces as shapes and merges every open piece into
   one shape named `open`. The same probe now takes 25 s, load included. A regression test
   covers it: 2 boxes plus 3,000 fragments load as 3 shapes; the old code gives 3,002.

Bugs 1-3 existed before this step; bug 4 appeared once bug 3 was fixed. They matter more now than before, because the new prompt pushes hollow, thin-walled parts and helper shapes. Bugs 1-3 were fixed before the experiment. Bug 4 was fixed during it: the one run it broke was re-run, and no other run spent more than 39 s outside model calls.

### Measurement (DeepSeek builder = deepseek-v4-pro, $1 cap, 2026-11-24 03:57-06:31 UTC)

- **Setup:** 4 new prompts × 3 runs × 3 variants, 4 runs at a time, jobs interleaved
  (repetition, then prompt, then variant). Command:
  `scripts/step10_experiment.py --runs 3 --jobs 4`; tables from
  `scripts/step10_report.py`, saved in `runs/step10/report.md`.
- **The variants:**
  - **A:** step 9's system prompt and docs;
  - **B:** the v1 prompt and docs;
  - **C:** B, plus `reasoning_effort=low` until the first compile.

  All three use context policy v1, adaptive effort and the 15-minute limit.
- **Excluded and re-run: 5 of the 36 runs.** They were broken by something outside the
  variant:
  - **The laptop went into Modern Standby at 05:12-05:34 UTC** (Windows System log: a power
    source change, network disconnected, sleep). 3 runs ended with
    `APIConnectionError` after 5 retries, and 1 compile sat frozen for 903 s.
  - **Bug 4 above:** 1 run (B cabinet) lost 13 minutes to one probe.

  These 5 runs are in `runs/step10/excluded/`, with the reasons in
  `excluded/reasons.json`. They were re-run after the loader fix, with
  `--redo VARIANT:PROMPT:REP`. The experiment script now also asks Windows not to sleep
  while it runs (`SetThreadExecutionState`).
- **No tool time left after that:** in every counted run, the time outside model calls is
  at most 39 s.
- **Spend:** $4.73 in all: $4.58 for the counted runs and $0.15 for the excluded ones.

**Per variant (12 runs each):**

| Variant | Success | Turns | Failed compiles | Quickstart re-reads | Doc page reads | Reasoning before 1st compile (mean / median) | Planning turn (median) | Reasoning after 1st compile (mean) | Reasoning / run | Cost / run | Cost / success | Time / run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: step 9 prompt + docs | **9/12** | 8.8 | 1.1 | 4 | 14 | 28,272 / **15,178** | 12,017 | 7,550 | 35,822 | **$0.106** | **$0.141** | **528 s** |
| B: v1 prompt + docs | 7/12 | 9.6 | 1.8 | **0** | 21 | 31,181 / 24,515 | 13,021 | 15,895 | 47,076 | $0.136 | $0.233 | 673 s |
| C: B + low effort before the 1st compile | 7/12 | 10.6 | 2.0 | **0** | 19 | 27,536 / 24,136 | **7,456** | 27,478 | 55,015 | $0.140 | $0.240 | 701 s |

- **"Planning turn"** is the reasoning of the turn that first writes `main.py`.
- **Runs that never compile** count all their reasoning as "before the 1st compile", so the
  median is the fairer number.
- **End states:**
  - A: 9 success, 3 time_exceeded;
  - B: 7 success, 5 time_exceeded;
  - C: 7 success, 4 time_exceeded, 1 no_progress.

**Quality of the successful builds** (means; triangles: median):

| Variant | Builds | Bodies | Shapes | Joints | Non-box shapes | Helpers / run | Triangles | Helpers used (any run) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 9 | 3.0 | 10.2 | 2.0 | 46% | 0.4 | 584 | Extrude, RoundedBox |
| B | 7 | 3.1 | **14.0** | 2.1 | **78%** | **2.3** | 30,944 | Extrude, ExtrudeWithHoles, Lathe, RoundedBox, boolean_difference, boolean_union, sample_catmull_rom_spline_2d, wire_from_points |
| C | 7 | 3.6 | 13.9 | 2.9 | 67% | 2.4 | 31,024 | Capsule, Extrude, ExtrudeWithHoles, Lathe, RoundedBox, rounded_rect_profile |

**Per prompt** (success; median reasoning before the 1st compile is in the report):

| Prompt | A | B | C |
| --- | --- | --- | --- |
| kitchen cabinet with two doors and a drawer | 3/3, $0.072 | 2/3, $0.149 | 3/3, $0.095 |
| folding table | 3/3, $0.121 | 3/3, $0.111 | 1/3, $0.170 |
| crank-slider pump | 0/3, $0.197 | 0/3, $0.180 | **2/3**, $0.163 |
| pair of pliers | 3/3, $0.034 | 2/3, $0.104 | 1/3, $0.132 |

**Renders:** `runs/step10/renders/` holds:
- `<prompt>__iso.png`: one sheet per prompt, a row per variant, a column per run, each cell
  labelled with status, turns and cost;
- `<prompt>__<variant>-<n>.png`: each run's 4 views.

What they show:
- **Folding table: clearly better in B and C.** All three A tables are a flat top with their
  legs folded under it at rest. Most B and C tables stand on tubular or panel leg frames;
  B-2 has cross-braced tube frames.
- **Pump: more detail in B and C.** Rounded bases, bored cylinders, crank discs and pins;
  C-3 is a recognizable crank-slider.
- **Pliers: mixed.** B-1 has curved handles and a pivot pin; the others are mostly bars.
- **Cabinet: alike in all variants.** The iso camera shows the back or side of most
  cabinets, so the doors hardly show.

**What the numbers say, goal by goal:**

1. **Less planning reasoning: not achieved.**
   - **The written plan did not shorten planning.** Median reasoning before the 1st compile
     went up: A 15.2k → B 24.5k → C 24.1k.
   - **Where that reasoning happens:** before the write turn. In most runs, turn 1 reasons
     8-25k tokens and then reads docs, or reads docs first and reasons on the next turn.
   - **Low effort shortens the write turn** (median 13.0k → 7.5k), but not the turns before
     it.
   - **It moves reasoning rather than saving it.** C's worse first drafts take more fixing:
     reasoning after the 1st compile is 27.5k per run, against 15.9k (B) and 7.6k (A).
   - **The planning share of reasoning fell (79% → 66% → 50%),** but only because the fixing
     grew.
2. **More realistic, detailed objects: achieved, measured by code and seen in the renders.**
   - Non-box shapes 46% → 78% (B) and 67% (C).
   - Helpers per build 0.4 → 2.3-2.4.
   - Shapes 10.2 → 14.0.
   - The only variant with any pump success is C (2/3).

   The cost: triangles rose 50x (median 584 → 31k), and complex geometry fails more checks
   (failed compiles 1.1 → 1.8-2.0).
3. **No wasted turns: mixed.**
   - **Gained:** quickstart re-reads 4 → 0, and reminders 0.5 → 0.2.
   - **Lost:** turns rose 8.8 → 9.6 (B) and 10.6 (C), from failed compiles.
   - **Doc page reads 14 → 21:** the on-demand pages are read as intended, but each read
     costs a turn.
- **Success and cost:** B and C have 2 fewer successes than A (7 vs 9 of 12). Cost per
  success is $0.141 → $0.233 / $0.240, and runs take 30% longer. With 3 runs per cell,
  differences of 1 success per prompt are noise; the cost and reasoning shifts are
  consistent across prompts.

### Tests (38 new, no network)

- **Docs (`test_docs.py`, 22):**
  - the system prompt and each page are ≤ 2k tokens (2.9 characters per token, below every
    ratio DeepSeek measured);
  - the system prompt lists the 3 on-demand pages, one line each, and says the quickstart
    is in context;
  - only python/text code blocks;
  - every python block runs: the 3 complete programs (mailbox, four-bar, floor hatch) go
    through the real compiler with `warnings=0`, and every snippet's meshes are closed;
  - every call named in a text block or inline code exists in the SDK;
  - the quickstart example is the mailbox and matches no evaluation prompt; no docs program
    builds an evaluation object, except the four-bar, which is allowed by name;
  - the v0 prompt set is unchanged.
- **Quality (`test_quality.py`, 10):**
  - known examples: the mailbox gives 3 bodies, 8 shapes, 2 joints, 75% non-box and 4
    helpers; the laptop is all boxes, 48 triangles;
  - an import alone is not a helper use;
  - `render_final` writes the 5 images;
  - the prompt sets and their docs folders;
  - the planning effort applies only before the first compile;
  - the record fields; no build means no quality.
- **Others:**
  - `test_context.py` (2): a docs page read is kept whole and noted on re-read;
  - `test_checks.py` (1): an open shape compiles without networkx;
  - `test_export.py` (3): the thin shell keeps its volume; an open piece loads without
    repair; fragments load as one shape.

**374 tests pass, 1 skipped.**

### How Articraft does it (study §10)

- **The system prompt:** 3,402 tokens (2,861 text-only) in seven sections: `<role>`,
  `<quality_requirements>` (4 rules), `<workflow>` (survey the docs → brief → build →
  preview → compile → repair), `<authoring_contract>`, `<testing>`, `<tools>`,
  `<final_response>`. It includes the "count the pivots" guidance for loops.
- **Always sent:** `task.md` (256 tokens), the quickstart as a user message (1,210 tokens)
  and 8 tool schemas (about 1,150 tokens).
- **On demand:** about 192k tokens of docs. `common/` is 15.8k, `mesh/` 20.4k, the examples
  4.9k, and about 150k are generic build123d docs. A run reads about 10-20k of them.
- **Weak points:** the text-only prompt repeats its "Finish only when…" line, the trajectory
  text is sent on every run, and most of the docs are generic.

### What we changed vs Articraft, and why

| Change | Why |
| --- | --- |
| A 1.1k-token system prompt (Articraft: 3.4k), with an explicit plan size (at most about 15 lines in the reply) and "do not deliberate at length" | To cut planning reasoning. Measured: it did not; the model deliberates in hidden reasoning whatever the prompt says |
| 4 curated pages, ≤ 2k tokens each, 6.5k tokens in all (Articraft: about 192k, mostly generic build123d) | Every example in them is compiled by a test, so the docs cannot drift from the SDK |
| The quickstart example is not any evaluation object (a mailbox), checked by a test | So that results measure the model, not a copied example |
| Realism rules that name the helper for each kind of part | Measured: non-box shapes 46% → 78%, helpers per build 0.4 → 2.3 |
| Quality metrics computed by code and saved in `record.json`, plus 4 final renders | Articraft has no quality metric in the run record; realism can now be compared across steps |
| Planning effort (`--planning-effort`) as an option, off by default | Measured: it shortens the write turn by 43% but moves the reasoning to fixing, at a higher total cost |

**Defaults after this step:** prompt set v1 (the more realistic objects are the goal of the
replica), planning effort off.

### Open ideas

- **Planning reasoning is not controlled by the prompt.** Try:
  - a hard output cap on the turns before the first compile (for example 12k), with
    `<reply_cut_off>` asking for the plan and the write;
  - giving the turn-1 reasoning nothing to search: put the page list's key signatures in the
    quickstart, so no read turn starts a second deliberation;
  - a cheaper planning model (deepseek-flash writes the plan, pro writes main.py).
- **Triangle budget.** Detailed builds are about 31k triangles, against 584. A
  `RoundedBoxGeometry` writes about 15k triangles even for a 5 cm foot. Scale its segments
  to the radius, and drop zero-area triangles at the source.
- **Rasterizer artifacts.** The final renders show dark triangular wedges on flat faces in
  every variant (also on plain-box builds). The CPU rasterizer needs a depth or winding
  fix.
- **Render from the front.** The iso camera often shows the back of the object (most
  cabinets). Pick the view from the model's own front, or the side the joints open toward.
- **More runs:** 3 runs per cell cannot separate 1 success from noise. Step 12 needs 5 or
  more.
- **REQUIRED before any OpenAI evaluation:** the Responses API adapter (step 6).

### Follow-up: failure analysis and renderer fixes (no API calls)

- **Failure analysis:** `docs/STEP10_FAILURES.md`, made with `scripts/step10_failures.py`.
  - 9 of the 10 B/C failures are `time_exceeded`; 2 of those had a passing build and were
    only verifying.
  - Planning is 57% of the time, and model calls of 200 s or more are 56%.
  - Most collision failures are main parts (45 of 75), not details.
  - 9 of 18 floating failures are the prompt's 1-2 mm clearance.
  - The ranked fixes: the time limit and end state, docs/prompt joint contact and recipes,
    soft-failing helpers.
  - It also found a compiler bug on Windows (`os.replace` of the progress file, then a cached
    error). Not fixed yet.
- **Renderer** (`agent/tools/render.py`):
  - **A per-pixel depth buffer** replaces the painter's sort by triangle centroid, which drew
    hidden faces over near ones: the dark wedges. It uses scanline spans, vectorized with
    numpy. Final renders of 30-120k-triangle builds take 1-2.5 s for 4 views.
  - **The views face the object's front.** That is the horizontal side whose front view
    (plus a quarter of the iso view) shows the largest area of moving bodies, in m². -Y stays
    the default unless another side shows 10% more. Labels name the real directions; with a
    -Y front the views are the old fixed ones.
  - **The light comes from the upper left,** so the front (the iso view's left face) is lit.
  - 7 tests in `tests/test_render.py`: a hidden triangle that centroid sorting drew on top,
    a box with exactly 3 visible faces, the default views equal to the old ones, the facing
    for a door on +Y or -Y, no moving parts → -Y.
  - The final renders and the sheets in `runs/step10/renders/` were regenerated.
- **381 tests pass, 1 skipped.**

## Step 10b: fixes from the step 10 failure analysis

Built from `docs/STEP10_FAILURES.md`. Experiment run on 2026-11-24 (results below). **B2
is now the default setup** (prompt set v1, context policy v1, adaptive effort, the fixes
below); `--two-phase` stays an optional high-detail mode.

### What was built

| # | Change | Files |
| --- | --- | --- |
| bug | **Compiler on Windows.** Progress-file writes are best effort: 5 tries, then skipped. A failure whose traceback has no frame in the workspace (and is not a syntax or SDK error), or a crashed worker, is "internal": its message says so, its hint is "compile again without changes", and it is **never cached** | `compiler/worker.py`, `compiler/runner.py` |
| 1 | **Time limit 25 min** (was 15). When it runs out and the latest compile passed with no edit since: one final turn for the summary (tool calls in it are not run), and the run is a **success** either way. **Fix reserve:** a fix cycle (the turn after a failed compile) does not start with less than 120 s left. `record.json`: `time_limit` (`final_turn` / `stopped` / `no_fix_cycle`), `fix_reserve_s` | `agent/loop.py`, `--max-time`, `--fix-reserve` |
| 2 | **Contact rule** in the prompt (rule 3: joined parts touch, gap 0, or a pin in a barrel with `allow_overlap`; rule 4: clearance only for parts that move past each other). `joints_and_loops.md`: a "Contact and clearance" section and **two recipes as complete programs**: a door on a carcass with 18 mm panels (`wall_cupboard`), and nesting folding legs (`camp_bench`). Both compile and pass every check, swept collisions included, in `test_docs.py`. The page was compacted to make room: the patterns table became one bullet, and the four-bar text was shortened | `prompts/system.md`, `docs/sdk/joints_and_loops.md` |
| 3 | **Soft-failing helpers:** `RoundedBoxGeometry` clamps a radius of half the smallest size or more to 0.45 × that size and leaves a notice; the compiler turns notices into `helper_adjusted` **warnings** at the calling line of main.py (`sdk/notices.py`). A self-intersecting profile names the crossing edges, their points and the crossing point; it is checked before ear clipping, which sometimes "succeeded" on such outlines. `ExtrudeWithHolesGeometry` names the hole outside the outline (with a point) or the two overlapping holes. A duplicate shape name suggests `f'foot_{i}'` | `sdk/primitives.py`, `sdk/mesh_geometry.py`, `sdk/solids.py`, `sdk/body.py`, `sdk/notices.py`, `compiler/feedback.py` |
| 5 | **`--two-phase`:** the task asks for a skeleton (plain primitives at real sizes, the real joints; plan in at most about 10 lines). Its first passing, fresh compile (every check, the motion checks included) adds a `<phase_2>` message asking for detail in groups, compiling after each. The skeleton's version stays the best build unless a phase 2 compile passes. The detail-writing turns are not "routine" (no low effort, no 16k cap) until phase 2 compiles. `record.json`: `two_phase` (phase, skeleton version/turn/minutes, detail compiles) | `agent/loop.py` |
| 6 | **`scripts/step10b_experiment.py`:** B2 (v1 + fixes) and D (B2 + `--two-phase`), the 4 step 10 prompts × 3 runs, 4 at a time, keep-awake, stop on HTTP 402, `runs/step10b/`. `step10_report.py` takes `--folder` and `--variants`, and adds a time-limit and two-phase table | `scripts/` |

**Sizes** (2.9 characters per token, the test's conservative count; real DeepSeek counts not
measured, since that needs an API call):
- `prompts/system.md`: 4,680 characters, at most 1,614 tokens (about 1,190 at step 10's
  measured ratio);
- `joints_and_loops.md`: 5,718 characters, at most 1,972 (about 1,740).

### Tests (27 new, no network): 408 pass, 1 skipped

- **Compiler (4):**
  - a compile passes while every progress write is denied;
  - internal failures are told apart from main.py's;
  - internal failures and crashes are not cached;
  - normal failures still are.
- **Loop (12, ScriptedModel with a fake clock):**
  - the defaults;
  - the final turn at the limit (with text; with tool calls, which are not run);
  - `time_exceeded` with a failing compile, or with an edit after the pass;
  - the fix reserve stops and starts;
  - two-phase: the skeleton → detail messages, the fallback to the skeleton build, a failing
    skeleton staying in phase 1, and the effort of the detail turns;
  - no change without the flag.
- **SDK (8):**
  - the radius clamp and its notice;
  - the warning at the calling line through the real compiler;
  - crossing edges named;
  - concave profiles still fine;
  - a hole outside the outline, overlapping holes, and good holes;
  - the duplicate-name hint.
- **Docs (+2):** the 2 recipe programs, compiled.

### How Articraft does it

- **Time:** Articraft has no wall clock. Its loop runs up to 100 turns, with "no cost, time
  or repair budget" (study §8), so it never ends a passing build on time.
- **Helpers:** its SDK helpers raise on bad arguments, like ours did. There is no "notice"
  channel; warnings come only from the checks.
- **Phases:** there is no two-phase mode. Its workflow asks for a brief, then the build, in
  one pass. The study's replica idea was "explicit stages (plan → build → check → fix)"
  (§8); the two-phase option is one such stage split, measured on its own (variant D).

### Open ideas and caveats

- **The recipes overlap 2 of the 4 step 10 prompts.** They teach a door on a carcass and
  folding legs, which are parts of the kitchen-cabinet and folding-table prompts. The objects
  differ (a one-door wall cupboard, a bench), and the docs test still passes (no example is
  named after an evaluation object). Still, B2's gains on those 2 prompts may partly come
  from the docs. Compare the pump and pliers separately, and use new prompts in step 12.
- **Ear clipping is O(n³):** a 2,000-point outline takes about 30 s. The new crossing check
  is vectorized (0.17 s). A faster triangulation (or build123d's) would help dense spline
  outlines.
- **Phase 1 does not enforce "plain primitives":** a skeleton that uses helpers is accepted.
  `quality.helpers_used` could warn.
- **REQUIRED before any OpenAI evaluation:** the Responses API adapter (step 6).

### Measurement (DeepSeek builder = deepseek-v4-pro, $1 cap, 2026-11-24 10:07-11:38 UTC)

- **Setup:** the 4 step 10 prompts × 3 runs × 2 variants, 4 at a time, jobs interleaved,
  keep-awake on. Command: `scripts/step10b_experiment.py --runs 3 --jobs 4`; tables from
  `scripts/step10_report.py --folder runs/step10b --variants B2 D`, saved in
  `runs/step10b/report.md`. Render sheets: `runs/step10b/renders/` (one per successful run,
  one `<prompt>__iso.png` per prompt).
- **The variants:**
  - **B2:** step 10's B (prompt set v1) plus the fixes above (no flag needed);
  - **D:** B2 plus `--two-phase`.
- **No run was excluded:** no sleep, no connection errors, no HTTP 402.
- **Spend:** $3.12 (B2 $1.37, D $1.75).

**Per variant (12 runs each), with step 10 for comparison:**

| Variant | Success | Turns | Failed compiles | Reasoning before 1st compile (mean / median) | Cost / run | Cost / success | Time / run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A (step 10): step 9 prompt | 9/12 | 8.8 | 1.1 | 28,272 / 15,178 | $0.106 | $0.141 | 528 s |
| B (step 10): v1 prompt | 7/12 | 9.6 | 1.8 | 31,181 / 24,515 | $0.136 | $0.233 | 673 s |
| C (step 10): B + low planning effort | 7/12 | 10.6 | 2.0 | 27,536 / 24,136 | $0.140 | $0.240 | 701 s |
| **B2**: B + step 10b fixes | **10/12** | 10.3 | 1.2 | 24,889 / 20,986 | $0.114 | **$0.137** | 720 s |
| **D**: B2 + two-phase | 9/12 | 14.9 | 1.4 | 34,318 / 20,714 | $0.146 | $0.194 | 918 s |

- **End states:** B2: 10 success, 2 time_exceeded. D: 9 success, 3 time_exceeded. Every
  failure is the pump.
- **Two-phase (D):** the skeleton passed every check in 10 of 12 runs, after a median of
  5.8 min. One failed run (pump 3) has a passing skeleton as its best build: it passed at
  21.8 min, and the detail did not pass before the limit.
- **Time limit outcomes:** B2: 2 `stopped`. D: 2 `stopped`, 1 `no_fix_cycle`. No run needed
  the final turn: every success finished on its own.

**Quality of the successful builds** (means; triangles: median):

| Variant | Builds | Bodies | Shapes | Joints | Non-box shapes | Helpers / run | Triangles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B (step 10) | 7 | 3.1 | 14.0 | 2.1 | 78% | 2.3 | 30,944 |
| B2 | 10 | 3.1 | 13.9 | 2.2 | 75% | 1.7 | 31,728 |
| D | 9 | 3.2 | 15.6 | 2.2 | **89%** | **3.1** | 106,136 |

**Per prompt** (success, cost per run):

| Prompt | B (step 10) | B2 | D |
| --- | --- | --- | --- |
| kitchen cabinet with two doors and a drawer | 2/3, $0.149 | 3/3, $0.098 | 3/3, $0.130 |
| folding table | 3/3, $0.111 | 3/3, $0.047 | 3/3, $0.136 |
| crank-slider pump | 0/3, $0.180 | 1/3, $0.258 | 0/3, $0.220 |
| pair of pliers | 2/3, $0.104 | 3/3, $0.053 | 3/3, $0.098 |

**Reading:**
- **B2 is the best setup so far:** 10/12 at $0.137 per success, against B's 7/12 at
  $0.233 (and A's 9/12 at $0.141). It becomes the default.
- **Caveat (see above):** the new recipes match the cabinet and the folding table. On the
  two prompts they do not cover, the pliers went from 2/3 to 3/3; the pump stayed bad.
- **D buys detail, not success.** It has more shaped parts (89% vs 75% non-box) and more
  helpers (3.1 vs 1.7 per run), for 42% more cost per success and 28% more time per run. It
  stays as `--two-phase`, an optional high-detail mode.
- **The pump is the hard case, and it fails in planning.** Its runs spend 35k-95k reasoning
  tokens before the first compile (mean 46k in B2, 85k in D), against 7k-36k for the other
  three prompts. 5 of the 6 pump runs had 2-3 replies cut off at the 32k output limit, with
  no answer and no tool call. The first compile came after 9.4-24 minutes of model time.
  With one compile left to fix, the runs ran out of time. **The slider-crank is a target
  for a later step:** a solver or recipe that places the crank, rod and slider from a few
  numbers (crank radius, rod length, stroke axis), so the model does not reason the loop
  closure out by hand.
- **Two runs ran past the limit:** B2 pump 2 ended at 1852 s (+352 s) and D pump 2 at
  1592 s (+92 s). The limit was checked only between turns, and one cut-off reply takes
  7-9.5 minutes. Fixed below.

### Follow-up: each model call capped at the time left

- **What:** every model call gets the run's time left as its timeout, and an output cap the
  model can write in that time.
  - **Timeout:** `model.query(..., timeout_s=left)` covers the whole call, retries included.
    Each HTTP attempt gets `min(per-attempt timeout, time left)`. A backoff never sleeps past
    the deadline. No attempt starts with less than 5 s left. Running out raises
    `CallTimeLimit` (a `ProviderError`). Its call-log entry has cost `unknown`, since the
    provider may bill what it generated.
  - **Output cap:** `rate × (left − 15 s) × 0.9` tokens, at least 2,000, only when lower
    than the call's own cap. The rate is the median over the run's replies of 2k+ tokens,
    else 60 tokens/s. That default is the 10th percentile of 216 step 10/10b builder calls,
    whose median is 75 tokens/s.
  - **When a call is cut off:** if the latest compile passed and nothing changed since, the
    final turn follows, with its own 120 s; the run is a success even if that turn is cut
    off too. Otherwise the run ends `time_exceeded`, with `time_limit.outcome = "call_cut"`.
  - **`record.json`:** `per_turn[].timeout_s`, `output_cap` and `timed_out`.
- **Files:** `llm/protocol.py` (`CallTimeLimit`, `timeout_s`), `llm/base.py`,
  `llm/openai_chat.py` (deadline-aware `call_with_retries`), `llm/scripted.py`
  (`model.timeouts`), `agent/loop.py` (`timed`, `output_rate`, `FINAL_TURN_S`).
- **Tests (11, no network, `tests/test_time_cap.py`):**
  - the loop's timeouts follow the time left, and calls have no timeout without a limit;
  - the output cap;
  - a cut call ends `time_exceeded` with no overrun, or leads to the final turn after a
    passing build (also when that turn is cut off);
  - in the adapter: an attempt's timeout, a retry and its backoff within the deadline, no
    retry after it (cost unknown), a timeout with time left still retried;
  - ScriptedModel keeps timeouts apart from options.
- **Left over:**
  - The context policy's compaction call (the cheap model) and a compile are not capped by
    the time left. A compile has its own 300 s timeout (in step 10, at most 39 s of
    tool time per run), so a run can still end late in the worst case.
  - The cap does not tell the model that its output is short. It gets the `<reply_cut_off>`
    note as before.

## Step 11: Run storage

No live runs needed; everything is tested with ScriptedModel. Step 12 has not been started.

### What was built

| # | What | Files |
| --- | --- | --- |
| 1 | **Messages to and from dicts:** `llm.messages.to_dict` / `from_dict`, the format of a `conversation.jsonl` line (the loop's own `_message_dict` is gone). A tool call keeps the exact argument string the model sent, and a call whose JSON was broken comes back with the same parse error. A resumed conversation therefore sends the same bytes, and the provider's prompt cache still matches | `llm/messages.py` |
| 2 | **Resume:** `python -m agent --resume <run_dir>` (or `agent.loop.resume`). See below | `agent/storage.py`, `agent/loop.py`, `agent/__main__.py` |
| 3 | **`--list <runs>`**, one row per run (run, status, turns, cost, best version, sessions, prompt) and a total, with `crashed` for a run whose process died. **`runs/index.jsonl`:** one line per finished session (run path, prompt, status, reason, session, turns, cost, duration, best version, model, prompt set, two-phase, quality). Any run under `runs/`, experiment subfolders included, goes to `runs/index.jsonl`; a run elsewhere goes to the folder above it. Every version folder already holds its `main.py` (since step 2) | `agent/storage.py`, `agent/loop.py` |
| 4 | **Atomic writes** (`fileio.py`) wherever a crash could leave half a file. Whole files: temporary file + fsync + `os.replace`, retried while Windows has the target open. Covers `record.json`, the workspace files from write and edit, the starter `main.py`, the compile cache and failure history, `runs/lessons.md`, and render PNGs. Logs: one `write` per line, flushed and fsynced (`conversation.jsonl`, `llm_calls.jsonl`, `compile_log.jsonl`, `index.jsonl`); readers skip a last line cut short. Exports were already atomic (staging folder + rename) | `fileio.py`, `agent/`, `compiler/runner.py`, `llm/accounting.py` |

**How resume works:**
- **The allowance:** `--max-time`, `--max-cost`, `--max-calls` and `--max-turns` are the new
  session's. The role, prompt set, context policy, effort, two-phase, lessons and fix
  reserve come from `record["options"]`, which is new; they are inferred for older records.
- **What is refused:** a run that succeeded, or one whose `.lock` names a live other process
  (the loop writes the lock at the start and removes it at the end). Windows is asked
  through `OpenProcess`, since `os.kill(pid, 0)` there would terminate the process.
- **Clean-up first:** `clean_after_crash` deletes temporary files and export staging folders.
  A last `conversation.jsonl` line cut short is dropped from the file, so appending can go
  on.
- **Repair:** tool calls at the end with no result get one: `interrupted` ("may or may not
  have taken effect; read or compile") or `not_run` (the run stopped before running them).
  The repair is logged too, so the log stays valid for the next resume.
- **The conversation goes on:** the old messages unchanged, then a `<resumed>` note (why it
  stopped, the new allowance, "compile first"). The latest compile is not restored; the
  model compiles again, and the compile cache answers if nothing changed.
- **Counters continue:** turns (numbering goes on), tokens, cost, compiles, tool calls, doc
  reads, the two-phase state, and the context policy's counts and read pages.
  `duration_s` is the total over sessions. `record["sessions"]` has one entry per session:
  started, ended, status, reason, resumed from, turns, cost, duration, allowance.

**Example (`--list runs/step10b`, the 24 step 10b runs):** 24 rows, then
`24 runs: 19 success, 5 time_exceeded; $3.1188 spent`.

### Tests (13 new, no network, `tests/test_storage.py`; 1 existing test updated)

- **Messages:** every kind round-trips (exact argument bytes, broken JSON, images,
  reasoning); log keys are ignored; an unknown role is refused.
- **Writes:**
  - a failed atomic write keeps the old file and leaves no temp;
  - a locked target is retried;
  - a cut-off last JSONL line is skipped, damage in the middle is not;
  - a whole run leaves no `.tmp` and no lock.
- **Resume:**
  - after `time_exceeded`, it continues and succeeds. The same message prefix is sent, then
    the note; turns and compiles continue; there are 2 sessions and 2 index lines;
  - after a crash mid-turn (the compiler dies, plus a cut-off log line and a stale temp
    file): the compile call is answered `interrupted`, the first session is `crashed`, and
    the log is valid afterwards;
  - after `budget_exceeded`: a fresh budget;
  - calls a stopped run never ran are answered `not_run`;
  - a successful run and a run held by a live process are refused.
- **Index and list:** one index for the runs folder, the right rows (a crash shows as
  `crashed`), the `--list` output and its totals.
- **CLI:** a run stopped by its call cap is resumed with `--resume`. The sessions, turn
  numbering and saved options are checked, and a second resume is refused.
- **Updated:** the CLI test now finds `index.jsonl` beside the run folder.

### How Articraft does it (study §11)

- **Runs:** `runs/<timestamp>-<slug>[-2]/` holds `record.json`, `conversation.jsonl`,
  `workspace/` and `result/`. The record is rewritten after every model call and compile.
- **Versions:** the next USDZ index. There is no `main.py` snapshot.
- **Resume:** there is none. `replay` only redraws the log in the TUI.
- **Writes:** not atomic.
- **The study's ideas, all done here:** resume from `conversation.jsonl` (it "already holds
  everything a resume command would need"), atomic writes, and a `main.py` snapshot per
  version (ours since step 2).

### What we changed vs Articraft, and why

- **Resume.** A long run stopped by time, budget, sleep or a crash no longer has to start
  again. In step 10b, the 5 failed pump runs spent 24-31 minutes each; with resume they can
  go on.
- **Sessions in one record.** Reports can count a run once and still see what each session
  cost.
- **An index file.** Step 12 reads `runs/index.jsonl` instead of walking every run folder.
  `--list` walks the folders, so it also finds crashed runs and runs made before the index
  existed.

### Open ideas

- **The latest compile is not restored on resume,** so the model spends one turn compiling.
  It could be restored from the compile cache.
- **The compaction summary is not restored** (only the counters are). A long resumed run
  may compact again, at the cost of one cheap-model call.
- **A lock can outlive a crash under another process's id,** if Windows reuses the id. Then
  resume refuses until the lock file is deleted by hand.
- **`--list` could filter** by status or prompt, and a script could backfill
  `runs/index.jsonl` from older runs.

## Step 12: Evaluation harness

Built only: no live runs yet (about $2 left on DeepSeek, no top-up). Step 13 has not been
started.

### What was built

| # | What | Files |
| --- | --- | --- |
| 1 | **The benchmark:** 12 prompts in 5 categories, each with an id, an object, the prompt and a one-line expected motion (for the rater). None is a docs example or recipe, an `examples/` object, or an earlier evaluation prompt. A test checks the object words, and the prompts are added to `eval_prompts.json` (`step12`), so the docs test also keeps future docs examples away from them | `prompts/benchmark_v1.json`, `prompts/eval_prompts.json` |
| 2 | **`scripts/eval.py`:** prompts × runs as `python -m agent`, 2 at a time, into `runs/eval_<name>/jobs/<id>-r<rep>/`. It reads `runs/index.jsonl` to know which runs finished. See below | `scripts/eval.py`, `agent/__main__.py` (`--index`) |
| 3 | **`scripts/eval_report.py`:** writes `runs/eval_<name>/report.md`, with success per category and overall (Wilson 95% intervals, flagged indicative), cost per run and per success, time, turns, failed compiles, failure end states, failure kinds, quality, a per-run table, and one render sheet per category (`renders/<category>.png`). The key tables also go into BUILD_LOG.md between `<!-- eval_<name> -->` markers; a second report replaces them | `scripts/eval_report.py` |
| 4 | **Blind human rating:** `eval_rating.py make --names v1 [...]` pools the successful runs of one or more evaluations, shuffles them (fixed seed) and names them `item_NN`. Each sheet shows the rest pose (iso, front) and every driven joint moved 70% of the way to its farther limit, or less if a loop will not close, with no text at all (`rasterize(..., legend=False)`). `ratings.csv` has the prompt and expected motion, plus `realistic` (1-5), `motion_correct` (yes/no) and `notes`. `key.json` maps items to runs. `score` checks the CSV and gives the mean rating and the correct-motion share per evaluation and category | `scripts/eval_rating.py`, `agent/tools/render.py` |
| 5 | **The Articraft baseline adapter** (prepared, not run): the same prompts through the original Articraft in WSL, with its DeepSeek provider and `deepseek-v4-pro`. The cost is computed from its token counts, since it records $0 for DeepSeek. The run is stopped from outside at the cost cap or the time limit, because it has neither. Each run gets a replica-style `record.json`, so the same report works. Setup, differences and the USDZ question: `docs/ARTICRAFT_BASELINE.md` | `scripts/articraft_baseline.py`, `docs/ARTICRAFT_BASELINE.md` |

**The benchmark:**

| Category | Id | Prompt |
| --- | --- | --- |
| single hinge or slide | jewelry_box | a jewelry box with a lid hinged at the back |
| | sliding_window | a sliding window: a glass pane that slides sideways in its frame |
| multi-joint open chains | robot_arm | a robot arm with a rotating base, a shoulder, an elbow and a wrist |
| | boom_arm | a microphone boom arm clamped to a shelf edge, with a swivel and two jointed segments |
| | excavator_arm | an excavator arm on a turntable: boom, stick and bucket |
| several independent joints | grand_piano | a grand piano with a lid that opens, a key cover and three pedals |
| | pickup_truck | a toy pickup truck with four spinning wheels, an opening hood and a drop-down tailgate |
| closed loops | scissor_lift | a scissor lift: a platform raised by crossed arms |
| | windshield_wipers | a pair of car windshield wipers driven by one motor |
| | pantograph | a pantograph drawing copier: a linked parallelogram that copies a drawing at a larger scale |
| everyday tools | stapler | a stapler |
| | nutcracker | a nutcracker with two handles |

- **Prompts left out:** the suggested lamp prompts (a task lamp, a parallelogram lamp)
  clash with the step 9 desk lamp and `examples/desk_lamp.py`. A wardrobe with doors and
  a drawer is close to the step 10 kitchen cabinet, and a car door is close to the
  door-on-a-carcass recipe.
- **Closest to an earlier prompt:** the nutcracker. It is a different object from the step
  10 pliers, but the same mechanism (two handles on one pin).

**How `eval.py` handles money, resumes and stops:**
- **Caps:** `--max-cost` $0.30 per run and `--max-total` $1.50 for the evaluation.
- **Starting a run:** a run starts only if the spend so far, plus a full per-run cap for
  every run in flight, the new one included, stays within the total. Otherwise it waits
  for a run in flight to end and free its reservation, and stops when none is left.
- **How far the total can go over:** no run can start that could take the total past the
  cap. It can only go over by what one call overshoots a run's own cap, about $0.07 for a
  32k-token reply.
- **Order:** repetition first, then round-robin over the categories. A stop on the cap
  therefore leaves every category covered.
- **A rerun:**
  - finished runs (any end state) are kept;
  - unfinished ones (a crash, sleep, a connection error, a stop on HTTP 402) are resumed
    with what is left of their cost and time, at most 3 sessions per run;
  - the rest are started.
- **Stops and setup:** HTTP 402 stops new runs. The machine is kept awake. `--dry-run`
  shows the plan, the worst case, and whether DeepSeek is at peak price now (off-peak is
  half price; peak is 01-04 and 06-10 UTC, Monday to Friday).

### Expected cost of 12 prompts × 1 run (an estimate, not measured)

- **The basis:** step 10b's B2 ran off-peak (10:07-11:38 UTC). Per run it cost $0.047-0.098
  for the simpler prompts and $0.258 for the closed-loop pump; $0.114 on average.
- **This benchmark:** it has 3 closed-loop prompts, likely near the $0.30 cap, and 3
  multi-joint chains. Estimate: **about $1.3-2.0 off-peak**, most likely about $1.6.
- **Against the $1.50 cap:** the cap will probably stop 1-3 runs before they start. The
  total stays at or below about $1.50 plus each run's last-call overshoot, worst case
  about $1.64.
- **At peak hours,** every number doubles, and the cap would stop about half the runs.
- **Time:** about 1-1.5 hours with 2 at a time (B2's runs took 12 minutes on average; the
  loop runs take up to 25).

### Tests (15 new, no network or WSL, `tests/test_eval.py`)

- **Benchmark:** 12 prompts with 2-3 per category; no clash with earlier objects, examples
  or docs; registered for the docs test.
- **`eval.py`, with the real agent CLI in-process and ScriptedModel:**
  - every prompt runs, and a rerun keeps finished runs;
  - a crashed run is resumed with its remaining allowance (2 sessions);
  - the total cap (1 and 2 at a time; no run starts that could pass it);
  - HTTP 402 stops new runs;
  - the dry run.
- **Report:**
  - Wilson intervals;
  - the tables (success, end states, failure kinds, per run);
  - one sheet per category;
  - the BUILD_LOG section replaced, not duplicated.
- **Rating:** only successful runs are included, the CSV and key are right, the sheet has
  no text in the label corner, and the scores and CSV errors are right.
- **Articraft adapter:**
  - the WSL command carries no key, and `WSLENV` forwards it;
  - the path conversion;
  - converting a run's record (success, cost, budget, time, max turns, and the quality from
    `model.json`);
  - the watcher stops a run on cost and on time.

### How Articraft does it (study §12)

- **Its tests come in four lanes:** unit, warm compile, scripted agent (`ScriptedModel`) and
  one live test ("a simple box" must succeed). CI runs ruff, basedpyright and the non-live
  lanes. An SDK benchmark tracks mesh speed and quality.
- **There is no evaluation of output quality:** no prompt suite, no scores, no model
  comparison. The study's idea for the replica was 10-20 prompts, recording success rate,
  turns, dollars, failed compiles and the diagnostics left at the end, then a judge looking
  at the previews.

### What we changed vs Articraft, and why

- **A fixed benchmark across categories of mechanism,** so a change can be measured where
  it matters (closed loops are the hard case: step 10b's pump).
- **Spend control built into the harness** (per-run and total caps, resume), because the
  budget is real and small.
- **Intervals with every rate.** With 1 run per prompt, a difference of one or two runs is
  noise, and the report says so.
- **A blind human rating** instead of a model judge, for now. It costs nothing, and the
  sheets show motion as well as looks.

### Open ideas

- **A VLM judge** on the same blind sheets (the study's idea), once there is budget, checked
  against the human ratings.
- **Our checks on Articraft's outputs** need its USDZ meshes. See
  docs/ARTICRAFT_BASELINE.md for the two routes: approving `usd-core`, or converting inside
  WSL.
- **More runs per prompt** (3) would make the category numbers mean something: about $5 at
  off-peak prices.

### Budget plan (2026-11-24)

The DeepSeek balance is about **$12** and cannot be topped up again. It is split as follows:

| Use | Budget | When |
| --- | --- | --- |
| Our benchmark: `eval.py --name v1 --runs 3 --jobs 3 --max-total 5 --off-peak-only` (12 prompts × 3 runs) | $5 | now |
| The Articraft baseline (`scripts/articraft_baseline.py`) | $3 | later |
| Re-testing the slider-crank / closed-loop solver | $2 | later |
| Safety margin | $2 | kept |

- **The eval's own cap holds its share:** `--max-total` starts no run that could take the
  evaluation past its $5 share.
- **`--off-peak-only`** (added for this run) starts no run at DeepSeek's peak price
  (01-04 and 06-10 UTC on weekdays, double the off-peak rate). Runs already going finish,
  and a rerun of the same command resumes and starts the rest.

<!-- eval_v1 -->
### Evaluation eval_v1 (benchmark_v1)

From `scripts/eval_report.py --name v1`; the full report is `runs/eval_v1/report.md`.

#### Success (Wilson 95% intervals)

11 counted runs, 1-2 per prompt (2 prompts with none). With so few runs per category the intervals are wide: **indicative only**, not a measured difference.

| category | success | rate | 95% interval |
| --- | --- | --- | --- |
| single (a single hinge or slide) | 3/3 | 100% | 44%-100% |
| chain (multi-joint open chains) | 2/2 | 100% | 34%-100% |
| independent (several independent joints) | 2/2 | 100% | 34%-100% |
| loop (closed loops) | 0/2 | 0% | 0%-66% |
| tool (everyday tools and mechanisms) | 2/2 | 100% | 34%-100% |
| **overall** | 9/11 | 82% | 52%-95% |

#### Cost, time and turns (per run, means)

| category | runs | cost / run | cost / success | time / run | turns | failed compiles | spent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | 3 | $0.052 | $0.052 | 256 s | 9.7 | 1.3 | $0.155 |
| chain | 2 | $0.200 | $0.200 | 1049 s | 23.5 | 4.0 | $0.400 |
| independent | 2 | $0.134 | $0.134 | 657 s | 10.5 | 1.0 | $0.268 |
| loop | 2 | $0.268 | - | 1485 s | 11.5 | 2.5 | $0.537 |
| tool | 2 | $0.107 | $0.107 | 541 s | 9.5 | 0.5 | $0.214 |
| **overall** | 11 | $0.143 | $0.175 | 748 s | 12.6 | 1.8 | $1.574 |

#### Failure end states

| end state | runs |
| --- | --- |
| no_progress | 1 |
| time_exceeded | 1 |

#### Failure kinds

Runs with at least one failed compile of each kind (all runs), and the kind of the last failed compile of each failed run.

| kind | runs with it | last failure of a failed run |
| --- | --- | --- |
| validation | 4 | 0 |
| overlap | 3 | 1 |
| mesh_health | 1 | 0 |
| runtime_error | 1 | 0 |
| separation | 1 | 0 |
| swept_collision | 1 | 0 |
| test_failure | 1 | 0 |
<!-- /eval_v1 -->

### What happened in eval_v1 (2026-11-24, 15:59-17:02 UTC, off-peak)

- **Stopped by HTTP 402 after $2.09 of tracked spend,** 14 runs in (36 planned). The
  provider returned "Insufficient Balance" to 3 runs mid-run: excavator_arm-r1,
  pantograph-r1 and robot_arm-r2. `eval.py` then started no new run.
- **The balance was not $12.** DeepSeek's balance endpoint (`GET /user/balance`, no tokens
  used) afterwards reported **−$0.04 USD** (`is_available: false`). So the account held
  about $2.05 when the evaluation started, about the "$2 left" of step 12. The $12 in the
  budget plan above was never available; that plan cannot be carried out until the balance
  is confirmed.
- **Our cost accounting agrees with the provider's:** $2.09 tracked against a drop to
  −$0.04 from about $2, so within about $0.05. The price table (off-peak at half price) is
  confirmed by money actually spent.
- **Counted:** 11 runs; 9 success, 1 time_exceeded (windshield_wipers-r1), 1 no_progress
  (scissor_lift-r1). Every prompt except the loops succeeded at least once.
  - **No closed-loop success yet,** 0 of 2 counted.
  - **The scissor lift failed like step 10b's pump:** 3 replies cut off at the output limit
    while planning, and `main.py` was never written.
  - **The wipers** spent the 25 minutes on overlap failures.
- **Not counted:** the 3 runs cut off by the 402 are listed in the report as interrupted
  ($0.51). The 22 not started remain, mostly repetitions 2 and 3. A rerun of the same
  command resumes the interrupted ones with what is left of their allowance, and starts the
  rest.
- **Blind rating:** 9 sheets in `runs/rating_v1/sheets/`, to fill in
  `runs/rating_v1/ratings.csv`.
