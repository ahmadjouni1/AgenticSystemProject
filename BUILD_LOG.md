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

- **Responses API for OpenAI:** `previous_response_id` would send only new items each turn
  (Articraft does this). It would mean a second converter, so wait until the agent loop
  shows the cost of resending.
- **`prompt_cache_key`:** OpenAI routes requests with the same key to the same cache. Check
  its docs before using it.
- **Strict tool schemas:** DeepSeek's beta `strict` mode (at `/beta`) and OpenAI Structured
  Outputs would reject malformed arguments at the source.
- **Streaming:** show progress on long reasoning calls. It is not needed for correctness.
- **Price freshness check:** warn when `PRICES_CHECKED` is more than, say, 60 days old.
- **Run storage (step 11)** will want to save the neutral messages; add `to_dict` /
  `from_dict` then.
