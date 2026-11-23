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
