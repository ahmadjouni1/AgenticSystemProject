# Articraft shape helpers: what the replica keeps

Source: `Previous work/Agentic_Systems_Project/src/articraft/sdk/_mesh/*` (5,077 lines with
`sdk/mesh.py`). The inventory was made by AST parsing (no imports) of every name that
`articraft.sdk` or `articraft.sdk.mesh` exports from `_mesh`: **57 public helpers** in total.

**How to read the columns:**

- **Usage** counts the *files* that use the name, with the counts done by a word-boundary search:
  - **ex**: the 9 complete example files (`examples/*/main.py` and `sdk/docs/examples/*.py`);
  - **docs**: the 16 SDK reference pages (`sdk/docs/common`, `sdk/docs/mesh`), not counting
    the quickstart;
  - **qs**: the quickstart, and **prompt**: the system prompt. Both are counted in mentions
    and shown only when non-zero.

  Docs pages describe every helper by design, so "real usage" means at least one example
  file, the quickstart, or the prompt.
- **Usefulness** is a judgement of how much the helper helps build realistic everyday
  articulated objects: furniture, appliances, tools, lamps, vehicles, linkages, cams.
- **Port rule:**
  - real usage; or
  - high usefulness with no 1-3 line build123d equivalent; or
  - on the step-1 review's minimum list and cheap.

  Names, arguments and defaults match Articraft. Options whose behavior is not ported yet
  are accepted at their default value, and any other value raises `NotImplementedError`
  naming the option.

## Ported set (26 names in 17 helper families)

| Helper | Replica file | Replica lines | Articraft lines (own / with private helpers) | Usage | Notes |
| --- | --- | --- | --- | --- | --- |
| `BoxGeometry` | `sdk/primitives.py` | 5 | 7 / 18 | ex 2 · docs 2 | Used in 2 examples. 5 lines on `trimesh.creation.box`. |
| `LatheGeometry` | `sdk/solids.py` | 89 | 109 / 228 | ex 2 · docs 2 | Used in 2 examples. Mesh-native revolve, 58 + 31 lines. |
| `boolean_difference` | `sdk/booleans.py` | 2 | 9 / 56 | ex 2 · docs 2 | Used in 2 examples. Same engine as `boolean_union`. |
| `rounded_rect_profile` | `sdk/profiles.py` | 22 | 32 / 32 | ex 2 · docs 3 | Used in 2 examples; feeds extrude, pipe and sweep. |
| `sweep_profile_along_spline` | `sdk/sweeps.py` | 29 | 51 / 93 | ex 2 · docs 1 | Used in 2 examples. |
| `CylinderGeometry` | `sdk/primitives.py` | 9 | 14 / 18 | ex 1 · docs 2 | Used in 1 example (6 times). The open-tube option has no 1-line build123d form. |
| `MeshGeometry` | `sdk/mesh_geometry.py` | 144 | 252 / 403 | ex 1 · docs 9 · qs 1 | Base of every helper. `.translate` appears 7 times in examples, `.rotate_y` once. Step 1's `Mesh` is now an alias. |
| `RoundedBoxGeometry` | `sdk/primitives.py` | 22 | 27 / 34 | ex 1 · docs 1 | Used in 1 example and on almost every appliance. Keeps its exact solid, so booleans on it are exact and fast. |
| `boolean_union` | `sdk/booleans.py` | 2 + 40 | 7 / 54 | ex 1 · docs 1 | Used in 1 example. Runs on OpenCASCADE, not manifold3d (not an approved dependency); exact solids are used when available. |
| `tube_from_spline_points` | `sdk/sweeps.py` | 32 | 45 / 74 | ex 1 · docs 1 | Used in 1 example. `spline='bezier'` raises NotImplementedError. |
| `CapsuleGeometry` | `sdk/primitives.py` | 14 | 27 / 31 | ex 0 · docs 1 | On the step-1 review's minimum list; 14 lines on `trimesh.creation.capsule`. |
| `ExtrudeGeometry` | `sdk/solids.py` | 30 | 43 / 74 | ex 0 · docs 1 | On the step-1 review's minimum list. Takes the ported profiles directly. Mesh-native, 30 lines. |
| `ExtrudeWithHolesGeometry` | `sdk/solids.py` | 43 | 51 / 139 | ex 0 · docs 1 | On the step-1 review's minimum list; the canonical bracket builder. Built exactly with build123d inside, and keeps its solid. |
| `LoftGeometry` | `sdk/solids.py` | 49 | 192 / 490 | ex 0 · docs 4 | On the step-1 review's minimum list; it backs `section_loft`. `tension`, `cap_style='round'` and `cap_length` raise NotImplementedError. |
| `PipeGeometry` | `sdk/sweeps.py` | 32 + 53 | 138 / 738 | ex 0 · docs 1 | On the step-1 review's minimum list; there is no reliable build123d one-liner. Variable `sections=` raises NotImplementedError. |
| `SweepGeometry` | `sdk/sweeps.py` | 2 | 36 / 36 | ex 0 · docs 1 | An alias in Articraft; 2 lines. |
| `wire_from_points` | `sdk/sweeps.py` | 37 + 27 | 41 / 41 | ex 0 · docs 1 | High usefulness and no easy build123d equivalent. Used by the replica's four-bar and folding-chair examples. |
| `LoftSection` | `sdk/solids.py` | 5 | 5 / 26 | ex 2 · docs 1 | Used in 2 examples (it is the input type of `section_loft`). |
| `SectionLoftSpec` | `sdk/solids.py` | 29 | 77 / 98 | ex 2 · docs 1 | Used in 2 examples. Ported: sections, cap, interpolation, samples, close_path, parameterization. `path`, `symmetry`, `repair` and `orient_to_path` raise NotImplementedError. |
| `section_loft` | `sdk/solids.py` | 28 | 28 / 300 | ex 2 · docs 2 | Used in 2 examples. A thin layer over `LoftGeometry`. |
| `superellipse_profile` | `sdk/profiles.py` | 15 | 19 / 19 | ex 1 · docs 1 | Used in 1 example; 15 lines. |
| `ConeGeometry` | `sdk/primitives.py` | 12 | 17 / 21 | ex 0 · docs 1 | On the step-1 review's minimum list; 12 lines, same placement as Articraft. |
| `TorusGeometry` | `sdk/primitives.py` | 15 | 21 / 25 | ex 0 · docs 1 | On the step-1 review's minimum list; 15 lines. |
| `boolean_intersection` | `sdk/booleans.py` | 2 | 8 / 55 | ex 0 · docs 1 | Completes the set; 2 lines once the other two exist. |
| `sample_catmull_rom_spline_3d` | `sdk/profiles.py` | 7 + 25 | 17 / 74 | ex 0 · docs 1 | Needed by the tubes, sweeps and smooth lofts. |
| `sample_catmull_rom_spline_2d` | `sdk/profiles.py` | 7 | 17 / 74 | ex 0 · docs 1 | Free once the 3D version exists (7 lines). |

Shared machinery, written once and used by many helpers: `triangulate_polygon` (ear
clipping, 37 lines), `skin` (rings to a closed solid, 40), `from_created` (merge, clean and
orient, 10), and `sdk/occ.py` (fast OpenCASCADE tessellation and moving exact solids, 72).
The whole ported mesh layer is **1,419 lines, docstrings included, against 5,077**.

## Full ranked table

Ranked by decision (ported, skipped, later), then usefulness, then example usage.

| # | Helper | What it does | Usage | Usefulness | build123d equivalent (1-3 lines) | Decision and why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `BoxGeometry` | Box mesh centered on the origin | ex 2 · docs 2 | high | yes: `Box(x, y, z)` | **PORTED**: Used in 2 examples. 5 lines on `trimesh.creation.box`. |
| 2 | `LatheGeometry` | Revolve an (r, z) profile; partial angles, x/y/z axis, `from_shell_profiles` for thin walls | ex 2 · docs 2 | high (lamps, knobs, bottles, wheels, cups) | 2 lines: `revolve(make_face(...))` | **PORTED**: Used in 2 examples. Mesh-native revolve, 58 + 31 lines. |
| 3 | `boolean_difference` | Cut one solid from another | ex 2 · docs 2 | high (bores, slots, openings) | for build123d shapes yes (`a - b`); not for meshes | **PORTED**: Used in 2 examples. Same engine as `boolean_union`. |
| 4 | `rounded_rect_profile` | Rounded rectangle as a 2D point list | ex 2 · docs 3 | high | `RectangleRounded`, but that is a sketch, not points | **PORTED**: Used in 2 examples; feeds extrude, pipe and sweep. |
| 5 | `sweep_profile_along_spline` | Any profile along a smooth curve | ex 2 · docs 1 | high (mug handles, cables, gaskets) | as above | **PORTED**: Used in 2 examples. |
| 6 | `CylinderGeometry` | Cylinder along z; `closed=False` gives an open tube | ex 1 · docs 2 | high | yes: `Cylinder(r, h)` | **PORTED**: Used in 1 example (6 times). The open-tube option has no 1-line build123d form. |
| 7 | `MeshGeometry` | Mutable triangle mesh with chainable translate / rotate / scale / merge / copy | ex 1 · docs 9 · qs 1 | high | n/a: it is the mesh type itself | **PORTED**: Base of every helper. `.translate` appears 7 times in examples, `.rotate_y` once. Step 1's `Mesh` is now an alias. |
| 8 | `RoundedBoxGeometry` | Box with every edge filleted | ex 1 · docs 1 | high (housings, appliances, cushions) | 2 lines: `Box` then `fillet(edges)` | **PORTED**: Used in 1 example and on almost every appliance. Keeps its exact solid, so booleans on it are exact and fast. |
| 9 | `boolean_union` | Union of two closed solids | ex 1 · docs 1 | high (merging parts into one shape) | for build123d shapes yes (`a + b`); not for meshes | **PORTED**: Used in 1 example. Runs on OpenCASCADE, not manifold3d (not an approved dependency); exact solids are used when available. |
| 10 | `tube_from_spline_points` | Smooth round tube through control points (Catmull-Rom) | ex 1 · docs 1 | high (lamp arms, cables, hoses) | 2 lines: `Spline` then `sweep`, but fragile at tight bends | **PORTED**: Used in 1 example. `spline='bezier'` raises NotImplementedError. |
| 11 | `CapsuleGeometry` | Cylinder with hemispherical ends | ex 0 · docs 1 | high (handles, rollers, pins, grips) | 1 line, but a BRep union of a cylinder and 2 spheres | **PORTED**: On the step-1 review's minimum list; 14 lines on `trimesh.creation.capsule`. |
| 12 | `ExtrudeGeometry` | Prism from a 2D profile; `.centered` / `.from_z0` | ex 0 · docs 1 | high (plates, cams, panels, outlines) | 2 lines: `extrude(make_face(Polyline(...)))` | **PORTED**: On the step-1 review's minimum list. Takes the ported profiles directly. Mesh-native, 30 lines. |
| 13 | `ExtrudeWithHolesGeometry` | Prism from an outer profile minus hole profiles | ex 0 · docs 1 | high (brackets, bolt plates, hinge leaves) | 3 lines: face minus hole faces, then `extrude` | **PORTED**: On the step-1 review's minimum list; the canonical bracket builder. Built exactly with build123d inside, and keeps its solid. |
| 14 | `LoftGeometry` | Skin through 3D rings with equal point counts; linear or Catmull-Rom between rings | ex 0 · docs 4 | high (vehicle bodies, shades, mice, handles) | 2-3 lines: `loft` of sketched wires, but each section needs its own plane | **PORTED**: On the step-1 review's minimum list; it backs `section_loft`. `tension`, `cap_style='round'` and `cap_length` raise NotImplementedError. |
| 15 | `PipeGeometry` | Sweep a 2D profile along a polyline: parallel-transport frames, mitered corners | ex 0 · docs 1 | high (frames, rails, gaskets, slats) | fragile: `sweep` along a polyline with sharp corners often fails | **PORTED**: On the step-1 review's minimum list; there is no reliable build123d one-liner. Variable `sections=` raises NotImplementedError. |
| 16 | `SweepGeometry` | Second name for `PipeGeometry` | ex 0 · docs 1 | high | as above | **PORTED**: An alias in Articraft; 2 lines. |
| 17 | `wire_from_points` | Round rod through polyline points, with miter / bevel / fillet corners | ex 0 · docs 1 | high (chair frames, bent handles, wire racks) | fragile: sweep along a filleted polyline | **PORTED**: High usefulness and no easy build123d equivalent. Used by the replica's four-bar and folding-chair examples. |
| 18 | `LoftSection` | One closed 3D ring for `section_loft` | ex 2 · docs 1 | medium | n/a | **PORTED**: Used in 2 examples (it is the input type of `section_loft`). |
| 19 | `SectionLoftSpec` | Options for `section_loft` (same 17 fields as Articraft) | ex 2 · docs 1 | medium | n/a | **PORTED (subset)**: Used in 2 examples. Ported: sections, cap, interpolation, samples, close_path, parameterization. `path`, `symmetry`, `repair` and `orient_to_path` raise NotImplementedError. |
| 20 | `section_loft` | Loft through `LoftSection`s; rings are resampled to a common count | ex 2 · docs 2 | medium-high | as for `LoftGeometry` | **PORTED (subset)**: Used in 2 examples. A thin layer over `LoftGeometry`. |
| 21 | `superellipse_profile` | Soft rectangle, |x/a|^n + |y/b|^n = 1 | ex 1 · docs 1 | medium (mice, kettles, phones, cushions) | no | **PORTED**: Used in 1 example; 15 lines. |
| 22 | `ConeGeometry` | Pointed cone along z, centered | ex 0 · docs 1 | medium (shades, nozzles, pointers) | yes: `Cone(r, 0, h)` | **PORTED**: On the step-1 review's minimum list; 12 lines, same placement as Articraft. |
| 23 | `TorusGeometry` | Ring in the xy plane | ex 0 · docs 1 | medium (knob rings, tyres, O-rings, handles) | yes: `Torus(R, r)` | **PORTED**: On the step-1 review's minimum list; 15 lines. |
| 24 | `boolean_intersection` | Common volume of two solids | ex 0 · docs 1 | medium | for build123d shapes yes (`a & b`) | **PORTED**: Completes the set; 2 lines once the other two exist. |
| 25 | `sample_catmull_rom_spline_3d` | Points on a centripetal Catmull-Rom curve (3D) | ex 0 · docs 1 | medium | `Spline(*pts).positions(...)` | **PORTED**: Needed by the tubes, sweeps and smooth lofts. |
| 26 | `sample_catmull_rom_spline_2d` | Same, in 2D | ex 0 · docs 1 | low-medium | same | **PORTED**: Free once the 3D version exists (7 lines). |
| 27 | `WirePolylineGeometry` | The class behind `wire_from_points` | ex 0 · docs 1 | high | as `wire_from_points` | **SKIPPED (name only)**: Not used in any example; the ported function covers it. |
| 28 | `weld` | Fuse solids into one smoothly blended mesh (signed-distance field) | ex 2 · docs 1 · prompt 1 | medium (organic blends) | partly: union, then build123d `fillet` | **SKIPPED**: Used in 2 examples, but needs libigl and manifold3d (neither approved) and about 300 lines of SDF meshing. Articraft's system prompt itself warns against it on precise surfaces. Open idea. |
| 29 | `SweepSection` | Profile scale, rotation and offset at a position along a sweep | ex 1 · docs 1 | medium (tapering handles, horns) | no | **SKIPPED**: Used in 1 example, but variable sections need the sweep core rewritten. Open idea. |
| 30 | `ArcPipeGeometry` | Profile swept along a circular arc | ex 0 · docs 2 | medium (curved rails, handles) | 2 lines: `sweep` along `CenterArc` | **SKIPPED**: Not used in any example; `PipeGeometry` along sampled arc points does the same. |
| 31 | `DomeGeometry` | Hemisphere or ellipsoidal cap | ex 0 · docs 1 | medium (domes, buttons, caps) | yes: `Sphere(r, arc_size1=0)` or a lathe | **SKIPPED**: Not used in any example; `LatheGeometry` covers it. |
| 32 | `SphereGeometry` | UV sphere | ex 0 · docs 2 | medium | yes: `Sphere(r)` | **SKIPPED**: Not used in any example; step 1's `Sphere` covers it. |
| 33 | `WirePath` | Turtle-style path builder (line, arc, ...) | ex 0 · docs 1 | medium | `Polyline` / `Line` + `JernArc` | **SKIPPED**: Not used in any example. |
| 34 | `cut_opening_on_face` | Cut a tapered opening into a shell face | ex 0 · docs 1 | medium (vents, ports) | partly: `boolean_difference` / build123d `-` | **SKIPPED**: Not used in any example; booleans cover most cases. |
| 35 | `sample_arc_3d` | Points on a 3D circular arc | ex 0 · docs 2 | medium | `CenterArc(...).positions(...)` | **SKIPPED**: Not used in any example. |
| 36 | `smooth_difference` | Subtract with smooth blended edges | ex 0 · docs 1 | medium | partly: `-` then `fillet` | **SKIPPED**: Not used in any example; same dependencies as `weld`. |
| 37 | `tube_network_from_paths` | Several tubes unioned at their junctions | ex 0 · docs 1 | medium (racks, frames with tees) | no | **SKIPPED**: Not used in any example. Needs many unions, which are slow without manifold3d; compose `wire_from_points` with `boolean_union`. |
| 38 | `ShellPartitionRegion` | One region of a shell partition | ex 0 · docs 1 | low | no | **SKIPPED**: Belongs to `partition_shell`. |
| 39 | `ShellPartitionSpec` | Options for `partition_shell` | ex 0 · docs 1 | low | no | **SKIPPED**: Belongs to `partition_shell`. |
| 40 | `SuperellipsoidGeometry` | 3D superellipsoid blob | ex 0 · docs 1 | low | no | **SKIPPED**: Not used in any example; niche. |
| 41 | `build123d_to_mesh` | Tessellate a build123d shape to a mesh | ex 0 · docs 1 | low | yes: `Build123dShape(s).to_trimesh()`, automatic in `Body.add` | **SKIPPED**: Already covered by step 1. |
| 42 | `partition_shell` | Split a shell into named panels (doors, lids) | ex 0 · docs 1 | low | no | **SKIPPED**: Not used in any example; about 460 lines. |
| 43 | `refine_mesh` | Split triangles until edges are short | ex 0 · docs 1 | low | no (`trimesh.remesh.subdivide_to_size`) | **SKIPPED**: Not used in any example; trimesh has it. |
| 44 | `repair_loft` | Re-mesh a self-intersecting loft | ex 0 · docs 1 | low | no | **SKIPPED**: Not used in any example; repair tooling. |
| 45 | `resample_side_sections` | Smooth a list of side-profile sections | ex 0 · docs 1 | low | no | **SKIPPED**: Not used in any example; vehicle-body niche. |
| 46 | `sample_cubic_bezier_spline_2d` | Points on a cubic Bezier chain (2D) | ex 0 · docs 1 | low-medium | `Bezier(...)` | **SKIPPED**: Not used in any example; Catmull-Rom covers smooth paths. |
| 47 | `sample_cubic_bezier_spline_3d` | Points on a cubic Bezier chain (3D) | ex 0 · docs 1 | low-medium | `Bezier(...)` | **SKIPPED**: Not used in any example. |
| 48 | `smooth_mesh` | Taubin smoothing | ex 0 · docs 1 | low | no (`trimesh.smoothing.filter_taubin`) | **SKIPPED**: Not used in any example; trimesh has it. |
| 49 | `snap_to` | Move a piece until it overlaps an anchor by a set depth | ex 0 · docs 1 | low-medium | no | **SKIPPED**: Not used in any example; the replica places parts with frames and `frame_in`. |
| 50 | `split_superellipse_side_loft` | Same, split into two parts at a y value | ex 0 · docs 1 | low | no | **SKIPPED**: Not used in any example. |
| 51 | `subdivide_mesh` | Split every triangle into four | ex 0 · docs 1 | low | no (`trimesh.remesh.subdivide`) | **SKIPPED**: Not used in any example; trimesh has it. |
| 52 | `superellipse_side_loft` | Loft superellipse sections from a side profile | ex 0 · docs 1 | low-medium (car / boat bodies) | no | **SKIPPED**: Not used in any example; `section_loft` with `superellipse_profile` rings covers it (see the computer mouse example). |
| 53 | `SnapRefused` | Error raised by `snap_to` | ex 0 · docs 1 | n/a | n/a | **SKIPPED**: Belongs to `snap_to`. |
| 54 | `analyze_mesh_health` | Topology and triangle-quality report | ex 0 · docs 2 | high for checks, not for shape-making | no | **LATER (step 4)**: A compiler check, not a shape helper; ported with the checks in step 4. |
| 55 | `MeshHealthFinding` | One counted defect | ex 0 · docs 1 | n/a | n/a | **LATER (step 4)**: Goes with `analyze_mesh_health`. |
| 56 | `MeshHealthIssue` | Kinds of mesh defects | ex 0 · docs 1 | n/a | n/a | **LATER (step 4)**: Goes with `analyze_mesh_health`. |
| 57 | `MeshHealthReport` | Result of `analyze_mesh_health` | ex 0 · docs 1 | n/a | n/a | **LATER (step 4)**: Goes with `analyze_mesh_health`. |

`MeshGeometry` methods: `translate` (ex 7), `rotate_y` (ex 1), and `rotate`, `rotate_x`,
`rotate_z`, `scale`, `merge`, `copy`, `add_vertex`, `add_face`, `bounds`, `is_watertight`,
`from_trimesh` and `to_trimesh` (docs only) are **ported**. `set_normal_crease`,
`use_smooth_normals`, `use_hard_normals` and `to_obj` (docs only) are **skipped** for now,
because they are export concerns (step 2). `LoftGeometry.from_shell_profiles` (docs only) is
**skipped**; `LatheGeometry.from_shell_profiles` and `ExtrudeGeometry.centered/from_z0` are
**ported**.

## Inventory: size and dependencies

Lines are measured from the AST. "Own" is the definition itself; "with private helpers" adds
the private functions it calls, but not the shared `MeshGeometry` base (403 lines), which every
helper inherits. Dependencies cover the helper and its private callees.

| Helper | Articraft file | Own lines | With private helpers | Depends on |
| --- | --- | --- | --- | --- |
| `boolean_difference` | `boolean.py` | 9 | 56 | manifold3d, numpy |
| `boolean_intersection` | `boolean.py` | 8 | 55 | manifold3d, numpy |
| `boolean_union` | `boolean.py` | 7 | 54 | manifold3d, numpy |
| `cut_opening_on_face` | `boolean.py` | 67 | 98 | (none) |
| `BoxGeometry` | `core.py` | 7 | 18 | trimesh |
| `CapsuleGeometry` | `core.py` | 27 | 31 | trimesh |
| `ConeGeometry` | `core.py` | 17 | 21 | numpy, trimesh |
| `CylinderGeometry` | `core.py` | 14 | 18 | numpy, trimesh |
| `DomeGeometry` | `core.py` | 25 | 36 | numpy, trimesh |
| `ExtrudeGeometry` | `core.py` | 43 | 74 | (none) |
| `ExtrudeWithHolesGeometry` | `core.py` | 51 | 139 | manifold3d, numpy |
| `LatheGeometry` | `core.py` | 109 | 228 | manifold3d, numpy, trimesh |
| `LoftGeometry` | `core.py` | 192 | 490 | manifold3d, numpy, scipy |
| `MeshGeometry` | `core.py` | 252 | 403 | numpy, trimesh |
| `RoundedBoxGeometry` | `core.py` | 27 | 34 | build123d |
| `SphereGeometry` | `core.py` | 8 | 12 | trimesh |
| `SuperellipsoidGeometry` | `core.py` | 72 | 81 | (none) |
| `TorusGeometry` | `core.py` | 21 | 25 | trimesh |
| `build123d_to_mesh` | `core.py` | 33 | 33 | build123d |
| `MeshHealthFinding` | `health.py` | 7 | 7 | (none) |
| `MeshHealthIssue` | `health.py` | 13 | 13 | (none) |
| `MeshHealthReport` | `health.py` | 18 | 18 | (none) |
| `analyze_mesh_health` | `health.py` | 195 | 371 | libigl, numpy, scipy, trimesh |
| `WirePath` | `profiles.py` | 53 | 68 | (none) |
| `resample_side_sections` | `profiles.py` | 59 | 59 | (none) |
| `rounded_rect_profile` | `profiles.py` | 32 | 32 | (none) |
| `sample_arc_3d` | `profiles.py` | 24 | 24 | numpy, scipy |
| `sample_catmull_rom_spline_2d` | `profiles.py` | 17 | 74 | numpy, scipy |
| `sample_catmull_rom_spline_3d` | `profiles.py` | 17 | 74 | numpy, scipy |
| `sample_cubic_bezier_spline_2d` | `profiles.py` | 9 | 30 | numpy, scipy |
| `sample_cubic_bezier_spline_3d` | `profiles.py` | 9 | 30 | numpy, scipy |
| `split_superellipse_side_loft` | `profiles.py` | 58 | 64 | (none) |
| `superellipse_profile` | `profiles.py` | 19 | 19 | (none) |
| `superellipse_side_loft` | `profiles.py` | 37 | 37 | (none) |
| `refine_mesh` | `refine.py` | 28 | 51 | numpy, trimesh |
| `smooth_mesh` | `refine.py` | 32 | 61 | numpy, trimesh |
| `subdivide_mesh` | `refine.py` | 31 | 54 | numpy, trimesh |
| `LoftSection` | `section_loft.py` | 5 | 26 | numpy |
| `SectionLoftSpec` | `section_loft.py` | 77 | 98 | numpy |
| `repair_loft` | `section_loft.py` | 13 | 42 | numpy, trimesh |
| `section_loft` | `section_loft.py` | 28 | 300 | numpy, trimesh |
| `ShellPartitionRegion` | `shell_partition.py` | 32 | 32 | (none) |
| `ShellPartitionSpec` | `shell_partition.py` | 45 | 45 | (none) |
| `partition_shell` | `shell_partition.py` | 30 | 61 | (none) |
| `ArcPipeGeometry` | `sweeps.py` | 44 | 44 | (none) |
| `PipeGeometry` | `sweeps.py` | 138 | 738 | manifold3d, numpy, scipy |
| `SweepGeometry` | `sweeps.py` | 36 | 36 | (none) |
| `SweepSection` | `sweeps.py` | 49 | 69 | (none) |
| `WirePolylineGeometry` | `sweeps.py` | 73 | 148 | (none) |
| `sweep_profile_along_spline` | `sweeps.py` | 51 | 93 | (none) |
| `tube_from_spline_points` | `sweeps.py` | 45 | 74 | (none) |
| `tube_network_from_paths` | `sweeps.py` | 76 | 150 | manifold3d, numpy |
| `wire_from_points` | `sweeps.py` | 41 | 41 | (none) |
| `SnapRefused` | `weld.py` | 2 | 2 | (none) |
| `smooth_difference` | `weld.py` | 43 | 292 | libigl, manifold3d, numpy, scipy, trimesh |
| `snap_to` | `weld.py` | 55 | 88 | numpy, trimesh |
| `weld` | `weld.py` | 46 | 295 | libigl, manifold3d, numpy, scipy, trimesh |

Not installed in the replica (and not approved): `manifold3d` (Articraft's booleans, lathe
and extrude caps, loft caps), `libigl` (welds, mesh health), and trimesh's polygon
triangulators (`mapbox_earcut`, `triangle`, `shapely`). The replica replaces them with a
small ear-clipping triangulator and OpenCASCADE (inside build123d).

Regenerate from the project root (Python 3.11+, no packages needed):

```shell
python -B -I replica/scripts/inventory_mesh_helpers.py "Previous work/Agentic_Systems_Project" inventory.json
python -B -I replica/scripts/make_shape_helpers_doc.py inventory.json replica/docs/SHAPE_HELPERS.md
```
