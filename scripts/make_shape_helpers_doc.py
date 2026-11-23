"""Generate replica/docs/SHAPE_HELPERS.md from the AST inventory plus hand ratings."""
import json
import sys

rows = {r["name"]: r for r in json.load(open(sys.argv[1]))}
out_path = sys.argv[2]

# name -> (what it does, usefulness, build123d equivalent, decision, why, replica lines)
R = {
    # --- ported -------------------------------------------------------------------------
    "MeshGeometry": ("Mutable triangle mesh with chainable translate / rotate / scale / merge / copy", "high",
        "n/a: it is the mesh type itself", "PORTED",
        "Base of every helper. `.translate` appears 7 times in examples, `.rotate_y` once. Step 1's `Mesh` is now an alias.", "144"),
    "BoxGeometry": ("Box mesh centered on the origin", "high", "yes: `Box(x, y, z)`", "PORTED",
        "Used in 2 examples. 5 lines on `trimesh.creation.box`.", "5"),
    "CylinderGeometry": ("Cylinder along z; `closed=False` gives an open tube", "high", "yes: `Cylinder(r, h)`", "PORTED",
        "Used in 1 example (6 times). The open-tube option has no 1-line build123d form.", "9"),
    "RoundedBoxGeometry": ("Box with every edge filleted", "high (housings, appliances, cushions)",
        "2 lines: `Box` then `fillet(edges)`", "PORTED",
        "Used in 1 example and on almost every appliance. Keeps its exact solid, so booleans on it are exact and fast.", "22"),
    "CapsuleGeometry": ("Cylinder with hemispherical ends", "high (handles, rollers, pins, grips)",
        "1 line, but a BRep union of a cylinder and 2 spheres", "PORTED",
        "On the step-1 review's minimum list; 14 lines on `trimesh.creation.capsule`.", "14"),
    "ConeGeometry": ("Pointed cone along z, centered", "medium (shades, nozzles, pointers)", "yes: `Cone(r, 0, h)`",
        "PORTED", "On the step-1 review's minimum list; 12 lines, same placement as Articraft.", "12"),
    "TorusGeometry": ("Ring in the xy plane", "medium (knob rings, tyres, O-rings, handles)", "yes: `Torus(R, r)`",
        "PORTED", "On the step-1 review's minimum list; 15 lines.", "15"),
    "ExtrudeGeometry": ("Prism from a 2D profile; `.centered` / `.from_z0`", "high (plates, cams, panels, outlines)",
        "2 lines: `extrude(make_face(Polyline(...)))`", "PORTED",
        "On the step-1 review's minimum list. Takes the ported profiles directly. Mesh-native, 30 lines.", "30"),
    "ExtrudeWithHolesGeometry": ("Prism from an outer profile minus hole profiles",
        "high (brackets, bolt plates, hinge leaves)", "3 lines: face minus hole faces, then `extrude`", "PORTED",
        "On the step-1 review's minimum list; the canonical bracket builder. Built exactly with build123d inside, and keeps its solid.", "43"),
    "LatheGeometry": ("Revolve an (r, z) profile; partial angles, x/y/z axis, `from_shell_profiles` for thin walls",
        "high (lamps, knobs, bottles, wheels, cups)", "2 lines: `revolve(make_face(...))`", "PORTED",
        "Used in 2 examples. Mesh-native revolve, 58 + 31 lines.", "89"),
    "LoftGeometry": ("Skin through 3D rings with equal point counts; linear or Catmull-Rom between rings",
        "high (vehicle bodies, shades, mice, handles)",
        "2-3 lines: `loft` of sketched wires, but each section needs its own plane", "PORTED",
        "On the step-1 review's minimum list; it backs `section_loft`. `tension`, `cap_style='round'` and `cap_length` raise NotImplementedError.", "49"),
    "LoftSection": ("One closed 3D ring for `section_loft`", "medium", "n/a", "PORTED",
        "Used in 2 examples (it is the input type of `section_loft`).", "5"),
    "SectionLoftSpec": ("Options for `section_loft` (same 17 fields as Articraft)", "medium", "n/a", "PORTED (subset)",
        "Used in 2 examples. Ported: sections, cap, interpolation, samples, close_path, parameterization. `path`, `symmetry`, `repair` and `orient_to_path` raise NotImplementedError.", "29"),
    "section_loft": ("Loft through `LoftSection`s; rings are resampled to a common count", "medium-high",
        "as for `LoftGeometry`", "PORTED (subset)", "Used in 2 examples. A thin layer over `LoftGeometry`.", "28"),
    "PipeGeometry": ("Sweep a 2D profile along a polyline: parallel-transport frames, mitered corners",
        "high (frames, rails, gaskets, slats)",
        "fragile: `sweep` along a polyline with sharp corners often fails", "PORTED",
        "On the step-1 review's minimum list; there is no reliable build123d one-liner. Variable `sections=` raises NotImplementedError.", "32 + 53"),
    "SweepGeometry": ("Second name for `PipeGeometry`", "high", "as above", "PORTED", "An alias in Articraft; 2 lines.", "2"),
    "wire_from_points": ("Round rod through polyline points, with miter / bevel / fillet corners",
        "high (chair frames, bent handles, wire racks)", "fragile: sweep along a filleted polyline", "PORTED",
        "High usefulness and no easy build123d equivalent. Used by the replica's four-bar and folding-chair examples.", "37 + 27"),
    "tube_from_spline_points": ("Smooth round tube through control points (Catmull-Rom)",
        "high (lamp arms, cables, hoses)", "2 lines: `Spline` then `sweep`, but fragile at tight bends", "PORTED",
        "Used in 1 example. `spline='bezier'` raises NotImplementedError.", "32"),
    "sweep_profile_along_spline": ("Any profile along a smooth curve", "high (mug handles, cables, gaskets)",
        "as above", "PORTED", "Used in 2 examples.", "29"),
    "boolean_union": ("Union of two closed solids", "high (merging parts into one shape)",
        "for build123d shapes yes (`a + b`); not for meshes", "PORTED",
        "Used in 1 example. Runs on OpenCASCADE, not manifold3d (not an approved dependency); exact solids are used when available.", "2 + 40"),
    "boolean_difference": ("Cut one solid from another", "high (bores, slots, openings)",
        "for build123d shapes yes (`a - b`); not for meshes", "PORTED",
        "Used in 2 examples. Same engine as `boolean_union`.", "2"),
    "boolean_intersection": ("Common volume of two solids", "medium", "for build123d shapes yes (`a & b`)", "PORTED",
        "Completes the set; 2 lines once the other two exist.", "2"),
    "rounded_rect_profile": ("Rounded rectangle as a 2D point list", "high",
        "`RectangleRounded`, but that is a sketch, not points", "PORTED",
        "Used in 2 examples; feeds extrude, pipe and sweep.", "22"),
    "superellipse_profile": ("Soft rectangle, |x/a|^n + |y/b|^n = 1", "medium (mice, kettles, phones, cushions)",
        "no", "PORTED", "Used in 1 example; 15 lines.", "15"),
    "sample_catmull_rom_spline_3d": ("Points on a centripetal Catmull-Rom curve (3D)", "medium",
        "`Spline(*pts).positions(...)`", "PORTED", "Needed by the tubes, sweeps and smooth lofts.", "7 + 25"),
    "sample_catmull_rom_spline_2d": ("Same, in 2D", "low-medium", "same", "PORTED",
        "Free once the 3D version exists (7 lines).", "7"),
    # --- skipped ------------------------------------------------------------------------
    "SphereGeometry": ("UV sphere", "medium", "yes: `Sphere(r)`", "SKIPPED",
        "Not used in any example; step 1's `Sphere` covers it.", ""),
    "DomeGeometry": ("Hemisphere or ellipsoidal cap", "medium (domes, buttons, caps)",
        "yes: `Sphere(r, arc_size1=0)` or a lathe", "SKIPPED", "Not used in any example; `LatheGeometry` covers it.", ""),
    "SuperellipsoidGeometry": ("3D superellipsoid blob", "low", "no", "SKIPPED", "Not used in any example; niche.", ""),
    "ArcPipeGeometry": ("Profile swept along a circular arc", "medium (curved rails, handles)",
        "2 lines: `sweep` along `CenterArc`", "SKIPPED",
        "Not used in any example; `PipeGeometry` along sampled arc points does the same.", ""),
    "WirePolylineGeometry": ("The class behind `wire_from_points`", "high", "as `wire_from_points`",
        "SKIPPED (name only)", "Not used in any example; the ported function covers it.", ""),
    "tube_network_from_paths": ("Several tubes unioned at their junctions", "medium (racks, frames with tees)", "no",
        "SKIPPED",
        "Not used in any example. Needs many unions, which are slow without manifold3d; compose `wire_from_points` with `boolean_union`.", ""),
    "SweepSection": ("Profile scale, rotation and offset at a position along a sweep",
        "medium (tapering handles, horns)", "no", "SKIPPED",
        "Used in 1 example, but variable sections need the sweep core rewritten. Open idea.", ""),
    "repair_loft": ("Re-mesh a self-intersecting loft", "low", "no", "SKIPPED",
        "Not used in any example; repair tooling.", ""),
    "cut_opening_on_face": ("Cut a tapered opening into a shell face", "medium (vents, ports)",
        "partly: `boolean_difference` / build123d `-`", "SKIPPED", "Not used in any example; booleans cover most cases.", ""),
    "weld": ("Fuse solids into one smoothly blended mesh (signed-distance field)", "medium (organic blends)",
        "partly: union, then build123d `fillet`", "SKIPPED",
        "Used in 2 examples, but needs libigl and manifold3d (neither approved) and about 300 lines of SDF meshing. Articraft's system prompt itself warns against it on precise surfaces. Open idea.", ""),
    "smooth_difference": ("Subtract with smooth blended edges", "medium", "partly: `-` then `fillet`", "SKIPPED",
        "Not used in any example; same dependencies as `weld`.", ""),
    "snap_to": ("Move a piece until it overlaps an anchor by a set depth", "low-medium", "no", "SKIPPED",
        "Not used in any example; the replica places parts with frames and `frame_in`.", ""),
    "SnapRefused": ("Error raised by `snap_to`", "n/a", "n/a", "SKIPPED", "Belongs to `snap_to`.", ""),
    "partition_shell": ("Split a shell into named panels (doors, lids)", "low", "no", "SKIPPED",
        "Not used in any example; about 460 lines.", ""),
    "ShellPartitionSpec": ("Options for `partition_shell`", "low", "no", "SKIPPED", "Belongs to `partition_shell`.", ""),
    "ShellPartitionRegion": ("One region of a shell partition", "low", "no", "SKIPPED", "Belongs to `partition_shell`.", ""),
    "refine_mesh": ("Split triangles until edges are short", "low", "no (`trimesh.remesh.subdivide_to_size`)",
        "SKIPPED", "Not used in any example; trimesh has it.", ""),
    "subdivide_mesh": ("Split every triangle into four", "low", "no (`trimesh.remesh.subdivide`)", "SKIPPED",
        "Not used in any example; trimesh has it.", ""),
    "smooth_mesh": ("Taubin smoothing", "low", "no (`trimesh.smoothing.filter_taubin`)", "SKIPPED",
        "Not used in any example; trimesh has it.", ""),
    "build123d_to_mesh": ("Tessellate a build123d shape to a mesh", "low",
        "yes: `Build123dShape(s).to_trimesh()`, automatic in `Body.add`", "SKIPPED", "Already covered by step 1.", ""),
    "sample_cubic_bezier_spline_2d": ("Points on a cubic Bezier chain (2D)", "low-medium", "`Bezier(...)`", "SKIPPED",
        "Not used in any example; Catmull-Rom covers smooth paths.", ""),
    "sample_cubic_bezier_spline_3d": ("Points on a cubic Bezier chain (3D)", "low-medium", "`Bezier(...)`", "SKIPPED",
        "Not used in any example.", ""),
    "sample_arc_3d": ("Points on a 3D circular arc", "medium", "`CenterArc(...).positions(...)`", "SKIPPED",
        "Not used in any example.", ""),
    "WirePath": ("Turtle-style path builder (line, arc, ...)", "medium", "`Polyline` / `Line` + `JernArc`", "SKIPPED",
        "Not used in any example.", ""),
    "resample_side_sections": ("Smooth a list of side-profile sections", "low", "no", "SKIPPED",
        "Not used in any example; vehicle-body niche.", ""),
    "superellipse_side_loft": ("Loft superellipse sections from a side profile", "low-medium (car / boat bodies)", "no",
        "SKIPPED",
        "Not used in any example; `section_loft` with `superellipse_profile` rings covers it (see the computer mouse example).", ""),
    "split_superellipse_side_loft": ("Same, split into two parts at a y value", "low", "no", "SKIPPED",
        "Not used in any example.", ""),
    "analyze_mesh_health": ("Topology and triangle-quality report", "high for checks, not for shape-making", "no",
        "LATER (step 4)", "A compiler check, not a shape helper; ported with the checks in step 4.", ""),
    "MeshHealthReport": ("Result of `analyze_mesh_health`", "n/a", "n/a", "LATER (step 4)",
        "Goes with `analyze_mesh_health`.", ""),
    "MeshHealthIssue": ("Kinds of mesh defects", "n/a", "n/a", "LATER (step 4)", "Goes with `analyze_mesh_health`.", ""),
    "MeshHealthFinding": ("One counted defect", "n/a", "n/a", "LATER (step 4)", "Goes with `analyze_mesh_health`.", ""),
}
missing = set(rows) - set(R)
extra = set(R) - set(rows)
assert not missing and not extra, (missing, extra)

FILES = {
    "solids.py": {"ExtrudeGeometry", "ExtrudeWithHolesGeometry", "LatheGeometry", "LoftGeometry",
                  "LoftSection", "SectionLoftSpec", "section_loft"},
    "sweeps.py": {"PipeGeometry", "SweepGeometry", "wire_from_points", "tube_from_spline_points",
                  "sweep_profile_along_spline"},
    "booleans.py": {"boolean_union", "boolean_difference", "boolean_intersection"},
    "profiles.py": {"rounded_rect_profile", "superellipse_profile", "sample_catmull_rom_spline_2d",
                    "sample_catmull_rom_spline_3d"},
    "mesh_geometry.py": {"MeshGeometry"},
}


def replica_file(name: str) -> str:
    for f, names in FILES.items():
        if name in names:
            return f
    return "primitives.py"


def level(u: str) -> int:
    return 3 if u.startswith("high") else 2 if u.startswith("medium") else 1 if u.startswith("low") else 0


def usage(r: dict) -> str:
    u, f = r["usage"], r["usage_files"]
    parts = [f"ex {f['examples']}", f"docs {f['docs']}"]
    if u["quickstart"]:
        parts.append(f"qs {u['quickstart']}")
    if u["system_prompt"]:
        parts.append(f"prompt {u['system_prompt']}")
    return " · ".join(parts)


def order(name: str):
    d = R[name][3]
    group = 0 if d.startswith("PORTED") else 2 if d.startswith("LATER") else 1
    return (group, -level(R[name][1]), -rows[name]["usage_files"]["examples"], name)


ranked = sorted(R, key=order)
ported = [n for n in ranked if R[n][3].startswith("PORTED")]
L = []
P = L.append
P("# Articraft shape helpers: what the replica keeps")
P("")
P("Source: `Previous work/Agentic_Systems_Project/src/articraft/sdk/_mesh/*` (5,077 lines with")
P("`sdk/mesh.py`). The inventory was made by AST parsing (no imports) of every name that")
P(f"`articraft.sdk` or `articraft.sdk.mesh` exports from `_mesh`: **{len(R)} public helpers** in total.")
P("")
P("**How to read the columns:**")
P("")
P("- **Usage** counts the *files* that use the name, with the counts done by a word-boundary search:")
P("  - **ex**: the 9 complete example files (`examples/*/main.py` and `sdk/docs/examples/*.py`);")
P("  - **docs**: the 16 SDK reference pages (`sdk/docs/common`, `sdk/docs/mesh`), not counting")
P("    the quickstart;")
P("  - **qs**: the quickstart, and **prompt**: the system prompt. Both are counted in mentions")
P("    and shown only when non-zero.")
P("")
P("  Docs pages describe every helper by design, so \"real usage\" means at least one example")
P("  file, the quickstart, or the prompt.")
P("- **Usefulness** is a judgement of how much the helper helps build realistic everyday")
P("  articulated objects: furniture, appliances, tools, lamps, vehicles, linkages, cams.")
P("- **Port rule:**")
P("  - real usage; or")
P("  - high usefulness with no 1-3 line build123d equivalent; or")
P("  - on the step-1 review's minimum list and cheap.")
P("")
P("  Names, arguments and defaults match Articraft. Options whose behavior is not ported yet")
P("  are accepted at their default value, and any other value raises `NotImplementedError`")
P("  naming the option.")
P("")
P(f"## Ported set ({len(ported)} names in 17 helper families)")
P("")
P("| Helper | Replica file | Replica lines | Articraft lines (own / with private helpers) | Usage | Notes |")
P("| --- | --- | --- | --- | --- | --- |")
for n in ported:
    r = rows[n]
    P(f"| `{n}` | `sdk/{replica_file(n)}` | {R[n][5]} | {r['own']} / {r['total']} | {usage(r)} | {R[n][4]} |")
P("")
P("Shared machinery, written once and used by many helpers: `triangulate_polygon` (ear")
P("clipping, 37 lines), `skin` (rings to a closed solid, 40), `from_created` (merge, clean and")
P("orient, 10), and `sdk/occ.py` (fast OpenCASCADE tessellation and moving exact solids, 72).")
P("The whole ported mesh layer is **1,419 lines, docstrings included, against 5,077**.")
P("")
P("## Full ranked table")
P("")
P("Ranked by decision (ported, skipped, later), then usefulness, then example usage.")
P("")
P("| # | Helper | What it does | Usage | Usefulness | build123d equivalent (1-3 lines) | Decision and why |")
P("| --- | --- | --- | --- | --- | --- | --- |")
for i, n in enumerate(ranked, 1):
    what, useful, b123d, decision, why, _ = R[n]
    P(f"| {i} | `{n}` | {what} | {usage(rows[n])} | {useful} | {b123d} | **{decision}**: {why} |")
P("")
P("`MeshGeometry` methods: `translate` (ex 7), `rotate_y` (ex 1), and `rotate`, `rotate_x`,")
P("`rotate_z`, `scale`, `merge`, `copy`, `add_vertex`, `add_face`, `bounds`, `is_watertight`,")
P("`from_trimesh` and `to_trimesh` (docs only) are **ported**. `set_normal_crease`,")
P("`use_smooth_normals`, `use_hard_normals` and `to_obj` (docs only) are **skipped** for now,")
P("because they are export concerns (step 2). `LoftGeometry.from_shell_profiles` (docs only) is")
P("**skipped**; `LatheGeometry.from_shell_profiles` and `ExtrudeGeometry.centered/from_z0` are")
P("**ported**.")
P("")
P("## Inventory: size and dependencies")
P("")
P("Lines are measured from the AST. \"Own\" is the definition itself; \"with private helpers\" adds")
P("the private functions it calls, but not the shared `MeshGeometry` base (403 lines), which every")
P("helper inherits. Dependencies cover the helper and its private callees.")
P("")
P("| Helper | Articraft file | Own lines | With private helpers | Depends on |")
P("| --- | --- | --- | --- | --- |")
for n in sorted(rows, key=lambda n: (rows[n]["file"], n)):
    r = rows[n]
    P(f"| `{n}` | `{r['file']}` | {r['own']} | {r['total']} | {', '.join(r['deps']) or '(none)'} |")
P("")
P("Not installed in the replica (and not approved): `manifold3d` (Articraft's booleans, lathe")
P("and extrude caps, loft caps), `libigl` (welds, mesh health), and trimesh's polygon")
P("triangulators (`mapbox_earcut`, `triangle`, `shapely`). The replica replaces them with a")
P("small ear-clipping triangulator and OpenCASCADE (inside build123d).")
P("")
P("Regenerate from the project root (Python 3.11+, no packages needed):")
P("")
P("```shell")
P('python -B -I replica/scripts/inventory_mesh_helpers.py "Previous work/Agentic_Systems_Project" inventory.json')
P("python -B -I replica/scripts/make_shape_helpers_doc.py inventory.json replica/docs/SHAPE_HELPERS.md")
P("```")
open(out_path, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("ported", len(ported), "of", len(R))
