"""Quality metrics of a finished build, and its final renders (step 10).

    quality(run_dir / "result" / "v0003")  # -> {"bodies": 3, "shapes": 8, ...}
    render_final(run_dir, best)            # -> renders/final/{front,side,top,iso,views}.png

Everything is computed by code from what the compile saved: ``model.json`` (bodies, shapes
with their geometry type and triangle count, joints) and that version's ``main.py``.

- bodies, shapes, joints: counts (joints include loop closures; ``movable_joints`` excludes
  fixed ones);
- non_box_share: the share of shapes that are not a plain box (``Box`` or ``BoxGeometry``);
- helpers: the distinct SDK shape helpers ``main.py`` uses, i.e. every geometry constructor or
  mesh function except the plain primitives (Box, Cylinder, Sphere and their mesh forms);
- triangles: visual triangles over all bodies.

A helper counts when ``main.py`` refers to it outside an import (a call, a classmethod such
as ``ExtrudeGeometry.centered``, or a function passed along), so a ``rounded_rect_profile``
that feeds an extrusion counts too, even though the shape's type is ``ExtrudeGeometry``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

PLAIN_BOX = {"Box", "BoxGeometry"}
PRIMITIVES = {"Box", "Cylinder", "Sphere", "BoxGeometry", "CylinderGeometry"}
HELPERS = {
    "CapsuleGeometry", "ConeGeometry", "TorusGeometry", "RoundedBoxGeometry",
    "ExtrudeGeometry", "ExtrudeWithHolesGeometry", "LatheGeometry", "LoftGeometry",
    "PipeGeometry", "SweepGeometry", "section_loft", "wire_from_points",
    "tube_from_spline_points", "sweep_profile_along_spline", "rounded_rect_profile",
    "superellipse_profile", "sample_catmull_rom_spline_2d", "sample_catmull_rom_spline_3d",
    "boolean_union", "boolean_difference", "boolean_intersection", "Build123dShape",
}
VIEWS = ("front", "side", "top", "iso")


def helpers_used(source: str) -> list[str]:
    """SDK shape helpers ``source`` refers to (imports alone do not count)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in HELPERS:
            used.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in HELPERS:
            used.add(node.attr)  # sdk.LatheGeometry(...)
    return sorted(used)


def quality(version_dir: str | Path) -> dict[str, Any]:
    folder = Path(version_dir)
    model = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    shapes = [s for b in model["bodies"] for s in b["shapes"]]
    types: dict[str, int] = {}
    for s in shapes:
        types[s["type"]] = types.get(s["type"], 0) + 1
    main = folder / "main.py"
    helpers = helpers_used(main.read_text(encoding="utf-8")) if main.is_file() else []
    boxes = sum(n for t, n in types.items() if t in PLAIN_BOX)
    return {
        "bodies": len(model["bodies"]),
        "shapes": len(shapes),
        "joints": len(model["joints"]),
        "movable_joints": sum(j["type"] != "fixed" for j in model["joints"]),
        "non_box_share": round(1 - boxes / len(shapes), 4) if shapes else 0.0,
        "helpers": helpers,
        "helper_count": len(helpers),
        "triangles": sum(b["triangles"] for b in model["bodies"]),
        "shape_types": dict(sorted(types.items())),
    }


def render_final(run_dir: str | Path, best: dict[str, Any] | None) -> list[str]:
    """The best build at rest, 4 views plus a 2x2 sheet, in ``renders/final/``. Returns the
    paths (relative to the run folder); [] if there is no build."""
    from PIL import Image

    from agent.tools.builds import Build, world_meshes
    from agent.tools.render import HEIGHT, WIDTH, build_front, rasterize, save_png, views
    from viewer.loader import load_version

    if best is None:
        return []
    run_dir = Path(run_dir)
    build = Build(best["version"], load_version(run_dir / best["path"]))
    meshes = world_meshes(build, build.loaded.assembly.solve({}).transforms)
    directions = views(build_front(build))  # facing the side with the moving parts
    folder = run_dir / "renders" / "final"
    folder.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (2 * WIDTH, 2 * HEIGHT), (255, 255, 255))
    paths = []
    for i, view in enumerate(VIEWS):
        forward, right, label = directions[view]
        image = rasterize(meshes, forward, f"{best['version']}  {label}", right)
        save_png(image, folder / f"{view}.png")
        sheet.paste(image, ((i % 2) * WIDTH, (i // 2) * HEIGHT))
        paths.append(f"renders/final/{view}.png")
    save_png(sheet, folder / "views.png")
    paths.append("renders/final/views.png")
    return paths


__all__ = ["HELPERS", "helpers_used", "quality", "render_final"]
