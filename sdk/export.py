"""Export an assembly as URDF + GLB + meshes + a JSON manifest, into a new version folder.

    result/
      v0000/  model.urdf  model.glb  model.json  meshes/<body>.obj  meshes/<body>_collision.obj
              <source file snapshot>
      v0001/  ...

Every call writes a new ``vNNNN`` folder. It is built in a temporary folder and renamed
into place at the end, so a version is either complete or absent, and never overwritten.

**Frames.** A URDF child link's frame sits on its joint frame, but our bodies keep their
own frames (joint = parent attachment ``F_p`` + child attachment ``F_c``). So each
link's frame is ``F_c`` in body coordinates (identity for the root), and the body's
meshes and inertia are placed in the link with origin ``inverse(F_c)``. A joint's origin
is ``inverse(F_c(parent link)) @ F_p``. URDF forward kinematics then gives exactly the
body poses that ``Assembly.solve`` gives.

**Loops.** URDF is a tree. Loop-closure joints are left out of the URDF and listed in
the manifest (``closures``), with their frames, so a loop-aware loader can add them back.

**GLB.** One node per body, nested like the joint tree, with the rest pose as node
transforms. To pose it, set each body node's local matrix to
``F_p @ motion(q) @ inverse(F_c)`` (all three are in the manifest). glTF is Y-up, so
the bodies hang under one root node, ``z_up_to_y_up``, that turns our Z-up frame into
Y-up.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from sdk import kinematics
from sdk.assembly import Assembly
from sdk.errors import ValidationError
from sdk.frames import Frame
from sdk.joint import Joint, JointType
from sdk.transforms import invert, rpy_from_matrix

SCHEMA_VERSION = 1
DEFAULT_DENSITY = 1000.0
"""kg/m^3 for bodies without their own density (about water / light plastic)."""
COLLISION_MAX_FACES = 5000
"""A body with more visual triangles than this gets a simplified collision mesh."""
DEFAULT_EFFORT = 100.0
"""URDF ``<limit effort>`` (N or N*m). Required by URDF; not modeled yet."""
DEFAULT_VELOCITY = 2.0
"""URDF ``<limit velocity>`` (m/s or rad/s). Required by URDF; not modeled yet."""
UNLIMITED_TRAVEL = 1000.0
"""URDF prismatic joints must have limits; an unlimited slide gets +-this many meters."""
GLB_ROOT = "z_up_to_y_up"
URDF_FILE, GLB_FILE, MANIFEST_FILE = "model.urdf", "model.glb", "model.json"
_VERSION = re.compile(r"v(\d{4,})")
_Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class ExportResult:
    version: int
    directory: Path
    urdf: Path
    glb: Path
    manifest: Path
    meshes: dict[str, Path]
    """Body name -> visual OBJ. The collision OBJ sits next to it (``<body>_collision.obj``)."""
    source: Path | None


def export(
    assembly: Assembly,
    out_dir: str | os.PathLike[str],
    version: int | None = None,
    *,
    source: str | os.PathLike[str] | None = None,
    density: float = DEFAULT_DENSITY,
    densities: Mapping[str, float] | None = None,
    collision_max_faces: int = COLLISION_MAX_FACES,
) -> ExportResult:
    """Validate ``assembly`` and write it to ``out_dir/vNNNN``.

    ``version`` defaults to the next free number. An explicit version that already
    exists raises ``FileExistsError``. ``source`` is a file (later ``main.py``) that is
    copied next to the export and hashed into the manifest. ``density`` (kg/m^3) gives
    every body's mass from its volume, and ``densities`` overrides it per body.
    """
    art = assembly.validate()  # refuses an invalid assembly before anything is written
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    number = _next_version(root) if version is None else int(version)
    final = root / f"v{number:04d}"
    if final.exists():
        raise FileExistsError(f"export version already exists: {final}")
    source_path = None if source is None else Path(source)
    if source_path is not None and not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")
    if density <= 0 or not math.isfinite(density):
        raise ValidationError("density must be finite and positive")

    staging = root / f".v{number:04d}-{uuid.uuid4().hex[:8]}.tmp"
    staging.mkdir()
    try:
        _write_version(assembly, art, staging, number, source_path, density,
                       dict(densities or {}), collision_max_faces)
        os.rename(staging, final)  # fails if another export took the number meanwhile
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ExportResult(
        version=number,
        directory=final,
        urdf=final / URDF_FILE,
        glb=final / GLB_FILE,
        manifest=final / MANIFEST_FILE,
        meshes={b.name: final / "meshes" / f"{_file_stem(b.name, assembly)}.obj"
                for b in assembly.bodies},
        source=None if source_path is None else final / source_path.name,
    )


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a ``model.json`` (file or version folder)."""
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8"))


# ---- one version -----------------------------------------------------------------------


def _write_version(assembly, art, folder: Path, number: int, source: Path | None,
                   density: float, densities: dict[str, float], max_faces: int) -> None:
    unknown = sorted(set(densities) - {b.name for b in assembly.bodies})
    if unknown:
        raise ValidationError(f"densities given for unknown bodies: {unknown}")
    (folder / "meshes").mkdir()
    # The rest pose (every joint at 0, mimics at their offset). Computed directly so the
    # export does not disturb the assembly's warm start; validate() proved it closes.
    zeros = {j.name: 0.0 for j in art.tree if j.movable and j.mimic is None}
    rest = kinematics.solve(art, zeros, [], {})
    link_frames = {art.root: np.eye(4)}
    for joint in art.tree:
        link_frames[joint.child.body.name] = joint.child.frame.matrix

    bodies = []
    for body in assembly.bodies:
        stem = _file_stem(body.name, assembly)
        visual = body.trimesh()
        visual.merge_vertices()
        collision, method = _collision_mesh(body, max_faces)
        _write_obj(visual, folder / "meshes" / f"{stem}.obj")
        _write_obj(collision, folder / "meshes" / f"{stem}_collision.obj")
        rho = float(densities.get(body.name, density))
        mass, com, inertia = _mass_properties(body, rho)
        bodies.append({
            "name": body.name,
            "visual_mesh": f"meshes/{stem}.obj",
            "collision_mesh": f"meshes/{stem}_collision.obj",
            "collision_method": method,
            "triangles": int(len(visual.faces)),
            "collision_triangles": int(len(collision.faces)),
            "link_frame": _frame_dict(link_frames[body.name]),
            "density": rho,
            "mass": mass,
            "center_of_mass": com,
            "inertia": inertia,
            "shapes": [{"name": s.name, "type": type(s.geometry).__name__,
                        "xyz": list(s.frame.xyz), "rpy": list(s.frame.rpy),
                        "triangles": int(len(s.to_trimesh().faces))} for s in body.shapes],
        })

    source_info = None
    if source is not None:
        data = source.read_bytes()
        (folder / source.name).write_bytes(data)
        source_info = {"file": source.name, "sha256": hashlib.sha256(data).hexdigest()}

    urdf = _urdf(assembly, art, bodies, link_frames)
    ET.indent(urdf)
    (folder / URDF_FILE).write_bytes(
        ET.tostring(urdf, encoding="utf-8", xml_declaration=True) + b"\n"
    )
    _glb(assembly, art, rest.transforms, folder / GLB_FILE)
    manifest = _manifest(assembly, art, rest, bodies, number, source_info)
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _next_version(root: Path) -> int:
    numbers = [int(m.group(1)) for p in root.iterdir() if (m := _VERSION.fullmatch(p.name))]
    return max(numbers) + 1 if numbers else 0


def _file_stem(name: str, assembly: Assembly) -> str:
    """Body names differing only in case would collide on Windows; number the later ones."""
    lowered = [b.name.lower() for b in assembly.bodies]
    index = [b.name for b in assembly.bodies].index(name)
    clashes = lowered[:index].count(name.lower())
    return name if clashes == 0 else f"{name}_{clashes + 1}"


def _write_obj(mesh: trimesh.Trimesh, path: Path) -> None:
    text = trimesh.exchange.obj.export_obj(
        mesh, include_normals=False, include_color=False, include_texture=False
    )
    path.write_text(text, encoding="utf-8")


# ---- meshes and mass -----------------------------------------------------------------------


def _collision_mesh(body, max_faces: int) -> tuple[trimesh.Trimesh, str]:
    """The visual mesh, or (if larger than ``max_faces``) each big shape simplified."""
    meshes = [shape.to_trimesh() for shape in body.shapes]
    total = sum(len(m.faces) for m in meshes)
    if total <= max_faces:
        combined = trimesh.util.concatenate(meshes)
        combined.merge_vertices()
        return combined, "copy"
    simplified, methods = [], set()
    for mesh in meshes:
        budget = max(12, int(max_faces * len(mesh.faces) / total))
        if len(mesh.faces) <= budget:
            simplified.append(mesh)
            continue
        reduced, method = _simplify(mesh, budget)
        simplified.append(reduced)
        methods.add(method)
    combined = trimesh.util.concatenate(simplified)
    combined.merge_vertices()
    return combined, "+".join(sorted(methods))


def _simplify(mesh: trimesh.Trimesh, budget: int) -> tuple[trimesh.Trimesh, str]:
    """Shrink a closed mesh with manifold3d's simplify (surfaces move less than the
    tolerance, which doubles until the budget is met); an open mesh gets its convex hull."""
    from sdk.booleans import from_manifold, to_manifold

    mesh = mesh.copy()
    mesh.merge_vertices()
    if not mesh.is_watertight:
        return mesh.convex_hull, "convex_hull"
    solid = to_manifold(mesh)
    tolerance = 1e-4 * float(np.linalg.norm(mesh.extents))
    for _ in range(24):
        reduced = solid.simplify(tolerance)
        if reduced.num_tri() <= budget:
            break
        tolerance *= 2
    result = from_manifold(reduced, "simplify").to_trimesh()
    return result, "manifold_simplify"


def _mass_properties(body, density: float) -> tuple[float, list[float], list[list[float]]]:
    """Mass from volume x density; center of mass and inertia (about the center of mass,
    body axes). Open shapes use their convex hull; overlapping shapes are counted twice."""
    solids = []
    for shape in body.shapes:
        mesh = shape.to_trimesh()
        mesh.merge_vertices()
        if not mesh.is_watertight or mesh.volume <= 0:
            try:
                mesh = mesh.convex_hull
            except Exception:  # qhull refuses flat or degenerate point sets
                mesh = None
            if mesh is None or not mesh.volume > 1e-15:
                raise ValidationError(
                    f"body {body.name!r} shape {shape.name!r} encloses no volume (an open or "
                    "flat mesh); give it thickness so it has a mass"
                )
        solids.append(mesh)
    combined = trimesh.util.concatenate(solids)
    properties = trimesh.triangles.mass_properties(combined.triangles, density=density)
    mass = float(properties["mass"])
    if not math.isfinite(mass) or mass <= 0:
        raise ValidationError(f"body {body.name!r} has no volume to give it mass")
    return (mass, [float(v) for v in properties["center_mass"]],
            [[float(v) for v in row] for row in properties["inertia"]])


# ---- URDF ------------------------------------------------------------------------------------


def _number(value: float) -> str:
    """12 significant digits; float noise below 1e-15 (and -0) is written as 0."""
    value = float(value)
    return "0" if abs(value) < 1e-15 else f"{value:.12g}"


def _xyz_rpy(transform: np.ndarray) -> dict[str, str]:
    rpy = rpy_from_matrix(transform[:3, :3])
    return {"xyz": " ".join(_number(v) for v in transform[:3, 3]),
            "rpy": " ".join(_number(v) for v in rpy)}


def _urdf_type(joint: Joint) -> str:
    if joint.type is JointType.REVOLUTE and joint.limits is None:
        return "continuous"
    return joint.type.value


def _urdf(assembly, art, bodies: list[dict], link_frames: dict[str, np.ndarray]) -> ET.Element:
    robot = ET.Element("robot", name=assembly.name)
    for info in bodies:
        name = info["name"]
        to_link = invert(link_frames[name])  # body coordinates -> link coordinates
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        com = np.eye(4)
        com[:3, 3] = info["center_of_mass"]
        ET.SubElement(inertial, "origin", _xyz_rpy(to_link @ com))
        ET.SubElement(inertial, "mass", value=_number(info["mass"]))
        i = info["inertia"]
        ET.SubElement(inertial, "inertia", ixx=_number(i[0][0]), ixy=_number(i[0][1]),
                      ixz=_number(i[0][2]), iyy=_number(i[1][1]), iyz=_number(i[1][2]),
                      izz=_number(i[2][2]))
        for tag, key in (("visual", "visual_mesh"), ("collision", "collision_mesh")):
            element = ET.SubElement(link, tag)
            ET.SubElement(element, "origin", _xyz_rpy(to_link))
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "mesh", filename=info[key])
    for joint in art.tree:
        element = ET.SubElement(robot, "joint", name=joint.name, type=_urdf_type(joint))
        ET.SubElement(element, "parent", link=joint.parent.body.name)
        ET.SubElement(element, "child", link=joint.child.body.name)
        origin = invert(link_frames[joint.parent.body.name]) @ joint.parent.frame.matrix
        ET.SubElement(element, "origin", _xyz_rpy(origin))
        if not joint.movable:
            continue
        ET.SubElement(element, "axis", xyz=" ".join(_number(v) for v in joint.axis))
        limits = joint.limits
        if joint.type is JointType.PRISMATIC and limits is None:
            limits = (-UNLIMITED_TRAVEL, UNLIMITED_TRAVEL)
        attributes = {"effort": _number(DEFAULT_EFFORT), "velocity": _number(DEFAULT_VELOCITY)}
        if limits is not None:
            attributes = {"lower": _number(limits[0]), "upper": _number(limits[1]), **attributes}
        ET.SubElement(element, "limit", attributes)
        if joint.mimic is not None:
            ET.SubElement(element, "mimic", joint=joint.mimic.source,
                          multiplier=_number(joint.mimic.multiplier),
                          offset=_number(joint.mimic.offset))
    return robot


# ---- GLB ------------------------------------------------------------------------------------


def _glb(assembly, art, rest: Mapping[str, np.ndarray], path: Path) -> None:
    scene = trimesh.Scene()
    scene.graph.update(frame_from=scene.graph.base_frame, frame_to=GLB_ROOT, matrix=_Z_UP_TO_Y_UP)
    parents = {art.root: GLB_ROOT}
    for joint in art.tree:
        parents[joint.child.body.name] = joint.parent.body.name
    order = [art.root] + [j.child.body.name for j in art.tree]  # every parent first
    for name in order:
        parent = parents[name]
        local = rest[name] if parent == GLB_ROOT else invert(rest[parent]) @ rest[name]
        mesh = assembly.get_body(name).trimesh()
        mesh.merge_vertices()
        scene.add_geometry(mesh, node_name=name, geom_name=name, parent_node_name=parent,
                           transform=local)
    path.write_bytes(scene.export(file_type="glb"))


# ---- manifest ----------------------------------------------------------------------------------


def _frame_dict(matrix: np.ndarray) -> dict[str, list[float]]:
    frame = Frame.from_matrix(matrix)
    return {"xyz": list(frame.xyz), "rpy": list(frame.rpy)}


def _manifest(assembly, art, rest, bodies, number: int, source_info) -> dict[str, Any]:
    closures = set(art.closures)

    def joint_dict(joint: Joint) -> dict[str, Any]:
        return {
            "name": joint.name,
            "type": joint.type.value,
            "role": "closure" if joint in closures else "tree",
            "urdf_type": None if joint in closures else _urdf_type(joint),
            "parent": joint.parent.body.name,
            "child": joint.child.body.name,
            "parent_frame": {"xyz": list(joint.parent.frame.xyz),
                             "rpy": list(joint.parent.frame.rpy)},
            "child_frame": {"xyz": list(joint.child.frame.xyz), "rpy": list(joint.child.frame.rpy)},
            "axis": None if joint.axis is None else list(joint.axis),
            "limits": None if joint.limits is None else list(joint.limits),
            "mimic": None if joint.mimic is None else {
                "source": joint.mimic.source, "multiplier": joint.mimic.multiplier,
                "offset": joint.mimic.offset},
        }

    notes = ["URDF holds the joint tree only; loop closures are listed under 'closures'."]
    if any(j.type is JointType.PRISMATIC and j.limits is None for j in art.tree):
        notes.append(f"Unlimited prismatic joints are written to URDF as +-{UNLIMITED_TRAVEL} m.")
    return {
        "schema_version": SCHEMA_VERSION,
        "name": assembly.name,
        "version": number,
        "units": {"length": "m", "angle": "rad", "mass": "kg", "density": "kg/m^3"},
        "up_axis": "Z",
        "files": {"urdf": URDF_FILE, "glb": GLB_FILE,
                  "source": None if source_info is None else source_info["file"]},
        "source": source_info,
        "bodies": bodies,
        "joints": [joint_dict(j) for j in assembly.joints],
        "articulation": {"root": art.root, "tree": [j.name for j in art.tree],
                         "closures": [j.name for j in art.closures]},
        "closures": [joint_dict(j) for j in art.closures],
        "mimics": [{"joint": j.name, "source": j.mimic.source, "multiplier": j.mimic.multiplier,
                    "offset": j.mimic.offset} for j in art.mimic_order],
        "rest_pose": {
            "joint_values": dict(rest.joint_values),
            "body_transforms": {name: [[float(v) for v in row] for row in matrix]
                                for name, matrix in rest.transforms.items()},
        },
        "glb": {"up_axis": "Y", "root_node": GLB_ROOT,
                "node_matrix": "parent_frame @ motion(q) @ inverse(child_frame)"},
        "urdf": {"effort": DEFAULT_EFFORT, "velocity": DEFAULT_VELOCITY,
                 "link_frame": "child_frame of the joint placing the body (identity for root)"},
        "notes": notes,
    }
