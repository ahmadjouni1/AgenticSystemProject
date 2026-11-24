"""Read a version folder (``model.json`` + ``model.glb``) back into a kinematic model.

The rebuilt ``Assembly`` has the same bodies, joints, mimics and loop closures as the
one that was exported. Each body's geometry comes from its GLB mesh, split into connected
pieces (closed solids again), so the model can be solved, posed, or checked again.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh

from sdk import Assembly, Frame, MeshGeometry
from sdk.kinematics import loop_unknowns
from sdk.testing import drivable_joints

MANIFEST_FILE = "model.json"
GLB_FILE = "model.glb"
SIGNALS_FILE = "signals.json"
UNLIMITED_SLIDE = 1.0  # meters each way for a slider on a prismatic joint without limits


@dataclass
class LoadedVersion:
    directory: Path
    manifest: dict[str, Any]
    assembly: Assembly
    drivers: list[dict[str, Any]]
    """One entry per slider: name, type, lower, upper."""
    mode: str
    """"tree" when every pose follows from the drivers alone (the browser can pose it), or
    "loop" when a loop solve is needed (the server poses it)."""
    signals: list[dict[str, Any]]
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def solve(self, values: dict[str, float]) -> dict[str, Any]:
        """Pose the model: world transforms (row-major 4x4, Z-up) and every joint value.
        Raises ``PoseError`` (``LoopClosureError``) if the pose cannot be reached."""
        with self.lock:  # the walking warm start is per model; poses come in order
            pose = self.assembly.solve(values)
        return {
            "bodies": {name: [float(v) for v in matrix.flatten()]
                       for name, matrix in pose.transforms.items()},
            "joints": {name: float(v) for name, v in pose.joint_values.items()},
        }


def load_version(directory: str | Path) -> LoadedVersion:
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    meshes = _glb_meshes(directory / GLB_FILE)
    assembly = Assembly(manifest["name"])
    for body in manifest["bodies"]:
        added = assembly.body(body["name"])
        for name, part in _pieces(meshes[body["name"]]):
            added.add(name, MeshGeometry(part.vertices, part.faces))
    for joint in manifest["joints"]:
        parent = assembly.get_body(joint["parent"])
        child = assembly.get_body(joint["child"])
        mimic = joint["mimic"]
        assembly.joint(
            joint["name"], joint["type"],
            parent.at(Frame(joint["parent_frame"]["xyz"], joint["parent_frame"]["rpy"])),
            child.at(Frame(joint["child_frame"]["xyz"], joint["child_frame"]["rpy"])),
            axis=None if joint["axis"] is None else tuple(joint["axis"]),
            limits=None if joint["limits"] is None else tuple(joint["limits"]),
            mimic=None if mimic is None else (mimic["source"], mimic["multiplier"],
                                              mimic["offset"]),
        )
    assembly.articulation(root=manifest["articulation"]["root"],
                          closures=manifest["articulation"]["closures"])
    signals_path = directory / SIGNALS_FILE
    signals = json.loads(signals_path.read_text(encoding="utf-8")) if signals_path.is_file() \
        else []
    drivers, mode = _sliders(assembly)
    return LoadedVersion(directory, manifest, assembly, drivers, mode, signals)


def _glb_meshes(path: Path) -> dict[str, trimesh.Trimesh]:
    """Body name -> its mesh in body coordinates (the GLB node's own geometry)."""
    scene = trimesh.load(path, force="scene")
    meshes = {}
    for node in scene.graph.nodes_geometry:
        _transform, geometry_name = scene.graph[node]
        meshes[node] = scene.geometry[geometry_name]
    return meshes


def _pieces(mesh: trimesh.Trimesh) -> list[tuple[str, trimesh.Trimesh]]:
    """A body's GLB mesh split back into its connected pieces.

    The export merges a body's shapes into one mesh, which is not watertight where shapes
    overlap; each piece of a closed shape is. Collision depth needs closed solids, so the
    checks and the agent's probe get the pieces, as the original model had separate shapes."""
    # repair=False: never fill holes (that needs networkx, and would change the geometry).
    parts = mesh.split(only_watertight=False, repair=False)
    if len(parts) <= 1:
        return [("visual", mesh)]
    # Open pieces become one shape: a mesh with zero-area triangles can split into thousands
    # of one-triangle fragments, and the collision scene tests shapes pairwise (step 10: a
    # rounded-box cabinet loaded as 35,095 shapes, and one probe took 13 minutes).
    closed = [part for part in parts if part.is_watertight]
    open_parts = [part for part in parts if not part.is_watertight]
    out = [(f"piece{i}", part) for i, part in enumerate(closed, 1)]
    if open_parts:
        out.append(("open", trimesh.util.concatenate(open_parts)))
    return out


def _sliders(assembly: Assembly) -> tuple[list[dict[str, Any]], str]:
    drivers, _members = drivable_joints(assembly)
    art = assembly.validate()
    by_name = {j.name: j for j in assembly.joints}
    needs_solver = bool(art.closures) and bool(loop_unknowns(art, set(drivers), by_name))
    sliders = []
    for joint in drivers.values():
        if joint.limits is not None:
            lower, upper = joint.limits
        elif joint.type.value == "revolute":
            lower, upper = -math.pi, math.pi
        else:
            lower, upper = -UNLIMITED_SLIDE, UNLIMITED_SLIDE
        sliders.append({"name": joint.name, "type": joint.type.value, "lower": lower,
                        "upper": upper})
    return sliders, "loop" if needs_solver else "tree"
