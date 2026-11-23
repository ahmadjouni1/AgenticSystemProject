"""Small OpenCASCADE helpers used by the mesh code (through build123d's OCP bindings)."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from sdk.errors import ValidationError


def tessellate(shape: Any, tolerance: float, angular_tolerance: float) -> trimesh.Trimesh:
    """Triangulate a build123d shape (or a TopoDS shape) into one trimesh.

    Reads OpenCASCADE's per-face triangulations directly. On shapes with thousands of
    faces (a faceted boolean result) this is about 8x faster than ``Shape.tessellate``.
    The caller still merges seam vertices.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    topods = getattr(shape, "wrapped", shape)
    BRepMesh_IncrementalMesh(topods, float(tolerance), False, float(angular_tolerance), True)
    vertices, faces, offset = [], [], 0
    explorer = TopExp_Explorer(topods, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        explorer.Next()
        if triangulation is None:
            continue
        trsf = location.Transformation()
        count = triangulation.NbNodes()
        points = [triangulation.Node(i).Transformed(trsf) for i in range(1, count + 1)]
        vertices.append(np.array([(p.X(), p.Y(), p.Z()) for p in points]))
        triangles = np.array(
            [triangulation.Triangle(i).Get() for i in range(1, triangulation.NbTriangles() + 1)]
        ) - 1
        if face.Orientation() == TopAbs_REVERSED:
            triangles = triangles[:, ::-1]
        faces.append(triangles + offset)
        offset += count
    if not faces:
        raise ValidationError("shape tessellated to an empty mesh")
    return trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=False)


def transformed(shape: Any, transform: np.ndarray) -> Any | None:
    """The build123d shape moved by a rigid (or uniformly scaled) 4x4, else ``None``."""
    from build123d import Shape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    t = np.asarray(transform, dtype=float)
    linear = t[:3, :3]
    scale = np.cbrt(np.linalg.det(linear))
    if scale <= 0 or not np.allclose(linear.T @ linear, np.eye(3) * scale**2, atol=1e-9):
        return None  # mirror or non-uniform scale: let the caller drop the exact solid
    trsf = gp_Trsf()
    rotation = linear / scale
    trsf.SetValues(*rotation[0], t[0, 3], *rotation[1], t[1, 3], *rotation[2], t[2, 3])
    if not np.isclose(scale, 1.0):
        scaling = gp_Trsf()
        scaling.SetScaleFactor(float(scale))
        trsf.Multiply(scaling)
    return Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, trsf, True).Shape())
