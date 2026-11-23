# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Boolean operations on solids: union, difference, intersection (Articraft's names).

Two engines, chosen per call:

- **OpenCASCADE** (inside build123d), when *every* input is exact: a build123d shape, or
  a ``MeshGeometry`` that still carries its exact solid (``RoundedBoxGeometry``,
  ``ExtrudeWithHolesGeometry``, an earlier exact result). The result is exact and keeps
  its solid, so chained cuts stay exact.
- **manifold3d** (Articraft's engine), otherwise. Inputs are meshes; a build123d input
  is tessellated first. It works in float64 (``Mesh64``) and takes milliseconds.

``boolean(..., engine="occ")`` forces OpenCASCADE everywhere: each mesh becomes a sewn
polyhedral solid, which is exact but about 1 ms per triangle. It is only kept for
benchmarks and as a fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from sdk.errors import ValidationError
from sdk.geometry import DEFAULT_TOLERANCE, as_geometry, is_build123d
from sdk.mesh_geometry import MeshGeometry, from_created
from sdk.occ import tessellate

_OPERATIONS = ("union", "difference", "intersection")


def _exact(geometry: Any) -> Any | None:
    if is_build123d(geometry) and hasattr(geometry, "wrapped"):
        return geometry
    if isinstance(geometry, MeshGeometry):
        return geometry.brep()
    return None


def _closed_mesh(geometry: Any, name: str) -> trimesh.Trimesh:
    if is_build123d(geometry) and hasattr(geometry, "wrapped"):
        vertices, faces = from_created(tessellate(geometry, DEFAULT_TOLERANCE, 0.1))
        mesh = trimesh.Trimesh(vertices, faces, process=False)
    else:
        mesh = as_geometry(geometry).to_trimesh()
        mesh.merge_vertices()
    if not mesh.is_watertight:
        raise ValidationError(f"boolean input {name} must be a closed (watertight) solid")
    return mesh


# ---- manifold3d ------------------------------------------------------------------------


def to_manifold(mesh: trimesh.Trimesh, name: str = "mesh") -> Any:
    import manifold3d

    result = manifold3d.Manifold(manifold3d.Mesh64(
        vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint64),
    ))
    if result.status() != manifold3d.Error.NoError:
        raise ValidationError(f"{name} is not a closed manifold solid ({result.status().name})")
    return result


def from_manifold(solid: Any, operation: str) -> MeshGeometry:
    out = solid.to_mesh64()
    faces = np.asarray(out.tri_verts, dtype=np.int64)
    if len(faces) == 0:
        raise ValidationError(f"{operation} produced an empty result")
    vertices = np.asarray(out.vert_properties, dtype=np.float64)[:, :3]
    return MeshGeometry(*from_created(trimesh.Trimesh(vertices, faces, process=False)))


def _with_manifold(operation: str, a: Any, b: Any) -> MeshGeometry:
    ma = to_manifold(_closed_mesh(a, "a"), "boolean input a")
    mb = to_manifold(_closed_mesh(b, "b"), "boolean input b")
    solid = {"union": ma + mb, "difference": ma - mb, "intersection": ma ^ mb}[operation]
    return from_manifold(solid, f"boolean_{operation}")


# ---- OpenCASCADE -----------------------------------------------------------------------


def _polyhedron(mesh: trimesh.Trimesh) -> Any:
    from build123d import Solid
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    points = [gp_Pnt(*map(float, v)) for v in mesh.vertices]
    sewing = BRepBuilderAPI_Sewing(1e-9)
    for a, b, c in mesh.faces:
        polygon = BRepBuilderAPI_MakePolygon(points[a], points[b], points[c], True)
        sewing.Add(BRepBuilderAPI_MakeFace(polygon.Wire(), True).Face())
    sewing.Perform()
    shells = TopExp_Explorer(sewing.SewedShape(), TopAbs_SHELL)
    if not shells.More():
        raise ValidationError("mesh could not be sewn into a closed shell")
    solid = Solid(BRepBuilderAPI_MakeSolid(TopoDS.Shell(shells.Current())).Solid())
    if solid.volume < 0:  # faces wound inward
        solid = Solid(solid.wrapped.Reversed())
    return solid


def _with_occ(operation: str, a: Any, b: Any) -> MeshGeometry:
    sa = _exact(a) or _polyhedron(_closed_mesh(a, "a"))
    sb = _exact(b) or _polyhedron(_closed_mesh(b, "b"))
    shape = {"union": sa + sb, "difference": sa - sb, "intersection": sa & sb}[operation]
    if not shape.faces():
        raise ValidationError(f"boolean_{operation} produced an empty result")
    result = MeshGeometry(*from_created(tessellate(shape, DEFAULT_TOLERANCE, 0.1)))
    return result.attach_brep(shape)


# ---- public ----------------------------------------------------------------------------


def boolean(operation: str, a: Any, b: Any, *, engine: str = "auto") -> MeshGeometry:
    """``engine``: "auto" (OpenCASCADE if both inputs are exact, else manifold3d),
    "manifold" or "occ" to force one."""
    if operation not in _OPERATIONS:
        raise ValidationError(f"boolean operation must be one of {_OPERATIONS}")
    if engine == "auto":
        engine = "occ" if _exact(a) is not None and _exact(b) is not None else "manifold"
    if engine == "occ":
        return _with_occ(operation, a, b)
    if engine == "manifold":
        return _with_manifold(operation, a, b)
    raise ValidationError("engine must be 'auto', 'manifold' or 'occ'")


def boolean_union(a: Any, b: Any) -> MeshGeometry:
    return boolean("union", a, b)


def boolean_difference(a: Any, b: Any) -> MeshGeometry:
    return boolean("difference", a, b)


def boolean_intersection(a: Any, b: Any) -> MeshGeometry:
    return boolean("intersection", a, b)
