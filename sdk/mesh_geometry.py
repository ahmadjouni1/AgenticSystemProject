# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""``MeshGeometry``: a small, mutable triangle mesh in meters (Articraft's name and methods).

Every ported mesh helper (lathe, loft, pipe, ...) is a ``MeshGeometry``. Transform
methods change the mesh in place and return it, so they chain:
``LatheGeometry(profile).rotate_y(math.pi / 2).translate(0, 0, 0.1)``.

Also here: the shared machinery the helpers use, which is

- ``triangulate_polygon``: ear clipping for simple 2D polygons (cap faces);
- ``skin``: join rings of matching points into a tube, with optional end caps.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import trimesh

from sdk.errors import ValidationError
from sdk.transforms import axis_angle_matrix, unit, vec3

_EPS = 1e-12


class MeshGeometry:
    def __init__(
        self,
        vertices: Iterable[Sequence[float]] = (),
        faces: Iterable[Sequence[int]] = (),
        normal_crease_angle: float | None = None,
    ) -> None:
        v = np.asarray(list(vertices) if not isinstance(vertices, np.ndarray) else vertices,
                       dtype=float)
        f = np.asarray(list(faces) if not isinstance(faces, np.ndarray) else faces)
        v = v.reshape(0, 3) if v.size == 0 else v
        f = f.reshape(0, 3).astype(np.int64) if f.size == 0 else f
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValidationError("mesh vertices must have shape (N, 3)")
        if not np.all(np.isfinite(v)):
            raise ValidationError("mesh vertices must be finite")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValidationError("mesh faces must have shape (M, 3)")
        if len(f) and not np.issubdtype(f.dtype, np.integer):
            raise ValidationError("mesh faces must be integer vertex indices")
        if len(f) and (f.min() < 0 or f.max() >= len(v)):
            raise ValidationError("mesh faces reference a vertex index out of range")
        self.vertices: np.ndarray = v.copy()
        self.faces: np.ndarray = f.astype(np.int64)
        self.normal_crease_angle = normal_crease_angle
        self._brep: object | None = None
        self._brep_key: tuple[int, int] | None = None

    # ---- the exact solid behind a mesh (when it came from build123d) ------------------

    def _fingerprint(self) -> tuple[int, int]:
        return hash(self.vertices.tobytes()), hash(self.faces.tobytes())

    def attach_brep(self, shape: object) -> MeshGeometry:
        """Remember the exact build123d solid this mesh was tessellated from.

        Booleans use it instead of the triangles (exact and much faster). It is kept
        only while the mesh is unchanged; a rigid transform moves it along, and any
        other edit (merge, scale, editing vertices) quietly drops it.
        """
        self._brep, self._brep_key = shape, self._fingerprint()
        return self

    def brep(self) -> object | None:
        if self._brep is None or self._brep_key != self._fingerprint():
            return None
        return self._brep

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vertices={len(self.vertices)}, faces={len(self.faces)})"

    # ---- conversion -------------------------------------------------------------

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh) -> MeshGeometry:
        result = cls.__new__(cls)
        MeshGeometry.__init__(result, mesh.vertices, mesh.faces)
        return result

    def to_trimesh(self, *, process: bool = False) -> trimesh.Trimesh:
        if len(self.faces) == 0:
            raise ValidationError(f"{type(self).__name__} is empty")
        return trimesh.Trimesh(self.vertices.copy(), self.faces.copy(), process=process)

    def copy(self) -> MeshGeometry:
        duplicate = MeshGeometry(self.vertices, self.faces, self.normal_crease_angle)
        exact = self.brep()
        return duplicate.attach_brep(exact) if exact is not None else duplicate

    # ---- queries ------------------------------------------------------------------

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if len(self.vertices) == 0:
            raise ValidationError("an empty mesh has no bounds")
        return tuple(self.vertices.min(axis=0)), tuple(self.vertices.max(axis=0))

    def is_watertight(self) -> bool:
        return len(self.faces) > 0 and self.to_trimesh().is_watertight

    # ---- editing (in place, chainable) --------------------------------------------

    def add_vertex(self, x: float, y: float, z: float) -> int:
        self.vertices = np.vstack([self.vertices, vec3((x, y, z), field="vertex")])
        return len(self.vertices) - 1

    def add_face(self, a: int, b: int, c: int) -> None:
        face = np.array([[a, b, c]], dtype=np.int64)
        if face.min() < 0 or face.max() >= len(self.vertices):
            raise ValidationError("face references a vertex index out of range")
        self.faces = np.vstack([self.faces, face])

    def merge(self, other: MeshGeometry) -> MeshGeometry:
        if not isinstance(other, MeshGeometry):
            raise ValidationError("merge needs a MeshGeometry")
        offset = len(self.vertices)
        self.vertices = np.vstack([self.vertices, other.vertices])
        self.faces = np.vstack([self.faces, other.faces + offset])
        return self

    def apply_transform(self, transform: np.ndarray) -> MeshGeometry:
        t = np.asarray(transform, dtype=float)
        exact = self.brep()
        self.vertices = self.vertices @ t[:3, :3].T + t[:3, 3]
        if np.linalg.det(t[:3, :3]) < 0:  # a mirror turns the mesh inside out
            self.faces = self.faces[:, ::-1].copy()
        if exact is not None:
            from sdk.occ import transformed

            moved = transformed(exact, t)
            if moved is not None:
                self.attach_brep(moved)
        return self

    def translate(self, dx: float, dy: float, dz: float) -> MeshGeometry:
        offset = vec3((dx, dy, dz), field="translation")
        return self.apply_transform(np.array(
            [[1, 0, 0, offset[0]], [0, 1, 0, offset[1]], [0, 0, 1, offset[2]], [0, 0, 0, 1]],
            dtype=float,
        ))

    def scale(self, sx: float, sy: float | None = None, sz: float | None = None) -> MeshGeometry:
        factors = vec3((sx, sx if sy is None else sy, sx if sz is None else sz), field="scale")
        if any(abs(f) < _EPS for f in factors):
            raise ValidationError("scale factors must be non-zero")
        return self.apply_transform(np.diag([*factors, 1.0]))

    def rotate(
        self, axis: Sequence[float], angle: float, *, origin: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> MeshGeometry:
        rotation = axis_angle_matrix(unit(axis, field="rotation axis"), float(angle))
        pivot = np.asarray(vec3(origin, field="rotation origin"))
        t = np.eye(4)
        t[:3, :3] = rotation
        t[:3, 3] = pivot - rotation @ pivot
        return self.apply_transform(t)

    def rotate_x(self, angle: float) -> MeshGeometry:
        return self.rotate((1.0, 0.0, 0.0), angle)

    def rotate_y(self, angle: float) -> MeshGeometry:
        return self.rotate((0.0, 1.0, 0.0), angle)

    def rotate_z(self, angle: float) -> MeshGeometry:
        return self.rotate((0.0, 0.0, 1.0), angle)


def from_created(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Clean a freshly generated trimesh: merge seams, drop slivers, face outward."""
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    faces = mesh.faces
    if mesh.is_watertight and mesh.volume < 0:
        faces = faces[:, ::-1]
    return np.asarray(mesh.vertices), np.asarray(faces)


# ---- 2D profiles --------------------------------------------------------------------


def profile_2d(profile: Iterable[Sequence[float]], *, minimum: int = 3, name: str = "profile"
               ) -> np.ndarray:
    """Validate a 2D point list; drop a repeated closing point."""
    try:
        points = np.asarray([(float(p[0]), float(p[1])) for p in profile], dtype=float)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValidationError(f"{name} must be a list of (x, y) points") from exc
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < minimum or not np.all(np.isfinite(points)):
        raise ValidationError(f"{name} needs at least {minimum} finite points")
    return points


def signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    return points if signed_area(points) >= 0 else points[::-1].copy()


def triangulate_polygon(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clip a simple 2D polygon. Triangles are counter-clockwise; indices are into
    ``points``, and every point is used, so a cap shares every edge with its tube."""
    points = np.asarray(points, dtype=float)
    order = list(range(len(points)))
    if signed_area(points) < 0:
        order.reverse()

    def cross(o: int, a: int, b: int) -> float:
        (ox, oy), (ax, ay), (bx, by) = points[o], points[a], points[b]
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)

    def inside(p: int, a: int, b: int, c: int) -> bool:
        return cross(a, b, p) >= -_EPS and cross(b, c, p) >= -_EPS and cross(c, a, p) >= -_EPS

    triangles: list[tuple[int, int, int]] = []
    while len(order) > 3:
        count = len(order)
        ear = None
        for k in range(count):
            a, b, c = order[k - 1], order[k], order[(k + 1) % count]
            if cross(a, b, c) <= _EPS:
                continue  # reflex or straight: not an ear
            others = (p for p in order if p not in (a, b, c))
            if not any(inside(p, a, b, c) for p in others):
                ear = k
                break
        if ear is None:  # only straight runs left: clip one anyway (zero-area triangle)
            ear = next((k for k in range(count)
                        if abs(cross(order[k - 1], order[k], order[(k + 1) % count])) <= _EPS),
                       None)
            if ear is None:
                raise ValidationError("profile polygon intersects itself")
        triangles.append((order[ear - 1], order[ear], order[(ear + 1) % count]))
        del order[ear]
    triangles.append((order[0], order[1], order[2]))
    return triangles


# ---- rings -> solid ------------------------------------------------------------------


def newell_normal(ring: np.ndarray) -> np.ndarray:
    nxt = np.roll(ring, -1, axis=0)
    normal = np.cross(ring, nxt).sum(axis=0)
    norm = np.linalg.norm(normal)
    return normal / norm if norm > _EPS else normal


def _cap_triangles(ring: np.ndarray, outward: np.ndarray) -> list[tuple[int, int, int]]:
    normal = newell_normal(ring)
    if np.linalg.norm(normal) < _EPS:
        raise ValidationError("a cap profile has no area")
    u = ring[1] - ring[0]
    u = u - u.dot(normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    flat = np.column_stack([(ring - ring[0]) @ u, (ring - ring[0]) @ v])
    triangles = triangulate_polygon(flat)  # counter-clockwise about ``normal``
    if normal.dot(outward) < 0:
        triangles = [(a, c, b) for a, b, c in triangles]
    return triangles


def skin(
    rings: Sequence[np.ndarray],
    *,
    closed_profile: bool = True,
    closed_path: bool = False,
    cap: bool = True,
) -> MeshGeometry:
    """Join rings with the same point count into one surface. Point ``j`` of each ring
    connects to point ``j`` of the next. With caps, the result is a closed solid."""
    rings = [np.asarray(r, dtype=float) for r in rings]
    count = len(rings[0])
    if len(rings) < 2 or any(len(r) != count for r in rings):
        raise ValidationError("rings must be at least two, with the same point count")
    # Point every ring's winding along the direction of travel, so the sides face out.
    travel = rings[1].mean(axis=0) - rings[0].mean(axis=0)
    if closed_profile and newell_normal(rings[0]).dot(travel) < 0:
        rings = [r[::-1].copy() for r in rings]

    vertices = np.vstack(rings)
    spans = len(rings) if closed_path else len(rings) - 1
    columns = count if closed_profile else count - 1
    faces: list[tuple[int, int, int]] = []
    for i in range(spans):
        here, there = i * count, ((i + 1) % len(rings)) * count
        for j in range(columns):
            k = (j + 1) % count
            faces.append((here + j, here + k, there + k))
            faces.append((here + j, there + k, there + j))
    if cap and closed_profile and not closed_path:
        first, last = rings[0], rings[-1]
        faces += _cap_triangles(first, first.mean(axis=0) - rings[1].mean(axis=0))
        offset = (len(rings) - 1) * count
        end = _cap_triangles(last, last.mean(axis=0) - rings[-2].mean(axis=0))
        faces += [(a + offset, b + offset, c + offset) for a, b, c in end]
    mesh = MeshGeometry(vertices, faces)
    if cap or closed_path:
        tri = mesh.to_trimesh()
        if tri.is_watertight and tri.volume < 0:
            mesh.faces = mesh.faces[:, ::-1].copy()
    return mesh


def circle_profile(radius: float, segments: int) -> list[tuple[float, float]]:
    segments = max(3, int(segments))
    return [
        (radius * math.cos(2 * math.pi * k / segments), radius * math.sin(2 * math.pi * k / segments))
        for k in range(segments)
    ]


Mesh = MeshGeometry
"""Step-1 name for a raw triangle mesh; kept as an alias."""
