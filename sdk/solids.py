# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Solids built from profiles: extrude, extrude with holes, lathe, loft, section loft.

Articraft's names and arguments are kept. Options whose behavior is not ported yet
are still accepted at their default value. Any other value raises
``NotImplementedError`` naming the option, so nothing is silently ignored.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import trimesh

from sdk.errors import ValidationError
from sdk.mesh_geometry import (
    MeshGeometry,
    ensure_ccw,
    from_created,
    profile_2d,
    skin,
    triangulate_polygon,
)
from sdk.occ import tessellate
from sdk.profiles import _catmull_rom

Vec3 = tuple[float, float, float]
_EPS = 1e-12


def not_ported(helper: str, **options: tuple[Any, Any]) -> None:
    """Raise if an option (value, default) was changed from its default."""
    for name, (value, default) in options.items():
        if value != default:
            raise NotImplementedError(
                f"{helper}({name}={value!r}) is not ported to the replica yet; "
                f"only the default {default!r} is supported"
            )


# ---- extrusions -----------------------------------------------------------------------


class ExtrudeGeometry(MeshGeometry):
    """Extrude a 2D ``profile`` (xy) along z by ``height``. ``center`` puts z = 0 mid-way."""

    @classmethod
    def centered(cls, profile, height, *, cap=True, closed=True) -> ExtrudeGeometry:
        return cls(profile, height, cap=cap, center=True, closed=closed)

    @classmethod
    def from_z0(cls, profile, height, *, cap=True, closed=True) -> ExtrudeGeometry:
        return cls(profile, height, cap=cap, center=False, closed=closed)

    def __init__(
        self,
        profile: Iterable[Sequence[float]],
        height: float,
        *,
        cap: bool = True,
        center: bool = True,
        closed: bool = True,
    ):
        height = float(height)
        if not math.isfinite(height) or height <= 0.0:
            raise ValidationError("extrude height must be finite and positive")
        points = profile_2d(profile, minimum=3 if closed else 2)
        if closed:
            points = ensure_ccw(points)
        z0 = -height / 2 if center else 0.0
        rings = [np.column_stack([points, np.full(len(points), z)]) for z in (z0, z0 + height)]
        mesh = skin(rings, closed_profile=closed, cap=cap and closed)
        super().__init__(mesh.vertices, mesh.faces)


class ExtrudeWithHolesGeometry(MeshGeometry):
    """Extrude an outer profile with holes cut through it (plates, brackets, panels).

    build123d builds the exact face (outer minus holes) and extrudes it, then the
    solid is tessellated. Straight-sided profiles tessellate without approximation.
    """

    def __init__(
        self,
        outer_profile: Iterable[Sequence[float]],
        hole_profiles: Iterable[Iterable[Sequence[float]]],
        height: float,
        *,
        cap: bool = True,
        center: bool = True,
        closed: bool = True,
    ):
        from build123d import Polyline, Pos, extrude, make_face

        not_ported("ExtrudeWithHolesGeometry", closed=(closed, True))
        height = float(height)
        if not math.isfinite(height) or height <= 0.0:
            raise ValidationError("extrude height must be finite and positive")
        outer = profile_2d(outer_profile, name="outer_profile")
        holes = [profile_2d(h, name=f"hole_profiles[{i}]") for i, h in enumerate(hole_profiles)]

        def face(points: np.ndarray):
            return make_face(Polyline(*[(float(x), float(y)) for x, y in points], close=True))

        section = face(outer)
        for hole in holes:
            section = section - face(hole)
        if abs(section.area - abs(_area(outer)) + sum(abs(_area(h)) for h in holes)) > 1e-9:
            raise ValidationError("every hole must lie inside the outer profile, apart")
        solid = extrude(section, amount=height)
        if center:
            solid = Pos(0, 0, -height / 2) * solid
        mesh = tessellate(solid, 1e-5, 0.1)
        if not cap:  # drop the two flat ends, leaving the walls
            mesh.update_faces(abs(mesh.face_normals[:, 2]) < 0.999)
        super().__init__(*from_created(mesh))
        if cap:
            self.attach_brep(solid)


def _area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# ---- lathe -------------------------------------------------------------------------------


def _cap_connector(start, end, mode: str, samples: int, *, outward_z: float) -> list:
    """Points joining two profile ends: a straight lip or a rounded (quadratic) one."""
    if mode not in {"flat", "round"}:
        raise ValidationError("lathe cap mode must be 'flat' or 'round'")
    if mode == "flat":
        return [start, end]
    distance = math.hypot(start[0] - end[0], start[1] - end[1])
    radial = 1.0 if start[0] >= end[0] else -1.0
    control = ((start[0] + end[0]) / 2 + radial * distance * 0.35,
               (start[1] + end[1]) / 2 + math.copysign(distance / 2, outward_z))
    samples = max(2, int(samples))
    points = []
    for k in range(samples + 1):
        t = k / samples
        points.append(tuple((1 - t) ** 2 * s + 2 * (1 - t) * t * c + t**2 * e
                            for s, c, e in zip(start, control, end, strict=True)))
    return points


class LatheGeometry(MeshGeometry):
    """Revolve a profile of ``(radius, z)`` points about the z axis.

    ``closed=True`` (default) treats the profile as a closed outline, so the result is a
    solid. ``angle`` < 2*pi makes a partial revolve, capped at both ends when
    ``cap_ends``. ``axis`` turns the finished part so its axis is x, y or z.
    """

    @classmethod
    def from_shell_profiles(
        cls,
        outer_profile: Iterable[Sequence[float]],
        inner_profile: Iterable[Sequence[float]],
        *,
        segments: int = 32,
        start_cap: str = "flat",
        end_cap: str = "flat",
        lip_samples: int = 6,
    ) -> LatheGeometry:
        """A thin-walled vessel (cup, shade, bottle) from matching outer and inner walls."""
        outer = [tuple(p) for p in profile_2d(outer_profile, minimum=2, name="outer_profile")]
        inner = [tuple(p) for p in profile_2d(inner_profile, minimum=2, name="inner_profile")]
        upward = 1.0 if outer[-1][1] + inner[-1][1] >= outer[0][1] + inner[0][1] else -1.0
        end = _cap_connector(outer[-1], inner[-1], end_cap, lip_samples, outward_z=upward)
        start = _cap_connector(inner[0], outer[0], start_cap, lip_samples, outward_z=-upward)
        outline = [*outer, *end[1:-1], *reversed(inner), *start[1:-1]]
        return cls(outline, segments=segments, closed=True)

    def __init__(
        self,
        profile: Iterable[Sequence[float]],
        *,
        segments: int = 32,
        closed: bool = True,
        angle: float = math.tau,
        start_angle: float = 0.0,
        axis: str = "z",
        cap_ends: bool = True,
    ):
        points = profile_2d(profile, minimum=3 if closed else 2)
        if closed:
            points = ensure_ccw(points)
        if np.any(points[:, 0] < -1e-9):
            raise ValidationError("lathe profile radii must be non-negative")
        angle, start_angle = float(angle), float(start_angle)
        if not (math.isfinite(angle) and 0.0 < angle <= math.tau):
            raise ValidationError("lathe angle must be greater than 0 and at most 2*pi")
        if str(axis).lower() not in {"x", "y", "z"}:
            raise ValidationError("lathe axis must be 'x', 'y', or 'z'")
        mesh = _revolve(points, angle, max(3, int(segments)), closed=closed, caps=cap_ends)
        rotation = trimesh.transformations.rotation_matrix
        if start_angle:
            mesh.apply_transform(rotation(start_angle, (0, 0, 1)))
        if axis.lower() == "x":
            mesh.apply_transform(rotation(math.pi / 2, (0, 1, 0)))
        elif axis.lower() == "y":
            mesh.apply_transform(rotation(-math.pi / 2, (1, 0, 0)))
        super().__init__(*from_created(mesh))


def _revolve(points: np.ndarray, angle: float, steps: int, *, closed: bool, caps: bool
             ) -> trimesh.Trimesh:
    """Sweep ``(r, z)`` points about z. A point on the axis becomes one shared vertex."""
    full = math.isclose(angle, math.tau, abs_tol=1e-10)
    thetas = np.linspace(0.0, angle, steps + 1)[: steps if full else steps + 1]
    vertices: list[tuple[float, float, float]] = []
    rows: list[list[int]] = []  # rows[j][k]: vertex of profile point j at angle k
    for r, z in points:
        if abs(r) <= 1e-12:
            vertices.append((0.0, 0.0, float(z)))
            rows.append([len(vertices) - 1] * len(thetas))
            continue
        start = len(vertices)
        vertices += [(r * math.cos(t), r * math.sin(t), float(z)) for t in thetas]
        rows.append(list(range(start, start + len(thetas))))

    faces: list[tuple[int, int, int]] = []
    edges = len(points) if closed else len(points) - 1
    spans = len(thetas) if full else len(thetas) - 1
    for j in range(edges):
        a, b = rows[j], rows[(j + 1) % len(points)]
        for k in range(spans):
            k2 = (k + 1) % len(thetas)
            quad = ((a[k], b[k], b[k2]), (a[k], b[k2], a[k2]))
            faces += [tri for tri in quad if len(set(tri)) == 3]  # no slivers at the axis
    if not full and closed and caps:
        cap = triangulate_polygon(points)
        first, last = [row[0] for row in rows], [row[-1] for row in rows]
        faces += [(first[i], first[k], first[j]) for i, j, k in cap]
        faces += [(last[i], last[j], last[k]) for i, j, k in cap]
    return trimesh.Trimesh(vertices, faces, process=False)


# ---- loft ------------------------------------------------------------------------------------


_ALPHA = {"uniform": 0.0, "centripetal": 0.5, "chord": 1.0}


class LoftGeometry(MeshGeometry):
    """Skin a surface through 3D profile rings that all have the same point count.

    Point ``j`` of every ring is connected to point ``j`` of the next ring.
    ``interpolation="catmull_rom"`` adds ``samples_per_span - 1`` smooth rings
    between each pair of given rings.
    """

    def __init__(
        self,
        profiles: Iterable[Iterable[Sequence[float]]],
        *,
        cap: bool = True,
        closed: bool = True,
        interpolation: str = "linear",
        samples_per_span: int = 1,
        close_path: bool = False,
        parameterization: str = "uniform",
        tension: float = 0.0,
        cap_style: str = "flat",
        cap_segments: int = 6,
        cap_length: float | None = None,
    ):
        not_ported("LoftGeometry", tension=(float(tension), 0.0), cap_style=(cap_style, "flat"),
                   cap_length=(cap_length, None))
        rings = [np.asarray([tuple(map(float, p)) for p in ring], dtype=float)
                 for ring in profiles]
        if len(rings) < 2:
            raise ValidationError("loft requires at least two profiles")
        if any(r.ndim != 2 or r.shape[1] != 3 for r in rings):
            raise ValidationError("loft profiles must be lists of (x, y, z) points")
        count = len(rings[0])
        if count < (3 if closed else 2) or any(len(r) != count for r in rings):
            raise ValidationError("loft profiles must have the same point count (3 or more)")
        if close_path and len(rings) < 3:
            raise ValidationError("a closed loft path requires at least three profiles")
        if interpolation not in {"linear", "catmull_rom"}:
            raise ValidationError("loft interpolation must be 'linear' or 'catmull_rom'")
        if parameterization not in _ALPHA:
            raise ValidationError("parameterization must be 'uniform', 'chord' or 'centripetal'")
        samples = max(1, int(samples_per_span))
        if interpolation == "catmull_rom" and samples > 1:
            stacked = np.stack(rings)  # (rings, points, 3): interpolate each point's track
            tracks = [_catmull_rom(stacked[:, j], samples, close_path, _ALPHA[parameterization])
                      for j in range(count)]
            rings = [np.stack([track[i] for track in tracks]) for i in range(len(tracks[0]))]
        mesh = skin(rings, closed_profile=closed, closed_path=close_path,
                    cap=cap and closed and not close_path)
        super().__init__(mesh.vertices, mesh.faces)


def _loop(points: Sequence[Sequence[float]]) -> tuple[Vec3, ...]:
    result: list[Vec3] = []
    for p in points:
        value = tuple(float(v) for v in p)
        if len(value) != 3 or not all(math.isfinite(v) for v in value):
            raise ValidationError("section points must be finite (x, y, z)")
        if not result or value != result[-1]:
            result.append(value)  # type: ignore[arg-type]
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    if len(result) < 3:
        raise ValidationError("a loft section needs at least 3 distinct points")
    return tuple(result)


@dataclass(frozen=True)
class LoftSection:
    points: tuple[Vec3, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _loop(self.points))


@dataclass(frozen=True)
class SectionLoftSpec:
    """Same fields as Articraft. Ported: ``sections``, ``cap``, ``interpolation``,
    ``samples_per_span``, ``close_path``, ``parameterization``. Every other field
    must stay at its default."""

    sections: tuple[LoftSection, ...]
    path: tuple[Vec3, ...] | None = None
    cap: bool = True
    symmetry: str | None = None
    repair: str = "off"
    interpolation: str = "linear"
    samples_per_span: int = 1
    close_path: bool = False
    align_sections: bool = True
    parameterization: str = "uniform"
    tension: float = 0.0
    cap_style: str = "flat"
    cap_segments: int = 6
    cap_length: float | None = None
    orient_to_path: bool = False
    frame_mode: str = "parallel_transport"
    up_hint: Vec3 = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        sections = tuple(s if isinstance(s, LoftSection) else LoftSection(tuple(s))
                         for s in self.sections)
        if len(sections) < 2:
            raise ValidationError("SectionLoftSpec requires at least two sections")
        object.__setattr__(self, "sections", sections)


_SECTION_LOFT_PORTED = {"sections", "cap", "interpolation", "samples_per_span", "close_path",
                        "parameterization", "tension", "cap_style", "cap_segments", "cap_length"}


def section_loft(
    spec: SectionLoftSpec | Sequence[LoftSection | Sequence[Sequence[float]]], /,
    **overrides: object,
) -> MeshGeometry:
    """Loft through ``LoftSection`` rings. Sections are resampled to a common point count
    when they differ (each ring is re-spaced evenly along its perimeter)."""
    if not isinstance(spec, SectionLoftSpec):
        spec = SectionLoftSpec(sections=tuple(spec))
    spec = replace(spec, **overrides) if overrides else spec
    defaults = SectionLoftSpec(sections=spec.sections)
    for f in fields(SectionLoftSpec):
        if f.name not in _SECTION_LOFT_PORTED:
            not_ported("section_loft", **{f.name: (getattr(spec, f.name), getattr(defaults, f.name))})
    rings = [np.asarray(s.points) for s in spec.sections]
    count = max(len(r) for r in rings)
    rings = [r if len(r) == count else _resample_ring(r, count) for r in rings]
    return LoftGeometry(
        rings, cap=spec.cap, interpolation=spec.interpolation,
        samples_per_span=spec.samples_per_span, close_path=spec.close_path,
        parameterization=spec.parameterization, tension=spec.tension,
        cap_style=spec.cap_style, cap_segments=spec.cap_segments, cap_length=spec.cap_length,
    )


def _resample_ring(ring: np.ndarray, count: int) -> np.ndarray:
    closed = np.vstack([ring, ring[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(lengths)])
    targets = np.linspace(0.0, distance[-1], count, endpoint=False)
    return np.stack([np.interp(targets, distance, closed[:, axis]) for axis in range(3)], axis=1)
