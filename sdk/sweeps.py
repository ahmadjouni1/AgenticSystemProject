# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Sweeps: a 2D profile carried along a 3D path (tubes, rods, rails, handles, cables).

The profile's ``(x, y)`` is placed in a frame that travels with the path: ``y`` along
the "up" direction (``up_hint`` projected off the tangent), ``x`` to the side. Frames
are carried by parallel transport (no sudden twists). At a corner, the ring lies in the
miter plane, so the tube keeps its full thickness through the bend.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from sdk.errors import ValidationError
from sdk.mesh_geometry import MeshGeometry, circle_profile, ensure_ccw, profile_2d, skin
from sdk.profiles import sample_catmull_rom_spline_3d
from sdk.solids import not_ported
from sdk.transforms import axis_angle_matrix

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
_EPS = 1e-12


def _path(points: Iterable[Sequence[float]], closed: bool, max_segment_length: float | None,
          min_segment_length: float = 1e-9) -> np.ndarray:
    raw = np.asarray([tuple(map(float, p)) for p in points], dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.all(np.isfinite(raw)):
        raise ValidationError("path must be a list of finite (x, y, z) points")
    keep = [raw[0]]
    for p in raw[1:]:
        if np.linalg.norm(p - keep[-1]) > min_segment_length:
            keep.append(p)
    path = np.asarray(keep)
    if closed and len(path) > 1 and np.linalg.norm(path[0] - path[-1]) <= min_segment_length:
        path = path[:-1]
    if len(path) < (3 if closed else 2):
        raise ValidationError("path needs at least 2 distinct points (3 when closed)")
    if max_segment_length is not None:
        limit = float(max_segment_length)
        if limit <= 0:
            raise ValidationError("max_segment_length must be positive")
        ends = np.vstack([path, path[:1]]) if closed else path
        dense = []
        for a, b in zip(ends[:-1], ends[1:], strict=True):
            pieces = max(1, math.ceil(np.linalg.norm(b - a) / limit))
            dense += [a + (b - a) * k / pieces for k in range(pieces)]
        path = np.asarray(dense if closed else [*dense, ends[-1]])
    return path


def _minimal_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < _EPS:
        return np.eye(3)
    return axis_angle_matrix(axis / s, math.atan2(s, float(np.dot(a, b))))


def sweep_rings(
    profile: np.ndarray, path: np.ndarray, *, closed_path: bool, up_hint: Sequence[float],
    frame_mode: str,
) -> list[np.ndarray]:
    """One ring of profile points per path point."""
    if frame_mode not in {"parallel_transport", "fixed_up"}:
        raise ValidationError("frame_mode must be 'parallel_transport' or 'fixed_up'")
    up = np.asarray(up_hint, dtype=float)
    if np.linalg.norm(up) < _EPS:
        raise ValidationError("up_hint must be non-zero")
    ends = np.vstack([path, path[:1]]) if closed_path else path
    tangents = np.diff(ends, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    def perpendicular_up(t: np.ndarray) -> np.ndarray:
        u = up - up.dot(t) * t
        if np.linalg.norm(u) < 1e-6:  # up_hint along the path: pick any perpendicular
            u = np.cross(t, [1.0, 0.0, 0.0] if abs(t[0]) < 0.9 else [0.0, 1.0, 0.0])
        return u / np.linalg.norm(u)

    ups = [perpendicular_up(tangents[0])]
    for k in range(1, len(tangents)):
        if frame_mode == "fixed_up":
            ups.append(perpendicular_up(tangents[k]))
        else:
            u = _minimal_rotation(tangents[k - 1], tangents[k]) @ ups[-1]
            ups.append(u / np.linalg.norm(u))
    if closed_path and frame_mode == "parallel_transport":
        # Spread the leftover twist around the loop so the last ring meets the first.
        wrapped = _minimal_rotation(tangents[-1], tangents[0]) @ ups[-1]
        twist = math.atan2(np.cross(wrapped, ups[0]).dot(tangents[0]), wrapped.dot(ups[0]))
        ups = [axis_angle_matrix(t, twist * k / len(tangents)) @ u
               for k, (t, u) in enumerate(zip(tangents, ups, strict=True))]

    rings = []
    count = len(path)
    for i in range(count):
        if closed_path or 0 < i < count - 1:
            k_in = (i - 1) % len(tangents)
            t_in, t_out = tangents[k_in], tangents[i % len(tangents)]
        else:
            k_in = 0 if i == 0 else len(tangents) - 1
            t_in = t_out = tangents[k_in]
        u_in = ups[k_in]
        side = np.cross(u_in, t_in)
        offsets = profile[:, :1] * side + profile[:, 1:2] * u_in  # in the plane across t_in
        miter = t_in + t_out
        if np.linalg.norm(miter) < 1e-6:
            raise ValidationError(f"the path doubles back on itself at point {i}")
        miter /= np.linalg.norm(miter)
        offsets = offsets - np.outer(offsets @ miter / t_in.dot(miter), t_in)
        rings.append(path[i] + offsets)
    return rings


class PipeGeometry(MeshGeometry):
    """Sweep a closed (or open, with ``closed=False``) profile along a polyline ``path``."""

    def __init__(
        self,
        profile: Iterable[Sequence[float]],
        path: Iterable[Sequence[float]],
        *,
        cap: bool = False,
        closed: bool = True,
        path_closed: bool = False,
        up_hint: Vec3 = (0.0, 0.0, 1.0),
        frame_mode: str = "parallel_transport",
        sections: Sequence[object] = (),
        section_interpolation: str = "linear",
        section_tension: float = 0.0,
        max_segment_length: float | None = None,
        cap_style: str = "flat",
        cap_segments: int = 6,
        cap_length: float | None = None,
    ):
        not_ported(type(self).__name__, sections=(tuple(sections), ()),
                   cap_style=(cap_style, "flat"), cap_length=(cap_length, None))
        shape = profile_2d(profile, minimum=3 if closed else 2)
        if closed:
            shape = ensure_ccw(shape)
        points = _path(path, path_closed, max_segment_length)
        rings = sweep_rings(shape, points, closed_path=path_closed, up_hint=up_hint,
                            frame_mode=frame_mode)
        mesh = skin(rings, closed_profile=closed, closed_path=path_closed,
                    cap=cap and closed and not path_closed)
        super().__init__(mesh.vertices, mesh.faces)


class SweepGeometry(PipeGeometry):
    """Articraft keeps this as a second name for ``PipeGeometry``."""


def _round_corners(path: np.ndarray, *, closed: bool, mode: str, radius: float,
                   segments: int) -> np.ndarray:
    """Trim each corner by ``radius`` (at most 45% of either side) and bevel or bend it."""
    if mode not in {"miter", "bevel", "fillet"}:
        raise ValidationError("corner_mode must be 'miter', 'bevel', or 'fillet'")
    if radius < 0 or not math.isfinite(radius):
        raise ValidationError("corner_radius must be finite and non-negative")
    if mode == "miter" or radius <= 0.0:
        return path
    count = len(path)
    out: list[np.ndarray] = []
    for i in range(count):
        if not closed and i in (0, count - 1):
            out.append(path[i])
            continue
        before, here, after = path[i - 1], path[i], path[(i + 1) % count]
        d_in, d_out = here - before, after - here
        trim = min(radius, 0.45 * np.linalg.norm(d_in), 0.45 * np.linalg.norm(d_out))
        start = here - d_in / np.linalg.norm(d_in) * trim
        end = here + d_out / np.linalg.norm(d_out) * trim
        if mode == "bevel":
            out += [start, end]
        else:  # quadratic bend with the corner as its control point
            n = max(2, int(segments))
            out += [(1 - t) ** 2 * start + 2 * (1 - t) * t * here + t**2 * end
                    for t in np.linspace(0.0, 1.0, n + 1)]
    return np.asarray(out)


def wire_from_points(
    points: Iterable[Sequence[float]],
    *,
    radius: float,
    radial_segments: int = 16,
    closed_path: bool = False,
    cap_ends: bool = False,
    corner_mode: str = "fillet",
    corner_radius: float = 0.0,
    corner_segments: int = 8,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[object] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
) -> MeshGeometry:
    """A round rod through polyline points (frames, rails, bent handles, wire racks).

    ``corner_radius`` is how far each corner is trimmed back before ``corner_mode``
    bends ("fillet") or cuts ("bevel") it. 0 keeps sharp mitered corners.
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValidationError("radius must be positive")
    path = _path(points, closed_path, None, min_segment_length)
    path = _round_corners(path, closed=closed_path, mode=corner_mode,
                          radius=float(corner_radius), segments=corner_segments)
    return PipeGeometry(
        circle_profile(radius, radial_segments), path, cap=cap_ends, path_closed=closed_path,
        up_hint=up_hint, frame_mode=frame_mode, sections=sections,
        max_segment_length=max_segment_length, cap_style=cap_style, cap_length=cap_length,
    )


def _spline(points, *, spline: str, samples: int, closed: bool, alpha: float,
            min_segment_length: float) -> np.ndarray:
    key = spline.strip().lower().replace("-", "_")
    if key not in {"catmull_rom", "catmullrom"}:
        raise NotImplementedError(f"spline={spline!r} is not ported; use 'catmull_rom'")
    sampled = sample_catmull_rom_spline_3d(list(points), samples_per_segment=samples,
                                           closed=closed, alpha=alpha)
    return _path(sampled, closed, None, min_segment_length)


def tube_from_spline_points(
    points: Iterable[Vec3],
    *,
    radius: float,
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    radial_segments: int = 16,
    cap_ends: bool = True,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[object] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
) -> MeshGeometry:
    """A smooth round tube through control points (lamp arms, cables, hoses, handles)."""
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValidationError("radius must be positive")
    path = _spline(points, spline=spline, samples=samples_per_segment, closed=closed_spline,
                   alpha=alpha, min_segment_length=min_segment_length)
    return PipeGeometry(
        circle_profile(radius, radial_segments), path, cap=cap_ends, path_closed=closed_spline,
        up_hint=up_hint, frame_mode=frame_mode, sections=sections,
        max_segment_length=max_segment_length, cap_style=cap_style, cap_length=cap_length,
    )


def sweep_profile_along_spline(
    points: Iterable[Vec3],
    *,
    profile: Iterable[Vec2],
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    cap_profile: bool = True,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[object] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
) -> MeshGeometry:
    """Any profile (rounded rectangle, superellipse, ...) along a smooth curve: mug
    handles, chair rails, gaskets."""
    path = _spline(points, spline=spline, samples=samples_per_segment, closed=closed_spline,
                   alpha=alpha, min_segment_length=min_segment_length)
    return PipeGeometry(
        profile, path, cap=cap_profile, path_closed=closed_spline, up_hint=up_hint,
        frame_mode=frame_mode, sections=sections, max_segment_length=max_segment_length,
        cap_style=cap_style, cap_length=cap_length,
    )
