# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""2D profiles and curve sampling (Articraft names and arguments).

Profiles are counter-clockwise lists of ``(x, y)`` points, centered on the origin,
ready for ``ExtrudeGeometry``, ``LoftGeometry`` rings, or a ``PipeGeometry`` profile.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from sdk.errors import ValidationError

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


def rounded_rect_profile(
    width: float, height: float, radius: float, *, corner_segments: int = 6
) -> list[Vec2]:
    width, height, radius = float(width), float(height), float(radius)
    if width <= 0.0 or height <= 0.0:
        raise ValidationError("width and height must be positive")
    if radius < 0.0 or radius > min(width, height) * 0.5:
        raise ValidationError("radius must fit inside the rectangle")
    w, h = width / 2, height / 2
    if radius <= 1e-12:
        return [(-w, -h), (w, -h), (w, h), (-w, h)]
    count = max(1, int(corner_segments))
    corners = ((w - radius, -h + radius, -math.pi / 2), (w - radius, h - radius, 0.0),
               (-w + radius, h - radius, math.pi / 2), (-w + radius, -h + radius, math.pi))
    points: list[Vec2] = []
    for cx, cy, start in corners:
        for k in range(count + 1):
            angle = start + (math.pi / 2) * k / count
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    # Where a straight side has zero length (radius == half the size), corners touch.
    deduped = [p for i, p in enumerate(points) if not np.allclose(p, points[i - 1])]
    return deduped


def superellipse_profile(
    width: float, height: float, exponent: float = 2.6, *, segments: int = 48
) -> list[Vec2]:
    """``|x/a|^n + |y/b|^n = 1``: 2 is an ellipse, larger values approach a rectangle."""
    width, height, exponent = float(width), float(height), float(exponent)
    if width <= 0.0 or height <= 0.0 or exponent <= 0.0:
        raise ValidationError("width, height, and exponent must be positive")
    count = max(12, int(segments))
    power = 2.0 / exponent
    points = []
    for k in range(count):
        c, s = math.cos(2 * math.pi * k / count), math.sin(2 * math.pi * k / count)
        points.append((math.copysign(width / 2 * abs(c) ** power, c),
                       math.copysign(height / 2 * abs(s) ** power, s)))
    return points


def _catmull_rom(points: np.ndarray, samples: int, closed: bool, alpha: float) -> np.ndarray:
    """Centripetal (alpha=0.5) Catmull-Rom through every control point."""
    if len(points) < 2:
        raise ValidationError("a spline needs at least two points")
    if closed:
        controls = np.vstack([points[-1], points, points[0], points[1]])
    else:  # mirror the end points so the curve starts and ends on them
        controls = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    spans = len(points) if closed else len(points) - 1
    out = [points[0]]
    for i in range(spans):
        p0, p1, p2, p3 = controls[i : i + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0) ** alpha, 1e-12)
        t2 = t1 + max(np.linalg.norm(p2 - p1) ** alpha, 1e-12)
        t3 = t2 + max(np.linalg.norm(p3 - p2) ** alpha, 1e-12)
        for t in np.linspace(t1, t2, samples + 1)[1:]:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    result = np.asarray(out)
    return result[:-1] if closed else result  # a closed curve does not repeat its start


def sample_catmull_rom_spline_3d(
    points: Sequence[Vec3], *, samples_per_segment: int = 12, closed: bool = False,
    alpha: float = 0.5,
) -> list[Vec3]:
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    sampled = _catmull_rom(values, max(1, int(samples_per_segment)), closed, float(alpha))
    return [tuple(map(float, p)) for p in sampled]


def sample_catmull_rom_spline_2d(
    points: Sequence[Vec2], *, samples_per_segment: int = 12, closed: bool = False,
    alpha: float = 0.5,
) -> list[Vec2]:
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    sampled = _catmull_rom(values, max(1, int(samples_per_segment)), closed, float(alpha))
    return [tuple(map(float, p)) for p in sampled]
