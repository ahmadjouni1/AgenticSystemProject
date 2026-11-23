# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Frames and attachments.

A ``Frame`` is a pose (``xyz`` in meters, ``rpy`` in radians) inside one body.
An ``Attachment`` is a frame bound to a body; joints connect two attachments.

build123d is the only place degrees appear: a build123d ``Location`` is converted
through its transformation matrix, so no Euler-angle convention is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from sdk.errors import ValidationError
from sdk.geometry import is_build123d
from sdk.transforms import Vec3, make_transform, rpy_from_matrix, rpy_matrix, vec3

if TYPE_CHECKING:
    from sdk.body import Body


@dataclass(frozen=True)
class Frame:
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz", vec3(self.xyz, field="frame xyz"))
        object.__setattr__(self, "rpy", vec3(self.rpy, field="frame rpy"))

    @cached_property
    def matrix(self) -> np.ndarray:
        """4x4 pose of this frame in its body (cached; do not modify it in place)."""
        return make_transform(self.xyz, rpy_matrix(self.rpy))

    @classmethod
    def from_matrix(cls, transform: np.ndarray) -> Frame:
        t = np.asarray(transform, dtype=float)
        return cls(tuple(t[:3, 3]), rpy_from_matrix(t[:3, :3]))


@dataclass(frozen=True, eq=False)
class Attachment:
    """A frame on a body, made with ``body.at(...)``."""

    body: Body
    frame: Frame

    def __repr__(self) -> str:
        return f"{self.body.name}.at(xyz={self.frame.xyz}, rpy={self.frame.rpy})"


def to_frame(point: Any = (0.0, 0.0, 0.0), rpy: Any = None) -> Frame:
    """Build a frame from a point and rpy, or from a build123d object.

    build123d objects follow Articraft's rules. A feature's natural axis becomes the
    frame's local z, so a hinge can simply be "about z":

    - ``Location``, ``Plane``, ``Axis``: their own pose.
    - flat ``Face``: its center, z along the outward normal.
    - round ``Face`` (cylinder, cone, sphere): its axis of symmetry, at the face's height
      along it. A hinge attached to a bore spins about the bore.
    - straight ``Edge``: its midpoint, z along the edge.
    - circular ``Edge`` (a hole rim): its center, z along the circle's normal.
    - ``Vertex``: its position. Any other shape or a mesh: its bounding-box center.

    Features are read in body coordinates, which is the case when the shape was added
    without ``xyz``/``rpy``.
    """
    if isinstance(point, Frame):
        if rpy is not None:
            raise ValidationError("pass rpy inside the Frame, not next to it")
        return point
    if is_build123d(point):
        frame = _build123d_frame(point)
        if rpy is not None:
            if frame.rpy != (0.0, 0.0, 0.0) or type(point).__name__ not in {"Vector", "Vertex"}:
                raise ValidationError(
                    f"a build123d {type(point).__name__} already has an orientation"
                )
            return Frame(frame.xyz, rpy)
        return frame
    if hasattr(point, "vertices") and hasattr(point, "faces") and hasattr(point, "to_trimesh"):
        vertices = np.asarray(point.vertices, dtype=float)
        if len(vertices) == 0:
            raise ValidationError("cannot attach at an empty mesh")
        return Frame(tuple((vertices.min(axis=0) + vertices.max(axis=0)) / 2))
    return Frame(point, (0.0, 0.0, 0.0) if rpy is None else rpy)


def _build123d_frame(obj: Any) -> Frame:
    from build123d import Axis, Edge, Face, Location, Plane, Shape, Vector, Vertex

    if isinstance(obj, Vector):
        return Frame((obj.X, obj.Y, obj.Z))
    if isinstance(obj, Location):
        return Frame.from_matrix(_location_matrix(obj))
    if isinstance(obj, (Plane, Axis)):
        return Frame.from_matrix(_location_matrix(obj.location))
    if isinstance(obj, Face):
        axis = obj.axis_of_rotation
        if axis is None:
            location = Plane(origin=obj.center(), z_dir=obj.normal_at()).location
        else:
            direction = Vector(axis.direction).normalized()
            offset = Vector(obj.center()) - Vector(axis.position)
            origin = Vector(axis.position) + direction * offset.dot(direction)
            location = Axis(origin, direction).location
        return Frame.from_matrix(_location_matrix(location))
    if isinstance(obj, Edge):
        if str(obj.geom_type).rsplit(".", 1)[-1] in {"CIRCLE", "ELLIPSE"}:
            location = Axis(obj.arc_center, obj.normal()).location
        else:
            location = Axis(obj.center(), obj.tangent_at()).location
        return Frame.from_matrix(_location_matrix(location))
    if isinstance(obj, Vertex):
        c = obj.center()
        return Frame((c.X, c.Y, c.Z))
    if isinstance(obj, Shape):
        c = obj.bounding_box().center()
        return Frame((c.X, c.Y, c.Z))
    raise ValidationError(f"cannot attach at a build123d {type(obj).__name__}")


def _location_matrix(location: Any) -> np.ndarray:
    """4x4 matrix of a build123d Location (its degrees never reach our code)."""
    trsf = location.wrapped.Transformation()
    t = np.eye(4)
    for row in range(3):
        for col in range(4):
            t[row, col] = trsf.Value(row + 1, col + 1)
    return t
