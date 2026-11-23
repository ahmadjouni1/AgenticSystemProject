"""A rigid body: named shapes placed in the body's own frame."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import trimesh

from sdk.errors import ValidationError
from sdk.frames import Attachment, Frame, to_frame
from sdk.geometry import Build123dShape, Geometry, as_geometry, is_build123d

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def check_name(name: object, *, kind: str) -> str:
    """Names become USD/URDF identifiers later, so keep them simple now."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValidationError(
            f"{kind} name must be a letter or _ followed by letters, digits or _, got {name!r}"
        )
    return name


def _is_topology(obj: Any) -> bool:
    from build123d import Shape

    return isinstance(obj, Shape)


def _contains(shape: Any, feature: Any) -> bool:
    """True if ``feature`` is ``shape`` or one of its sub-shapes (same TShape and location)."""
    from OCP.TopExp import TopExp_Explorer

    target = feature.wrapped
    if shape.wrapped.IsSame(target):
        return True
    explorer = TopExp_Explorer(shape.wrapped, target.ShapeType())
    while explorer.More():
        if explorer.Current().IsSame(target):
            return True
        explorer.Next()
    return False


@dataclass(frozen=True)
class PlacedShape:
    name: str
    geometry: Geometry
    frame: Frame

    def to_trimesh(self) -> trimesh.Trimesh:
        """The shape's mesh in body coordinates."""
        mesh = self.geometry.to_trimesh().copy()
        mesh.apply_transform(self.frame.matrix)
        return mesh


class Body:
    """Create bodies with ``Assembly.body(name)`` so the assembly knows about them."""

    def __init__(self, name: str) -> None:
        self.name = check_name(name, kind="body")
        self._shapes: dict[str, PlacedShape] = {}

    def __repr__(self) -> str:
        return f"Body({self.name!r}, shapes={list(self._shapes)})"

    def add(
        self,
        name: str,
        geometry: Any,
        *,
        xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> PlacedShape:
        """Add a named shape, placed at ``xyz`` (m) and ``rpy`` (rad) in this body."""
        check_name(name, kind=f"shape on body {self.name!r}")
        if name in self._shapes:
            raise ValidationError(f"body {self.name!r} already has a shape named {name!r}")
        shape = PlacedShape(name, as_geometry(geometry), Frame(xyz, rpy))
        self._shapes[name] = shape
        return shape

    @property
    def shapes(self) -> tuple[PlacedShape, ...]:
        return tuple(self._shapes.values())

    def shape(self, name: str) -> PlacedShape:
        try:
            return self._shapes[name]
        except KeyError:
            raise ValidationError(f"body {self.name!r} has no shape {name!r}") from None

    def at(
        self, point: Any = (0.0, 0.0, 0.0), rpy: Any = None, *, shape: str | None = None
    ) -> Attachment:
        """A frame on this body: a point (m) plus rpy (rad), or a build123d object.

        A build123d feature (face, edge, vertex) or a mesh added with ``xyz``/``rpy`` is
        found among this body's shapes, and that shape's placement is applied. So a face
        picked from the shape you passed to ``add`` gives a frame in body coordinates.
        Pass ``shape="name"`` to read ``point`` in that shape's coordinates explicitly
        (needed for an ``Axis``/``Plane``/``Location``, or for a shape added twice).
        """
        frame = to_frame(point, rpy)
        placement = self.shape(shape).frame.matrix if shape is not None else self._owner(point)
        if placement is not None:
            frame = Frame.from_matrix(placement @ frame.matrix)
        return Attachment(self, frame)

    def _owner(self, feature: Any) -> Any:
        """Placement matrix of the one shape that contains ``feature``, if any."""
        if isinstance(feature, Geometry) and not is_build123d(feature):
            owners = [s for s in self.shapes if s.geometry is feature]
        elif is_build123d(feature) and _is_topology(feature):
            owners = [s for s in self.shapes if isinstance(s.geometry, Build123dShape)
                      and _contains(s.geometry.shape, feature)]
        else:
            return None
        placements = {s.frame for s in owners}
        if len(placements) > 1:
            names = [s.name for s in owners]
            raise ValidationError(
                f"that feature belongs to several shapes on body {self.name!r} ({names}) at "
                "different placements; pass shape='name' to choose one"
            )
        return owners[0].frame.matrix if owners else None

    def trimesh(self) -> trimesh.Trimesh:
        """All shapes merged into one mesh, in body coordinates."""
        if not self._shapes:
            raise ValidationError(f"body {self.name!r} has no shapes")
        return trimesh.util.concatenate([shape.to_trimesh() for shape in self.shapes])
