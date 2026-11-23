"""Shapes a body can hold. Each one turns itself into a ``trimesh.Trimesh`` in meters.

Primitives are centered on their local origin:

- ``Box(size)``: full extents along x, y, z.
- ``Cylinder(radius, height)``: axis along local z.
- ``Sphere(radius)``.
- Any build123d shape is accepted by ``Body.add`` and wrapped in ``Build123dShape``.
  build123d has no units; its lengths are read as meters.

Raw triangle meshes and the Articraft mesh helpers (lathe, loft, pipe, ...) are
``MeshGeometry`` values, in ``sdk/mesh_geometry.py`` and its neighbours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import trimesh

from sdk.errors import ValidationError
from sdk.transforms import Vec3, vec3

DEFAULT_TOLERANCE = 0.0005
"""Tessellation chord tolerance for build123d shapes: 0.5 mm."""


@runtime_checkable
class Geometry(Protocol):
    def to_trimesh(self) -> trimesh.Trimesh: ...


def _positive(value: float, field: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{field} must be a positive number, got {value!r}")
    return value


@dataclass(frozen=True)
class Box:
    size: Vec3

    def __post_init__(self) -> None:
        size = vec3(self.size, field="Box size")
        for v in size:
            _positive(v, "Box size")
        object.__setattr__(self, "size", size)

    def to_trimesh(self) -> trimesh.Trimesh:
        return trimesh.creation.box(extents=self.size)


@dataclass(frozen=True)
class Cylinder:
    radius: float
    height: float
    sections: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive(self.radius, "Cylinder radius"))
        object.__setattr__(self, "height", _positive(self.height, "Cylinder height"))
        if int(self.sections) < 3:
            raise ValidationError("Cylinder sections must be at least 3")

    def to_trimesh(self) -> trimesh.Trimesh:
        return trimesh.creation.cylinder(
            radius=self.radius, height=self.height, sections=int(self.sections)
        )


@dataclass(frozen=True)
class Sphere:
    radius: float
    subdivisions: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive(self.radius, "Sphere radius"))

    def to_trimesh(self) -> trimesh.Trimesh:
        return trimesh.creation.icosphere(subdivisions=int(self.subdivisions), radius=self.radius)


@dataclass(frozen=True, eq=False)
class Build123dShape:
    """A build123d shape, tessellated on demand."""

    shape: Any
    tolerance: float = DEFAULT_TOLERANCE
    angular_tolerance: float = 0.1

    def to_trimesh(self) -> trimesh.Trimesh:
        vertices, faces = self.shape.tessellate(self.tolerance, self.angular_tolerance)
        if not vertices or not faces:
            raise ValidationError("build123d shape tessellated to an empty mesh")
        points = np.array([[v.X, v.Y, v.Z] for v in vertices], dtype=float)
        mesh = trimesh.Trimesh(vertices=points, faces=np.asarray(faces), process=False)
        mesh.merge_vertices()  # tessellation duplicates vertices along face seams
        return mesh


def is_build123d(obj: object) -> bool:
    """True for build123d objects, checked without importing build123d (slow import)."""
    return type(obj).__module__.split(".")[0] == "build123d"


def as_geometry(obj: object) -> Geometry:
    if isinstance(obj, Geometry):
        return obj
    if is_build123d(obj) and hasattr(obj, "tessellate"):
        return Build123dShape(obj)
    raise ValidationError(
        f"unsupported geometry {type(obj).__name__}; use Box, Cylinder, Sphere, a "
        "MeshGeometry helper, or a build123d shape"
    )
