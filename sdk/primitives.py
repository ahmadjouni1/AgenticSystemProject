# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Mesh primitives with Articraft's names and arguments. All are centered on the origin;
round ones have their axis along z."""

from __future__ import annotations

import math
from collections.abc import Sequence

import trimesh

from sdk.errors import ValidationError
from sdk.mesh_geometry import MeshGeometry, from_created
from sdk.notices import notice
from sdk.occ import tessellate
from sdk.transforms import vec3


def _positive(*values: float, what: str) -> None:
    if any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise ValidationError(f"{what} must be finite and positive")


def _without_end_caps(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop flat faces facing +-z (``closed=False`` leaves an open tube or shade)."""
    mesh.update_faces(abs(mesh.face_normals[:, 2]) < 0.999)
    mesh.remove_unreferenced_vertices()
    return mesh


class BoxGeometry(MeshGeometry):
    def __init__(self, size: Sequence[float]):
        size = vec3(size, field="box size")
        _positive(*size, what="box size values")
        super().__init__(*from_created(trimesh.creation.box(extents=size)))


class CylinderGeometry(MeshGeometry):
    def __init__(
        self, radius: float, height: float, *, radial_segments: int = 24, closed: bool = True
    ):
        _positive(radius, height, what="cylinder radius and height")
        mesh = trimesh.creation.cylinder(
            radius=radius, height=height, sections=max(3, int(radial_segments))
        )
        super().__init__(*from_created(mesh if closed else _without_end_caps(mesh)))


class ConeGeometry(MeshGeometry):
    """Base circle at z = -height/2, apex at z = +height/2."""

    def __init__(
        self, radius: float, height: float, *, radial_segments: int = 24, closed: bool = True
    ):
        _positive(radius, height, what="cone radius and height")
        mesh = trimesh.creation.cone(
            radius=radius, height=height, sections=max(3, int(radial_segments))
        )
        mesh.apply_translation((0.0, 0.0, -height / 2))
        super().__init__(*from_created(mesh if closed else _without_end_caps(mesh)))


class CapsuleGeometry(MeshGeometry):
    """A cylinder of ``length`` with hemispherical ends: total height ``length + 2*radius``."""

    def __init__(
        self, radius: float, length: float, *, radial_segments: int = 24, height_segments: int = 8
    ):
        _positive(radius, what="capsule radius")
        if not math.isfinite(length) or length < 0.0:
            raise ValidationError("capsule length must be finite and non-negative")
        mesh = trimesh.creation.capsule(
            height=length, radius=radius,
            count=(max(8, int(radial_segments)), max(4, int(height_segments))),
        )
        super().__init__(*from_created(mesh))


class TorusGeometry(MeshGeometry):
    """Ring in the xy plane: ``radius`` to the tube center, ``tube`` is the tube radius."""

    def __init__(
        self, radius: float, tube: float, *, radial_segments: int = 16, tubular_segments: int = 32
    ):
        _positive(radius, tube, what="torus radius and tube")
        if tube >= radius:
            raise ValidationError("torus tube must be smaller than its radius")
        mesh = trimesh.creation.torus(
            major_radius=radius, minor_radius=tube,
            major_sections=max(3, int(tubular_segments)),
            minor_sections=max(3, int(radial_segments)),
        )
        super().__init__(*from_created(mesh))


RADIUS_CLAMP = 0.45  # a too-large rounded-box radius becomes this times the smallest size


class RoundedBoxGeometry(MeshGeometry):
    """A box with every edge filleted (build123d does the fillet, then it is tessellated)."""

    def __init__(
        self,
        size: Sequence[float],
        radius: float,
        *,
        tolerance: float = 0.0005,
        angular_tolerance: float = 0.08,
    ):
        from build123d import Box

        size = vec3(size, field="rounded box size")
        _positive(*size, what="rounded box size values")
        _positive(radius, what="rounded box radius")
        if radius >= min(size) / 2:  # step 10b: repair it, with a warning, instead of failing
            clamped = RADIUS_CLAMP * min(size)
            notice(f"rounded box radius {radius:g} is not less than half its smallest size "
                   f"{min(size):g}; it was reduced to {clamped:g}",
                   "Pass a radius below half the smallest size to choose the rounding yourself.")
            radius = clamped
        box = Box(*size)
        rounded = box.fillet(radius, box.edges())
        super().__init__(*from_created(tessellate(rounded, tolerance, angular_tolerance)))
        self.attach_brep(rounded)  # booleans on this box stay exact and fast
