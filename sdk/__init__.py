"""Replica geometry SDK: bodies, frames, joints, assemblies, forward kinematics, and the
Articraft mesh helpers worth keeping (see ``docs/SHAPE_HELPERS.md``).

Units: meters and radians everywhere. Degrees only appear inside build123d and are
converted where a build123d object enters ``body.at(...)``.

Mesh operations (booleans, profiles, tubes, section lofts) live in ``sdk.mesh``, as
Articraft's live in ``articraft.sdk.mesh``. They are also re-exported here.
"""

from __future__ import annotations

from sdk.assembly import Assembly
from sdk.body import Body, PlacedShape
from sdk.errors import LoopClosureError, PoseError, SDKError, ValidationError
from sdk.frames import Attachment, Frame
from sdk.geometry import Box, Build123dShape, Cylinder, Geometry, Sphere
from sdk.joint import Joint, JointType, Mimic
from sdk.kinematics import LOOP_TOLERANCE, Articulation, Pose
from sdk.mesh import (
    LoftSection,
    SectionLoftSpec,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    rounded_rect_profile,
    sample_catmull_rom_spline_2d,
    sample_catmull_rom_spline_3d,
    section_loft,
    superellipse_profile,
    sweep_profile_along_spline,
    tube_from_spline_points,
    wire_from_points,
)
from sdk.mesh_geometry import Mesh, MeshGeometry
from sdk.primitives import (
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    RoundedBoxGeometry,
    TorusGeometry,
)
from sdk.solids import ExtrudeGeometry, ExtrudeWithHolesGeometry, LatheGeometry, LoftGeometry
from sdk.sweeps import PipeGeometry, SweepGeometry

__all__ = [
    "LOOP_TOLERANCE",
    "Articulation",
    "Assembly",
    "Attachment",
    "Body",
    "Box",
    "BoxGeometry",
    "Build123dShape",
    "CapsuleGeometry",
    "ConeGeometry",
    "Cylinder",
    "CylinderGeometry",
    "ExtrudeGeometry",
    "ExtrudeWithHolesGeometry",
    "Frame",
    "Geometry",
    "Joint",
    "JointType",
    "LatheGeometry",
    "LoftGeometry",
    "LoftSection",
    "LoopClosureError",
    "Mesh",
    "MeshGeometry",
    "Mimic",
    "PipeGeometry",
    "PlacedShape",
    "Pose",
    "PoseError",
    "RoundedBoxGeometry",
    "SDKError",
    "SectionLoftSpec",
    "Sphere",
    "SweepGeometry",
    "TorusGeometry",
    "ValidationError",
    "boolean_difference",
    "boolean_intersection",
    "boolean_union",
    "rounded_rect_profile",
    "sample_catmull_rom_spline_2d",
    "sample_catmull_rom_spline_3d",
    "section_loft",
    "superellipse_profile",
    "sweep_profile_along_spline",
    "tube_from_spline_points",
    "wire_from_points",
]
