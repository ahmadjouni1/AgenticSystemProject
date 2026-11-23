"""Mesh operations and recipe helpers, as in Articraft's ``articraft.sdk.mesh``.

Geometry classes (``LatheGeometry``, ``PipeGeometry``, ...) are importable from ``sdk``;
this module adds the functions: booleans, profiles, spline sampling, wires and tubes,
section lofts. See ``docs/SHAPE_HELPERS.md`` for what was ported and why.
"""

from __future__ import annotations

from sdk.booleans import boolean_difference, boolean_intersection, boolean_union
from sdk.profiles import (
    rounded_rect_profile,
    sample_catmull_rom_spline_2d,
    sample_catmull_rom_spline_3d,
    superellipse_profile,
)
from sdk.solids import LoftSection, SectionLoftSpec, section_loft
from sdk.sweeps import sweep_profile_along_spline, tube_from_spline_points, wire_from_points

__all__ = [
    "LoftSection",
    "SectionLoftSpec",
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
