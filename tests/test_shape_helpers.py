"""Every ported mesh helper: a closed, outward-facing mesh with the expected bounding box."""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box as CadBox

from sdk import (
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    ExtrudeGeometry,
    ExtrudeWithHolesGeometry,
    LatheGeometry,
    LoftGeometry,
    LoftSection,
    MeshGeometry,
    PipeGeometry,
    RoundedBoxGeometry,
    SectionLoftSpec,
    SweepGeometry,
    TorusGeometry,
    ValidationError,
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


def circle(r: float, n: int = 32, cx: float = 0.0, cy: float = 0.0) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def rect_section(width: float, depth: float, z: float) -> LoftSection:
    x, y = width / 2, depth / 2
    return LoftSection(((-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)))


SQUARE_RINGS = [[(-0.05, -0.05, 0.0), (0.05, -0.05, 0.0), (0.05, 0.05, 0.0), (-0.05, 0.05, 0.0)],
                [(-0.07, -0.07, 0.1), (0.07, -0.07, 0.1), (0.07, 0.07, 0.1), (-0.07, 0.07, 0.1)]]
LATHE_PROFILE = [(0.0, 0.0), (0.05, 0.0), (0.05, 0.02), (0.02, 0.05), (0.0, 0.05)]

# name -> (factory, expected (min, max) bounds, tolerance)
CASES = {
    "BoxGeometry": (lambda: BoxGeometry((0.2, 0.1, 0.05)),
                    ((-0.1, -0.05, -0.025), (0.1, 0.05, 0.025)), 1e-12),
    "CylinderGeometry": (lambda: CylinderGeometry(0.05, 0.2),
                         ((-0.05, -0.05, -0.1), (0.05, 0.05, 0.1)), 1e-12),
    "ConeGeometry": (lambda: ConeGeometry(0.05, 0.2),
                     ((-0.05, -0.05, -0.1), (0.05, 0.05, 0.1)), 1e-12),
    "CapsuleGeometry": (lambda: CapsuleGeometry(0.02, 0.1),
                        ((-0.02, -0.02, -0.07), (0.02, 0.02, 0.07)), 1e-12),
    "TorusGeometry": (lambda: TorusGeometry(0.1, 0.02),
                      ((-0.12, -0.12, -0.02), (0.12, 0.12, 0.02)), 1e-3),
    "RoundedBoxGeometry": (lambda: RoundedBoxGeometry((0.2, 0.1, 0.05), 0.01),
                           ((-0.1, -0.05, -0.025), (0.1, 0.05, 0.025)), 1e-6),
    "ExtrudeGeometry": (lambda: ExtrudeGeometry(rounded_rect_profile(0.1, 0.06, 0.01), 0.02),
                        ((-0.05, -0.03, -0.01), (0.05, 0.03, 0.01)), 1e-12),
    "ExtrudeGeometry.from_z0": (lambda: ExtrudeGeometry.from_z0(circle(0.03), 0.02),
                                ((-0.03, -0.03, 0.0), (0.03, 0.03, 0.02)), 1e-12),
    "ExtrudeWithHolesGeometry": (
        lambda: ExtrudeWithHolesGeometry(
            rounded_rect_profile(0.1, 0.06, 0.01),
            [circle(0.005, 24, cx) for cx in (-0.03, 0.03)], 0.01),
        ((-0.05, -0.03, -0.005), (0.05, 0.03, 0.005)), 1e-9),
    "LatheGeometry": (lambda: LatheGeometry(LATHE_PROFILE, segments=48),
                      ((-0.05, -0.05, 0.0), (0.05, 0.05, 0.05)), 1e-12),
    "LatheGeometry axis=x": (lambda: LatheGeometry(LATHE_PROFILE, segments=48, axis="x"),
                             ((0.0, -0.05, -0.05), (0.05, 0.05, 0.05)), 1e-12),
    "LatheGeometry half turn": (
        lambda: LatheGeometry([(0.03, 0), (0.05, 0), (0.05, 0.02), (0.03, 0.02)], angle=math.pi),
        ((-0.05, 0.0, 0.0), (0.05, 0.05, 0.02)), 1e-12),
    "LatheGeometry.from_shell_profiles": (
        lambda: LatheGeometry.from_shell_profiles(
            [(0.04, 0.0), (0.05, 0.1)], [(0.035, 0.005), (0.045, 0.1)], segments=48),
        ((-0.05, -0.05, 0.0), (0.05, 0.05, 0.1)), 1e-12),
    "LoftGeometry": (lambda: LoftGeometry(SQUARE_RINGS),
                     ((-0.07, -0.07, 0.0), (0.07, 0.07, 0.1)), 1e-12),
    "section_loft": (
        lambda: section_loft(SectionLoftSpec(sections=(
            rect_section(0.16, 0.10, 0.0), rect_section(0.20, 0.13, 0.15),
            rect_section(0.14, 0.09, 0.30)))),
        ((-0.10, -0.065, 0.0), (0.10, 0.065, 0.30)), 1e-12),
    "PipeGeometry": (
        lambda: PipeGeometry(rounded_rect_profile(0.02, 0.02, 0.0),
                             [(0, 0, 0), (0.2, 0, 0)], cap=True),
        ((0.0, -0.01, -0.01), (0.2, 0.01, 0.01)), 1e-12),
    "SweepGeometry": (
        lambda: SweepGeometry(circle(0.01, 16), [(0, 0, 0), (0, 0, 0.1), (0, 0, 0.2)], cap=True),
        ((-0.01, -0.01, 0.0), (0.01, 0.01, 0.2)), 1e-12),
    "wire_from_points": (
        lambda: wire_from_points([(0, 0, 0), (0.2, 0, 0), (0.2, 0.2, 0)], radius=0.01,
                                 cap_ends=True, corner_radius=0.05),
        ((0.0, -0.01, -0.01), (0.21, 0.2, 0.01)), 2e-3),
    "tube_from_spline_points": (
        lambda: tube_from_spline_points([(0, 0, 0), (0, 0, 0.1), (0, 0, 0.2)], radius=0.01),
        ((-0.01, -0.01, 0.0), (0.01, 0.01, 0.2)), 1e-9),
    "sweep_profile_along_spline": (
        lambda: sweep_profile_along_spline([(0, 0, 0), (0, 0.1, 0), (0, 0.2, 0)],
                                           profile=rounded_rect_profile(0.02, 0.01, 0.002)),
        ((-0.01, 0.0, -0.005), (0.01, 0.2, 0.005)), 1e-9),
    "boolean_union": (
        lambda: boolean_union(BoxGeometry((0.1, 0.1, 0.1)), BoxGeometry((0.05, 0.05, 0.2))),
        ((-0.05, -0.05, -0.1), (0.05, 0.05, 0.1)), 1e-9),
    "boolean_difference": (
        lambda: boolean_difference(BoxGeometry((0.1, 0.1, 0.1)), CylinderGeometry(0.02, 0.2)),
        ((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)), 1e-9),
    "boolean_intersection": (
        lambda: boolean_intersection(BoxGeometry((0.1, 0.1, 0.1)), CylinderGeometry(0.06, 0.2)),
        ((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)), 1e-9),
}


@pytest.mark.parametrize("name", list(CASES))
def test_helper_is_watertight_with_expected_bounds(name: str) -> None:
    factory, (lower, upper), tolerance = CASES[name]
    geometry = factory()
    assert isinstance(geometry, MeshGeometry)
    mesh = geometry.to_trimesh()
    assert mesh.is_watertight, f"{name} is not watertight"
    assert mesh.is_winding_consistent
    assert mesh.volume > 0, f"{name} faces point inward"
    low, high = geometry.bounds()
    assert low == pytest.approx(lower, abs=tolerance)
    assert high == pytest.approx(upper, abs=tolerance)


# ---- values, not just shapes ------------------------------------------------------------


def test_volumes_match_the_math() -> None:
    plate = ExtrudeWithHolesGeometry(rounded_rect_profile(0.1, 0.06, 0.0), [circle(0.01, 64)], 0.01)
    hole_area = 0.5 * 64 * 0.01**2 * math.sin(2 * math.pi / 64)  # the 64-gon cut out
    assert plate.to_trimesh().volume == pytest.approx((0.006 - hole_area) * 0.01, rel=1e-9)
    bored = boolean_difference(BoxGeometry((0.1, 0.1, 0.1)), CylinderGeometry(0.02, 0.2,
                                                                              radial_segments=64))
    ring = 0.5 * 64 * 0.02**2 * math.sin(2 * math.pi / 64)
    assert bored.to_trimesh().volume == pytest.approx(0.001 - ring * 0.1, rel=1e-9)
    frustum = LoftGeometry(SQUARE_RINGS).to_trimesh().volume
    assert frustum == pytest.approx(0.1 / 3 * (0.01 + 0.0196 + math.sqrt(0.01 * 0.0196)))


def test_profiles() -> None:
    rect = np.asarray(rounded_rect_profile(0.1, 0.06, 0.01, corner_segments=8))
    assert rect.min(axis=0) == pytest.approx((-0.05, -0.03))
    assert rect.max(axis=0) == pytest.approx((0.05, 0.03))
    blob = np.asarray(superellipse_profile(0.1, 0.06, 4.0))
    assert np.max(np.abs(blob), axis=0) == pytest.approx((0.05, 0.03))
    with pytest.raises(ValidationError, match="radius must fit"):
        rounded_rect_profile(0.1, 0.06, 0.04)


def test_catmull_rom_passes_through_its_points() -> None:
    points = [(0, 0, 0), (0.1, 0.05, 0), (0.2, 0, 0.1), (0.3, 0.02, 0)]
    sampled = sample_catmull_rom_spline_3d(points, samples_per_segment=5)
    assert len(sampled) == 3 * 5 + 1
    for k, point in enumerate(points):
        assert sampled[5 * k] == pytest.approx(point)
    closed = sample_catmull_rom_spline_2d([(0, 0), (1, 0), (1, 1), (0, 1)], samples_per_segment=4,
                                          closed=True)
    assert len(closed) == 16 and closed[0] == pytest.approx((0, 0))


def test_lathe_axis_point_is_a_single_vertex() -> None:
    lathe = LatheGeometry(LATHE_PROFILE, segments=48)
    on_axis = np.sum(np.linalg.norm(lathe.vertices[:, :2], axis=1) < 1e-12)
    assert on_axis == 2  # one at the bottom center, one at the top center


def test_closed_paths() -> None:
    ring = wire_from_points([(0, 0, 0), (0.2, 0, 0), (0.2, 0.2, 0), (0, 0.2, 0)], radius=0.005,
                            closed_path=True, corner_radius=0.03)
    hoop = tube_from_spline_points([(0.1, 0, 0), (0, 0.1, 0.02), (-0.1, 0, 0), (0, -0.1, -0.02)],
                                   radius=0.008, closed_spline=True)
    for mesh in (ring, hoop):
        assert mesh.is_watertight()
        assert mesh.to_trimesh().volume > 0


def test_corner_modes_keep_the_tube_thick() -> None:
    path = [(0, 0, 0), (0.1, 0, 0), (0.1, 0.1, 0)]
    sharp = wire_from_points(path, radius=0.01, cap_ends=True, corner_mode="miter")
    # A mitered corner reaches the outside corner of the bend: (0.11, -0.01).
    assert sharp.bounds()[1][0] == pytest.approx(0.11)
    assert sharp.bounds()[0][1] == pytest.approx(-0.01)
    for mode in ("bevel", "fillet"):
        assert wire_from_points(path, radius=0.01, cap_ends=True, corner_mode=mode,
                                corner_radius=0.03).is_watertight()


# ---- MeshGeometry --------------------------------------------------------------------------


def test_mesh_geometry_transforms_chain() -> None:
    box = BoxGeometry((0.2, 0.1, 0.05)).rotate_z(math.pi / 2).translate(1, 0, 0).scale(2)
    low, high = box.bounds()
    assert low == pytest.approx((2 - 0.1, -0.2, -0.05))
    assert high == pytest.approx((2 + 0.1, 0.2, 0.05))
    merged = BoxGeometry((0.1, 0.1, 0.1)).merge(BoxGeometry((0.1, 0.1, 0.1)).translate(1, 0, 0))
    assert len(merged.vertices) == 16 and len(merged.faces) == 24
    mirrored = BoxGeometry((0.1, 0.1, 0.1)).scale(-1, 1, 1)
    assert mirrored.to_trimesh().volume > 0  # winding flipped with the mirror


def test_exact_solid_follows_rigid_moves_and_drops_on_edits() -> None:
    box = RoundedBoxGeometry((0.1, 0.1, 0.1), 0.01)
    assert box.brep() is not None
    box.rotate_x(0.3).translate(0, 0, 1)
    assert box.brep() is not None
    assert box.brep().bounding_box().center().Z == pytest.approx(1.0)
    box.vertices[0, 0] += 0.001  # an in-place edit is detected by the fingerprint
    assert box.brep() is None


def test_booleans_accept_build123d_shapes() -> None:
    result = boolean_difference(CadBox(0.1, 0.1, 0.1), BoxGeometry((0.05, 0.05, 0.2)))
    assert result.is_watertight()
    assert result.to_trimesh().volume == pytest.approx(0.001 - 0.05 * 0.05 * 0.1)


def test_boolean_needs_closed_input() -> None:
    open_tube = CylinderGeometry(0.05, 0.1, closed=False)
    with pytest.raises(ValidationError, match="closed"):
        boolean_union(open_tube, BoxGeometry((0.1, 0.1, 0.1)))


@pytest.mark.parametrize(
    ("call", "option"),
    [
        (lambda: LoftGeometry(SQUARE_RINGS, cap_style="round"), "cap_style='round'"),
        (lambda: section_loft(SectionLoftSpec(sections=(rect_section(0.1, 0.1, 0),
                                                        rect_section(0.1, 0.1, 0.1)),
                                              path=((0, 0, 0), (0, 0, 0.1)))), "path="),
        (lambda: PipeGeometry(circle(0.01), [(0, 0, 0), (0, 0, 1)], sections=(object(),)),
         "sections="),
        (lambda: tube_from_spline_points([(0, 0, 0), (0, 0, 1)], radius=0.01, spline="bezier"),
         "spline='bezier'"),
    ],
)
def test_unported_options_fail_loudly(call, option: str) -> None:
    with pytest.raises(NotImplementedError, match=option.replace("(", r"\(")):
        call()
