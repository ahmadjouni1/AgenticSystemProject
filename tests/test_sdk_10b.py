"""Step 10b SDK fixes: helpers that repair or explain bad arguments (from the step 10 failure
analysis: 10 failed compiles, 2 lost runs)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from compiler import compile as compile_run
from sdk import Assembly, Box, ExtrudeGeometry, ExtrudeWithHolesGeometry, RoundedBoxGeometry
from sdk.errors import ValidationError
from sdk.mesh_geometry import first_crossing
from sdk.notices import drain


def square(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]


def test_a_too_large_rounded_box_radius_is_clamped_with_a_notice() -> None:
    drain()
    box = RoundedBoxGeometry((0.05, 0.05, 0.02), 0.012)  # 0.012 >= 0.02 / 2
    (item,) = drain()
    assert "reduced to 0.009" in item.message
    mesh = box.to_trimesh(process=True)
    assert mesh.is_watertight
    assert np.allclose(mesh.bounds[1] - mesh.bounds[0], (0.05, 0.05, 0.02), atol=1e-4)
    RoundedBoxGeometry((0.05, 0.05, 0.02), 0.004)
    assert drain() == []  # a valid radius: no notice


def test_the_compiler_reports_the_clamp_as_a_warning_at_the_calling_line(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "workspace").mkdir(parents=True)
    (run_dir / "workspace" / "main.py").write_text(textwrap.dedent("""
        from sdk import Assembly, RoundedBoxGeometry

        model = Assembly("puck")
        body = model.body("puck")
        body.add("disc", RoundedBoxGeometry((0.06, 0.06, 0.01), 0.008))
        object_model = model
    """).lstrip(), encoding="utf-8")
    result = compile_run(run_dir, timeout=120)
    assert result.ok, result.signals_text  # a warning never blocks
    (warning,) = [s for s in result.signals if s.kind == "helper_adjusted"]
    assert warning.severity == "warning" and warning.line == 5 and "reduced to" in warning.message
    assert "helper_adjusted" in result.signals_text


def test_a_self_intersecting_profile_names_the_crossing_edges_and_point() -> None:
    crossed = [(0, 0), (4, 0), (4, 2), (2, -1), (1, 3), (0, 2)]
    with pytest.raises(ValidationError) as info:
        ExtrudeGeometry(crossed, 0.1)
    text = str(info.value)
    assert "edge 0 (0, 0) -> (4, 0) crosses edge 2 (4, 2) -> (2, -1) at (2.667, 0)" in text
    assert first_crossing(np.array(crossed, dtype=float))[:2] == (0, 2)


def test_simple_concave_profiles_still_extrude() -> None:
    ell = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    assert first_crossing(np.array(ell, dtype=float)) is None
    assert ExtrudeGeometry(ell, 0.1).to_trimesh(process=True).is_watertight


def test_a_hole_outside_the_outline_is_named() -> None:
    with pytest.raises(ValidationError, match=r"hole_profiles\[1\] is not inside the outer "
                                              r"profile: its point \(1\.05, -0\.1\)"):
        ExtrudeWithHolesGeometry(square(0, 0, 1), [square(0, 0, 0.1), square(0.95, 0, 0.1)], 0.1)


def test_overlapping_holes_are_named() -> None:
    with pytest.raises(ValidationError, match=r"hole_profiles\[0\] and hole_profiles\[1\] "
                                              "overlap"):
        ExtrudeWithHolesGeometry(square(0, 0, 1), [square(0, 0, 0.2), square(0.1, 0, 0.2)], 0.1)


def test_good_holes_still_work() -> None:
    plate = ExtrudeWithHolesGeometry(square(0, 0, 1), [square(-0.5, 0, 0.1),
                                                       square(0.5, 0, 0.1)], 0.1)
    assert plate.to_trimesh(process=True).is_watertight


def test_a_duplicate_shape_name_suggests_a_numbered_name() -> None:
    body = Assembly("bench").body("frame")
    body.add("foot", Box((0.1, 0.1, 0.1)))
    with pytest.raises(ValidationError, match=r"in a loop: f'foot_\{i\}'"):
        body.add("foot", Box((0.1, 0.1, 0.1)))
