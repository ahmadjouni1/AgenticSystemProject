"""Baseline checks and the run_tests(ctx) API, on broken fixtures with known defects."""

from __future__ import annotations

import importlib
import math
import time

import pytest

import check_fixtures as fx
from sdk import Assembly, Box, MeshGeometry, ValidationError
from sdk.testing import CheckConfig, TestContext, call_run_tests, run_checks


def only(report, check: str):
    found = [f for f in report.findings if f.check == check]
    assert len(found) == 1, report.text()
    return found[0]


# ---- each defect is found, with names, pose and size ------------------------------------------


def test_lid_inside_the_base_overlaps_at_rest() -> None:
    finding = only(run_checks(fx.laptop(lid_sink=0.005)), "overlap")
    assert finding.blocking
    assert finding.bodies == ("base", "lid") and finding.shapes == ("deck", "panel")
    assert finding.amount == pytest.approx(0.005, abs=1e-6)
    assert "overlap by 5.00 mm" in finding.message and finding.pose == {}


def test_floating_knob() -> None:
    finding = only(run_checks(fx.floating_knob(gap=0.010)), "floating")
    assert finding.blocking and finding.bodies == ("knob",)
    assert finding.amount == pytest.approx(0.010, abs=1e-6)
    assert "10.00 mm from 'knob'" in finding.message


def test_door_hits_its_frame_at_90_degrees() -> None:
    finding = only(run_checks(fx.door_in_frame()), "swept_collision")
    assert finding.blocking and finding.joint == "hinge"
    assert set(finding.bodies) == {"frame", "door"}
    assert math.degrees(finding.pose["hinge"]) == pytest.approx(90.0, abs=0.5)
    assert "first contact at 90.0 deg" in finding.message
    # Limited to 90 degrees, the same door is fine.
    assert run_checks(fx.door_in_frame(limit_degrees=89.0)).passed


def test_drawer_collides_when_fully_extended() -> None:
    """Prismatic joints are swept too (Articraft skips them)."""
    finding = only(run_checks(fx.drawer()), "swept_collision")
    assert finding.blocking and finding.joint == "slide"
    assert finding.shapes == ("lip", "back")
    assert finding.pose["slide"] == pytest.approx(0.26, abs=0.002)  # back panel meets the lip
    assert finding.amount == pytest.approx(0.020, abs=1e-6)  # fully overlapped at 280 mm
    assert "first contact at 260.0 mm" in finding.message


def test_hinge_on_the_wrong_axis_separates() -> None:
    finding = only(run_checks(fx.laptop(hinge_offset=(0.05, 0.05))), "separation")
    assert finding.blocking and finding.joint == "hinge"
    assert finding.bodies == ("base", "lid")
    assert finding.amount > 0.05  # the lid lifts clean off the base
    assert run_checks(fx.laptop()).passed  # the same laptop with the hinge on the edge


def test_four_bar_links_cross_mid_cycle() -> None:
    report = run_checks(fx.four_bar())
    crossings = [f for f in report.findings if f.check == "swept_collision"]
    assert crossings and all(f.blocking for f in crossings), report.text()
    assert {frozenset(f.bodies) for f in crossings} == {frozenset(("crank", "rocker"))}
    for finding in crossings:  # posed by the loop solver while the crank turns
        assert finding.joint == "crank_joint"
        assert -40.0 < math.degrees(finding.pose["crank_joint"]) < -5.0
        assert "crank_joint" in finding.message
    assert not any(f.check == "overlap" for f in report.findings)  # fine at rest


def test_loop_that_cannot_turn_fully_is_a_failure() -> None:
    finding = only(run_checks(fx.four_bar(coupler_degrees=40.0)), "loop")  # not Grashof
    assert finding.blocking and finding.joint == "crank_joint"
    assert "the loop cannot close for crank_joint" in finding.message


def test_joints_that_collide_only_together_are_caught_with_the_full_pose() -> None:
    report = run_checks(fx.two_arms())
    assert not [f for f in report.findings if f.check == "swept_collision"]  # each alone is clear
    finding = only(report, "combined_collision")
    assert finding.blocking and set(finding.bodies) == {"left_arm", "right_arm"}
    assert finding.pose == pytest.approx({"left_arm_swing": math.radians(60),
                                          "right_arm_swing": math.radians(60)})
    assert "left_arm_swing=60.0 deg, right_arm_swing=60.0 deg" in finding.message
    assert run_checks(fx.two_arms(spread=0.8)).passed  # far enough apart: clear


def test_corners_are_capped_and_in_gray_code_order() -> None:
    import numpy as np

    from sdk.testing import _corners

    corners = _corners([(0, 1)] * 3, 64, np.random.default_rng(0))
    assert len(corners) == 8
    assert all(sum(a != b for a, b in zip(p, q)) == 1 for p, q in zip(corners, corners[1:]))
    assert len(_corners([(0, 1)] * 8, 64, np.random.default_rng(0))) == 64  # 256 capped


# ---- allowances ----------------------------------------------------------------------------------


def test_allow_overlap_with_a_reason_passes_and_reports_it() -> None:
    model = fx.laptop(lid_sink=0.005)
    ctx = TestContext(model)
    ctx.allow_overlap("base", "lid", reason="the lid is recessed into the deck by design")
    report = run_checks(model, ctx)
    assert report.passed
    finding = only(report, "overlap")
    assert not finding.blocking and finding.allowed == "the lid is recessed into the deck by design"
    assert "recessed into the deck by design" in report.text()


def test_allowance_needs_a_reason_and_an_unused_one_is_reported() -> None:
    ctx = TestContext(fx.laptop())
    with pytest.raises(ValidationError, match="needs a reason"):
        ctx.allow_overlap("base", "lid", reason="")
    with pytest.raises(ValidationError, match="needs a reason"):
        ctx.allow_isolated("lid", reason="  ")
    ctx.allow_isolated("lid", reason="not actually needed")
    finding = only(run_checks(fx.laptop(), ctx), "allowance")
    assert not finding.blocking and "allowance not needed" in finding.message


def test_shape_level_allowance_does_not_cover_other_shapes() -> None:
    model = fx.laptop(lid_sink=0.005)
    model.get_body("lid").add("stand", Box((0.02, 0.02, 0.02)), xyz=(0.1, -0.1, -0.01))
    ctx = TestContext(model)
    ctx.allow_overlap("base", "lid", shape_b="stand", reason="the stand is let into the deck")
    report = run_checks(model, ctx)
    blocking = [f for f in report.findings if f.blocking]
    assert [f.shapes for f in blocking] == [("deck", "panel")]


def test_allow_isolated() -> None:
    model = fx.floating_knob()
    ctx = TestContext(model)
    ctx.allow_isolated("knob", reason="the knob is shown pulled off its shaft")
    report = run_checks(model, ctx)
    assert report.passed and only(report, "floating").allowed


# ---- scale-relative tolerances ---------------------------------------------------------------------


def test_tolerances_scale_with_the_object() -> None:
    small_gap, big_gap = run_checks(fx.stacked_blocks(0.02, 0.0005)), \
        run_checks(fx.stacked_blocks(2.0, 0.0005))
    assert small_gap.tolerances["contact"] == pytest.approx(1e-4)  # the floor wins at 2 cm
    assert big_gap.tolerances["contact"] == pytest.approx(1e-3 * big_gap.size)  # 3.5 mm at 2 m
    assert only(small_gap, "floating").blocking  # 0.5 mm is a real gap on a 2 cm part
    assert big_gap.passed  # ... and contact on a 2 m part
    small_sink, big_sink = run_checks(fx.stacked_blocks(0.02, -0.001)), \
        run_checks(fx.stacked_blocks(2.0, -0.001))
    assert only(small_sink, "overlap").blocking  # 1 mm into a 2 cm part
    assert big_sink.passed  # 1 mm into a 2 m part is within 6.9 mm


def test_all_thresholds_come_from_one_config() -> None:
    strict = CheckConfig(contact_floor=1e-6, contact_fraction=0.0)
    report = run_checks(fx.stacked_blocks(2.0, 0.0005), TestContext(
        fx.stacked_blocks(2.0, 0.0005), strict))
    assert only(report, "floating").blocking


# ---- mesh health, islands, dimensions ---------------------------------------------------------


def test_mesh_health_blocks_bad_winding() -> None:
    box = Box((0.1, 0.1, 0.1)).to_trimesh()
    faces = box.faces.copy()
    faces[0] = faces[0][::-1]  # one triangle flipped
    model = Assembly("broken")
    model.body("base").add("flipped", MeshGeometry(box.vertices, faces))
    report = run_checks(model)
    finding = only(report, "mesh_health")
    assert finding.blocking and "inconsistent triangle winding" in finding.message


def test_islands_and_odd_dimensions_only_warn() -> None:
    model = Assembly("loose")
    base = model.body("base")
    base.add("block", Box((0.1, 0.1, 0.1)))
    base.add("far_piece", Box((0.01, 0.01, 0.01)), xyz=(0.5, 0, 0))  # not touching: an island
    report = run_checks(model)
    islands = only(report, "islands")
    assert not islands.blocking and "far_piece" in islands.message and report.passed

    model = Assembly("huge")
    base = model.body("base")
    base.add("slab", Box((100.0, 0.01, 0.01)))  # absurd span (a unit slip: 100 m)
    report = run_checks(model)
    dimensions = only(report, "dimensions")
    assert not dimensions.blocking and "'base'/slab spans 100 m" in dimensions.message
    assert report.passed


# ---- the authored API ----------------------------------------------------------------------------


def test_expectations_record_failures_without_raising() -> None:
    model = fx.laptop()
    ctx = TestContext(model)
    assert ctx.expect_contact("base", "lid")
    assert ctx.expect_contact("base", "lid", pose={"hinge": 1.0})  # still touching at the hinge
    assert not ctx.expect_gap("base", "lid", min_gap=0.01)  # closed, so no 10 mm gap
    assert ctx.expect_gap("base", "lid", max_gap=0.001, pose={"hinge": 1.0})
    corner = ("lid", (0.16, -0.22, 0.0))
    assert ctx.expect_distance(corner, ("base", (0.16, -0.11, 0.015)), 0.0)
    assert not ctx.expect_distance(corner, ("base", (0.16, -0.11, 0.015)), 0.05)
    assert ctx.expect_no_collision_at_poses([{"hinge": 0.5}, {"hinge": 2.0}])
    report = run_checks(model, ctx)
    authored = [f for f in report.findings if f.check == "authored"]
    assert len(authored) == 2 and all(f.blocking for f in authored)


def test_sample_joint_and_track_point() -> None:
    ctx = TestContext(fx.laptop())
    samples = ctx.sample_joint("hinge", 4)
    assert [round(math.degrees(v)) for v, _ in samples] == [0, 45, 90, 135]
    path = ctx.track_point("lid", (0.16, -0.22, 0.008), "hinge", 3)
    tops = [point[2] for _, point in path]
    assert tops[0] == pytest.approx(0.023) and max(tops) > 0.2  # the corner rises


def test_expect_no_collision_reports_the_pose() -> None:
    ctx = TestContext(fx.door_in_frame())
    assert not ctx.expect_no_collision_at_poses([{"hinge": math.radians(110)}])
    assert "hinge=110.0 deg" in ctx.findings[0].message


def test_run_tests_with_and_without_context() -> None:
    ctx = TestContext(fx.laptop())
    assert call_run_tests(lambda: "old style", ctx) == "old style"
    assert call_run_tests(lambda c: c is ctx, ctx) is True


# ---- the good examples, and speed -------------------------------------------------------------

EXAMPLES = ["laptop", "four_bar", "gantry", "toaster", "door_hinge", "desk_lamp",
            "computer_mouse", "folding_chair"]


def check_example(name: str):
    module = importlib.import_module(f"examples.{name}")
    model = module.build()
    ctx = TestContext(model)
    if hasattr(module, "run_tests"):
        call_run_tests(module.run_tests, ctx)
    return run_checks(model, ctx)


@pytest.mark.parametrize("name", EXAMPLES)
def test_good_examples_have_no_blocking_findings(name: str) -> None:
    report = check_example(name)
    assert report.passed, report.text()
    assert not [f for f in report.findings if f.check == "allowance"]  # every allowance is used


def test_checks_for_all_examples_stay_within_budget() -> None:
    for name in EXAMPLES:  # build first: import and geometry time is not check time
        importlib.import_module(f"examples.{name}").build()
    start = time.perf_counter()
    for name in EXAMPLES:
        check_example(name)
    assert time.perf_counter() - start < 3.0 * 2  # 3 s target; slack for slow CI machines
