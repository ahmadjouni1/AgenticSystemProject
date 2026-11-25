"""Step 13: the teammate's nested solver, the plug-ins, the synthesize tool, the trajectory check.

No network. The solver tests in the first section were written for the replica after the
README's description of the teammate's eight tests, which were never committed (see
solvers/nested/ATTRIBUTION.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import pytest

from agent.loop import PROMPT_SETS, run
from agent.tools import ToolContext, run_tool, schemas
from agent.tools.synthesize import LINKAGE_RULE
from llm import Budget, ScriptedModel
from compiler import compile as compile_run
from sdk.errors import ValidationError
from sdk.target_path import check_target_path, distance_to_path
from solvers.base import (REPLICA_ROOT, SolverError, SolverTimeout, Target, solve,
                          solve_isolated)
from solvers.nested import NestedSynthesis, SerialURDF, Settings, Task, load_problem
from solvers.planar import LINKAGES, sweep
from solvers.search import Synthesis

NESTED = REPLICA_ROOT / "solvers" / "nested"
EXAMPLE = NESTED / "examples" / "spatial_3r.json"
KNOWN_3R = np.array([0.28, 0.42, 0.33])  # the design the example's waypoints come from
# SHA-256 of the teammate's files (Previous work/Agentic_Systems_Project/Project_extra_tools)
TEAMMATE_FILES = {
    "synthesis.py": "4f1ecbb81d1724e66cb783d2eb1da352df83e606365bf11c8251b62b0b54db65",
    "README.md": "97f3cb3cde04bce3c7849e149584e637181bcacbabdc321090ec6ee1691fc2e4",
}


def test_the_teammates_files_are_unmodified_copies() -> None:
    for name, digest in TEAMMATE_FILES.items():
        assert hashlib.sha256((NESTED / name).read_bytes()).hexdigest() == digest, name


# ---- the teammate's solver ------------------------------------------------------------------


class TwoLink:
    """A custom model with no URDF: a planar 2R arm, design = (upper, lower) lengths."""
    joint_names = ["shoulder", "elbow"]
    lower = np.array([-math.pi, -math.pi])
    upper = np.array([math.pi, math.pi])

    def fk(self, design, q):
        q = np.atleast_2d(q)
        a, b = q[:, 0], q[:, 0] + q[:, 1]
        elbow = np.c_[design[0] * np.cos(a), design[0] * np.sin(a), np.zeros(len(q))]
        tip = elbow + np.c_[design[1] * np.cos(b), design[1] * np.sin(b), np.zeros(len(q))]
        nodes = np.stack([np.zeros_like(tip), elbow, tip], axis=1)
        return tip, np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy(), nodes


def two_link_task(true=(0.5, 0.3), n=8) -> Task:
    s = np.linspace(0, 1, n)
    q = np.c_[0.2 + 0.9 * s, 0.4 + 0.6 * np.sin(np.pi * s)]
    return Task(TwoLink().fk(np.array(true), q)[0], q[0], q[-1], np.full(2, 0.4),
                position_tolerance=1e-3, length_scale=0.8)


def two_link_solver(task: Task | None = None, cls=NestedSynthesis, **kwargs) -> NestedSynthesis:
    return cls(TwoLink(), task or two_link_task(), np.array([0.4, 0.4]), np.array([0.1, 0.1]),
               np.array([1.0, 1.0]), Settings(outer_maxeval=300), **kwargs)


def test_a_custom_model_without_urdf_finds_the_known_design() -> None:
    best, stats = two_link_solver(cls=Synthesis).run()
    assert best.feasible and best.max_error < 1e-5, stats
    assert np.allclose(best.design, [0.5, 0.3], atol=1e-4)


def test_supplied_endpoints_stay_exact() -> None:
    task = two_link_task()
    best, _ = two_link_solver(task, cls=Synthesis).run()
    assert np.array_equal(best.q[0], task.q_start) and np.array_equal(best.q[-1], task.q_end)


def test_an_infeasible_task_returns_a_graded_violation_instead_of_raising() -> None:
    task = two_link_task()
    task.extra_margins = lambda design, q: np.full(len(q), -0.5)  # never satisfiable
    best, _ = two_link_solver(task).solve(verbose=False)
    assert not best.feasible and best.min_margin == pytest.approx(-0.5)


def test_the_solve_is_repeatable() -> None:
    first, _ = two_link_solver().solve(verbose=False)
    second, _ = two_link_solver().solve(verbose=False)
    assert np.array_equal(first.design, second.design) and np.array_equal(first.q, second.q)


def test_endpoints_the_step_limits_cannot_connect_are_rejected() -> None:
    task = two_link_task()
    task.max_step = np.full(2, 0.01)
    with pytest.raises(ValueError, match="cannot be connected"):
        two_link_solver(task)


def test_the_urdf_adapter_matches_a_hand_computed_spatial_3r() -> None:
    spec = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    model = SerialURDF(EXAMPLE.parent / spec["urdf"], "base_link", "tool",
                       spec["design_parameters"])
    h, a, b = 0.29, 0.41, 0.37
    yaw, shoulder, elbow = 0.7, -0.4, 0.9
    tip = model.fk(np.array([h, a, b]), np.array([[yaw, shoulder, elbow]]))[0][0]
    # Pitch about +y turns +x toward -z: reach r and height z in the arm's vertical plane.
    r = a * math.cos(shoulder) + b * math.cos(shoulder + elbow)
    z = h - a * math.sin(shoulder) - b * math.sin(shoulder + elbow)
    assert np.allclose(tip, [r * math.cos(yaw), r * math.sin(yaw), z], atol=1e-12)


def test_the_exported_kinematic_urdf_reloads_to_the_same_kinematics(tmp_path) -> None:
    solver, spec = load_problem(EXAMPLE)
    q = np.array([[0.3, -0.2, 0.5], [1.0, 0.4, -0.3]])
    out = tmp_path / "optimized.urdf"
    solver.model.write_kinematic_urdf(KNOWN_3R, out)
    params = [{**p, "initial": float(v)} for p, v in zip(spec["design_parameters"], KNOWN_3R,
                                                          strict=True)]
    reloaded = SerialURDF(out, "base_link", "tool", params)
    assert np.allclose(reloaded.fk(KNOWN_3R, q)[0], solver.model.fk(KNOWN_3R, q)[0], atol=1e-10)


def test_spatial_3r_the_nested_search_alone_stalls_and_the_polish_finds_the_known_design() -> None:
    solver, _ = load_problem(EXAMPLE)
    initial_error = solver.solve_internal(solver.initial).max_error
    nested, _ = solver.solve(verbose=False)  # the teammate's own loop, unchanged
    assert nested.feasible and nested.max_error < initial_error
    assert nested.max_error > solver.task.position_tolerance  # the corner stall (search.py)
    ours = Synthesis(solver.model, solver.task, solver.initial, solver.lower, solver.upper,
                     solver.settings)
    best, stats = ours.run()
    assert best.feasible and best.max_error < 1e-5, stats
    assert np.allclose(best.design, KNOWN_3R, atol=1e-4)


# ---- the plug-in interface ------------------------------------------------------------------


@pytest.mark.parametrize("data", [
    {"points": "__import__('os').system('echo hi')"},
    {"points": [["0.1", 0, 0], [0.2, 0, 0], [0.3, 0, 0]]},
    {"points": [[0.1, 0, 0], [0.2, 0, 0], [float("nan"), 0, 0]]},
    {"points": [[0.1, 0, 0], [0.2, 0, 0]]},
    {"points": [[0.1, 0, 0]] * 3},
    {"points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]], "function": "lambda t: t"},
    {"points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]], "tolerance": -1},
], ids=["code", "string", "nan", "two", "same", "extra", "tolerance"])
def test_targets_are_checked_as_data_and_never_run(data) -> None:
    with pytest.raises(SolverError):
        Target.from_data(data)


def test_an_isolated_solve_is_stopped_at_its_timeout() -> None:
    points = [[0.1 * math.cos(a), 0.1 * math.sin(a), 0] for a in np.linspace(0, 6, 20)]
    started = time.perf_counter()
    with pytest.raises(SolverTimeout, match="did not finish in 3 s"):
        solve_isolated("four_bar", {"points": points}, {"time_limit_s": 200}, timeout_s=3)
    assert time.perf_counter() - started < 15


def test_isolated_solver_errors_come_back_as_solver_errors() -> None:
    with pytest.raises(SolverError, match="unsupported mechanism .hexapod."):
        solve_isolated("hexapod", {"points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}, {})
    with pytest.raises(SolverError, match="unknown constraints"):
        solve_isolated("four_bar", {"points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
                       {"crank": 1})


# ---- plug-in A: serial arms ------------------------------------------------------------------


def arm_constraints(**extra) -> tuple[dict, dict]:
    spec = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    target = {"points": spec["waypoints"]["positions_m"],
              "tolerance": spec["position_tolerance_m"]}
    constraints = {"urdf": (EXAMPLE.parent / spec["urdf"]).read_text(encoding="utf-8"),
                   "base_link": "base_link", "tip_link": "tool",
                   "parameters": spec["design_parameters"], **extra}
    return target, constraints


def test_plugin_a_solves_the_spatial_3r_example() -> None:
    spec = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    target, constraints = arm_constraints(q_start=spec["q_start"], q_end=spec["q_end"],
                                          max_step=spec["max_joint_step"])
    result = solve("serial_arm", target, constraints)
    assert result.within_tolerance and result.error["max"] < 2e-4, result.message
    assert np.allclose(list(result.parameters.values()), KNOWN_3R, atol=2e-3)
    assert result.details["joint_origins"]["elbow"]["xyz"][0] == pytest.approx(
        result.parameters["upper_arm_length_m"])


def test_plugin_a_finds_its_own_endpoints_by_inverse_kinematics() -> None:
    target, constraints = arm_constraints(max_evaluations=200)
    result = solve("serial_arm", target, constraints)
    assert result.within_tolerance, result.message


def test_plugin_a_keeps_outer_maxeval_configurable() -> None:
    target, constraints = arm_constraints(max_evaluations=5, polish=False)
    result = solve("serial_arm", target, constraints)
    assert result.details["geometry_evaluations"] <= 6  # COBYLA may check its last point


# ---- plug-in B: closed loops -----------------------------------------------------------------

KNOWN_FOUR_BAR = {"crank_pivot_x": 0.0, "crank_pivot_y": 0.0, "crank_length": 0.03,
                  "coupler_length": 0.09, "ground_length": 0.10, "ground_angle": 0.0,
                  "coupler_angle": math.radians(50), "crank_phase": 0.0, "point_x": 0.05,
                  "point_y": 0.03}
KNOWN_SLIDER_CRANK = {"crank_pivot_x": 0.0, "crank_pivot_y": 0.0, "crank_length": 0.025,
                      "rod_length": 0.08, "rod_angle": math.radians(-20), "slider_angle": 0.0,
                      "crank_phase": math.radians(20), "point_x": 0.04, "point_y": 0.015}


def recorded_path(name: str, params: dict, samples: int = 13) -> np.ndarray:
    """Build the known linkage, turn its crank once, and record the output point (in the
    linkage's own plane, xy)."""
    assembly, body = LINKAGES[name].build(params, 2 * math.pi)
    curve = sweep(assembly, body, (params["point_x"], params["point_y"]), 2 * math.pi)
    assert curve.reach == 1.0
    return curve.at(np.linspace(0, 1, samples))


def perturbed(params: dict) -> dict:
    return {k: v * 1.08 if "length" in k else v + 0.1 if ("angle" in k or "phase" in k)
            else v + 0.003 for k, v in params.items()}


def reproduces(name: str, params: dict, values: list[float], target: np.ndarray,
               tolerance: float) -> dict:
    """Independent check: build the solved linkage again and run the compile's trajectory
    check (values mode) on its output point."""
    assembly, body = LINKAGES[name].build(params, 2 * math.pi)
    spec = {"body": body, "point": [params["point_x"], params["point_y"], 0.0],
            "joint": "crank_joint", "points": target.tolist(), "tolerance": tolerance,
            "values": values}
    findings, summary = check_target_path(assembly, spec)
    assert not findings, findings
    return summary


@pytest.mark.parametrize("name,known,plane", [("four_bar", KNOWN_FOUR_BAR, "xy"),
                                              ("slider_crank", KNOWN_SLIDER_CRANK, "xz")])
def test_plugin_b_recovers_dimensions_that_reproduce_a_recorded_path(name, known, plane) -> None:
    path = recorded_path(name, known)
    target = (np.c_[path, np.zeros(len(path))] if plane == "xy"
              else np.c_[path[:, 0], np.zeros(len(path)), path[:, 1]])
    tolerance = 0.001
    result = solve(name, {"points": target.tolist(), "tolerance": tolerance},
                   {"plane": plane, "initial": perturbed(known), "time_limit_s": 150})
    assert result.within_tolerance, (result.message, result.details)
    # The start is 8% off. The search stops within a tenth of the tolerance, where several
    # nearby linkages trace the path: the lengths come out 1-8% from the known ones.
    for key in LINKAGES[name].lengths:
        assert result.parameters[key] == pytest.approx(known[key], rel=0.10), key
    planar = (target if plane == "xy" else
              np.c_[target[:, 0], target[:, 2], np.zeros(len(target))])
    summary = reproduces(name, result.parameters, result.details["crank_values"], planar,
                         tolerance)
    assert summary["max_error_mm"] <= tolerance * 1000
    assert result.details["seconds_per_pose"] < 0.05


def test_plugin_b_slider_output_tracks_the_slider_pin() -> None:
    known = {**KNOWN_SLIDER_CRANK, "point_x": KNOWN_SLIDER_CRANK["rod_length"], "point_y": 0.0}
    path = recorded_path("slider_crank", known)
    assert np.ptp(path[:, 1]) < 1e-9  # the pin runs along the rail (here along x)
    fixed = {k: known[k] for k in ("crank_pivot_x", "crank_pivot_y", "slider_angle",
                                   "crank_phase", "rod_angle")}
    result = solve("slider_crank", {"points": np.c_[path, np.zeros(len(path))].tolist(),
                                    "tolerance": 0.0005},
                   {"output": "slider", "fixed": fixed,
                    "initial": {"crank_length": 0.03, "rod_length": 0.07}})
    assert result.within_tolerance, result.message
    assert result.parameters["point_x"] == result.parameters["rod_length"]


def test_plugin_b_reports_a_linkage_that_cannot_turn_through_the_span() -> None:
    # A crank longer than the rest of the loop allows: it cannot make a full turn.
    params = {**KNOWN_FOUR_BAR, "crank_length": 0.09}
    assembly, body = LINKAGES["four_bar"].build(params, 2 * math.pi)
    curve = sweep(assembly, body, (0.05, 0.03), 2 * math.pi)
    assert 0 < curve.reach < 1


# ---- the synthesize tool ---------------------------------------------------------------------


def test_the_tool_is_offered_only_in_prompt_set_v2() -> None:
    assert "synthesize" not in [s.name for s in schemas()]
    assert [s.name for s in schemas(synthesis=True)][-1] == "synthesize"
    v1, v2 = PROMPT_SETS["v1"].system_text(), PROMPT_SETS["v2"].system_text()
    assert "synthesis.md" not in v1 and v2.count("synthesis.md") == 1
    lines = v2.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("- `testing.md`"))
    assert lines[at + 1].startswith("- `synthesis.md`")
    workflow = lines.index("## Workflow")
    assert lines[workflow + 2].startswith("**Linkages:** for four-bars and slider-cranks only")
    assert LINKAGE_RULE.count("\n") == 1 and "skeleton" in LINKAGE_RULE  # two lines
    assert v2.replace(lines[at + 1] + "\n", "").replace(LINKAGE_RULE + "\n\n", "") == v1


def test_the_run_record_counts_synthesize_calls_and_the_final_target_path(tmp_path) -> None:
    _model, spec = four_bar_target()
    spec.pop("_values")
    main = ("from examples.four_bar import build\n\nobject_model = build()\n"
            f"target_path = {json.dumps(spec)}\n")
    same = {"points": [[0, 0, 0]] * 3}
    replies = [{"tool_calls": [{"name": "synthesize", "arguments": {
                   "mechanism": "four_bar", "target": same}}]},
               {"tool_calls": [{"name": "write", "arguments": {"path": "main.py",
                                                               "content": main}}]},
               {"tool_calls": [{"name": "compile", "arguments": {}}]},
               "Done: a four-bar whose coupler point follows the target."]
    budget = Budget(max_cost_usd=None, max_calls=None)
    result = run("a four-bar", ScriptedModel(replies, budget=budget), budget,
                 run_dir=tmp_path / "v2", prompt_set="v2")
    record = json.loads((result.run_dir / "record.json").read_text(encoding="utf-8"))
    synthesis = record["synthesis"]
    assert synthesis["count"] == 1 and synthesis["calls"][0]["outcome"] == "solver_error"
    assert synthesis["calls"][0]["wall_s"] > 0 and synthesis["final_build"] == "v0000"
    assert synthesis["final_declared_target_path"] is True
    assert synthesis["final_target_path"]["passed"] is True
    # Prompt set v1 does not offer the tool (the scripted model refuses the call, as a real
    # provider would reject an undeclared tool), and nothing is solved.
    v1 = run("a four-bar", ScriptedModel(replies, budget=budget), budget,
             run_dir=tmp_path / "v1", prompt_set="v1")
    assert "not one of the tools offered" in v1.reason
    record = json.loads((v1.run_dir / "record.json").read_text(encoding="utf-8"))
    assert record["synthesis"]["count"] == 0


@pytest.fixture()
def ctx(tmp_path):
    return ToolContext(tmp_path / "run")


def tool_error(ctx, args) -> dict:
    out = run_tool(ctx, "synthesize", args)
    assert out.is_error, out.text
    return out.data


def test_the_tool_reports_bad_requests_as_structured_errors(ctx, monkeypatch) -> None:
    line = {"points": [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]}
    started = time.perf_counter()
    scissor = tool_error(ctx, {"mechanism": "scissor_lift", "target": line})
    assert scissor["error_kind"] == "unsupported" and "design" in scissor["hint"]
    assert time.perf_counter() - started < 0.5  # answered at once, no solver process
    assert ctx.cache["synthesis_calls"][-1]["outcome"] == "unsupported"
    bad = tool_error(ctx, {"mechanism": "four_bar", "target": {"points": [[0, 0, "x"]] * 3}})
    assert bad["error_kind"] == "bad_arguments"  # the schema says numbers
    same = tool_error(ctx, {"mechanism": "four_bar", "target": {"points": [[0, 0, 0]] * 3}})
    assert same["error_kind"] == "solver_error" and "same point" in same["message"]
    unknown = tool_error(ctx, {"mechanism": "four_bar", "target": line,
                               "constraints": {"fixed": {"wheel": 1}}})
    assert "unknown parameters" in unknown["message"]
    assert tool_error(ctx, {"mechanism": "serial_arm", "target": line})["error_kind"] == \
        "no_build"
    limit = tool_error(ctx, {"mechanism": "four_bar", "target": line,
                             "constraints": {"time_limit_s": 1000}})
    assert "time_limit_s" in limit["message"]
    monkeypatch.setattr("agent.tools.synthesize.STARTUP_S", -4.0)  # a 1 s process timeout
    slow = tool_error(ctx, {"mechanism": "four_bar", "target": line,
                            "constraints": {"time_limit_s": 5}})
    assert slow["error_kind"] == "solver_timeout"


def test_the_tool_returns_numbers_and_saves_the_result(ctx) -> None:
    known = {**KNOWN_SLIDER_CRANK, "point_x": KNOWN_SLIDER_CRANK["rod_length"], "point_y": 0.0}
    path = recorded_path("slider_crank", known, samples=9)
    fixed = {k: known[k] for k in ("crank_pivot_x", "crank_pivot_y", "slider_angle",
                                   "crank_phase", "rod_angle")}
    out = run_tool(ctx, "synthesize", {
        "mechanism": "slider_crank",
        "target": {"points": np.c_[path[:, 0], np.full(9, 0.2), path[:, 1]].tolist(),
                   "tolerance": 0.001},
        "constraints": {"plane": "xz", "output": "slider", "fixed": fixed}})
    assert not out.is_error, out.text
    data = out.data
    assert data["within_tolerance"] and data["error_mm"]["max"] <= 1.0
    assert data["parameters"]["crank_length"] == pytest.approx(0.025, rel=0.02)
    assert set(data["rest_points"]) == {"crank_joint", "rod_joint", "slide", "pin",
                                        "output_point"}
    assert data["rest_points"]["crank_joint"][1] == pytest.approx(0.2)  # the plane's y
    saved = json.loads((ctx.run_dir / data["saved"]).read_text(encoding="utf-8"))
    assert saved["result"]["parameters"]["rod_length"] == pytest.approx(
        data["parameters"]["rod_length"], abs=1e-6)


ARM = """import math

from sdk import Assembly, Box, Cylinder

model = Assembly("arm")
base = model.body("base")
base.add("plate", Box((0.2, 0.2, 0.05)), xyz=(0, 0, 0.025))
base.add("post", Box((0.03, 0.08, 0.07)), xyz=(0, 0, 0.035))
upper = model.body("upper")
upper.add("bar", Box((UPPER, 0.04, 0.04)), xyz=(UPPER / 2, 0, 0))
upper.add("hub", Cylinder(radius=0.025, height=0.05), rpy=(math.pi / 2, 0, 0))
fore = model.body("fore")
fore.add("bar", Box((0.25, 0.03, 0.03)), xyz=(0.125, 0, 0))
model.joint("shoulder", "revolute", base.at((0, 0, 0.07)), upper.at(), axis=(0, 1, 0),
            limits=(-1.5, 1.5))
model.joint("elbow", "revolute", upper.at((UPPER, 0, 0)), fore.at(), axis=(0, 1, 0),
            limits=(-2.5, 2.5))
object_model = model


def run_tests(ctx):
    ctx.allow_overlap("base", "upper", reason="the shoulder pin sits in the base")
    ctx.allow_overlap("upper", "fore", reason="the elbow pin sits in the upper arm")
"""


def test_the_tool_sizes_a_built_arm_and_converts_frames(ctx) -> None:
    (ctx.workspace / "main.py").write_text(ARM.replace("UPPER", "0.3"), encoding="utf-8")
    compiled = run_tool(ctx, "compile", {})
    assert "Compile passed" in compiled.text, compiled.text
    # Where a 0.36 m upper arm puts the forearm's tip (0.25 m) for a few joint values.
    q = np.c_[np.linspace(-0.6, 0.2, 7), np.linspace(1.2, 0.4, 7)]
    a, b = q[:, 0], q[:, 0] + q[:, 1]
    tip = np.c_[0.36 * np.cos(a) + 0.25 * np.cos(b), np.zeros(7),
                0.07 - 0.36 * np.sin(a) - 0.25 * np.sin(b)]
    out = run_tool(ctx, "synthesize", {
        "mechanism": "serial_arm", "target": {"points": tip.tolist(), "tolerance": 0.001},
        "constraints": {"tip_body": "fore", "tip_point": [0.25, 0, 0], "q_start": q[0].tolist(),
                        "q_end": q[-1].tolist(),
                        "parameters": [{"name": "upper_arm", "initial": 0.3, "lower": 0.2,
                                        "upper": 0.5,
                                        "bindings": [{"joint": "elbow", "xyz": [1, 0, 0]}]}]}})
    assert not out.is_error, out.text
    assert out.data["within_tolerance"], out.data["message"]
    assert out.data["parameters"]["upper_arm"] == pytest.approx(0.36, abs=1e-3)
    assert out.data["joint_origins"]["elbow"]["parent"] == "upper"
    assert out.data["joint_origins"]["elbow"]["xyz"] == pytest.approx([0.36, 0, 0], abs=1e-3)


# ---- the trajectory check --------------------------------------------------------------------


def four_bar_target(shift: float = 0.0, count: int = 12) -> tuple[object, dict]:
    from examples.four_bar import COUPLER, THICK, build

    model = build()
    point = [COUPLER / 2, 0.01, THICK]
    values = np.linspace(-math.pi, math.pi, count, endpoint=False)
    points = []
    for value in values:
        matrix = model.solve({"crank_joint": float(value)}).transforms["coupler"]
        points.append(matrix[:3, :3] @ point + matrix[:3, 3] + [0, 0, shift])  # off-plane
    spec = {"body": "coupler", "point": point, "joint": "crank_joint",
            "points": np.array(points).tolist(), "tolerance": 0.001}
    return build(), {**spec, "_values": values.tolist()}


def test_the_trajectory_check_passes_on_the_built_path_and_fails_off_it() -> None:
    model, spec = four_bar_target()
    values = spec.pop("_values")
    findings, summary = check_target_path(model, spec)
    assert not findings and summary["passed"] and summary["max_error_mm"] < 0.5
    findings, summary = check_target_path(model, {**spec, "values": values})
    assert not findings and summary["max_error_mm"] < 1e-3
    model, spec = four_bar_target(shift=0.004)
    spec.pop("_values")
    findings, summary = check_target_path(model, spec)
    assert not summary["passed"] and len(findings) == 1
    finding = findings[0]
    assert finding.blocking and finding.check == "target_path" and "crank_joint" in finding.pose
    assert 3.9 < finding.amount * 1000 < 4.1  # the 4 mm shift out of the linkage's plane
    assert "target path missed" in finding.message


def test_the_trajectory_check_rms_tolerance_and_declaration_errors() -> None:
    model, spec = four_bar_target(shift=0.0008)
    spec.pop("_values")
    assert check_target_path(model, spec)[1]["passed"]
    assert not check_target_path(model, {**spec, "rms_tolerance": 0.0002})[1]["passed"]
    with pytest.raises(ValidationError, match="the loop's driver is 'crank_joint'"):
        check_target_path(model, {**spec, "joint": "rocker_joint"})
    with pytest.raises(ValidationError, match="missing"):
        check_target_path(model, {"body": "coupler"})
    with pytest.raises(ValidationError, match="one driver value per point"):
        check_target_path(model, {**spec, "values": [0.0]})


def test_distance_to_a_polyline() -> None:
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float)
    distance, segment = distance_to_path(np.array([[0.5, 0.2, 0], [1.3, 0.5, 0]]), path)
    assert np.allclose(distance, [0.2, 0.3]) and list(segment) == [0, 1]


@pytest.mark.parametrize("shift,passes", [(0.0, True), (0.004, False)])
def test_compile_blocks_a_missed_target_path(tmp_path, shift, passes) -> None:
    _model, spec = four_bar_target(shift)
    spec.pop("_values")
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "main.py").write_text(
        "from examples.four_bar import build\n\nobject_model = build()\n"
        f"target_path = {json.dumps(spec)}\n", encoding="utf-8")
    result = compile_run(tmp_path, timeout=180)
    if passes:
        assert result.ok and "target_path: max error" in result.signals_text, result.signals_text
    else:
        assert not result.ok
        assert [s.kind for s in result.signals if s.severity == "failure"] == ["target_path"]
        assert "synthesize" in result.signals_text  # the hint names the tool


# ---- function generation (after synth_v1): output motion vs crank angle --------------------


def known_motion(name: str, params: dict, crank=None):
    from solvers.function import motion

    return motion(LINKAGES[name], params, 2 * math.pi, step=math.radians(1.0), at=crank)


def rebuilt_check(name: str, params: dict, spec: dict) -> dict:
    """Independent check of a solved design: build it again, run the compile's check."""
    from sdk.target_path import check_target_path as check

    assembly, _ = LINKAGES[name].build(params, 2 * math.pi)
    findings, summary = check(assembly, {"joint": "crank_joint", **spec})
    assert not findings, findings
    return summary


@pytest.mark.parametrize("data,match", [
    ({"sweep": -1.0}, "positive"),
    ({"sweep": 1.0, "stroke": 0.05}, "one of"),
    ({"stroke": "60 mm"}, "positive"),
    ({"sweep": 1.0, "points": 3}, "points"),
    ({"crank": [0, 1, 2], "output": [0, 0.1, 0.2, 0.3]}, "same length"),
    ({"crank": [0, 1, 2], "output": [0, 0.1, 0.2], "lambda": "x"}, "unknown fields"),
], ids=["negative", "two", "string", "mixed", "lengths", "extra"])
def test_function_targets_are_checked_as_data(data, match) -> None:
    from solvers.base import parse_target

    with pytest.raises(SolverError, match=match):
        parse_target(data)


def test_function_targets_go_only_to_mechanisms_that_have_that_output() -> None:
    with pytest.raises(SolverError, match="takes a \"sweep\" target"):
        solve("four_bar", {"stroke": 0.06}, {})
    with pytest.raises(SolverError, match="takes a \"stroke\" target"):
        solve("slider_crank", {"sweep": 1.0}, {})
    with pytest.raises(SolverError, match="path target"):
        solve("serial_arm", {"sweep": 1.0}, {})
    with pytest.raises(SolverError, match="unsupported mechanism 'pantograph'"):
        solve("pantograph", {"sweep": 1.0}, {})


def test_four_bar_sweep_of_a_known_linkage_is_recovered() -> None:
    known = known_motion("four_bar", KNOWN_FOUR_BAR)
    tolerance = math.radians(0.5)
    result = solve("four_bar", {"sweep": known.extent, "tolerance": tolerance}, {})
    assert result.within_tolerance, result.message
    assert result.details["derived"]["min_transmission_deg"] >= 34.5
    summary = rebuilt_check("four_bar", result.parameters, {
        "output_joint": "rocker_joint", "sweep": known.extent, "tolerance": tolerance})
    assert summary["passed"] and summary["achieved"] == pytest.approx(
        math.degrees(known.extent), abs=0.5)


def test_a_wiper_sweep_of_90_degrees_without_jamming() -> None:
    started = time.perf_counter()
    result = solve("four_bar", {"sweep": math.radians(90), "tolerance": math.radians(0.5)},
                   {"initial": {"ground_length": 0.25}})
    assert result.within_tolerance and result.feasible, result.message
    assert result.details["derived"]["grashof"]
    assert result.details["derived"]["min_transmission_deg"] >= 34.5
    assert time.perf_counter() - started < 60
    rebuilt_check("four_bar", result.parameters, {"output_joint": "rocker_joint",
                                                  "sweep": math.radians(90),
                                                  "tolerance": math.radians(0.5)})


def test_four_bar_function_of_a_known_linkage_is_recovered() -> None:
    crank = np.radians(np.arange(0.0, 360.0, 30.0))
    known = known_motion("four_bar", KNOWN_FOUR_BAR, crank)
    start = {**KNOWN_FOUR_BAR, "crank_length": 0.034, "coupler_length": 0.085,
             "coupler_angle": 0.95}  # not a uniform scale: angles change
    tolerance = math.radians(0.5)
    result = solve("four_bar", {"crank": crank.tolist(), "output": known.output.tolist(),
                                "tolerance": tolerance}, {"initial": start})
    assert result.within_tolerance, result.message
    ratio = lambda p, k: p[k] / p["ground_length"]  # noqa: E731 - angles fix only ratios
    for key in ("crank_length", "coupler_length"):
        assert ratio(result.parameters, key) == pytest.approx(ratio(KNOWN_FOUR_BAR, key),
                                                              rel=0.03), key
    rebuilt_check("four_bar", result.parameters, {
        "output_joint": "rocker_joint", "values": crank.tolist(),
        "outputs": known.output.tolist(), "tolerance": tolerance})


def test_slider_stroke_and_function_are_recovered() -> None:
    result = solve("slider_crank", {"stroke": 0.06, "tolerance": 0.0005}, {})
    assert result.within_tolerance, result.message
    rebuilt_check("slider_crank", result.parameters, {"output_joint": "slide",
                                                      "stroke": 0.06, "tolerance": 0.0005})
    crank = np.radians(np.arange(0.0, 360.0, 30.0))
    known = known_motion("slider_crank", KNOWN_SLIDER_CRANK, crank)
    start = {**KNOWN_SLIDER_CRANK, "crank_length": 0.028, "rod_length": 0.07}
    result = solve("slider_crank", {"crank": crank.tolist(), "output": known.output.tolist(),
                                    "tolerance": 0.0005}, {"initial": start})
    assert result.within_tolerance, result.message
    assert result.parameters["crank_length"] == pytest.approx(0.025, rel=0.05)
    rebuilt_check("slider_crank", result.parameters, {
        "output_joint": "slide", "values": crank.tolist(), "outputs": known.output.tolist(),
        "tolerance": 0.0005})


def test_the_tool_answers_a_sweep_in_degrees_with_a_target_path_to_declare(ctx) -> None:
    out = run_tool(ctx, "synthesize", {"mechanism": "four_bar",
                                       "target": {"sweep": 0.9, "tolerance": 0.01},
                                       "constraints": {"plane": "xz", "height": 0.1}})
    assert not out.is_error, out.text
    data = out.data
    assert data["within_tolerance"] and data["error_deg"]["max"] <= 0.573
    assert data["target_path"] == {"joint": "crank_joint", "output_joint": "rocker_joint",
                                   "sweep": 0.9, "tolerance": 0.01}
    assert data["rest_points"]["crank_joint"][1] == pytest.approx(0.1)  # the plane's y
    assert data["predicted_motion"]["columns"][1] == "rocker_joint (rad)"


def test_the_function_check_passes_fails_and_rejects_the_wrong_kind() -> None:
    from examples.four_bar import build

    model = build()
    known = known_motion("four_bar", {**KNOWN_FOUR_BAR})  # the example's own linkage
    spec = {"joint": "crank_joint", "output_joint": "rocker_joint", "sweep": known.extent,
            "tolerance": 0.01}
    findings, summary = check_target_path(model, spec)
    assert not findings and summary["passed"] and summary["unit"] == "deg"
    findings, summary = check_target_path(model, {**spec, "sweep": known.extent + 0.2})
    assert len(findings) == 1 and findings[0].blocking and not summary["passed"]
    assert "target sweep missed" in findings[0].message and "deg" in findings[0].message
    with pytest.raises(ValidationError, match="use sweep for a revolute"):
        check_target_path(model, {**{k: v for k, v in spec.items() if k != "sweep"},
                                  "stroke": 0.05})
    with pytest.raises(ValidationError, match="not a joint you can drive"):
        check_target_path(model, {**spec, "joint": "rocker_joint"})


@pytest.mark.parametrize("extra,passes", [(0.0, True), (0.2, False)])
def test_compile_blocks_a_missed_sweep(tmp_path, extra, passes) -> None:
    known = known_motion("four_bar", KNOWN_FOUR_BAR)
    spec = {"joint": "crank_joint", "output_joint": "rocker_joint",
            "sweep": known.extent + extra, "tolerance": 0.01}
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "main.py").write_text(
        "from examples.four_bar import build\n\nobject_model = build()\n"
        f"target_path = {json.dumps(spec)}\n", encoding="utf-8")
    result = compile_run(tmp_path, timeout=180)
    assert result.ok == passes, result.signals_text
    if passes:
        assert "rocker_joint sweep" in result.signals_text
        assert result.target_path["kind"] == "sweep" and result.target_path["passed"]
    else:
        assert [s.kind for s in result.signals if s.severity == "failure"] == ["target_path"]


# ---- the skeleton main.py (after synth_v2) ---------------------------------------------------


def compiled_skeleton(tmp_path, mechanism: str, target: dict, constraints: dict):
    from solvers.skeleton import skeleton_source

    result = solve(mechanism, target, constraints)
    assert result.within_tolerance, result.message
    source = skeleton_source(mechanism, result, target)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "main.py").write_text(source, encoding="utf-8")
    compiled = compile_run(tmp_path, timeout=180)
    assert compiled.ok, compiled.signals_text
    assert "warnings=0" in compiled.signals_text, compiled.signals_text
    assert compiled.target_path and compiled.target_path["passed"], compiled.target_path
    return source, compiled


@pytest.mark.parametrize("mechanism,target,constraints", [
    ("four_bar", {"sweep": math.radians(90), "tolerance": 0.01},
     {"plane": "xz", "height": 0.1, "initial": {"ground_length": 0.25}}),
    ("four_bar", {"sweep": 0.9035, "tolerance": 0.01}, {"plane": "xy"}),
    ("slider_crank", {"stroke": 0.06, "tolerance": 0.001}, {"plane": "yz", "height": -0.05}),
], ids=["wiper-90deg-xz", "sweep-xy", "pump-60mm-yz"])
def test_a_function_skeleton_compiles_and_passes_every_check(tmp_path, mechanism, target,
                                                              constraints) -> None:
    source, compiled = compiled_skeleton(tmp_path, mechanism, target, constraints)
    assert "target_path = " in source and "frame_in" in source
    assert compiled.target_path["kind"] in ("sweep", "stroke")


def test_a_function_target_skeleton_checks_the_outputs(tmp_path) -> None:
    crank = np.radians(np.arange(0.0, 360.0, 30.0))
    known = known_motion("slider_crank", KNOWN_SLIDER_CRANK, crank)
    _source, compiled = compiled_skeleton(
        tmp_path, "slider_crank",
        {"crank": crank.tolist(), "output": known.output.tolist(), "tolerance": 0.0005},
        {"initial": {**KNOWN_SLIDER_CRANK, "crank_length": 0.028}})
    assert compiled.target_path["kind"] == "values"


@pytest.mark.parametrize("name,known,free", [
    ("four_bar", KNOWN_FOUR_BAR, ("crank_length", "coupler_length")),
    ("slider_crank", KNOWN_SLIDER_CRANK, ("crank_length", "rod_length")),
])
def test_a_path_skeleton_follows_the_recorded_path(tmp_path, name, known, free) -> None:
    path = recorded_path(name, known)
    target = {"points": np.c_[path[:, 0], np.full(len(path), 0.3), path[:, 1]].tolist(),
              "tolerance": 0.001}
    fixed = {k: v for k, v in known.items() if k not in free}
    initial = {k: known[k] * 1.05 for k in free}
    source, compiled = compiled_skeleton(tmp_path, name, target,
                                         {"plane": "xz", "fixed": fixed, "initial": initial})
    assert '"body": "coupler"' in source.replace("'", '"') or name == "slider_crank"
    assert compiled.target_path["max_error_mm"] <= 1.0


def test_a_linkage_the_layers_cannot_clear_gets_no_skeleton() -> None:
    from solvers.skeleton import SkeletonError, skeleton_source

    short = {**KNOWN_SLIDER_CRANK, "rod_length": 0.045}  # less than twice the crank
    result = solve("slider_crank", {"stroke": 0.05, "tolerance": 0.01},
                   {"fixed": {k: short[k] for k in ("crank_length", "rod_length",
                                                     "crank_phase")}, "time_limit_s": 30})
    with pytest.raises(SkeletonError, match="rod is too short"):
        skeleton_source("slider_crank", result, {"stroke": 0.05})


def test_the_tool_returns_a_skeleton_that_already_compiles(ctx) -> None:
    out = run_tool(ctx, "synthesize", {"mechanism": "slider_crank",
                                       "target": {"stroke": 0.06, "tolerance": 0.001}})
    assert not out.is_error, out.text
    data = out.data
    assert data["skeleton_compile"] == "passed every check, target_path included"
    assert data["skeleton_main_py"].startswith('"""slider-crank skeleton')
    assert "write skeleton_main_py as main.py" in data["next"]
    assert not (ctx.workspace / "main.py").exists()  # the workspace is the model's
    assert list((ctx.run_dir / "synthesis").glob("*-skeleton/workspace/main.py"))
