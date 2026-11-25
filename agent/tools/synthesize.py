"""synthesize: compute a mechanism's dimensions from a target path, so the model gets numbers
instead of working them out in its reasoning.

    synthesize(mechanism="four_bar", target={"points": [[x, y, z], ...], "tolerance": 0.002},
               constraints={"plane": "xz"})

Every solve runs in its own process with a time limit (``solvers.base.solve_isolated``).
Mechanisms:

- ``four_bar`` / ``slider_crank`` (plug-in B): the result gives every pivot and pin at
  q = 0 (``rest_points``, world coordinates) and the link lengths, so the model can author
  the linkage directly.
- ``serial_arm`` (plug-in A): works on the newest build. The model names the tip body and the
  dimensions to change (``parameters`` with ``bindings`` in parent-body coordinates); the
  tool hands the build's exported URDF to the teammate's URDF adapter, and converts the
  optimized joint origins back to ``parent.at(xyz=..., rpy=...)`` values.

The whole result is saved in the run folder (``synthesis/NN-<mechanism>.json``).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from agent.tools.builds import newest_build, round_list
from agent.tools.core import AgentTool, ToolContext, ToolError
from sdk.frames import Frame
from sdk.transforms import invert
from solvers.base import SolverError, SolverResult, SolverTimeout, solve_isolated

MECHANISMS = ("four_bar", "slider_crank", "serial_arm")
DEFAULT_TIME_LIMIT_S = 100.0
MAX_TIME_LIMIT_S = 240.0
STARTUP_S = 60.0  # the solver process starts, imports, and verifies the final design
MAX_PATH_POINTS = 40
SYSTEM_LINE = ("- `synthesis.md`: the `synthesize` tool, which computes a four-bar's, a "
               "slider-crank's or a serial arm's dimensions from a target path, and "
               "`target_path`, which makes compile check that path.")


def _frames(build) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Link frame of each body (the child frame of the joint placing it), and joints by name."""
    assembly = build.loaded.assembly
    art = assembly.validate()
    link = {art.root: np.eye(4)}
    for joint in art.tree:
        link[joint.child.body.name] = joint.child.frame.matrix
    return link, {j.name: j for j in art.tree}


def _arm_constraints(ctx: ToolContext, constraints: dict[str, Any]) -> tuple[dict, dict]:
    """The model's body-frame request -> the URDF adapter's link-frame constraints."""
    build = newest_build(ctx)
    link, joints = _frames(build)
    body_names = [b.name for b in build.loaded.assembly.bodies]
    request = dict(constraints)
    tip = request.pop("tip_body", None)
    if tip not in body_names:
        raise ToolError("bad_arguments", f"serial_arm needs constraints.tip_body, one of "
                        f"{body_names}", "the body that carries the point to move")
    base = request.pop("base_body", build.loaded.assembly.validate().root)
    if base not in body_names:
        raise ToolError("bad_arguments", f"base_body {base!r} is not a body", f"{body_names}")
    point = request.pop("tip_point", [0.0, 0.0, 0.0])
    request["tip_point"] = (invert(link[tip]) @ np.r_[np.asarray(point, float), 1.0])[:3].tolist()
    parameters = []
    for raw in request.get("parameters") or []:
        item = dict(raw)
        bindings = []
        for binding in item.get("bindings") or []:
            joint = joints.get(binding.get("joint"))
            if joint is None:
                raise ToolError("bad_arguments", f"parameter {item.get('name')!r} binds unknown "
                                f"joint {binding.get('joint')!r}", f"joints: {sorted(joints)}")
            moved = dict(binding)
            if "xyz" in binding:  # parent-body frame -> parent-link frame
                rotation = link[joint.parent.body.name][:3, :3]
                moved["xyz"] = (rotation.T @ np.asarray(binding["xyz"], float)).tolist()
            bindings.append(moved)
        item["bindings"] = bindings
        parameters.append(item)
    request["parameters"] = parameters
    request.update(urdf=(build.loaded.directory / "model.urdf").read_text(encoding="utf-8"),
                   base_link=base, tip_link=tip)
    return request, {"link": link, "joints": joints, "build": build.label}


def _joint_origins(result: SolverResult, frames: dict[str, Any]) -> dict[str, Any]:
    """Optimized URDF origins (parent-link frame) -> ``parent.at(xyz, rpy)`` values."""
    out = {}
    for name, origin in result.details.get("joint_origins", {}).items():
        joint = frames["joints"].get(name)
        if joint is None:  # the tool's own tip mount
            continue
        matrix = frames["link"][joint.parent.body.name] @ Frame(
            tuple(origin["xyz"]), tuple(origin["rpy"])).matrix
        frame = Frame.from_matrix(matrix)
        out[name] = {"parent": joint.parent.body.name, "xyz": round_list(frame.xyz),
                     "rpy": round_list(frame.rpy)}
    return out


def _save(ctx: ToolContext, mechanism: str, request: dict[str, Any],
          result: SolverResult) -> str:
    folder = ctx.run_dir / "synthesis"
    folder.mkdir(exist_ok=True)
    number = len(list(folder.glob("*.json"))) + 1
    path = folder / f"{number:02d}-{mechanism}.json"
    request = {k: v for k, v in request.items() if k != "urdf"}
    path.write_text(json.dumps({"mechanism": mechanism, "request": request,
                                "result": result.to_dict()}, indent=2), encoding="utf-8")
    return path.relative_to(ctx.run_dir).as_posix()


def run_synthesize(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    mechanism, target = args["mechanism"], args["target"]
    constraints = dict(args.get("constraints") or {})
    limit = constraints.get("time_limit_s", DEFAULT_TIME_LIMIT_S)
    if not isinstance(limit, (int, float)) or not 5 <= limit <= MAX_TIME_LIMIT_S:
        raise ToolError("bad_arguments", f"time_limit_s must be 5 to {MAX_TIME_LIMIT_S:.0f} s",
                        "the default is 100 s")
    constraints["time_limit_s"] = float(limit)
    frames = None
    if mechanism == "serial_arm":
        constraints, frames = _arm_constraints(ctx, constraints)
    try:
        result = solve_isolated(mechanism, target, constraints, timeout_s=limit + STARTUP_S)
    except SolverTimeout as exc:
        raise ToolError("solver_timeout", str(exc), "give a closer initial design, fix some "
                        "parameters, or lower max_evaluations") from None
    except SolverError as exc:
        raise ToolError("solver_error", str(exc), "fix the target or constraints as the "
                        "message says; see docs/sdk/synthesis.md") from None
    details = result.details
    path = result.trajectory
    step = max(1, -(-len(path) // MAX_PATH_POINTS))
    out: dict[str, Any] = {
        "mechanism": mechanism,
        "within_tolerance": result.within_tolerance,
        "feasible": result.feasible,
        "message": result.message,
        "error_mm": {"max": round(result.error["max"] * 1000, 4),
                     "rms": round(result.error["rms"] * 1000, 4)},
        "parameters": {k: round(v, 6) for k, v in result.parameters.items()},
        "predicted_path": [round_list(p, 5) for p in path[::step]],
        "seconds": details.get("process_seconds"),
    }
    if mechanism == "serial_arm":
        out["joint_origins"] = _joint_origins(result, frames)
        out["joint_path_at_targets"] = {
            "joints": details["joint_order"],
            "values": [round_list(q, 5) for q in details["joint_path"][::step]]}
        out["next"] = ("Set each joint's parent.at(xyz=..., rpy=...) to joint_origins, adjust "
                       "the links' geometry to the new lengths, then compile.")
    else:
        out["derived"] = {k: round(v, 6) if isinstance(v, float) else v
                          for k, v in details["derived"].items()}
        out["rest_points"] = {k: round_list(v) for k, v in details["rest_points"].items()}
        out["crank_values_at_targets"] = round_list(details["crank_values"][::step], 4)
        out["next"] = ("Build the linkage with these pivots and pins at q = 0 (the crank "
                       "turns from 0 to crank_span), then declare target_path so compile "
                       "checks the path.")
    out["saved"] = _save(ctx, mechanism, constraints, result)
    return out


_POINTS = {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}

SYNTHESIZE = AgentTool(
    "synthesize",
    "Compute a four_bar's, slider_crank's or serial_arm's dimensions so a point follows a "
    "target path (meters); returns the dimensions, the predicted path and its error.",
    {"type": "object", "properties": {
        "mechanism": {"type": "string", "enum": list(MECHANISMS)},
        "target": {"type": "object", "properties": {
            "points": {**_POINTS, "description": "[x, y, z] in meters, in order of motion"},
            "tolerance": {"type": "number", "description": "meters (default 0.002)"},
        }, "required": ["points"], "additionalProperties": False},
        "constraints": {"type": "object",
                        "description": "per mechanism; see docs/sdk/synthesis.md"},
    }, "required": ["mechanism", "target"], "additionalProperties": False},
    run_synthesize)
