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
import time
from typing import Any

import numpy as np

from agent.tools.builds import newest_build, round_list
from agent.tools.core import AgentTool, ToolContext, ToolError
from sdk.frames import Frame
from sdk.transforms import invert
from solvers.base import (SolverError, SolverResult, SolverTimeout, Unsupported,
                          check_supported, solve_isolated)
from solvers.skeleton import SkeletonError, skeleton_source

MECHANISMS = ("four_bar", "slider_crank", "serial_arm")
DEFAULT_TIME_LIMIT_S = 180.0  # the solve's own limit; it then returns its best design
MAX_TIME_LIMIT_S = 240.0
STARTUP_S = 60.0  # the solver process starts, imports, and verifies the final design
MAX_PATH_POINTS = 40
SYSTEM_LINE = ("- `synthesis.md`: the `synthesize` tool (four-bar and slider-crank dimensions "
               "from a sweep, stroke or path) and `target_path`, which makes compile check it.")
LINKAGE_RULE = ("**Linkages:** for four-bars and slider-cranks only (wipers, pumps): pick the "
                "sweep, stroke or path, call\n`synthesize`, write its skeleton as main.py, "
                "compile, then add detail in small edits. Others: as usual.")


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


CALL_LOG = "synthesis_calls"  # ctx.cache key: one entry per call, for the run record


def run_synthesize(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    entry: dict[str, Any] = {"mechanism": args.get("mechanism")}
    try:
        out = _synthesize(ctx, args)
        key = "error_deg" if "error_deg" in out else "error_mm"
        entry.update(outcome="within_tolerance" if out["within_tolerance"] else "outside",
                     target=next(k for k in ("points", "sweep", "stroke", "crank")
                                 if k in args["target"]),
                     **{f"{key}_max": out[key]["max"]}, solve_s=out["seconds"])
        return out
    except ToolError as exc:
        entry["outcome"] = exc.kind
        raise
    finally:
        entry["wall_s"] = round(time.perf_counter() - started, 2)
        ctx.cache.setdefault(CALL_LOG, []).append(entry)


def _synthesize(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    mechanism, target = args["mechanism"], args["target"]
    try:
        check_supported(mechanism)
    except Unsupported as exc:
        raise ToolError("unsupported", str(exc), "design this mechanism normally, without "
                        "synthesize") from None
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
    function = details.get("kind") in ("sweep", "stroke", "function")
    degrees = details.get("unit") == "rad"
    scale, unit = (57.29577951308232, "deg") if degrees else (1000.0, "mm")
    out: dict[str, Any] = {
        "mechanism": mechanism,
        "within_tolerance": result.within_tolerance,
        "feasible": result.feasible,
        "message": result.message,
        f"error_{unit}": {"max": round(result.error["max"] * scale, 4),
                          "rms": round(result.error["rms"] * scale, 4)},
        "parameters": {k: round(v, 6) for k, v in result.parameters.items()},
        "seconds": details.get("process_seconds"),
    }
    if function:
        out["predicted_motion"] = {
            "columns": ["crank (rad)", f"{details['output_joint']} ({details['unit']})"],
            "rows": [round_list(p, 5) for p in path[::step]]}
    else:
        out["predicted_path"] = [round_list(p, 5) for p in path[::step]]
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
        if function:
            out["target_path"] = _function_target_path(target, details)
            out["next"] = ("Build the linkage with these pivots and pins at q = 0, then "
                           "declare target_path (output_joint is your own joint's name) so "
                           "compile checks the motion.")
        else:
            out["crank_values_at_targets"] = round_list(details["crank_values"][::step], 4)
            out["next"] = ("Build the linkage with these pivots and pins at q = 0 (the crank "
                           "turns from 0 to crank_span), then declare target_path so compile "
                           "checks the path.")
    out["saved"] = _save(ctx, mechanism, constraints, result)
    if mechanism != "serial_arm":
        out.update(_skeleton(ctx, mechanism, result, target, out["saved"]))
    return out


def _skeleton(ctx: ToolContext, mechanism: str, result: SolverResult, target: dict[str, Any],
              saved: str) -> dict[str, Any]:
    """The skeleton main.py for this design, compiled once in its own folder (never the
    workspace) so the answer says whether it passes every check as it is."""
    try:
        source = skeleton_source(mechanism, result, target)
    except SkeletonError as exc:
        return {"skeleton_main_py": None, "skeleton_compile": f"no skeleton: {exc}"}
    folder = ctx.run_dir / saved.replace(".json", "-skeleton")
    (folder / "workspace").mkdir(parents=True, exist_ok=True)
    (folder / "workspace" / "main.py").write_text(source, encoding="utf-8")
    compiled = ctx.compiler(folder)
    if compiled.ok:
        verdict = "passed every check, target_path included"
    else:
        first = next((s for s in compiled.signals if s.severity == "failure"), None)
        verdict = f"failed: {first.kind}: {first.message}" if first else "failed"
    return {"skeleton_main_py": source, "skeleton_compile": verdict,
            "next": ("write skeleton_main_py as main.py and compile it; then add detail in "
                     "small edits (keep the pivots, lengths, joint names and target_path), "
                     "compiling after each")}


def _function_target_path(target: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    """The target_path to declare for this function target (the joint names are ours)."""
    spec: dict[str, Any] = {"joint": "crank_joint", "output_joint": details["output_joint"]}
    if details["kind"] == "function":
        spec.update(values=target["crank"], outputs=target["output"])
    else:
        spec[details["kind"]] = target[details["kind"]]
    spec["tolerance"] = details["tolerance"]
    return spec


_POINTS = {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
_NUMBERS = {"type": "array", "items": {"type": "number"}}

SYNTHESIZE = AgentTool(
    "synthesize",
    "Dimensions of a four_bar or slider_crank (a point path, a rocker sweep or a slider "
    "stroke) or a serial_arm (a path). Any other mechanism is unsupported: design it normally.",
    {"type": "object", "properties": {
        "mechanism": {"type": "string",
                      "description": "four_bar | slider_crank | serial_arm; nothing else"},
        "target": {"type": "object", "properties": {
            "points": {**_POINTS, "description": "a path: [x, y, z] in meters, in order"},
            "sweep": {"type": "number", "description": "four_bar: rocker swing, radians"},
            "stroke": {"type": "number", "description": "slider_crank: slider travel, meters"},
            "crank": {**_NUMBERS, "description": "with output: crank angles, radians"},
            "output": {**_NUMBERS, "description": "rocker (rad) or slider (m) at each angle"},
            "tolerance": {"type": "number", "description": "in the target's unit"},
        }, "additionalProperties": False},
        "constraints": {"type": "object",
                        "description": "per mechanism; see docs/sdk/synthesis.md"},
    }, "required": ["mechanism", "target"], "additionalProperties": False},
    run_synthesize)
