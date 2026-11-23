"""probe: pose the newest build (no compile) and answer geometric questions about it.

Loops are solved with the walking warm start, as in the checks. Queries:
- ``{"type": "position", "body": B}``: B's frame in the world (origin in m, rpy in deg);
- ``{"type": "point", "body": B, "point": [x, y, z]}``: a point given in B's frame, in world;
- ``{"type": "gap", "a": A, "b": B}``: the clearance between two bodies, or how deep they
  overlap;
- ``{"type": "collisions"}``: every pair of bodies that interpenetrates at this pose.
The solved value of every moving joint is always returned.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.tools.builds import (POSE_SCHEMA, Build, body_mesh, joint_report, newest_build,
                                round_list, solve_pose)
from agent.tools.core import AgentTool, ToolContext, ToolError
from sdk.testing import DEFAULT_CONFIG
from sdk.transforms import rpy_from_matrix

MAX_QUERIES = 20
MIN_DEPTH = 1e-6  # m: shallower intersections are touching faces, not penetration


def _rpy(matrix: np.ndarray) -> list[float]:
    """Roll, pitch, yaw in degrees, in the SDK's own convention (sdk.transforms)."""
    return round_list(np.degrees(rpy_from_matrix(matrix[:3, :3])), 3)


def _size(build: Build) -> float:
    """The rest-pose bounding-box diagonal: what the checks scale their tolerances by."""
    if build.size is None:
        rest = build.loaded.assembly.solve({})
        corners = []
        for body in build.loaded.assembly.bodies:
            mesh = body_mesh(body)
            corners.append(mesh.bounds @ rest.transforms[body.name][:3, :3].T
                           + rest.transforms[body.name][:3, 3])
        every = np.vstack(corners)
        build.size = float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))
    return build.size


def _collisions(scene) -> dict[frozenset, tuple[float, float]]:
    """Body pair -> (deepest penetration m, total overlap volume m^3)."""
    out: dict[frozenset, tuple[float, float]] = {}
    for a, b, depth, volume in scene.collisions(scene.candidate_pairs(0.0)):
        if depth < MIN_DEPTH:
            continue
        key = frozenset((a[0], b[0]))
        old = out.get(key, (0.0, 0.0))
        out[key] = (max(old[0], depth), old[1] + volume)
    return out


def run_probe(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    queries = args["queries"]
    if not queries or len(queries) > MAX_QUERIES:
        raise ToolError("bad_arguments", f"give 1 to {MAX_QUERIES} queries, got {len(queries)}",
                        'for example [{"type": "collisions"}]')
    build = newest_build(ctx)
    bodies = [b.name for b in build.loaded.assembly.bodies]
    for i, query in enumerate(queries):
        needed = {"position": ("body",), "point": ("body", "point"), "gap": ("a", "b"),
                  "collisions": ()}[query["type"]]
        for field_name in needed:
            if field_name not in query:
                raise ToolError("bad_arguments", f"queries[{i}] ({query['type']}) needs "
                                f"{field_name!r}", f"fields for {query['type']}: {list(needed)}")
        for field_name in ("body", "a", "b"):
            if field_name in query and query[field_name] not in bodies:
                raise ToolError("bad_arguments", f"queries[{i}]: no body named "
                                f"{query[field_name]!r}", f"bodies: {bodies}")
        if query["type"] == "point" and len(query["point"]) != 3:
            raise ToolError("bad_arguments", f"queries[{i}]: point must be [x, y, z] in m",
                            "a point in the body's own frame")
    pose = solve_pose(build, args.get("pose") or {})
    transforms = pose.transforms
    scene = None
    hits: dict[frozenset, tuple[float, float]] | None = None
    overlap_tol = DEFAULT_CONFIG.tolerance("overlap", _size(build))
    contact_tol = DEFAULT_CONFIG.tolerance("contact", _size(build))

    def collisions():
        nonlocal scene, hits
        if hits is None:
            scene = build.collision_scene()
            scene.place(transforms)
            hits = _collisions(scene)
        return hits

    results = []
    for query in queries:
        kind = query["type"]
        if kind == "position":
            matrix = transforms[query["body"]]
            results.append({"type": kind, "body": query["body"],
                            "origin_m": round_list(matrix[:3, 3]), "rpy_deg": _rpy(matrix)})
        elif kind == "point":
            matrix = transforms[query["body"]]
            world = matrix[:3, :3] @ np.asarray(query["point"], float) + matrix[:3, 3]
            results.append({"type": kind, "body": query["body"],
                            "local_m": round_list(query["point"]), "world_m": round_list(world)})
        elif kind == "gap":
            a, b = query["a"], query["b"]
            if a == b:
                raise ToolError("bad_arguments", "gap needs two different bodies", f"bodies: {bodies}")
            found = collisions()
            gap = scene.distance(a, b)
            depth, volume = found.get(frozenset((a, b)), (0.0, 0.0))
            state = ("penetrating" if depth > 0 else "touching" if gap <= contact_tol
                     else "apart")
            entry = {"type": kind, "a": a, "b": b, "state": state,
                     "gap_mm": round(gap * 1000, 3)}
            if depth > 0:
                entry.update(depth_mm=round(depth * 1000, 3),
                             over_tolerance=depth > overlap_tol)
            results.append(entry)
        else:
            found = collisions()
            results.append({"type": kind, "pairs": [
                {"a": sorted(pair)[0], "b": sorted(pair)[1], "depth_mm": round(d * 1000, 3),
                 "volume_mm3": round(v * 1e9, 3), "over_tolerance": d > overlap_tol}
                for pair, (d, v) in sorted(found.items(), key=lambda item: -item[1][0])],
                "tolerance_mm": round(overlap_tol * 1000, 3)})
    out = {"build": build.label, "joints": joint_report(build, pose.joint_values),
           "results": results}
    if build.note:
        out["note"] = build.note
    return out


_QUERY = {"type": "object", "properties": {
    "type": {"type": "string", "enum": ["position", "point", "gap", "collisions"]},
    "body": {"type": "string"}, "point": {"type": "array", "items": {"type": "number"}},
    "a": {"type": "string"}, "b": {"type": "string"},
}, "required": ["type"], "additionalProperties": False}

PROBE = AgentTool(
    "probe",
    "Pose the newest build without compiling and query body positions, points, gaps or "
    "collisions; also returns every joint value. Use to debug motion.",
    {"type": "object", "properties": {
        "pose": POSE_SCHEMA,
        "queries": {"type": "array", "items": _QUERY,
                    "description": "position{body}, point{body, point}, gap{a, b}, collisions{}"},
    }, "required": ["queries"], "additionalProperties": False},
    run_probe)  # not parallel: it poses one shared collision scene
