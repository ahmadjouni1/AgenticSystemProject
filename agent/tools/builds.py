"""The newest build of a run, posed: shared by render and probe.

The newest build is ``result/last_failed`` when it exists (it exists only when the latest
compile failed its checks; every compile removes it first), otherwise the latest
``result/vNNNN``. Builds are loaded once per folder and cached in the ToolContext.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from agent.tools.core import ToolContext, ToolError
from sdk.errors import LoopClosureError, PoseError
from viewer.loader import MANIFEST_FILE, LoadedVersion, load_version

LAST_FAILED = "last_failed"
_VERSION = re.compile(r"v\d{4,}")

POSE_SCHEMA = {"type": "object", "additionalProperties": {"type": "number"},
               "description": "joint -> value (rad or m); unset joints stay at 0"}


@dataclass
class Build:
    label: str  # "v0003" or "last_failed"
    loaded: LoadedVersion
    note: str | None = None
    scene: Any = None  # sdk.testing.CollisionScene, made on first use
    size: float | None = None  # rest-pose bounding-box diagonal, m
    front: Any = None  # the side the views face (agent.tools.render.build_front), made on first use

    def collision_scene(self):
        if self.scene is None:
            from sdk.testing import CollisionScene

            self.scene = CollisionScene(self.loaded.assembly)
        return self.scene


def newest_build(ctx: ToolContext) -> Build:
    result_dir = ctx.run_dir / "result"
    folder: Path | None = None
    if (result_dir / LAST_FAILED / MANIFEST_FILE).is_file():
        folder = result_dir / LAST_FAILED
    elif result_dir.is_dir():
        versions = sorted(p for p in result_dir.iterdir()
                          if p.is_dir() and _VERSION.fullmatch(p.name)
                          and (p / MANIFEST_FILE).is_file())
        folder = versions[-1] if versions else None
    if folder is None:
        raise ToolError("no_build", "there is no build yet",
                        "call compile first; a build exists once main.py builds a model")
    stamp = (folder / MANIFEST_FILE).stat().st_mtime_ns
    key = f"build:{folder}"
    cached = ctx.cache.get(key)
    if cached is None or cached[0] != stamp:
        cached = (stamp, Build(folder.name, load_version(folder)))
        ctx.cache[key] = cached
    build = cached[1]
    build.note = None
    if folder.name == LAST_FAILED:
        build.note = "the latest compile failed its checks; this is the model it built"
    elif ctx.last_compile is not None and not ctx.last_compile.ok:
        build.note = (f"the latest compile failed before a model was built; this is the last "
                      f"good build, {folder.name}, which is older than main.py")
    return build


def _describe(driver: dict[str, Any]) -> str:
    if driver["type"] == "revolute":
        return (f"{driver['name']} (revolute, {math.degrees(driver['lower']):.0f} to "
                f"{math.degrees(driver['upper']):.0f} deg)")
    return f"{driver['name']} (prismatic, {driver['lower']:.3f} to {driver['upper']:.3f} m)"


def solve_pose(build: Build, pose: dict[str, float]) -> Any:
    """``assembly.solve`` with structured errors. Only driven joints may be set."""
    drivers = {d["name"]: d for d in build.loaded.drivers}
    listing = "; ".join(_describe(d) for d in build.loaded.drivers) or "none (no moving joint)"
    for name in pose:
        if name not in drivers:
            joints = {j["name"] for j in build.loaded.manifest["joints"]}
            why = ("is moved by a mimic or loop and cannot be set" if name in joints
                   else "is not a joint of this model")
            raise ToolError("bad_arguments", f"joint {name!r} {why}",
                            f"set only the driven joints: {listing}")
    try:
        return build.loaded.assembly.solve(dict(pose))
    except LoopClosureError as exc:
        raise ToolError("pose_unreachable", f"the loop cannot close at this pose: {exc}",
                        f"try values closer to the rest pose; driven joints: {listing}") from None
    except PoseError as exc:
        raise ToolError("bad_pose", str(exc), f"driven joints: {listing}") from None


def joint_report(build: Build, joint_values: dict[str, float]) -> dict[str, Any]:
    """Every moving joint's value, in rad and deg (revolute) or m (prismatic)."""
    types = {j["name"]: j["type"] for j in build.loaded.manifest["joints"]}
    out: dict[str, Any] = {}
    for name, value in joint_values.items():
        if types.get(name) == "revolute":
            out[name] = {"rad": round(float(value), 6), "deg": round(math.degrees(value), 3)}
        elif types.get(name) == "prismatic":
            out[name] = {"m": round(float(value), 6)}
    return out


def body_mesh(body) -> trimesh.Trimesh:
    """All of a body's shapes as one mesh, in body coordinates."""
    return trimesh.util.concatenate([shape.to_trimesh() for shape in body.shapes])


def world_meshes(build: Build, transforms: dict[str, np.ndarray]):
    """(body name, mesh in world coordinates) for every body, in manifest order."""
    out = []
    for body in build.loaded.assembly.bodies:
        mesh = body_mesh(body)
        mesh.apply_transform(transforms[body.name])
        out.append((body.name, mesh))
    return out


def round_list(values, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]
