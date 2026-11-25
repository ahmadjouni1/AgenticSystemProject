"""Plug-in A: serial arms, through the teammate's URDF adapter (``SerialURDF``).

Constraints (JSON data):

- ``urdf``: the arm's URDF, as text. ``base_link`` and ``tip_link``: the chain's ends.
- ``tip_point``: optional [x, y, z] on the tip link (its frame), the point that should follow
  the target; default its origin.
- ``parameters``: the design, in the adapter's format: ``[{"name", "initial", "lower",
  "upper", "bindings": [{"joint", "xyz": [...], "rpy": [...]}]}]``. The URDF is drawn at
  the ``initial`` values; a binding moves that joint's origin by ``(value - initial) x xyz``
  (meters, in the joint's parent-link frame), and adds ``(value - initial) x rpy`` to its
  roll/pitch/yaw.
- ``q_start`` / ``q_end``: optional joint values at the first and last target points (the
  solver keeps them fixed). By default each is the inverse kinematics of that target point
  at the initial design.
- ``max_step``: optional largest joint change between consecutive points (default: 4 x
  range / points).
- ``joint_bounds``: optional {joint: [lower, upper]}, needed for continuous joints.
- ``max_evaluations`` (default 500): the outer search's geometry evaluations
  (``Settings.outer_maxeval``). ``polish`` (default true). ``time_limit_s`` (default 100).
"""

from __future__ import annotations

import math
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from solvers.base import SolverError, SolverResult, Target
from solvers.nested import SerialURDF, Settings, Task
from solvers.search import Synthesis

DEFAULT_MAX_EVALUATIONS = 500
DEFAULT_TIME_LIMIT_S = 100.0
TIP_LINK = "synthesis_tip"
_FIELDS = {"urdf", "base_link", "tip_link", "tip_point", "parameters", "q_start", "q_end",
           "max_step", "joint_bounds", "max_evaluations", "polish", "time_limit_s"}


def errors(points: np.ndarray, target: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(points - target, axis=1)
    return {"max": float(distance.max()), "rms": float(np.sqrt(np.mean(distance ** 2)))}


class SerialArm:
    name = "serial_arm"

    def solve(self, target: Target, constraints: dict[str, Any]) -> SolverResult:
        started = time.perf_counter()
        unknown = set(constraints) - _FIELDS
        if unknown:
            raise SolverError(f"serial_arm: unknown constraints {sorted(unknown)}; fields: "
                              f"{sorted(_FIELDS)}")
        for name in ("urdf", "base_link", "tip_link", "parameters"):
            if name not in constraints:
                raise SolverError(f"serial_arm needs constraints.{name}")
        parameters = constraints["parameters"]
        if not isinstance(parameters, list) or not parameters:
            raise SolverError("serial_arm: parameters must be a non-empty list")
        time_limit = float(constraints.get("time_limit_s", DEFAULT_TIME_LIMIT_S))
        model = _load(constraints)
        initial = np.array([p["initial"] for p in parameters], float)
        lower = np.array([p["lower"] for p in parameters], float)
        upper = np.array([p["upper"] for p in parameters], float)
        n, dof = len(target.points), len(model.lower)
        q_start = _endpoint(constraints, "q_start", model, initial, target.points[0], dof)
        q_end = _endpoint(constraints, "q_end", model, initial, target.points[-1], dof)
        if "max_step" in constraints:
            max_step = np.asarray(constraints["max_step"], float)
        else:
            needed = np.abs(q_end - q_start) / (n - 1) + 1e-9
            max_step = np.maximum((model.upper - model.lower) * 4 / (n - 1), needed)
        scale = max(float(np.ptp(target.points, axis=0).max()), target.tolerance)
        task = Task(target.points, q_start, q_end, max_step, target.tolerance, scale)
        settings = Settings(outer_maxeval=int(constraints.get("max_evaluations",
                                                              DEFAULT_MAX_EVALUATIONS)))
        try:
            solver = Synthesis(model, task, initial, lower, upper, settings,
                               deadline=started + time_limit, stop_error=target.tolerance / 10)
        except ValueError as exc:
            raise SolverError(f"serial_arm: {exc}") from None
        best, stats = solver.run(polish=bool(constraints.get("polish", True)))
        reached = model.fk(best.design, best.q)[0]
        error = errors(reached, target.points)
        names = [p["name"] for p in parameters]
        delta = best.design - model.reference
        origins = {}
        for i, joint in enumerate(model.joints):
            if np.any(model.xyz_basis[i]) or np.any(model.rpy_basis[i]):
                origins[joint.name] = {"xyz": (joint.xyz + delta @ model.xyz_basis[i]).tolist(),
                                       "rpy": (joint.rpy + delta @ model.rpy_basis[i]).tolist()}
        feasible = bool(best.feasible)
        within = feasible and error["max"] <= target.tolerance
        message = (f"max error {error['max'] * 1000:.3g} mm (tolerance "
                   f"{target.tolerance * 1000:.3g} mm): "
                   + ("the design reaches every target point" if within else
                      "no design within tolerance was found" if feasible else
                      "no design satisfies the joint limits and step bounds"))
        return SolverResult(
            parameters=dict(zip(names, best.design.tolist(), strict=True)),
            trajectory=reached.tolist(), error=error, feasible=feasible, message=message,
            details={"tolerance": target.tolerance, "joint_order": model.joint_names,
                     "joint_path": best.q.tolist(), "joint_origins": origins,
                     "initial_design": dict(zip(names, initial.tolist(), strict=True)),
                     **stats, "seconds": round(time.perf_counter() - started, 3)})


def _load(constraints: dict[str, Any]) -> SerialURDF:
    """Parse the URDF text, add the tip point as a fixed link, and build the adapter."""
    try:
        root = ET.fromstring(constraints["urdf"])
    except (ET.ParseError, TypeError) as exc:
        raise SolverError(f"serial_arm: urdf is not valid XML: {exc}") from None
    tip_link = constraints["tip_link"]
    point = constraints.get("tip_point")
    if point is not None:
        if len(point) != 3:
            raise SolverError("serial_arm: tip_point must be [x, y, z]")
        ET.SubElement(root, "link", name=TIP_LINK)
        joint = ET.SubElement(root, "joint", name=f"{TIP_LINK}_mount", type="fixed")
        ET.SubElement(joint, "parent", link=tip_link)
        ET.SubElement(joint, "child", link=TIP_LINK)
        ET.SubElement(joint, "origin", xyz=" ".join(repr(float(v)) for v in point),
                      rpy="0 0 0")
        tip_link = TIP_LINK
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "arm.urdf"
        path.write_bytes(ET.tostring(root))
        try:
            return SerialURDF(path, constraints["base_link"], tip_link,
                              constraints["parameters"], constraints.get("joint_bounds"))
        except (ValueError, KeyError) as exc:
            raise SolverError(f"serial_arm: {exc}") from None


def _endpoint(constraints: dict[str, Any], key: str, model: SerialURDF, design: np.ndarray,
              point: np.ndarray, dof: int) -> np.ndarray:
    if key in constraints:
        value = np.asarray(constraints[key], float)
        if value.shape != (dof,):
            raise SolverError(f"serial_arm: {key} needs {dof} values ({model.joint_names})")
        return value
    return inverse_kinematics(model, design, point)


def inverse_kinematics(model: SerialURDF, design: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Joint values (within bounds) that bring the tip closest to ``point``; a few seeds."""
    middle = (model.lower + model.upper) / 2
    seeds = [np.clip(np.zeros_like(middle), model.lower, model.upper), middle]
    seeds += [model.lower + (model.upper - model.lower) * f for f in (0.25, 0.75)]
    best, best_cost = seeds[0], math.inf
    for seed in seeds:
        fit = least_squares(lambda q: model.fk(design, q[None])[0][0] - point, seed,
                            bounds=(model.lower, model.upper))
        if fit.cost < best_cost:
            best, best_cost = fit.x, fit.cost
    return best
