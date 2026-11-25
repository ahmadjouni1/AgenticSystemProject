"""The trajectory check: does the built mechanism's output point follow the target path?

``main.py`` may declare, next to ``object_model``, plain data:

    target_path = {
        "body": "coupler",               # the body that carries the output point
        "point": [0.05, 0.03, 0.0],      # the output point, in that body's frame (m)
        "joint": "crank_joint",          # the driver: a joint you can drive
        "points": [[x, y, z], ...],      # the target path in the world (m), in order
        "tolerance": 0.002,              # m: the largest allowed miss
    }

Optional: ``"values"`` (one driver value per target point: then each point is compared with
the output at its own driver value) and ``"rms_tolerance"`` (default ``tolerance``).

The compiler drives the joint over its whole range (its limits; a full turn for an unlimited
hinge) in small steps, with the loop solver's walking warm start, and traces the output
point. Without ``values``, each target point's error is its distance from the traced path.
The check blocks the compile when the largest error is over ``tolerance`` or the RMS error is
over ``rms_tolerance``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from sdk.errors import LoopClosureError, PoseError, ValidationError
from sdk.joint import JointType
from sdk.testing import DEFAULT_CONFIG, CheckConfig, Finding, _joint_range, drivable_joints

SAMPLES = 361
_FIELDS = {"body", "point", "joint", "points", "tolerance", "values", "rms_tolerance"}


def _vector(value: Any, what: str) -> np.ndarray:
    array = np.asarray(value, dtype=float) if isinstance(value, (list, tuple)) else None
    if array is None or array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValidationError(f"target_path: {what} must be [x, y, z] in meters, got {value!r}"
                              [:200])
    return array


def read_spec(model, spec: Any) -> dict[str, Any]:
    """Check the declaration; raises ValidationError with a fix-it message."""
    if not isinstance(spec, Mapping):
        raise ValidationError("target_path must be a dict: body, point, joint, points, "
                              "tolerance")
    unknown = set(spec) - _FIELDS
    missing = {"body", "point", "joint", "points", "tolerance"} - set(spec)
    if unknown or missing:
        raise ValidationError(f"target_path: unknown fields {sorted(unknown)}, missing "
                              f"{sorted(missing)}; fields: {sorted(_FIELDS)}")
    model.get_body(spec["body"])
    drivers, members = drivable_joints(model)
    joint = spec["joint"]
    if joint not in drivers:
        driver = members.get(joint)
        hint = (f"; the loop's driver is {driver!r}" if driver else
                f"; drivable joints: {sorted(drivers)}")
        raise ValidationError(f"target_path: joint {joint!r} is not a joint you can drive{hint}")
    points = spec["points"]
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        raise ValidationError("target_path: points must be a list of at least 2 [x, y, z]")
    array = np.array([_vector(p, f"points[{i}]") for i, p in enumerate(points)])
    tolerance = spec["tolerance"]
    if not isinstance(tolerance, (int, float)) or not tolerance > 0:
        raise ValidationError(f"target_path: tolerance must be a positive number of meters, "
                              f"got {tolerance!r}")
    rms = spec.get("rms_tolerance", tolerance)
    if not isinstance(rms, (int, float)) or not rms > 0:
        raise ValidationError("target_path: rms_tolerance must be a positive number of meters")
    values = spec.get("values")
    if values is not None:
        values = np.asarray(values, dtype=float)
        if values.shape != (len(array),) or not np.all(np.isfinite(values)):
            raise ValidationError(f"target_path: values needs one driver value per point "
                                  f"({len(array)})")
    return {"body": spec["body"], "point": _vector(spec["point"], "point"), "joint": joint,
            "points": array, "tolerance": float(tolerance), "rms_tolerance": float(rms),
            "values": values}


def trace(model, body: str, point: np.ndarray, joint: str,
          values: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """The output point at each driver value, walking in order. Stops where the loop cannot
    close; returns (values reached, points, why it stopped or "")."""
    out, reached = [], []
    for value in values:
        try:
            pose = model.solve({joint: float(value)})
        except (LoopClosureError, PoseError) as exc:
            return np.array(reached), np.array(out).reshape(-1, 3), str(exc).splitlines()[0]
        matrix = pose.transforms[body]
        out.append(matrix[:3, :3] @ point + matrix[:3, 3])
        reached.append(float(value))
    return np.array(reached), np.array(out).reshape(-1, 3), ""


def distance_to_path(points: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point: the distance to the polyline ``path``, and the nearest segment index."""
    if len(path) == 1:
        return np.linalg.norm(points - path[0], axis=1), np.zeros(len(points), int)
    a, b = path[:-1], path[1:]
    ab = b - a
    length2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30)
    t = np.clip(np.einsum("kij,ij->ki", points[:, None, :] - a[None], ab) / length2, 0, 1)
    nearest = a[None] + t[..., None] * ab[None]
    distance = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    return distance.min(axis=1), distance.argmin(axis=1)


def check_target_path(model, spec: Any,
                      config: CheckConfig = DEFAULT_CONFIG) -> tuple[list[Finding], dict]:
    """Findings (one blocking finding if the path is missed) and a summary for the record."""
    s = read_spec(model, spec)
    joint = model.get_joint(s["joint"])
    low, high = _joint_range(joint, config, 1.0)
    if s["values"] is not None:
        values = s["values"]
        reached, traced, stopped = trace(model, s["body"], s["point"], s["joint"], values)
        if len(reached) < len(values):
            errors = np.full(len(values), math.inf)
        else:
            errors = np.linalg.norm(traced - s["points"], axis=1)
        nearest_value = values
    else:
        values = np.linspace(low, high, SAMPLES)
        reached, traced, stopped = trace(model, s["body"], s["point"], s["joint"], values)
        if not len(reached):
            errors, nearest_value = np.full(len(s["points"]), math.inf), np.full(1, low)
        else:
            errors, segment = distance_to_path(s["points"], traced)
            nearest_value = reached[np.minimum(segment, len(reached) - 1)]
    worst = int(np.argmax(errors))
    max_error = float(errors[worst])
    rms = float(np.sqrt(np.mean(errors ** 2)))
    passed = max_error <= s["tolerance"] and rms <= s["rms_tolerance"]
    unit = (math.degrees, "deg") if joint.type is JointType.REVOLUTE else (lambda v: v * 1000,
                                                                            "mm")
    summary = {"max_error_mm": _mm(max_error), "rms_error_mm": _mm(rms),
               "tolerance_mm": s["tolerance"] * 1000, "passed": passed,
               "points": len(s["points"]), "samples": len(reached),
               "joint": s["joint"], "range": [unit[0](low), unit[0](high)], "unit": unit[1]}
    if passed:
        return [], summary
    pose_value = float(nearest_value[min(worst, len(nearest_value) - 1)])
    if math.isfinite(max_error):
        where = (f"worst at target point {worst} {np.round(s['points'][worst], 4).tolist()}, "
                 f"nearest at {s['joint']}={unit[0](pose_value):.1f} {unit[1]}")
    else:
        where = "the output point could not be traced at every target"
    message = (f"target path missed: the output point of {s['body']!r} is up to "
               f"{_mm(max_error)} mm from the target (RMS {_mm(rms)} mm; tolerance "
               f"{s['tolerance'] * 1000:.3g} mm) over {s['joint']} "
               f"{unit[0](low):.4g}..{unit[0](high):.4g} {unit[1]}; {where}")
    if stopped:
        message += f"; the motion stopped early: {stopped}"[:200]
    finding = Finding("target_path", message, True, (s["body"],), joint=s["joint"],
                      pose={s["joint"]: pose_value} if len(reached) else None,
                      amount=max_error if math.isfinite(max_error) else None)
    return [finding], summary


def _mm(meters: float) -> float | None:
    return round(meters * 1000, 3) if math.isfinite(meters) else None
