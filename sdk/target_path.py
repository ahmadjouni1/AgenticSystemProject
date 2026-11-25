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

**Function targets** (step 13): the motion of an output *joint* against the driver, for a
wiper's rocker or a pump's slider:

    target_path = {"joint": "crank_joint", "output_joint": "rocker_joint",
                   "sweep": 1.5708, "tolerance": 0.01}       # radians, peak to peak
    target_path = {"joint": "crank_joint", "output_joint": "piston_slide",
                   "stroke": 0.06, "tolerance": 0.001}       # meters, peak to peak
    target_path = {"joint": "crank_joint", "output_joint": "rocker_joint",
                   "values": [0, 0.5, ...], "outputs": [0, 0.2, ...], "tolerance": 0.01}

``sweep`` is for a revolute output, ``stroke`` for a prismatic one; ``outputs`` are compared
as changes from the output at the first driver value.
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


_FUNCTION_FIELDS = {"joint", "output_joint", "sweep", "stroke", "values", "outputs",
                    "tolerance"}


def _driver(model, joint: Any) -> str:
    drivers, members = drivable_joints(model)
    if joint not in drivers:
        driver = members.get(joint)
        hint = (f"; the loop's driver is {driver!r}" if driver else
                f"; drivable joints: {sorted(drivers)}")
        raise ValidationError(f"target_path: joint {joint!r} is not a joint you can drive{hint}")
    return joint


def read_function_spec(model, spec: Mapping) -> dict[str, Any]:
    unknown = set(spec) - _FUNCTION_FIELDS
    kinds = [k for k in ("sweep", "stroke", "values") if k in spec]
    if unknown or "tolerance" not in spec or len(kinds) != 1:
        raise ValidationError(
            f"target_path with output_joint takes joint, output_joint, tolerance and one of "
            f"sweep (radians), stroke (meters) or values + outputs; unknown fields "
            f"{sorted(unknown)}")
    joint = _driver(model, spec["joint"])
    output = model.get_joint(spec["output_joint"])
    if not output.movable:
        raise ValidationError(f"target_path: output_joint {output.name!r} is fixed")
    kind = kinds[0]
    revolute = output.type is JointType.REVOLUTE
    if (kind == "sweep" and not revolute) or (kind == "stroke" and revolute):
        raise ValidationError(f"target_path: use sweep for a revolute output_joint and stroke "
                              f"for a prismatic one ({output.name!r} is {output.type.value})")
    tolerance = spec["tolerance"]
    if not isinstance(tolerance, (int, float)) or not tolerance > 0:
        raise ValidationError("target_path: tolerance must be a positive number")
    out = {"joint": joint, "output": output.name, "kind": kind, "revolute": revolute,
           "tolerance": float(tolerance)}
    if kind == "values":
        values = np.asarray(spec["values"], dtype=float)
        outputs = np.asarray(spec.get("outputs"), dtype=float)
        if values.ndim != 1 or len(values) < 2 or outputs.shape != values.shape:
            raise ValidationError("target_path: values and outputs must be lists of the same "
                                  "length (at least 2)")
        out.update(values=values, outputs=outputs)
    else:
        extent = spec[kind]
        if not isinstance(extent, (int, float)) or not extent > 0:
            raise ValidationError(f"target_path: {kind} must be a positive number")
        out["extent"] = float(extent)
    return out


def check_function_target(model, spec: Mapping,
                          config: CheckConfig = DEFAULT_CONFIG) -> tuple[list[Finding], dict]:
    """Drive the joint and read the output joint: its sweep or stroke, or its values."""
    s = read_function_spec(model, spec)
    driver = model.get_joint(s["joint"])
    low, high = _joint_range(driver, config, 1.0)
    values = s["values"] if s["kind"] == "values" else np.linspace(low, high, SAMPLES)
    got, stopped = [], ""
    for value in values:
        try:
            got.append(model.solve({s["joint"]: float(value)}).joint_values[s["output"]])
        except (LoopClosureError, PoseError) as exc:
            stopped = str(exc).splitlines()[0]
            break
    output = np.unwrap(np.array(got)) if s["revolute"] else np.array(got)
    scale, unit = (math.degrees, "deg") if s["revolute"] else ((lambda v: v * 1000), "mm")
    if s["kind"] == "values":
        if len(output) < len(values):
            errors = np.full(len(values), math.inf)
        else:
            errors = np.abs((output - output[0]) - (s["outputs"] - s["outputs"][0]))
        max_error = float(np.max(errors))
        rms = float(np.sqrt(np.mean(errors ** 2)))
        what = "output"
        achieved = None
    else:
        achieved = float(np.ptp(output)) if len(output) else 0.0
        max_error = rms = abs(achieved - s["extent"]) if not stopped else math.inf
        what = "sweep" if s["kind"] == "sweep" else "stroke"
    passed = max_error <= s["tolerance"]
    summary = {"kind": s["kind"], "output_joint": s["output"], "unit": unit,
               "max_error": _shown(scale, max_error), "rms_error": _shown(scale, rms),
               "tolerance": scale(s["tolerance"]), "passed": passed,
               "achieved": None if achieved is None else _shown(scale, achieved),
               "target": None if achieved is None else scale(s["extent"]), "joint": s["joint"],
               "samples": len(output)}
    summary.update(max_error_mm=summary["max_error"], rms_error_mm=summary["rms_error"],
                   tolerance_mm=summary["tolerance"])  # the compile summary's keys
    if passed:
        return [], summary
    if achieved is not None and not stopped:
        detail = (f"{s['output']!r} moves {scale(achieved):.4g} {unit} peak to peak, the target "
                  f"is {scale(s['extent']):.4g} {unit} (tolerance {scale(s['tolerance']):.3g} "
                  f"{unit})")
    else:
        detail = (f"{s['output']!r} misses the target by up to "
                  f"{_shown(scale, max_error)} {unit} (tolerance {scale(s['tolerance']):.3g} "
                  f"{unit})")
    hinge = driver.type is JointType.REVOLUTE
    driver_scale = math.degrees if hinge else (lambda v: v * 1000)
    message = (f"target {what} missed: {detail} over {s['joint']} "
               f"{_shown(driver_scale, low)}..{_shown(driver_scale, high)} "
               f"{'deg' if hinge else 'mm'}")
    if stopped:
        message += f"; the motion stopped early: {stopped}"[:200]
    return [Finding("target_path", message, True, (), joint=s["output"],
                    amount=None)], summary


def _shown(scale, value: float) -> float | None:
    return round(scale(value), 3) if math.isfinite(value) else None


def check_target_path(model, spec: Any,
                      config: CheckConfig = DEFAULT_CONFIG) -> tuple[list[Finding], dict]:
    """Findings (one blocking finding if the target is missed) and a summary for the record."""
    if isinstance(spec, Mapping) and "output_joint" in spec:
        return check_function_target(model, spec, config)
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
