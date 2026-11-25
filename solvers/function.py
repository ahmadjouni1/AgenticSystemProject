"""Plug-in B, function generation: the output joint's motion against the crank angle.

A wiper's rocker must swing 90 degrees; a pump's piston must travel 60 mm. That is a target
for the **output joint** (the rocker's hinge, the slider's rail) as the crank turns, not for a
point, and synth_v1 showed that path-only synthesis does not fit it (BUILD_LOG, step 13).

The crank angles are given, so there is no joint path to find and the teammate's inner loop
has nothing to do. The design is fitted directly with SciPy ``least_squares``. Every
evaluation builds the replica ``Assembly`` (``solvers.planar``) and poses it with our loop
solver (walking warm start) every 5 degrees of crank; the final design is checked again every
degree, and at each given crank angle directly.

Residuals, all dimensionless:
- the target: (output extent - target) / tolerance, or each output change / tolerance;
- reach: the crank must turn through the whole span with the loop closed, on one branch;
- transmission (four-bar): the coupler-rocker angle stays at least ``min_transmission_deg``
  (default 35) from a straight line, so the linkage does not jam;
- a weak pull toward the starting design, which picks one answer when several fit.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from sdk.errors import LoopClosureError, PoseError
from solvers.base import FunctionTarget, SolverError, SolverResult
from solvers.planar import FLIP_RADIANS, GRID_STEP, PLANES, LoopModel, rest_points

OUTPUT_JOINT = {"four_bar": "rocker_joint", "slider_crank": "slide"}
KINDS = {"four_bar": ("sweep", "function"), "slider_crank": ("stroke", "function")}
FREE = {"four_bar": ("crank_length", "coupler_length", "ground_length", "coupler_angle",
                     "crank_phase"),
        "slider_crank": ("crank_length", "rod_length", "rod_angle", "crank_phase")}
FIELDS = {"plane", "height", "crank_span", "initial", "bounds", "fixed", "min_transmission_deg",
          "time_limit_s", "max_evaluations"}
DEFAULT_MIN_TRANSMISSION_DEG = 35.0
DEFAULT_TIME_LIMIT_S = 100.0
DEFAULT_MAX_EVALUATIONS = 600
CHECK_STEP = math.radians(1.0)
REGULARIZATION = 0.02


class _OutOfTime(Exception):
    pass


class _GoodEnough(Exception):
    pass


MARGIN = math.radians(1.0)  # the fit aims this far above the transmission limit
ALLOWANCE = math.radians(0.5)  # and the final check allows this far below it
STOP = 0.2  # a fit within this fraction of the tolerance, with nothing violated, is done


@dataclass
class Motion:
    crank: np.ndarray  # crank values reached, from 0 toward the span
    output: np.ndarray  # output joint value from its value at q = 0 (unwrapped for a rocker)
    reach: float  # fraction of the span the crank turned through
    min_transmission: float  # radians; pi/2 for a slider-crank (not measured)
    poses: int

    def at(self, crank: np.ndarray) -> np.ndarray:
        if len(self.crank) < 2:
            return np.zeros(len(crank))
        clipped = np.clip(crank, min(self.crank[0], self.crank[-1]),
                          max(self.crank[0], self.crank[-1]))
        order = np.argsort(self.crank)
        return CubicSpline(self.crank[order], self.output[order])(clipped)

    @property
    def extent(self) -> float:
        return float(np.ptp(self.output)) if len(self.output) else 0.0


def transmission(pose, params: dict[str, float]) -> float:
    """The acute angle between the coupler and the rocker (pi/2 is ideal, 0 jams)."""
    coupler = pose.transforms["coupler"]
    pin = coupler[:3, :3] @ np.array([params["coupler_length"], 0.0, 0.0]) + coupler[:3, 3]
    rocker = pin - pose.transforms["rocker"][:3, 3]
    along = coupler[:3, 0]
    cosine = abs(float(along @ rocker)) / max(float(np.linalg.norm(rocker)), 1e-12)
    return math.acos(min(1.0, cosine))


def motion(linkage, params: dict[str, float], span: float, step: float = GRID_STEP,
           at: np.ndarray | None = None) -> Motion:
    """Pose the linkage from crank 0 to ``span`` (or at the crank values ``at``, in order)
    and read the output joint."""
    output_joint = OUTPUT_JOINT[linkage.name]
    if min(params[name] for name in linkage.lengths) <= 1e-6:
        return Motion(np.zeros(1), np.zeros(1), 0.0, 0.0, 0)
    assembly, _ = linkage.build(params, span)
    values = (np.linspace(0.0, span, max(2, math.ceil(abs(span) / step) + 1)) if at is None
              else np.asarray(at, float))
    reached, out, previous = [], [], None
    worst = math.pi / 2
    for value in values:
        try:
            pose = assembly.solve({"crank_joint": float(value)})
        except (LoopClosureError, PoseError):
            break
        loop = {k: v for k, v in pose.joint_values.items()
                if k != "crank_joint" and not k.startswith("slide")}
        if previous is not None and any(
                abs((loop[k] - previous[k] + math.pi) % (2 * math.pi) - math.pi) > FLIP_RADIANS
                for k in loop):
            break  # the loop flipped to its other branch
        previous = loop
        reached.append(float(value))
        out.append(pose.joint_values[output_joint])
        if linkage.name == "four_bar":
            worst = min(worst, transmission(pose, params))
    output = np.array(out)
    if linkage.name == "four_bar" and len(output):
        output = np.unwrap(output)
    if len(output):
        output = output - output[0] if at is None else output - _rest_output(assembly,
                                                                              output_joint)
    total = len(values) - 1
    reach = (len(reached) - 1) / total if total and reached else 0.0
    return Motion(np.array(reached), output, reach, worst, len(reached))


def _rest_output(assembly, joint: str) -> float:
    return float(assembly.solve({"crank_joint": 0.0}, warm_start={}).joint_values[joint])


def initial_design(linkage, target: FunctionTarget, given: dict[str, float]) -> dict[str, float]:
    """A plausible start from the target's size: a Grashof crank-rocker whose rocker swings
    about the target sweep, or a zero-offset slider-crank with half the stroke as its crank."""
    extent = target.extent if target.kind != "function" else float(np.ptp(target.output))
    values = {k: 0.0 for k in linkage.parameters}
    if linkage.name == "four_bar":
        ground = float(given.get("ground_length", 0.1))
        rocker = 0.6 * ground
        crank = min(rocker * math.sin(min(extent, math.radians(150)) / 2), 0.4 * ground)
        values.update(ground_length=ground, crank_length=max(crank, 0.05 * ground),
                      coupler_length=ground, crank_phase=0.0, point_x=ground / 2)
        values.update({k: float(v) for k, v in given.items()})
        values["coupler_angle"] = given.get("coupler_angle",
                                            _coupler_angle_for(values, rocker))
    else:
        crank = float(given.get("crank_length", max(extent / 2, 1e-3)))
        rod = float(given.get("rod_length", 3 * crank))
        phase = float(given.get("crank_phase", 0.4))
        values.update(crank_length=crank, rod_length=rod, crank_phase=phase,
                      rod_angle=-phase - math.asin(max(-1.0, min(1.0, crank * math.sin(phase)
                                                                  / rod))),
                      point_x=rod)
        values.update({k: float(v) for k, v in given.items()})
    return values


def _coupler_angle_for(p: dict[str, float], rocker: float) -> float:
    """The coupler's rest angle that makes the rocker about ``rocker`` long (above ground)."""
    phase, ground_angle = p["crank_phase"], p["ground_angle"]
    pin = np.array([p["crank_pivot_x"] + p["crank_length"] * math.cos(phase),
                    p["crank_pivot_y"] + p["crank_length"] * math.sin(phase)])
    pivot = np.array([p["crank_pivot_x"] + p["ground_length"] * math.cos(ground_angle),
                      p["crank_pivot_y"] + p["ground_length"] * math.sin(ground_angle)])
    best, best_score = 1.0, math.inf
    for beta in np.linspace(-math.pi, math.pi, 721):
        tip = pin + p["coupler_length"] * np.array([math.cos(phase + beta),
                                                    math.sin(phase + beta)])
        score = abs(float(np.linalg.norm(tip - pivot)) - rocker) + (0 if tip[1] >= pin[1]
                                                                    else 1.0)
        if score < best_score:
            best, best_score = float(beta), score
    return best


def solve_function(plugin, target: FunctionTarget, constraints: dict[str, Any]) -> SolverResult:
    started = time.perf_counter()
    linkage, name = plugin.linkage, plugin.name
    unknown = set(constraints) - FIELDS
    if unknown:
        raise SolverError(f"{name}: unknown constraints {sorted(unknown)} for a function "
                          f"target; fields: {sorted(FIELDS)}")
    if target.kind not in KINDS[name]:
        wanted = " or ".join(f'"{k}"' for k in KINDS[name] if k != "function")
        raise SolverError(f"{name} takes a {wanted} target (or crank/output), not "
                          f"\"{target.kind}\"")
    for key in ("initial", "bounds", "fixed"):
        extra = set(constraints.get(key) or {}) - set(linkage.parameters)
        if extra:
            raise SolverError(f"{name}: unknown parameters in {key}: {sorted(extra)}; "
                              f"parameters: {list(linkage.parameters)}")
    plane = constraints.get("plane", "xy")
    if plane not in PLANES:
        raise SolverError(f"{name}: plane must be one of {sorted(PLANES)}")
    span = float(constraints.get("crank_span", 2 * math.pi))
    if target.kind == "function":
        crank = target.crank
        if np.any(np.sign(crank) * np.sign(span) < 0) or np.max(np.abs(crank)) > abs(span) + 1e-9:
            raise SolverError(f"{name}: target.crank values must lie between 0 and crank_span "
                              f"({span:.4g} rad)")
    fixed = {k: float(v) for k, v in (constraints.get("fixed") or {}).items()}
    initial = initial_design(linkage, target, {**(constraints.get("initial") or {}), **fixed})
    free = [k for k in FREE[name] if k not in fixed]
    if not free:
        raise SolverError(f"{name}: every parameter that shapes the motion is fixed")
    size = max(initial[k] for k in linkage.lengths)
    lower, upper = plugin.bounds(initial, size, constraints.get("bounds") or {}, free)
    start = np.array([initial[k] for k in free])
    u0 = (start - lower) / (upper - lower)
    min_mu = math.radians(float(constraints.get("min_transmission_deg",
                                                DEFAULT_MIN_TRANSMISSION_DEG)))
    deadline = started + float(constraints.get("time_limit_s", DEFAULT_TIME_LIMIT_S))
    scale = np.array([size if k in linkage.lengths else 1.0 for k in free])
    best: dict[str, Any] = {"cost": math.inf, "u": u0}
    count = {"evaluations": 0, "poses": 0, "pose_seconds": 0.0}

    def params_of(u: np.ndarray) -> dict[str, float]:
        x = lower + (upper - lower) * np.clip(u, 0, 1)
        return {**initial, **dict(zip(free, map(float, x), strict=True))}

    def residuals(u: np.ndarray) -> np.ndarray:
        if time.perf_counter() > deadline and count["evaluations"]:
            raise _OutOfTime
        p = params_of(u)
        posed = time.perf_counter()
        m = motion(linkage, p, span)
        count["evaluations"] += 1
        count["poses"] += m.poses
        count["pose_seconds"] += time.perf_counter() - posed
        if target.kind == "function":
            fit = (m.at(target.crank) - m.at(target.crank[:1])
                   - (target.output - target.output[0]))[1:] / target.tolerance
        else:
            fit = np.array([(m.extent - target.extent) / target.tolerance])
        penalties = np.array([50.0 * (1.0 - m.reach),
                              max(0.0, min_mu + MARGIN - m.min_transmission) / math.radians(5)
                              if name == "four_bar" else 0.0])
        x = lower + (upper - lower) * np.clip(u, 0, 1)
        pull = REGULARIZATION * (x - start) / scale
        r = np.concatenate([fit, penalties, pull])
        cost = float(r @ r)
        if m.reach >= 1.0 and cost < best["cost"]:
            best.update(cost=cost, u=np.array(u, float))
            if np.all(np.abs(fit) <= STOP) and penalties.max() <= 0.01:  # 0.05 deg
                raise _GoodEnough  # within a fifth of the tolerance, nothing violated
        return r

    message = ""
    try:
        fit = least_squares(residuals, u0, bounds=(0.0, 1.0), diff_step=1e-4,
                            max_nfev=int(constraints.get("max_evaluations",
                                                         DEFAULT_MAX_EVALUATIONS)))
        message = f"least_squares: {fit.message}"
    except _OutOfTime:
        message = "stopped at the time limit"
    except _GoodEnough:
        message = "stopped: within a fifth of the tolerance"
    return _verify(plugin, target, params_of(best["u"]), span, min_mu, plane,
                   float(constraints.get("height", 0.0)), started, message, count)


def _verify(plugin, target: FunctionTarget, params: dict[str, float], span: float,
            min_mu: float, plane: str, height: float, started: float, message: str,
            count: dict[str, Any]) -> SolverResult:
    """Pose the final design every degree, and directly at each given crank angle."""
    linkage, name = plugin.linkage, plugin.name
    fine = motion(linkage, params, span, step=CHECK_STEP)
    unit = "rad" if name == "four_bar" else "m"
    if target.kind == "function":
        direct = motion(linkage, params, span, at=target.crank)
        reached = len(direct.output) == len(target.crank)
        got = direct.output - direct.output[0] if reached else np.full(len(target.crank),
                                                                       math.inf)
        miss = got - (target.output - target.output[0])
        trajectory = [[float(c), float(o)] for c, o in zip(target.crank, got, strict=True)]
        error = {"max": float(np.max(np.abs(miss))), "rms": float(np.sqrt(np.mean(miss ** 2)))}
    else:
        miss = abs(fine.extent - target.extent)
        error = {"max": miss, "rms": miss}
        step = max(1, len(fine.crank) // 36)
        trajectory = [[float(c), float(o)] for c, o in zip(fine.crank[::step],
                                                           fine.output[::step], strict=True)]
    jams = name == "four_bar" and fine.min_transmission < min_mu - ALLOWANCE
    feasible = fine.reach >= 1.0 and not jams
    within = feasible and error["max"] <= target.tolerance
    shown = (lambda v: f"{math.degrees(v):.3g} deg") if unit == "rad" else (
        lambda v: f"{v * 1000:.3g} mm")
    what = {"sweep": "sweep", "stroke": "stroke", "function": "output"}[target.kind]
    if within:
        verdict = "the linkage makes the motion"
    elif fine.reach < 1.0:
        verdict = "the best linkage cannot turn through the whole crank span"
    elif jams:
        verdict = (f"the best linkage's transmission angle drops to "
                   f"{math.degrees(fine.min_transmission):.0f} deg")
    else:
        verdict = "no linkage within tolerance was found"
    result_message = (f"{what} error {shown(error['max'])} (tolerance "
                      f"{shown(target.tolerance)}): {verdict}")
    model = LoopModel(linkage, [], params, span, plane, height,
                      slider_output=False)
    assembly, _ = linkage.build(params, span)
    derived = linkage.derived(params, assembly)
    derived["extent"] = fine.extent
    if name == "four_bar":
        derived["min_transmission_deg"] = math.degrees(fine.min_transmission)
    return SolverResult(
        parameters=params, trajectory=trajectory, error=error, feasible=bool(feasible),
        message=result_message,
        details={"tolerance": target.tolerance, "kind": target.kind, "unit": unit,
                 "output_joint": OUTPUT_JOINT[name], "crank_span": span, "plane": plane,
                 "height": height,
                 "derived": derived, "rest_points": rest_points(assembly, model, params),
                 "fit": message, "evaluations": count["evaluations"],
                 "geometry_evaluations": count["evaluations"],
                 "poses": count["poses"],
                 "seconds_per_pose": round(count["pose_seconds"] / max(1, count["poses"]), 5),
                 "seconds": round(time.perf_counter() - started, 3)})
