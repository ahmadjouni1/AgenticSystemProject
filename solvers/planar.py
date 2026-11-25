"""Plug-in B (experimental): closed loops, a four-bar and a slider-crank.

The design vector builds a real replica ``Assembly`` (bodies, joints, one loop closure,
authored closed with ``frame_in`` as in ``examples/four_bar.py``). Our loop solver poses
it, with the walking warm start (``Assembly.solve``), and the output point is read from
the posed bodies. The adapter goes into the teammate's solver through its custom-model
interface (``fk(design, q)`` with ``joint_names``, ``lower``, ``upper``).

**Speed.** One pose costs about 4 ms, and the inner optimizer asks for thousands of poses
per design. So each new design is posed once, at crank steps of at most 5 degrees over the
whole crank span (73 poses for a full turn), and ``fk`` interpolates that curve with a
cubic spline. The final design is then posed again directly at every solved crank angle,
and those poses, not the spline, give the reported trajectory and error.

**Reach.** A crank angle where the loop cannot close, or where it flips to its other
branch, ends the curve. ``extra_margins`` then makes target points beyond that angle
infeasible, so the outer search is steered back to linkages that move through the span.

Parameters (meters, radians; all in the mechanism's plane, ``plane`` = "xy", "xz" or "yz"):

four_bar: ``crank_pivot_x``, ``crank_pivot_y``, ``crank_length``, ``coupler_length`` (crank
pin to rocker pin), ``ground_length`` and ``ground_angle`` (crank pivot to rocker pivot),
``coupler_angle`` (the coupler's direction relative to the crank at q = 0), ``crank_phase``
(the crank's direction at q = 0), ``point_x``, ``point_y`` (the output point on the coupler:
x from the crank pin toward the rocker pin). The rocker's length follows (``derived``).

slider_crank: ``crank_pivot_x``, ``crank_pivot_y``, ``crank_length``, ``rod_length``,
``rod_angle`` (relative to the crank at q = 0), ``slider_angle`` (the rail's direction),
``crank_phase``, ``point_x``, ``point_y`` (on the rod; the slider pin is (rod_length, 0)).
The rail passes through the slider pin at q = 0; its offset from the crank pivot follows.

Other constraints: ``crank_span`` (radians the crank turns from the first target point to
the last, default 2 pi; negative turns the other way), ``initial`` / ``bounds`` / ``fixed``
({name: value} / {name: [lower, upper]} / {name: value}), ``max_evaluations`` (default 500),
``polish`` (default true), ``time_limit_s`` (default 100).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from sdk import Assembly, Box
from sdk.errors import LoopClosureError, PoseError
from solvers.base import SolverError, SolverResult, Target
from solvers.nested import Settings, Task
from solvers.search import Synthesis
from solvers.serial_arm import errors

GRID_STEP = math.radians(5.0)  # below the walk step (0.1 rad), so each grid pose is one solve
FLIP_RADIANS = 0.8  # a loop joint turning more than this in one grid step has changed branch
DEFAULT_MAX_EVALUATIONS = 500
DEFAULT_TIME_LIMIT_S = 100.0
POLISH_ITERATIONS = 60  # each polish iteration sweeps ~one new curve per free parameter
PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}  # (u, v, normal) axes
_FIELDS = {"plane", "crank_span", "initial", "bounds", "fixed", "max_evaluations", "polish",
           "time_limit_s", "output"}
_Z = (0.0, 0.0, 1.0)


def _bar(body, length: float) -> None:
    """Any geometry will do (the solver only poses frames); validate() needs a shape."""
    body.add("bar", Box((max(abs(length), 1e-3), 0.002, 0.002)), xyz=(length / 2, 0, 0))


def four_bar(p: dict[str, float], span: float) -> tuple[Assembly, str]:
    """A four-bar from the design, closed at q = 0. Returns (assembly, output body)."""
    m = Assembly("four_bar")
    ground, crank, coupler, rocker = (m.body(n) for n in ("ground", "crank", "coupler",
                                                         "rocker"))
    _bar(ground, p["ground_length"])
    _bar(crank, p["crank_length"])
    _bar(coupler, p["coupler_length"])
    m.joint("crank_joint", "revolute",
            ground.at((p["crank_pivot_x"], p["crank_pivot_y"], 0), rpy=(0, 0, p["crank_phase"])),
            crank.at(), axis=_Z, limits=(min(0.0, span), max(0.0, span)))
    m.joint("coupler_joint", "revolute",
            crank.at((p["crank_length"], 0, 0), rpy=(0, 0, p["coupler_angle"])), coupler.at(),
            axis=_Z)
    angle = p["ground_angle"]
    pivot = (p["crank_pivot_x"] + p["ground_length"] * math.cos(angle),
             p["crank_pivot_y"] + p["ground_length"] * math.sin(angle), 0)
    m.joint("rocker_joint", "revolute", ground.at(pivot), rocker.at(), axis=_Z)
    tip = coupler.at((p["coupler_length"], 0, 0))
    pin = m.frame_in(tip, rocker)
    _bar(rocker, float(np.linalg.norm(pin.frame.xyz)))
    m.joint("pin", "revolute", tip, pin, axis=_Z)
    m.articulation(root="ground", closures=["pin"])
    return m, "coupler"


def slider_crank(p: dict[str, float], span: float) -> tuple[Assembly, str]:
    m = Assembly("slider_crank")
    ground, crank, rod, slider = (m.body(n) for n in ("ground", "crank", "rod", "slider"))
    _bar(ground, 0.01)
    _bar(crank, p["crank_length"])
    _bar(rod, p["rod_length"])
    _bar(slider, 0.01)
    m.joint("crank_joint", "revolute",
            ground.at((p["crank_pivot_x"], p["crank_pivot_y"], 0), rpy=(0, 0, p["crank_phase"])),
            crank.at(), axis=_Z, limits=(min(0.0, span), max(0.0, span)))
    m.joint("rod_joint", "revolute",
            crank.at((p["crank_length"], 0, 0), rpy=(0, 0, p["rod_angle"])), rod.at(), axis=_Z)
    tip = rod.at((p["rod_length"], 0, 0))
    on_ground = m.frame_in(tip, ground).frame.xyz
    m.joint("slide", "prismatic", ground.at(on_ground, rpy=(0, 0, p["slider_angle"])),
            slider.at(), axis=(1.0, 0.0, 0.0))
    m.joint("pin", "revolute", tip, m.frame_in(tip, slider), axis=_Z)
    m.articulation(root="ground", closures=["pin"])
    return m, "rod"


def _four_bar_derived(p: dict[str, float], m: Assembly) -> dict[str, Any]:
    rocker = float(np.linalg.norm(m.get_joint("pin").child.frame.xyz))
    lengths = sorted([p["crank_length"], p["coupler_length"], p["ground_length"], rocker])
    return {"rocker_length": rocker,
            "grashof": lengths[0] + lengths[3] <= lengths[1] + lengths[2]}


def _slider_crank_derived(p: dict[str, float], m: Assembly) -> dict[str, Any]:
    rail = m.get_joint("slide").parent.frame
    direction = np.array([math.cos(p["slider_angle"]), math.sin(p["slider_angle"])])
    to_pivot = np.array([p["crank_pivot_x"], p["crank_pivot_y"]]) - np.asarray(rail.xyz[:2])
    return {"offset": float(direction[0] * to_pivot[1] - direction[1] * to_pivot[0])}


@dataclass(frozen=True)
class Linkage:
    name: str
    parameters: tuple[str, ...]
    lengths: tuple[str, ...]  # must stay positive
    angles: tuple[str, ...]
    build: Callable[[dict[str, float], float], tuple[Assembly, str]]
    derived: Callable[[dict[str, float], Assembly], dict[str, Any]]
    defaults: dict[str, float]  # initial values, lengths as multiples of the target's size


LINKAGES = {
    "four_bar": Linkage(
        "four_bar",
        ("crank_pivot_x", "crank_pivot_y", "crank_length", "coupler_length", "ground_length",
         "ground_angle", "coupler_angle", "crank_phase", "point_x", "point_y"),
        ("crank_length", "coupler_length", "ground_length"),
        ("ground_angle", "coupler_angle", "crank_phase"),
        four_bar, _four_bar_derived,
        {"crank_length": 0.4, "coupler_length": 1.2, "ground_length": 1.2, "ground_angle": 0.0,
         "coupler_angle": 1.2, "crank_phase": 0.0}),
    "slider_crank": Linkage(
        "slider_crank",
        ("crank_pivot_x", "crank_pivot_y", "crank_length", "rod_length", "rod_angle",
         "slider_angle", "crank_phase", "point_x", "point_y"),
        ("crank_length", "rod_length"),
        ("rod_angle", "slider_angle", "crank_phase"),
        slider_crank, _slider_crank_derived,
        {"crank_length": 0.4, "rod_length": 1.2, "rod_angle": -0.5, "slider_angle": 0.0,
         "crank_phase": 0.5}),
}


@dataclass
class Curve:
    t: np.ndarray  # crank fraction of the span, 0..reach
    points: np.ndarray  # (k, 2) output point in the plane
    reach: float  # how far (fraction of the span) the crank turned before the loop failed
    spline: CubicSpline | None
    poses: int
    seconds: float

    def at(self, t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, self.reach)
        if self.spline is None:
            return np.repeat(self.points[:1], len(t), axis=0)
        return self.spline(t)


def output_point(pose, body: str, point: tuple[float, float]) -> np.ndarray:
    matrix = pose.transforms[body]
    return (matrix[:3, :3] @ np.array([point[0], point[1], 0.0]) + matrix[:3, 3])[:2]


def sweep(assembly: Assembly, body: str, point: tuple[float, float], span: float) -> Curve:
    """Pose the linkage at crank steps of at most 5 degrees from 0 to ``span``."""
    started = time.perf_counter()
    count = max(2, math.ceil(abs(span) / GRID_STEP) + 1)
    fractions = np.linspace(0.0, 1.0, count)
    points, reached, previous = [], [], None
    for t in fractions:
        try:
            pose = assembly.solve({"crank_joint": float(t * span)})
        except (LoopClosureError, PoseError):
            break
        loop = {k: v for k, v in pose.joint_values.items()
                if k != "crank_joint" and not k.startswith("slide")}
        if previous is not None and any(  # angles are wrapped to +-pi: compare the turn
                abs((loop[k] - previous[k] + math.pi) % (2 * math.pi) - math.pi) > FLIP_RADIANS
                for k in loop):
            break  # the loop jumped to its other branch: the real linkage cannot do that
        previous = loop
        points.append(output_point(pose, body, point))
        reached.append(t)
    t = np.array(reached)
    pts = np.array(points).reshape(-1, 2)
    spline = CubicSpline(t, pts, axis=0) if len(t) >= 2 else None
    return Curve(t, pts, float(t[-1]) if len(t) else 0.0, spline, len(t),
                 time.perf_counter() - started)


class LoopModel:
    """The teammate's custom-model interface for one linkage: ``fk(design, q)`` gives the
    output point for each crank value, from the swept curve of that design."""

    def __init__(self, linkage: Linkage, free: list[str], fixed: dict[str, float],
                 span: float, plane: str, height: float, slider_output: bool = False) -> None:
        self.linkage, self.free, self.fixed = linkage, free, fixed
        self.slider_output = slider_output
        self.span, self.plane, self.height = span, plane, height
        self.joint_names = ["crank_joint"]
        self.lower = np.array([min(0.0, span)])
        self.upper = np.array([max(0.0, span)])
        self.curves: dict[bytes, Curve] = {}
        self.sweeps = 0
        self.sweep_seconds = 0.0
        self.poses = 0

    def full(self, design: np.ndarray) -> dict[str, float]:
        values = {**self.fixed, **dict(zip(self.free, map(float, design), strict=True))}
        if self.slider_output:  # the output is the slider pin, wherever the rod ends
            values.update(point_x=values["rod_length"], point_y=0.0)
        return values

    def build(self, design: np.ndarray) -> tuple[Assembly, str]:
        return self.linkage.build(self.full(design), self.span)

    def curve(self, design: np.ndarray) -> Curve:
        key = np.asarray(design, float).tobytes()
        if key not in self.curves:
            if len(self.curves) > 64:
                self.curves.clear()
            p = self.full(design)
            if min(p[name] for name in self.linkage.lengths) <= 1e-6:
                found = Curve(np.zeros(1), np.zeros((1, 2)), 0.0, None, 0, 0.0)
            else:
                assembly, body = self.linkage.build(p, self.span)
                found = sweep(assembly, body, (p["point_x"], p["point_y"]), self.span)
            self.curves[key] = found
            self.sweeps += 1
            self.sweep_seconds += found.seconds
            self.poses += found.poses
        return self.curves[key]

    def embed(self, uv: np.ndarray) -> np.ndarray:
        u, v, normal = PLANES[self.plane]
        out = np.empty((len(uv), 3))
        out[:, u], out[:, v], out[:, normal] = uv[:, 0], uv[:, 1], self.height
        return out

    def fk(self, design: np.ndarray, q: np.ndarray):
        q = np.atleast_2d(q)
        positions = self.embed(self.curve(design).at(q[:, 0] / self.span))
        n = len(q)
        return positions, np.broadcast_to(np.eye(3), (n, 3, 3)).copy(), np.zeros((n, 0, 3))

    def margins(self, design: np.ndarray, q: np.ndarray) -> np.ndarray:
        """>= 0 where the crank can reach the point's angle (fraction of the span)."""
        return self.curve(design).reach - np.atleast_2d(q)[:, 0] / self.span


class PlanarLoop:
    def __init__(self, linkage: Linkage) -> None:
        self.linkage = linkage
        self.name = linkage.name

    def solve(self, target: Target, constraints: dict[str, Any]) -> SolverResult:
        started = time.perf_counter()
        linkage = self.linkage
        unknown = set(constraints) - _FIELDS
        if unknown:
            raise SolverError(f"{self.name}: unknown constraints {sorted(unknown)}; fields: "
                              f"{sorted(_FIELDS)}")
        plane = constraints.get("plane", "xy")
        if plane not in PLANES:
            raise SolverError(f"{self.name}: plane must be one of {sorted(PLANES)}")
        span = float(constraints.get("crank_span", 2 * math.pi))
        if not 0.05 <= abs(span) <= 4 * math.pi:
            raise SolverError(f"{self.name}: crank_span must be 0.05 to 4 pi radians "
                              "(negative turns the other way)")
        for key in ("initial", "bounds", "fixed"):
            names = set(constraints.get(key) or {}) - set(linkage.parameters)
            if names:
                raise SolverError(f"{self.name}: unknown parameters in {key}: {sorted(names)}; "
                                  f"parameters: {list(linkage.parameters)}")
        output = constraints.get("output", "point")
        if output not in ("point", "slider") or (output == "slider"
                                                 and self.name != "slider_crank"):
            raise SolverError(f"{self.name}: output must be \"point\""
                              + (" or \"slider\"" if self.name == "slider_crank" else ""))
        u, v, normal = PLANES[plane]
        uv = target.points[:, [u, v]]
        height = float(np.mean(target.points[:, normal]))
        size = max(float(np.ptp(uv, axis=0).max()), target.tolerance)
        given = dict(constraints.get("initial") or {})
        fixed = {k: float(v) for k, v in (constraints.get("fixed") or {}).items()}
        if output == "slider":  # the output point follows the rod's length (LoopModel.full)
            given.update(point_x=0.0, point_y=0.0)
            fixed.update(point_x=0.0, point_y=0.0)
        initial = self.initial(uv, size, given, span)
        initial.update(fixed)
        free = [name for name in linkage.parameters if name not in fixed]
        if not free:
            raise SolverError(f"{self.name}: every parameter is fixed; nothing to solve")
        lower, upper = self.bounds(initial, size, constraints.get("bounds") or {}, free)
        model = LoopModel(linkage, free, fixed, span, plane, height, output == "slider")
        n = len(uv)
        task = Task(target.points, np.zeros(1), np.array([span]),
                    np.array([3 * abs(span) / (n - 1)]), target.tolerance, size,
                    extra_margins=model.margins)
        settings = Settings(outer_maxeval=int(constraints.get("max_evaluations",
                                                              DEFAULT_MAX_EVALUATIONS)))
        time_limit = float(constraints.get("time_limit_s", DEFAULT_TIME_LIMIT_S))
        try:
            solver = Synthesis(model, task, [initial[k] for k in free], lower, upper, settings,
                               deadline=started + time_limit, stop_error=target.tolerance / 10)
        except ValueError as exc:
            raise SolverError(f"{self.name}: {exc}") from None
        best, stats = solver.run(polish=bool(constraints.get("polish", True)),
                                 polish_iterations=POLISH_ITERATIONS)
        return self.verify(model, best, stats, target, started)

    def initial(self, uv: np.ndarray, size: float, given: dict[str, float],
                span: float) -> dict[str, float]:
        """A starting design: the defaults scaled to the target, the crank pivot below the
        target, and the output point placed on the first target point at q = 0."""
        linkage = self.linkage
        values = {k: v * size if k in linkage.lengths else v
                  for k, v in linkage.defaults.items()}
        centre = uv.mean(axis=0)
        values.update(crank_pivot_x=float(centre[0] - 0.3 * size),
                      crank_pivot_y=float(centre[1] - 1.0 * size), point_x=0.0, point_y=0.0)
        values.update({k: float(v) for k, v in given.items()})
        if "point_x" not in given and "point_y" not in given:
            assembly, body = linkage.build(values, span)
            matrix = assembly.solve({}).transforms[body]
            local = matrix[:3, :3].T @ (np.r_[uv[0], 0.0] - matrix[:3, 3])
            values.update(point_x=float(local[0]), point_y=float(local[1]))
        return values

    def bounds(self, initial: dict[str, float], size: float, given: dict[str, list[float]],
               free: list[str]) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = [], []
        for name in free:
            value = initial[name]
            if name in given:
                low, high = map(float, given[name])
            elif name in self.linkage.lengths:
                low, high = max(1e-3 * size, 0.2 * value), value + 2 * size
            elif name in self.linkage.angles:
                low, high = value - math.pi, value + math.pi
            else:
                low, high = value - 2 * size, value + 2 * size
            if not low <= value <= high or low >= high:
                raise SolverError(f"{self.name}: bounds for {name} ({low:.4g}, {high:.4g}) must "
                                  f"contain its initial value {value:.4g}")
            lower.append(low)
            upper.append(high)
        return np.array(lower), np.array(upper)

    def verify(self, model: LoopModel, best, stats: dict[str, Any], target: Target,
               started: float) -> SolverResult:
        """Pose the final design directly at every solved crank angle (no interpolation)."""
        params = model.full(best.design)
        assembly, body = model.build(best.design)
        values = best.q[:, 0]
        ok, posed_at = True, time.perf_counter()
        order = np.argsort(np.abs(values))  # walk out from q = 0, as the real crank turns
        found: dict[int, np.ndarray] = {}
        for i in order:
            try:
                pose = assembly.solve({"crank_joint": float(values[i])})
            except (LoopClosureError, PoseError):
                ok = False
                found[i] = model.curve(best.design).at(np.array([values[i] / model.span]))[0]
                continue
            found[i] = output_point(pose, body, (params["point_x"], params["point_y"]))
        pose_seconds = (time.perf_counter() - posed_at) / max(1, len(values))
        reached = model.embed(np.array([found[i] for i in range(len(values))]))
        error = errors(reached, target.points)
        feasible = bool(best.feasible and ok)
        within = feasible and error["max"] <= target.tolerance
        rest = assembly.solve({}, warm_start={})
        message = (f"max error {error['max'] * 1000:.3g} mm (tolerance "
                   f"{target.tolerance * 1000:.3g} mm): "
                   + ("the linkage passes through every target point" if within else
                      "no linkage within tolerance was found" if feasible else
                      "the best linkage cannot turn through the whole crank span"))
        return SolverResult(
            parameters=params, trajectory=reached.tolist(), error=error, feasible=feasible,
            message=message,
            details={"tolerance": target.tolerance, "plane": model.plane,
                     "crank_span": model.span,
                     "crank_values": values.tolist(),
                     "derived": self.linkage.derived(params, assembly),
                     "rest_points": rest_points(assembly, model, params),
                     "joint_values_at_rest": dict(rest.joint_values),
                     **stats,
                     "curve_sweeps": model.sweeps, "poses": model.poses,
                     "seconds_per_sweep": round(model.sweep_seconds / max(1, model.sweeps), 4),
                     "seconds_per_pose": round(model.sweep_seconds / max(1, model.poses), 5),
                     "verify_seconds_per_pose": round(pose_seconds, 5),
                     "seconds": round(time.perf_counter() - started, 3)})


def rest_points(assembly: Assembly, model: LoopModel, params: dict[str, float]) -> dict:
    """Where the pivots and pins are at q = 0, in world coordinates: what to build."""
    pose = assembly.solve({}, warm_start={})
    out = {}
    for joint in assembly.joints:
        matrix = pose.transforms[joint.parent.body.name] @ joint.parent.frame.matrix
        out[joint.name] = matrix[:2, 3]
    body = "coupler" if "coupler" in pose.transforms else "rod"
    out["output_point"] = output_point(pose, body, (params["point_x"], params["point_y"]))
    return {name: model.embed(np.asarray(p)[None])[0].tolist() for name, p in out.items()}


PLANAR = {name: PlanarLoop(linkage) for name, linkage in LINKAGES.items()}
