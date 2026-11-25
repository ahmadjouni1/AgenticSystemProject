"""The solver plug-in interface, and running one solve in its own process.

    result = solve_isolated("four_bar", target, constraints, timeout_s=120)
    result.parameters, result.trajectory, result.error, result.feasible, result.message

A plug-in turns a **target** (the points the output must pass through, in order, and a
tolerance) and **constraints** (the mechanism's settings) into a ``SolverResult``. Both are
JSON data only. Nothing in them is ever executed: Articraft's trajectory tool accepts Python
code for the target curve, this interface does not.

**Why a subprocess.** ``solve_isolated`` runs the solve in a fresh interpreter
(``python -m solvers.worker``), with the request on stdin and the result as JSON on stdout.
A solve that runs too long is killed with ``Popen.kill`` (TerminateProcess on Windows,
SIGKILL elsewhere), so the timeout works the same everywhere. A crash or a hang in the
optimizer cannot take the agent down with it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

REPLICA_ROOT = Path(__file__).resolve().parents[1]
MAX_POINTS = 400
DEFAULT_TIMEOUT_S = 120.0


class SolverError(Exception):
    """Bad input, or a solve that could not run. The message is meant for the model."""


class SolverTimeout(SolverError):
    pass


@dataclass(frozen=True)
class Target:
    points: np.ndarray
    """(N, 3) meters: where the output point should be, in order along the motion."""
    tolerance: float
    """Meters: the largest acceptable distance from any target point."""

    @classmethod
    def from_data(cls, data: Any) -> Target:
        """Check that ``data`` is plain data: {"points": [[x, y, z], ...], "tolerance": m}."""
        if not isinstance(data, dict):
            raise SolverError("target must be an object: {\"points\": [[x, y, z], ...], "
                              "\"tolerance\": meters}")
        unknown = set(data) - {"points", "tolerance"}
        if unknown:
            raise SolverError(f"target has unknown fields {sorted(unknown)}; it takes points "
                              "and tolerance")
        points = data.get("points")
        if not isinstance(points, list) or not 3 <= len(points) <= MAX_POINTS:
            raise SolverError(f"target.points must be a list of 3 to {MAX_POINTS} [x, y, z] "
                              "points in meters")
        for i, point in enumerate(points):
            if (not isinstance(point, list) or len(point) != 3
                    or not all(_is_number(v) for v in point)):
                raise SolverError(f"target.points[{i}] must be [x, y, z] (three finite numbers, "
                                  f"meters), got {point!r}"[:200])
        tolerance = data.get("tolerance", 0.002)
        if not _is_number(tolerance) or tolerance <= 0:
            raise SolverError(f"target.tolerance must be a positive number of meters, got "
                              f"{tolerance!r}")
        array = np.asarray(points, dtype=float)
        if np.ptp(array, axis=0).max() < 1e-6:
            raise SolverError("target.points are all the same point; give the path the output "
                              "should follow")
        return cls(array, float(tolerance))

    def to_data(self) -> dict[str, Any]:
        return {"points": self.points.tolist(), "tolerance": self.tolerance}


@dataclass(frozen=True)
class FunctionTarget:
    """Function generation: the output joint's motion against the crank angle (radians for a
    rocker, meters for a slider), not a point path.

    - ``kind="sweep"`` / ``"stroke"``: the output's peak-to-peak travel over the crank span
      (a wiper swinging 90 degrees, a piston moving 60 mm), ``extent``.
    - ``kind="function"``: the output at given crank angles, ``crank`` and ``output``, each
      measured from its value at the first crank angle.
    """
    kind: str
    tolerance: float
    extent: float | None = None
    crank: np.ndarray | None = None
    output: np.ndarray | None = None

    def to_data(self) -> dict[str, Any]:
        if self.kind == "function":
            return {"crank": self.crank.tolist(), "output": self.output.tolist(),
                    "tolerance": self.tolerance}
        return {self.kind: self.extent, "tolerance": self.tolerance}


FUNCTION_DEFAULT_TOLERANCE = {"sweep": math.radians(1.0), "stroke": 0.001}
TARGET_FORMS = ('{"points": [[x, y, z], ...]} (a path), {"sweep": radians} (a rocker), '
                '{"stroke": meters} (a slider), or {"crank": [...], "output": [...]}; each with '
                'an optional "tolerance"')


def parse_target(data: Any) -> Target | FunctionTarget:
    """A path target or a function target, checked as plain data."""
    if not isinstance(data, dict):
        raise SolverError(f"target must be an object: {TARGET_FORMS}")
    if "points" in data:
        return Target.from_data(data)
    kinds = [k for k in ("sweep", "stroke", "crank") if k in data]
    if len(kinds) != 1:
        raise SolverError(f"target must be one of {TARGET_FORMS}")
    kind = "function" if kinds[0] == "crank" else kinds[0]
    allowed = {"crank", "output", "tolerance"} if kind == "function" else {kind, "tolerance"}
    unknown = set(data) - allowed
    if unknown:
        raise SolverError(f"target has unknown fields {sorted(unknown)} for a {kind} target")
    tolerance = data.get("tolerance", FUNCTION_DEFAULT_TOLERANCE.get(kind, 0.01))
    if not _is_number(tolerance) or tolerance <= 0:
        raise SolverError(f"target.tolerance must be a positive number, got {tolerance!r}")
    if kind != "function":
        extent = data[kind]
        if not _is_number(extent) or extent <= 0:
            raise SolverError(f"target.{kind} must be a positive number "
                              f"({'radians' if kind == 'sweep' else 'meters'}), got {extent!r}")
        return FunctionTarget(kind, float(tolerance), extent=float(extent))
    crank, output = data.get("crank"), data.get("output")
    for name, values in (("crank", crank), ("output", output)):
        if (not isinstance(values, list) or not 3 <= len(values) <= MAX_POINTS
                or not all(_is_number(v) for v in values)):
            raise SolverError(f"target.{name} must be a list of 3 to {MAX_POINTS} numbers")
    if len(crank) != len(output):
        raise SolverError("target.crank and target.output need the same length")
    return FunctionTarget("function", float(tolerance), crank=np.asarray(crank, float),
                          output=np.asarray(output, float))


@dataclass
class SolverResult:
    parameters: dict[str, float]
    """The design: every dimension, by name (meters, radians)."""
    trajectory: list[list[float]]
    """The output point the design reaches for each target point (meters)."""
    error: dict[str, float]
    """{"max": m, "rms": m}: distances from the target points to ``trajectory``."""
    feasible: bool
    """The design and its motion satisfy every constraint (limits, loop closure)."""
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    """Plug-in specific: driver values, derived dimensions, timing."""

    @property
    def within_tolerance(self) -> bool:
        return self.feasible and self.error["max"] <= self.details.get("tolerance", math.inf)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolverResult:
        return cls(**data)


class Plugin(Protocol):
    name: str

    def solve(self, target: Target, constraints: dict[str, Any]) -> SolverResult: ...


SUPPORTED = ("four_bar", "slider_crank", "serial_arm")
UNSUPPORTED_HINT = ("synthesize sizes only a four_bar or a slider_crank (path, sweep or stroke "
                    "targets) and a serial_arm (path); design this mechanism normally")


class Unsupported(SolverError):
    """A mechanism no plug-in handles: answered at once, before any solve."""


def plugins() -> dict[str, Plugin]:
    """Every plug-in by mechanism name (imported on first use)."""
    from solvers.planar import PLANAR
    from solvers.serial_arm import SerialArm

    return {"serial_arm": SerialArm(), **PLANAR}


def check_supported(name: Any) -> None:
    if name not in SUPPORTED:
        raise Unsupported(f"unsupported mechanism {name!r}: {UNSUPPORTED_HINT}")


def solve(name: str, target: Any, constraints: Any) -> SolverResult:
    """One solve in this process (the worker and the tests use it)."""
    check_supported(name)
    found = plugins()
    if not isinstance(constraints, dict):
        raise SolverError("constraints must be an object")
    started = time.perf_counter()
    result = found[name].solve(parse_target(target), constraints)
    result.details.setdefault("seconds", round(time.perf_counter() - started, 3))
    return result


def solve_isolated(name: str, target: Any, constraints: Any,
                   timeout_s: float = DEFAULT_TIMEOUT_S) -> SolverResult:
    """``solve`` in a fresh interpreter, killed after ``timeout_s`` seconds."""
    check_supported(name)  # no process for a mechanism no plug-in handles
    request = json.dumps({"mechanism": name, "target": target, "constraints": constraints})
    env = {k: v for k, v in os.environ.items()  # the solver needs no credentials
           if not any(word in k.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, "-m", "solvers.worker"], cwd=REPLICA_ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env, text=True, encoding="utf-8")
    try:
        out, err = process.communicate(request, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise SolverTimeout(f"the solve did not finish in {timeout_s:.0f} s and was stopped") \
            from None
    lines = [line for line in out.splitlines() if line.strip()]
    try:
        reply = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        tail = err.strip().splitlines()[-1:] or ["no output"]
        raise SolverError(f"the solver process failed (exit {process.returncode}): "
                          f"{tail[0][:300]}") from None
    if "error" in reply:
        raise SolverError(reply["error"])
    result = SolverResult.from_dict(reply["result"])
    result.details["process_seconds"] = round(time.perf_counter() - started, 3)
    return result


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))
