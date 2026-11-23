# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Forward kinematics: tree propagation, mimic joints, loop closures.

Order of work for one pose:

1. Every independent tree joint gets a value: the caller's, the solver's, or its rest value.
2. Mimic joints copy ``multiplier * source + offset``, sources first.
3. Bodies are placed from the root down the tree.
4. If there are loop closures, ``least_squares`` adjusts the tree joints on each loop
   (the ones the caller did not drive) until every closure joint is satisfied again.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from sdk.errors import LoopClosureError, PoseError
from sdk.joint import Joint, JointType
from sdk.transforms import invert, rotation_vector

LOOP_TOLERANCE = 1e-6
"""Largest allowed closure error: meters for positions, radians for directions."""


@dataclass(frozen=True)
class Articulation:
    """The resolved structure of an assembly (made by ``Assembly.validate``)."""

    root: str
    tree: tuple[Joint, ...]
    """Tree joints, every parent placed before its child."""
    closures: tuple[Joint, ...]
    """Joints left out of the tree. They constrain the pose instead of placing a body."""
    parent_joint: Mapping[str, Joint]
    """Body name -> the tree joint that places it. The root has none."""
    mimic_order: tuple[Joint, ...]
    """Mimic joints, every source before its followers."""


@dataclass(frozen=True)
class Pose:
    transforms: dict[str, np.ndarray]
    """Body name -> 4x4 world transform."""
    joint_values: dict[str, float]
    """Every movable joint: driven, rest, mimic, solved, and closure values."""
    residual: float
    """Largest remaining loop-closure error (0 when there are no loops)."""


def place_bodies(art: Articulation, values: Mapping[str, float]) -> dict[str, np.ndarray]:
    transforms = {art.root: np.eye(4)}
    for joint in art.tree:
        motion = joint.motion(values[joint.name]) if joint.movable else np.eye(4)
        transforms[joint.child.body.name] = (
            transforms[joint.parent.body.name]
            @ joint.parent.frame.matrix
            @ motion
            @ invert(joint.child.frame.matrix)
        )
    return transforms


def apply_mimics(art: Articulation, values: dict[str, float]) -> None:
    for joint in art.mimic_order:
        mimic = joint.mimic
        values[joint.name] = mimic.multiplier * values[mimic.source] + mimic.offset


def _closure_motion(joint: Joint, transforms: Mapping[str, np.ndarray]) -> np.ndarray:
    """The motion a closure joint would need: parent frame -> child frame."""
    parent = transforms[joint.parent.body.name] @ joint.parent.frame.matrix
    child = transforms[joint.child.body.name] @ joint.child.frame.matrix
    return invert(parent) @ child


def closure_error(joint: Joint, transforms: Mapping[str, np.ndarray]) -> np.ndarray:
    """Residual rows that are zero exactly when the closure joint can be satisfied."""
    motion = _closure_motion(joint, transforms)
    rotation, offset = motion[:3, :3], motion[:3, 3]
    if joint.type is JointType.FIXED:
        return np.concatenate([offset, rotation_vector(rotation)])
    axis = np.asarray(joint.axis)
    if joint.type is JointType.REVOLUTE:
        # Frames must touch, and the hinge axis must map onto itself.
        return np.concatenate([offset, rotation @ axis - axis])
    # Prismatic: no rotation, and the offset may only run along the axis.
    return np.concatenate([offset - offset.dot(axis) * axis, rotation_vector(rotation)])


def closure_value(joint: Joint, transforms: Mapping[str, np.ndarray]) -> float:
    motion = _closure_motion(joint, transforms)
    axis = np.asarray(joint.axis)
    if joint.type is JointType.REVOLUTE:
        return float(rotation_vector(motion[:3, :3]).dot(axis))
    return float(motion[:3, 3].dot(axis))


def _ancestors(art: Articulation, body: str) -> list[Joint]:
    chain = []
    while body != art.root:
        joint = art.parent_joint[body]
        chain.append(joint)
        body = joint.parent.body.name
    return chain


def loop_path(art: Articulation, closure: Joint) -> list[Joint]:
    """Tree joints on the loop that ``closure`` closes."""
    a = _ancestors(art, closure.parent.body.name)
    b = _ancestors(art, closure.child.body.name)
    shared = set(a) & set(b)
    return [j for j in a if j not in shared] + [j for j in b if j not in shared]


def loop_unknowns(art: Articulation, driven: set[str], by_name: Mapping[str, Joint]) -> list[Joint]:
    """Independent joints the solver may move: on a loop (directly or through a mimic),
    not driven by the caller, and not locked by equal limits."""
    unknowns: list[Joint] = []
    for closure in art.closures:
        for joint in loop_path(art, closure):
            while joint.mimic is not None:  # a mimic moves with its source
                joint = by_name[joint.mimic.source]
            locked = joint.limits is not None and joint.limits[0] == joint.limits[1]
            if joint.movable and joint.name not in driven and not locked:
                if joint not in unknowns:
                    unknowns.append(joint)
    return unknowns


def solve(
    art: Articulation,
    values: Mapping[str, float],
    unknowns: list[Joint],
    seed: Mapping[str, float],
    tolerance: float = LOOP_TOLERANCE,
) -> Pose:
    """Solve the unknowns so every closure closes, then place every body."""
    lower = np.array([_bound(j, 0) for j in unknowns], dtype=float)
    upper = np.array([_bound(j, 1) for j in unknowns], dtype=float)

    def full_values(x: np.ndarray) -> dict[str, float]:
        out = dict(values)
        for joint, q in zip(unknowns, x, strict=True):
            out[joint.name] = float(q)
        apply_mimics(art, out)
        return out

    def residual(x: np.ndarray) -> np.ndarray:
        transforms = place_bodies(art, full_values(x))
        return np.concatenate([closure_error(c, transforms) for c in art.closures])

    x = np.zeros(0)
    if unknowns:
        start = np.clip([seed.get(j.name, values[j.name]) for j in unknowns], lower, upper)
        x = _least_squares(residual, start, lower, upper, tolerance)
        x = np.array([_wrap(j, q) for j, q in zip(unknowns, x, strict=True)])

    final = full_values(x)
    transforms = place_bodies(art, final)
    error = float(np.max(np.abs(residual(x)), initial=0.0)) if art.closures else 0.0
    if error > tolerance:
        raise LoopClosureError(_open_loop_message(art, unknowns, x, lower, upper, error, tolerance))
    for closure in art.closures:
        if closure.movable:
            final[closure.name] = closure_value(closure, transforms)
    for joint in (*art.tree, *art.closures):
        if joint.movable and not joint.within_limits(final[joint.name], 1e-6):
            raise PoseError(
                f"joint {joint.name!r} ends at {final[joint.name]:.6g}, outside its limits "
                f"{joint.limits}"
            )
    return Pose(transforms, final, error)


def _least_squares(residual, start: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   tolerance: float) -> np.ndarray:
    if float(np.max(np.abs(residual(start)), initial=0.0)) <= tolerance * 1e-3:
        # Already closed: do not let finite-difference noise nudge joints the loop does
        # not constrain (e.g. a rail drive whose two carriages are mimicked).
        return start
    # Solve for the offset from the start, not the joint values themselves. SciPy's "trf"
    # sizes its first trust region from the starting point (norm(x0)), so a start like
    # [0, 4e-16] gets a 4e-16 trust region and gives up at once. An offset always starts
    # at exactly 0, which SciPy turns into a first trust region of 1 x x_scale. x_scale=0.1
    # keeps that first step modest (~6 degrees / 10 cm), so a rough warm start does not
    # jump to the other branch of a loop; the region grows when bigger steps pay off.
    result = least_squares(lambda offset: residual(start + offset), np.zeros_like(start),
                           bounds=(lower - start, upper - start), method="trf", x_scale=0.1,
                           xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return start + result.x


def _open_loop_message(art: Articulation, unknowns: list[Joint], x: np.ndarray,
                       lower: np.ndarray, upper: np.ndarray, error: float,
                       tolerance: float) -> str:
    names = ", ".join(repr(c.name) for c in art.closures)
    if not unknowns:
        hint = "; no free joint is left on the loop to close it"
    else:
        pinned = [j.name for j, q, lo, hi in zip(unknowns, x, lower, upper, strict=True)
                  if math.isclose(q, lo, abs_tol=1e-9) or math.isclose(q, hi, abs_tol=1e-9)]
        hint = f"; solved joints stuck at a limit: {pinned}" if pinned else ""
    return f"loop closure {names} stays open: error {error:.3g} > {tolerance:g}{hint}"


def _bound(joint: Joint, side: int) -> float:
    if joint.limits is None or joint.periodic:
        return -math.inf if side == 0 else math.inf
    return joint.limits[side]


def _wrap(joint: Joint, value: float) -> float:
    """Bring a full-turn hinge back into its range; the pose does not change."""
    if not joint.periodic:
        return float(value)
    start = -math.pi if joint.limits is None else joint.limits[0]
    return start + (value - start) % (2.0 * math.pi)
