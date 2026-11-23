"""An assembly: bodies, the joints between them, and how they form a tree.

Every joint points from a parent body to a child body. Tree joints place their
child. The tree must have exactly one root (the only body that is no joint's child).

A mechanism with a ring of bodies (a four-bar, a scissor, a gantry bridge on two
carriages) has more joints than a tree can hold. The extra joints are *loop
closures*: they do not place a body, they constrain the pose, and forward
kinematics solves the free tree joints until every closure is satisfied.
Name them with ``articulation(closures=[...])``, or let the assembly pick them:
joints are taken in authored order, and a joint whose two bodies are already
connected becomes a closure.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from sdk import kinematics
from sdk.body import Body, check_name
from sdk.errors import LoopClosureError, PoseError, ValidationError
from sdk.frames import Attachment, Frame
from sdk.joint import Joint, JointType, Mimic
from sdk.kinematics import LOOP_TOLERANCE, Articulation, Pose
from sdk.transforms import invert


class Assembly:
    def __init__(self, name: str) -> None:
        self.name = check_name(name, kind="assembly")
        self._bodies: dict[str, Body] = {}
        self._joints: dict[str, Joint] = {}
        self._root: str | None = None
        self._closures: tuple[str, ...] | None = None
        self._warm_start: dict[str, float] = {}  # solved loop joints of the last pose
        self._last_values: dict[str, float] = {}  # independent joint values of the last pose

    def __repr__(self) -> str:
        return f"Assembly({self.name!r}, bodies={list(self._bodies)}, joints={list(self._joints)})"

    # ---- authoring -------------------------------------------------------------

    def body(self, name: str) -> Body:
        if name in self._bodies:
            raise ValidationError(f"duplicate body name {name!r}")
        body = Body(name)
        self._bodies[name] = body
        return body

    def joint(
        self,
        name: str,
        type: JointType | str,
        parent: Attachment,
        child: Attachment,
        *,
        axis: tuple[float, float, float] | None = None,
        limits: tuple[float, float] | None = None,
        mimic: Mimic | tuple[str, float, float] | None = None,
    ) -> Joint:
        """Connect ``parent`` to ``child`` (both ``body.at(...)``).

        ``axis`` is in the parent attachment frame. ``limits`` are radians (revolute)
        or meters (prismatic); ``None`` means unlimited. ``mimic`` is
        ``(source_joint, multiplier, offset)``.
        """
        check_name(name, kind="joint")
        if name in self._joints:
            raise ValidationError(f"duplicate joint name {name!r}")
        joint = Joint(name, type, parent, child, axis=axis, limits=limits, mimic=mimic)
        for end in (joint.parent, joint.child):
            if self._bodies.get(end.body.name) is not end.body:
                raise ValidationError(
                    f"joint {name!r} uses body {end.body.name!r}, which is not in this assembly"
                )
        self._joints[name] = joint
        return joint

    def articulation(
        self, *, root: str | None = None, closures: Iterable[str] | None = None
    ) -> Articulation:
        """Choose the root and the loop-closure joints, then validate."""
        self._root = root
        self._closures = None if closures is None else tuple(closures)
        return self.validate()

    def frame_in(self, attachment: Attachment, body: Body) -> Attachment:
        """The same physical frame as ``attachment``, expressed on ``body``, in the rest pose.

        Use it to author a loop closure: pin the coupler tip to wherever the rocker
        will meet it, without working out the rocker's angles by hand. Only the joints
        that connect the two bodies need to exist already.
        """
        root_a, pose_a = self._rest_pose(attachment.body)
        root_b, pose_b = self._rest_pose(body)
        if root_a != root_b:
            raise ValidationError(
                f"bodies {attachment.body.name!r} and {body.name!r} are not connected by "
                "joints yet; author the joints that place them first"
            )
        return body.at(Frame.from_matrix(invert(pose_b) @ pose_a @ attachment.frame.matrix))

    def _rest_pose(self, body: Body) -> tuple[str, np.ndarray]:
        """(root name, rest transform of ``body`` relative to that root) from the joints so far."""
        if self._bodies.get(body.name) is not body:
            raise ValidationError(f"body {body.name!r} is not in this assembly")
        closures = set(self._pick_closures())
        parents: dict[str, Joint] = {}
        for joint in self._joints.values():
            if joint not in closures:
                parents.setdefault(joint.child.body.name, joint)
        transform, name = np.eye(4), body.name
        while name in parents:
            joint = parents[name]
            motion = joint.motion(self._rest_value(joint)) if joint.movable else np.eye(4)
            transform = (
                joint.parent.frame.matrix @ motion @ invert(joint.child.frame.matrix) @ transform
            )
            name = joint.parent.body.name
        return name, transform

    def _rest_value(self, joint: Joint, depth: int = 0) -> float:
        """0, or for a mimic its value when everything upstream is at 0."""
        if joint.mimic is None or depth > len(self._joints):
            return 0.0
        source = self._joints.get(joint.mimic.source)
        upstream = 0.0 if source is None else self._rest_value(source, depth + 1)
        return joint.mimic.multiplier * upstream + joint.mimic.offset

    @property
    def bodies(self) -> tuple[Body, ...]:
        return tuple(self._bodies.values())

    @property
    def joints(self) -> tuple[Joint, ...]:
        return tuple(self._joints.values())

    def get_body(self, name: str) -> Body:
        try:
            return self._bodies[name]
        except KeyError:
            raise ValidationError(f"unknown body {name!r}") from None

    def get_joint(self, name: str) -> Joint:
        try:
            return self._joints[name]
        except KeyError:
            raise ValidationError(f"unknown joint {name!r}") from None

    # ---- validation ------------------------------------------------------------

    def validate(self) -> Articulation:
        """Check the structure and return the resolved tree. Raises ``ValidationError``."""
        if not self._bodies:
            raise ValidationError(f"assembly {self.name!r} has no bodies")
        empty = [name for name, body in self._bodies.items() if not body.shapes]
        if empty:
            raise ValidationError(
                f"bodies with no shapes: {empty}; give every body geometry with body.add(...)"
            )
        closures = self._pick_closures()
        tree = [j for j in self._joints.values() if j not in closures]

        parent_joint: dict[str, Joint] = {}
        for joint in tree:
            child = joint.child.body.name
            if child in parent_joint:
                raise ValidationError(
                    f"body {child!r} is the child of both {parent_joint[child].name!r} and "
                    f"{joint.name!r}; make one of them a loop closure"
                )
            parent_joint[child] = joint

        roots = [name for name in self._bodies if name not in parent_joint]
        if self._root is not None:
            if self._root not in self._bodies:
                raise ValidationError(f"root {self._root!r} is not a body of this assembly")
            if self._root in parent_joint:
                raise ValidationError(
                    f"root {self._root!r} is the child of joint {parent_joint[self._root].name!r}"
                )
        if len(roots) != 1:
            raise ValidationError(
                f"assembly needs exactly one root, found {len(roots)}: {roots}. "
                "Every body except the root must be the child of a tree joint"
            )
        root = roots[0]

        order = _tree_order(root, tree)
        unreached = sorted(set(self._bodies) - {root} - {j.child.body.name for j in order})
        if unreached:
            raise ValidationError(f"bodies not reachable from root {root!r}: {unreached}")

        mimic_order = self._mimic_order(closures)
        self._check_mimic_ranges(mimic_order)
        art = Articulation(root, tuple(order), tuple(closures), parent_joint, mimic_order)
        self._check_rest_pose_closes(art)
        return art

    def _check_mimic_ranges(self, mimic_order: tuple[Joint, ...]) -> None:
        """A mimic must stay inside its own limits over its source's whole range."""
        reach: dict[str, tuple[float, float] | None] = {}
        for joint in mimic_order:
            source = self._joints[joint.mimic.source]
            source_range = reach[source.name] if source.mimic is not None else source.limits
            if source_range is None:
                span = None
            else:
                ends = [joint.mimic.multiplier * v + joint.mimic.offset for v in source_range]
                span = (min(ends), max(ends))
            reach[joint.name] = span
            if joint.limits is None:
                continue
            if span is None:
                raise ValidationError(
                    f"mimic {joint.name!r} has limits {joint.limits} but its source "
                    f"{source.name!r} is unlimited, so it can be driven past them"
                )
            lower, upper = joint.limits
            if span[0] < lower - 1e-9 or span[1] > upper + 1e-9:
                raise ValidationError(
                    f"mimic {joint.name!r} reaches ({span[0]:.6g}, {span[1]:.6g}) over the "
                    f"range of {source.name!r}, outside its limits {joint.limits}"
                )

    def _check_rest_pose_closes(self, art: Articulation) -> None:
        """With every joint at 0 (the authored pose), each loop closure must be satisfied."""
        if not art.closures:
            return
        values = {j.name: 0.0 for j in art.tree if j.movable and j.mimic is None}
        kinematics.apply_mimics(art, values)
        transforms = kinematics.place_bodies(art, values)
        for closure in art.closures:
            error = float(np.max(np.abs(kinematics.closure_error(closure, transforms))))
            if error <= LOOP_TOLERANCE:
                continue
            ends = [
                transforms[end.body.name] @ end.frame.matrix
                for end in (closure.parent, closure.child)
            ]
            gap = float(np.linalg.norm(ends[0][:3, 3] - ends[1][:3, 3]))
            raise ValidationError(
                f"loop closure {closure.name!r} is open in the rest pose (error {error:.3g}; "
                f"its two frames are {gap * 1000:.3g} mm apart). With every joint at 0, "
                f"{closure.parent!r} and {closure.child!r} must coincide. Build one end with "
                "assembly.frame_in(...) so the authored pose is closed"
            )

    def _pick_closures(self) -> list[Joint]:
        groups = _UnionFind()
        if self._closures is None:
            closures = []
            for joint in self._joints.values():
                if not groups.union(joint.parent.body.name, joint.child.body.name):
                    closures.append(joint)
            return closures

        closures = [self.get_joint(name) for name in self._closures]
        for joint in self._joints.values():
            if joint in closures:
                continue
            if not groups.union(joint.parent.body.name, joint.child.body.name):
                raise ValidationError(
                    f"tree joints form a loop at {joint.name!r}; list one joint of that loop "
                    "in closures"
                )
        return closures

    def _mimic_order(self, closures: list[Joint]) -> tuple[Joint, ...]:
        ordered: list[Joint] = []
        state: dict[str, str] = {}  # "visiting" or "done"

        def visit(joint: Joint, chain: list[str]) -> None:
            if state.get(joint.name) == "done":
                return
            if state.get(joint.name) == "visiting":
                cycle = " -> ".join([*chain, joint.name])
                raise ValidationError(f"mimic cycle: {cycle}")
            state[joint.name] = "visiting"
            if joint.mimic is not None:
                visit(self._mimic_source(joint, closures), [*chain, joint.name])
                ordered.append(joint)
            state[joint.name] = "done"

        for joint in self._joints.values():
            visit(joint, [])
        return tuple(ordered)

    def _mimic_source(self, joint: Joint, closures: list[Joint]) -> Joint:
        name = joint.mimic.source
        source = self._joints.get(name)
        if source is None:
            raise ValidationError(f"joint {joint.name!r} mimics {name!r}, which does not exist")
        if source.type is JointType.FIXED:
            raise ValidationError(f"joint {joint.name!r} mimics fixed joint {name!r}")
        if joint in closures or source in closures:
            raise ValidationError(
                f"mimic {joint.name!r} <- {name!r}: loop-closure joints cannot mimic or be mimicked"
            )
        return source

    # ---- kinematics ------------------------------------------------------------

    def solve(
        self,
        joint_values: Mapping[str, float] | None = None,
        *,
        tolerance: float = LOOP_TOLERANCE,
        warm_start: Mapping[str, float] | None = None,
    ) -> Pose:
        """Full pose: body transforms plus every joint value (driven, mimic, solved).

        Joints you do not drive stay at 0, the authored rest pose (mimic joints follow
        their source), except loop joints, which the solver moves.

        A loop usually has two or more solutions for one input (a four-bar can close
        "elbow up" or "elbow down"). To stay on the branch the mechanism is on, the
        solver walks from the last solved pose (or the rest pose) to the requested one
        in small steps, warm-starting each step from the previous one. Pass
        ``warm_start`` to skip the walk and start the solver from your own guess.
        """
        art = self.validate()
        driven = self._check_driven(art, joint_values or {})
        values = {j.name: 0.0 for j in art.tree if j.movable and j.mimic is None}
        values.update(driven)
        unknowns = kinematics.loop_unknowns(art, set(driven), self._joints)

        if warm_start is not None:
            seed = dict(warm_start)
        else:
            seed = dict(self._warm_start)
            start = {name: self._last_values.get(name, 0.0) for name in values}
            for fraction in _walk_fractions(self._joints, start, values) if unknowns else ():
                step = {k: start[k] + (values[k] - start[k]) * fraction for k in values}
                try:
                    seed = kinematics.solve(art, step, unknowns, seed, tolerance).joint_values
                except LoopClosureError:
                    break  # an unreachable midpoint: finish with a direct solve

        pose = kinematics.solve(art, values, unknowns, seed, tolerance)
        self._warm_start.update({j.name: pose.joint_values[j.name] for j in unknowns})
        self._last_values = values
        return pose

    def forward_kinematics(
        self, joint_values: Mapping[str, float] | None = None
    ) -> dict[str, np.ndarray]:
        """Body name -> 4x4 world transform for the given joint values."""
        return self.solve(joint_values).transforms

    def _check_driven(self, art: Articulation, joint_values: Mapping[str, Any]) -> dict[str, float]:
        driven: dict[str, float] = {}
        for name, raw in joint_values.items():
            joint = self._joints.get(name)
            if joint is None:
                raise PoseError(f"unknown joint {name!r}; joints are {list(self._joints)}")
            if not joint.movable:
                raise PoseError(f"joint {name!r} is fixed and takes no value")
            if joint.mimic is not None:
                raise PoseError(f"joint {name!r} mimics {joint.mimic.source!r}; drive that instead")
            if joint in art.closures:
                raise PoseError(f"joint {name!r} closes a loop; its value comes from the solver")
            value = float(raw)
            if not np.isfinite(value):
                raise PoseError(f"joint {name!r} value must be finite, got {raw!r}")
            if not joint.within_limits(value):
                raise PoseError(f"joint {name!r} value {value:.6g} is outside {joint.limits}")
            driven[name] = value
        return driven


WALK_STEP_RADIANS = 0.1
WALK_STEP_METERS = 0.01
MAX_WALK_STEPS = 200


def _walk_fractions(
    joints: Mapping[str, Joint], start: Mapping[str, float], target: Mapping[str, float]
) -> list[float]:
    """Intermediate fractions (excluding 1.0) so no joint moves more than one step at a time."""
    steps = 1
    for name, value in target.items():
        joint = joints[name]
        size = WALK_STEP_RADIANS if joint.type is JointType.REVOLUTE else WALK_STEP_METERS
        steps = max(steps, math.ceil(abs(value - start[name]) / size))
    steps = min(steps, MAX_WALK_STEPS)
    return [k / steps for k in range(1, steps)]


def _tree_order(root: str, tree: list[Joint]) -> list[Joint]:
    """Breadth-first from the root, so every parent is placed before its child."""
    children: dict[str, list[Joint]] = {}
    for joint in tree:
        children.setdefault(joint.parent.body.name, []).append(joint)
    order: list[Joint] = []
    frontier = [root]
    while frontier:
        body = frontier.pop(0)
        for joint in children.get(body, []):
            order.append(joint)
            frontier.append(joint.child.body.name)
    return order


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: str, b: str) -> bool:
        """Join two groups; False if they were already joined (the edge makes a loop)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[ra] = rb
        return True
