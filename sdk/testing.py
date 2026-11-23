# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Checks: the authored test API (``TestContext``) and the baseline checks every compile runs.

    def run_tests(ctx):
        ctx.allow_overlap("arm", "head", reason="the arm tube is pushed into the socket",
                          shape_a="tube", shape_b="socket")
        ctx.expect_contact("lid", "base")
        ctx.expect_no_collision_at_poses([{"hinge": 1.0}, {"hinge": 2.0}])

Baseline checks (``run_checks``):

| Check | What it finds | Blocks the compile? |
| --- | --- | --- |
| mesh health | non-manifold edges, inconsistent or inside-out winding | yes |
| mesh health | degenerate triangles, tiny debris pieces, open boundaries | warning |
| overlap at rest | two bodies interpenetrate | yes, unless allowed |
| floating parts | a body not connected to the root by contacts | yes, unless allowed |
| swept collision | every drivable joint is swept over its range (revolute and prismatic), coarse then refined near contacts | yes, unless allowed |
| combined poses | all drivers at their range corners together (up to 64) plus 20 seeded random poses | yes, unless allowed |
| loop reach | a loop that cannot close somewhere in its driver's range | yes |
| joint separation | the gap at a hinge grows as it turns | yes |
| islands | shapes in one body that do not touch | warning |
| dimensions | absurd sizes and scale outliers | warning |

**Tolerances scale with the object:** ``max(floor, fraction x size)``, where size is the
diagonal of the rest-pose bounding box. All thresholds live in ``CheckConfig``.

**Collision pipeline:**
1. An AABB pre-check picks the shape pairs that could touch.
2. trimesh's ``CollisionManager`` (python-fcl) says which of them do.
3. manifold3d intersects each colliding pair to measure the penetration (millimetres) and
   the overlap volume.
4. Gaps come from fcl distance queries between bodies.
"""

from __future__ import annotations

import inspect
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from sdk.errors import PoseError, ValidationError
from sdk.joint import Joint, JointType
from sdk.kinematics import Pose, loop_path

# ---- configuration ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckConfig:
    """Every threshold the checks use. Lengths in meters, angles in radians."""

    contact_floor: float = 1e-4
    """Bodies closer than max(contact_floor, contact_fraction x size) are in contact."""
    contact_fraction: float = 1e-3
    overlap_floor: float = 2e-4
    """A penetration deeper than max(overlap_floor, overlap_fraction x size) is an overlap."""
    overlap_fraction: float = 2e-3
    separation_floor: float = 5e-4
    """A hinge gap may grow by at most max(separation_floor, separation_fraction x size)."""
    separation_fraction: float = 5e-3
    refine_fraction: float = 2e-2
    """Sweep intervals where bodies come within this x size (and were apart at rest) are
    sampled more finely."""
    debris_fraction: float = 0.02
    """A connected piece of a shape smaller than this x the shape's size is debris."""
    degenerate_area_fraction: float = 1e-10
    """Triangles with area below this x size^2 are degenerate."""
    max_dimension: float = 50.0
    """Shapes larger than this (m) get an absurd-size warning."""
    outlier_ratio: float = 100.0
    """Shapes more than this times the median shape size get a scale warning."""
    revolute_step: float = math.radians(15.0)
    """Coarse sweep spacing for hinges."""
    prismatic_step_fraction: float = 0.05
    """Coarse sweep spacing for slides, as a fraction of the object size."""
    min_samples: int = 5
    max_samples: int = 25
    refine_depth: int = 6
    """Bisection steps to locate where a collision starts."""
    unlimited_revolute: tuple[float, float] = (-math.pi, math.pi)
    max_corners: int = 64
    """Joints moving together: at most this many min/max corners of the joint ranges."""
    random_poses: int = 20
    """... plus this many random poses."""
    random_seed: int = 0

    def tolerance(self, kind: str, size: float) -> float:
        floor = getattr(self, f"{kind}_floor")
        return max(floor, getattr(self, f"{kind}_fraction") * size)


DEFAULT_CONFIG = CheckConfig()
_SLIVER_VOLUME = 1e-15
"""Intersection pieces below this volume (m^3) are touching faces, not penetration."""


# ---- results ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Allowance:
    kind: str
    """"overlap" or "isolated"."""
    bodies: tuple[str, ...]
    reason: str
    shapes: tuple[str | None, str | None] = (None, None)

    def covers_overlap(self, a: tuple[str, str], b: tuple[str, str]) -> bool:
        if self.kind != "overlap":
            return False
        for (body_a, shape_a), (body_b, shape_b) in ((a, b), (b, a)):
            if (body_a, body_b) == self.bodies and self.shapes[0] in (None, shape_a) \
                    and self.shapes[1] in (None, shape_b):
                return True
        return False

    def describe(self) -> str:
        if self.kind == "isolated":
            return f"{self.bodies[0]!r} may float: {self.reason}"
        names = [f"{b}/{s}" if s else b for b, s in zip(self.bodies, self.shapes, strict=True)]
        return f"{names[0]!r} may overlap {names[1]!r}: {self.reason}"


@dataclass(frozen=True)
class Finding:
    check: str
    message: str
    blocking: bool
    bodies: tuple[str, ...] = ()
    shapes: tuple[str, ...] = ()
    joint: str | None = None
    pose: dict[str, float] | None = None
    """Driven joint values where the problem happens."""
    amount: float | None = None
    """Meters: penetration depth, gap, or gap growth."""
    allowed: str | None = None
    """The allowance reason, when one covers this finding (it then does not block)."""


@dataclass
class CheckReport:
    findings: list[Finding]
    allowances: list[Allowance]
    timings: dict[str, float]
    size: float
    tolerances: dict[str, float]

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.blocking]

    @property
    def passed(self) -> bool:
        return not self.blocking

    def text(self) -> str:
        lines = [f"object size {self.size * 1000:.1f} mm; tolerances "
                 + ", ".join(f"{k} {v * 1000:.2f} mm" for k, v in self.tolerances.items())]
        for f in self.findings:
            tag = "FAIL" if f.blocking else ("allowed" if f.allowed else "warn")
            lines.append(f"[{tag}] {f.check}: {f.message}")
        lines += [f"[allowance] {a.describe()}" for a in self.allowances]
        return "\n".join(lines)


# ---- geometry scene --------------------------------------------------------------------------


@dataclass
class _Shape:
    key: tuple[str, str]
    mesh: trimesh.Trimesh  # body coordinates
    corners: np.ndarray  # 8 x 3 local bounding-box corners
    solid: Any  # manifold3d.Manifold, or None if the mesh is not closed
    fcl_object: Any


class _Scene:
    """Collision and distance queries for one assembly at changing poses."""

    def __init__(self, model) -> None:
        import fcl

        from sdk.booleans import to_manifold

        self.model = model
        self.manager = trimesh.collision.CollisionManager()
        self.shapes: dict[tuple[str, str], _Shape] = {}
        self.bodies: dict[str, Any] = {}
        self.body_of: dict[str, list[tuple[str, str]]] = {}
        for body in model.bodies:
            meshes = []
            for shape in body.shapes:
                mesh = shape.to_trimesh()
                mesh.merge_vertices()
                meshes.append(mesh)
                key = (body.name, shape.name)
                solid = None
                if mesh.is_watertight:
                    try:
                        solid = to_manifold(mesh)
                    except ValidationError:
                        solid = None
                self.manager.add_object(self._name(key), mesh)
                low, high = mesh.bounds
                corners = np.array([[x, y, z] for x in (low[0], high[0])
                                    for y in (low[1], high[1]) for z in (low[2], high[2])])
                bvh = trimesh.collision.mesh_to_BVH(mesh)
                self.shapes[key] = _Shape(key, mesh, corners, solid,
                                          fcl.CollisionObject(bvh, fcl.Transform()))
                self.body_of.setdefault(body.name, []).append(key)
            combined = trimesh.util.concatenate(meshes)
            self.bodies[body.name] = fcl.CollisionObject(trimesh.collision.mesh_to_BVH(combined),
                                                         fcl.Transform())
        self.transforms: dict[str, np.ndarray] = {}
        self.aabb: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _name(key: tuple[str, str]) -> str:
        return f"{key[0]}/{key[1]}"

    def place(self, transforms: Mapping[str, np.ndarray]) -> None:
        import fcl

        self.transforms = dict(transforms)
        for name, matrix in transforms.items():
            fcl_transform = fcl.Transform(matrix[:3, :3], matrix[:3, 3])
            self.bodies[name].setTransform(fcl_transform)
            for key in self.body_of[name]:
                shape = self.shapes[key]
                self.manager.set_transform(self._name(key), matrix)
                shape.fcl_object.setTransform(fcl_transform)
                world = shape.corners @ matrix[:3, :3].T + matrix[:3, 3]
                self.aabb[key] = (world.min(axis=0), world.max(axis=0))

    def candidate_pairs(self, margin: float, moved: set[str] | None = None):
        """Shape pairs of different bodies whose world boxes overlap (grown by ``margin``)."""
        keys = list(self.shapes)
        pairs = []
        for i, a in enumerate(keys):
            low_a, high_a = self.aabb[a]
            for b in keys[i + 1:]:
                if a[0] == b[0] or (moved is not None and a[0] not in moved and b[0] not in moved):
                    continue
                low_b, high_b = self.aabb[b]
                if np.all(low_a - margin <= high_b) and np.all(low_b - margin <= high_a):
                    pairs.append((a, b))
        return pairs

    def collisions(self, pairs) -> list[tuple[tuple[str, str], tuple[str, str], float, float]]:
        """(shape a, shape b, penetration depth, overlap volume) for colliding pairs."""
        if not pairs:
            return []
        hit, names = self.manager.in_collision_internal(return_names=True)
        if not hit:
            return []
        colliding = {frozenset(pair) for pair in names}
        out = []
        for a, b in pairs:
            if frozenset((self._name(a), self._name(b))) in colliding:
                depth, volume = self._penetration(a, b)
                out.append((a, b, depth, volume))
        return out

    def _penetration(self, a, b) -> tuple[float, float]:
        sa, sb = self.shapes[a], self.shapes[b]
        if sa.solid is not None and sb.solid is not None:
            ta, tb = self.transforms[a[0]], self.transforms[b[0]]
            common = sa.solid.transform(ta[:3, :4]) ^ sb.solid.transform(tb[:3, :4])
            if common.is_empty():
                return 0.0, 0.0
            # Measure each separate piece of the intersection: two faces that merely touch
            # in two places give two zero-volume slivers, whose joint box would look thick.
            depth = 0.0
            for piece in common.decompose():
                if piece.volume() < _SLIVER_VOLUME:
                    continue
                low_x, low_y, low_z, high_x, high_y, high_z = piece.bounding_box()
                depth = max(depth, min(high_x - low_x, high_y - low_y, high_z - low_z))
            return float(depth), float(common.volume())
        return self._contact_depth(sa, sb), 0.0

    @staticmethod
    def _contact_depth(sa: _Shape, sb: _Shape) -> float:
        import fcl

        request = fcl.CollisionRequest(num_max_contacts=64, enable_contact=True)
        result = fcl.CollisionResult()
        fcl.collide(sa.fcl_object, sb.fcl_object, request, result)
        return max((c.penetration_depth for c in result.contacts), default=0.0)

    def distance(self, body_a: str, body_b: str) -> float:
        import fcl

        value = fcl.distance(self.bodies[body_a], self.bodies[body_b], fcl.DistanceRequest(),
                             fcl.DistanceResult())
        return max(0.0, float(value))

    def shape_distance(self, a, b) -> float:
        import fcl

        value = fcl.distance(self.shapes[a].fcl_object, self.shapes[b].fcl_object,
                             fcl.DistanceRequest(), fcl.DistanceResult())
        return max(0.0, float(value))


# ---- the context ----------------------------------------------------------------------------


def _name(item: Any) -> str:
    return item if isinstance(item, str) else item.name


class TestContext:
    """What ``run_tests(ctx)`` receives. Expectations record findings (they do not raise),
    and allowances are handed to the baseline checks."""

    __test__ = False  # not a pytest class

    def __init__(self, model, config: CheckConfig = DEFAULT_CONFIG) -> None:
        self.model = model
        self.config = config
        self.allowances: list[Allowance] = []
        self.findings: list[Finding] = []
        self._scene: _Scene | None = None
        self._size: float | None = None

    # -- allowances --

    def allow_overlap(self, a: Any, b: Any, reason: str, *, shape_a: str | None = None,
                      shape_b: str | None = None) -> None:
        """Let two bodies (or two named shapes) interpenetrate, for a stated reason."""
        a, b = _name(a), _name(b)
        self._require_reason(reason, f"allow_overlap({a!r}, {b!r})")
        for body, shape in ((a, shape_a), (b, shape_b)):
            target = self.model.get_body(body)
            if shape is not None:
                target.shape(shape)
        self.allowances.append(Allowance("overlap", (a, b), reason.strip(), (shape_a, shape_b)))

    def allow_isolated(self, body: Any, reason: str) -> None:
        """Let a body float (not touch the rest), for a stated reason."""
        body = _name(body)
        self._require_reason(reason, f"allow_isolated({body!r})")
        self.model.get_body(body)
        self.allowances.append(Allowance("isolated", (body,), reason.strip()))

    @staticmethod
    def _require_reason(reason: object, call: str) -> None:
        if not isinstance(reason, str) or len(reason.strip()) < 3:
            raise ValidationError(f"{call} needs a reason, e.g. reason='the pin passes through "
                                  "the knuckle'")

    # -- posing helpers --

    @property
    def scene(self) -> _Scene:
        if self._scene is None:
            self._scene = _Scene(self.model)
        return self._scene

    @property
    def size(self) -> float:
        if self._size is None:
            rest = self.model.solve({}, warm_start={})
            self.scene.place(rest.transforms)
            boxes = np.array([corner for low, high in self.scene.aabb.values()
                              for corner in (low, high)])
            self._size = float(np.linalg.norm(boxes.max(axis=0) - boxes.min(axis=0)))
        return self._size

    def _pose(self, pose: Mapping[str, float] | None) -> Pose:
        result = self.model.solve(dict(pose or {}))
        self.scene.place(result.transforms)
        return result

    def sample_joint(self, joint: Any, samples: int = 5) -> list[tuple[float, Pose]]:
        """``samples`` evenly spaced values over a joint's range, with the solved pose."""
        joint = self.model.get_joint(_name(joint))
        low, high = _joint_range(joint, self.config, self.size)
        return [(float(v), self.model.solve({joint.name: float(v)}))
                for v in np.linspace(low, high, max(2, samples))]

    def track_point(self, body: Any, point: Sequence[float], joint: Any,
                    samples: int = 5) -> list[tuple[float, np.ndarray]]:
        """World positions of a body-local point as a joint moves over its range."""
        body = _name(body)
        local = np.asarray(point, dtype=float)
        return [(value, pose.transforms[body][:3, :3] @ local + pose.transforms[body][:3, 3])
                for value, pose in self.sample_joint(joint, samples)]

    # -- expectations --

    def check(self, name: str, ok: bool, details: str = "") -> bool:
        if not ok:
            self.findings.append(Finding("authored", f"{name}: {details}" if details else name,
                                         True))
        return bool(ok)

    def fail(self, name: str, details: str) -> bool:
        return self.check(name, False, details)

    def expect_contact(self, a: Any, b: Any, *, pose: Mapping[str, float] | None = None,
                       name: str | None = None) -> bool:
        a, b = _name(a), _name(b)
        self._pose(pose)
        gap = self.scene.distance(a, b)
        limit = self.config.tolerance("contact", self.size)
        return self._expect(name or f"expect_contact({a!r}, {b!r})", gap <= limit,
                            f"gap is {gap * 1000:.2f} mm (contact means <= {limit * 1000:.2f} mm)",
                            (a, b), pose, gap)

    def expect_gap(self, a: Any, b: Any, min_gap: float = 0.0, max_gap: float | None = None, *,
                   pose: Mapping[str, float] | None = None, name: str | None = None) -> bool:
        a, b = _name(a), _name(b)
        self._pose(pose)
        gap = self.scene.distance(a, b)
        ok = gap >= min_gap and (max_gap is None or gap <= max_gap)
        wanted = f">= {min_gap * 1000:.2f} mm" + (f" and <= {max_gap * 1000:.2f} mm"
                                                   if max_gap is not None else "")
        return self._expect(name or f"expect_gap({a!r}, {b!r})", ok,
                            f"gap is {gap * 1000:.2f} mm, expected {wanted}", (a, b), pose, gap)

    def expect_distance(self, point_a: tuple[Any, Sequence[float]],
                        point_b: tuple[Any, Sequence[float]], distance: float, *,
                        tolerance: float | None = None, pose: Mapping[str, float] | None = None,
                        name: str | None = None) -> bool:
        """Distance between two body-local points, e.g. ``(("lid", (0.16, -0.22, 0)),
        ("base", (0.16, -0.11, 0)), 0.2)``."""
        result = self._pose(pose)
        (body_a, local_a), (body_b, local_b) = point_a, point_b
        ta, tb = result.transforms[_name(body_a)], result.transforms[_name(body_b)]
        world_a = ta[:3, :3] @ np.asarray(local_a, float) + ta[:3, 3]
        world_b = tb[:3, :3] @ np.asarray(local_b, float) + tb[:3, 3]
        actual = float(np.linalg.norm(world_a - world_b))
        tolerance = self.config.tolerance("contact", self.size) if tolerance is None else tolerance
        return self._expect(name or f"expect_distance({_name(body_a)!r}, {_name(body_b)!r})",
                            abs(actual - distance) <= tolerance,
                            f"distance is {actual * 1000:.2f} mm, expected {distance * 1000:.2f} "
                            f"+- {tolerance * 1000:.2f} mm", (_name(body_a), _name(body_b)), pose,
                            actual)

    def expect_no_collision_at_poses(self, poses: Iterable[Mapping[str, float]], *,
                                     bodies: Iterable[Any] | None = None,
                                     name: str | None = None) -> bool:
        wanted = None if bodies is None else {_name(b) for b in bodies}
        limit = self.config.tolerance("overlap", self.size)
        ok = True
        for pose in poses:
            self._pose(pose)
            for a, b, depth, _volume in self.scene.collisions(self.scene.candidate_pairs(0.0)):
                if depth <= limit or (wanted and not {a[0], b[0]} & wanted):
                    continue
                if any(al.covers_overlap(a, b) for al in self.allowances):
                    continue
                ok = self._expect(name or "expect_no_collision_at_poses", False,
                                  f"{a[0]!r}/{a[1]} hits {b[0]!r}/{b[1]} by {depth * 1000:.2f} mm "
                                  f"at {_format_pose(pose, self.model)}", (a[0], b[0]),
                                  pose, depth)
        return ok

    def _expect(self, name, ok, details, bodies, pose, amount) -> bool:
        if not ok:
            self.findings.append(Finding("authored", f"{name}: {details}", True, tuple(bodies),
                                         pose=dict(pose) if pose else None, amount=amount))
        return bool(ok)


# ---- baseline checks ---------------------------------------------------------------------------


def run_checks(model, ctx: TestContext | None = None,
               config: CheckConfig | None = None) -> CheckReport:
    """All baseline checks, plus the authored findings and allowances in ``ctx``."""
    ctx = ctx or TestContext(model, config or DEFAULT_CONFIG)
    config = ctx.config
    timings: dict[str, float] = {}
    findings: list[Finding] = list(ctx.findings)
    used: set[int] = set()

    def timed(name: str, function: Callable[[], list[Finding]]) -> None:
        start = time.perf_counter()
        findings.extend(function())
        timings[name] = round(time.perf_counter() - start, 4)

    scene, size = ctx.scene, ctx.size
    rest = model.solve({}, warm_start={})
    tolerances = {kind: config.tolerance(kind, size)
                  for kind in ("contact", "overlap", "separation")}
    checker = _Checker(model, ctx, scene, rest, size, tolerances, used)
    timed("mesh_health", checker.mesh_health)
    timed("dimensions", checker.dimensions)
    timed("islands", checker.islands)
    timed("overlap_at_rest", checker.overlap_at_rest)
    timed("floating", checker.floating)
    timed("sweeps", checker.sweeps)
    timed("combined", checker.combinations)
    for index, allowance in enumerate(ctx.allowances):
        if index not in used:
            findings.append(Finding("allowance", f"allowance not needed: {allowance.describe()}",
                                    False, allowance.bodies))
    return CheckReport(findings, list(ctx.allowances), timings, size, tolerances)


def drivable_joints(model) -> tuple[dict[str, Joint], dict[str, str]]:
    """The joints a person (or a sweep) drives: movable tree joints that are not mimics.
    Of the joints on a loop, only the first one authored drives it; the solver moves the
    rest. Returns (drivers by name, loop member name -> its driver's name)."""
    art = model.validate()
    by_name = {j.name: j for j in model.joints}
    order = [j.name for j in model.joints]
    loop_members: dict[str, str] = {}
    for closure in art.closures:
        sources: list[Joint] = []
        for joint in loop_path(art, closure):
            while joint.mimic is not None:
                joint = by_name[joint.mimic.source]
            if joint.movable and joint not in sources:
                sources.append(joint)
        sources.sort(key=lambda j: order.index(j.name))
        for joint in sources:
            loop_members.setdefault(joint.name, sources[0].name)
    drivers = {joint.name: joint for joint in art.tree
               if joint.movable and joint.mimic is None
               and loop_members.get(joint.name, joint.name) == joint.name}
    return drivers, loop_members


def _joint_range(joint: Joint, config: CheckConfig, size: float) -> tuple[float, float]:
    if joint.limits is not None:
        return joint.limits
    if joint.type is JointType.REVOLUTE:
        return config.unlimited_revolute
    raise ValidationError(f"joint {joint.name!r} is an unlimited slide; give it limits so it "
                          "can be checked")


def _format_value(joint: Joint, value: float) -> str:
    if joint.type is JointType.REVOLUTE:
        return f"{math.degrees(value):.1f} deg"
    return f"{value * 1000:.1f} mm"


def _format_pose(pose: Mapping[str, float], model) -> str:
    parts = []
    for name, value in pose.items():
        parts.append(f"{name}={_format_value(model.get_joint(name), value)}")
    return ", ".join(parts) or "the rest pose"


@dataclass
class _PoseEval:
    value: float
    transforms: dict[str, np.ndarray] | None
    collisions: list = field(default_factory=list)
    """(shape a, shape b, depth, volume, allowance reason) deeper than the overlap tolerance."""
    raw: dict[frozenset, float] = field(default_factory=dict)
    """Penetration depth of every interpenetrating pair, however shallow (for first contact)."""
    near: bool = False
    loop_error: str | None = None
    gaps: dict[str, float] = field(default_factory=dict)


class _Checker:
    def __init__(self, model, ctx, scene, rest, size, tolerances, used) -> None:
        self.model, self.ctx, self.scene, self.rest = model, ctx, scene, rest
        self.size, self.tol, self.used = size, tolerances, used
        self.config: CheckConfig = ctx.config
        self.rest_overlaps: set[frozenset] = set()
        self.rest_contacts: set[frozenset] = set()
        self.reported_pairs: set[frozenset] = set()
        self.drivers: dict[str, Joint] = {}

    # -- allowances --

    def _overlap_allowance(self, a, b) -> str | None:
        for index, allowance in enumerate(self.ctx.allowances):
            if allowance.covers_overlap(a, b):
                self.used.add(index)
                return allowance.reason
        return None

    def _isolated_allowance(self, body: str) -> str | None:
        for index, allowance in enumerate(self.ctx.allowances):
            if allowance.kind == "isolated" and allowance.bodies == (body,):
                self.used.add(index)
                return allowance.reason
        return None

    # -- mesh health, dimensions, islands --

    def mesh_health(self) -> list[Finding]:
        out = []
        area_limit = self.config.degenerate_area_fraction * self.size ** 2
        for (body, name), shape in self.scene.shapes.items():
            mesh = shape.mesh
            label = f"{body!r}/{name}"
            edges = mesh.edges_sorted
            _, counts = np.unique(edges, axis=0, return_counts=True)
            if np.any(counts > 2):
                out.append(Finding("mesh_health", f"{label} has {int(np.sum(counts > 2))} "
                                   "non-manifold edges (shared by 3+ triangles)", True,
                                   (body,), (name,)))
            if not mesh.is_winding_consistent:
                out.append(Finding("mesh_health", f"{label} has inconsistent triangle winding",
                                   True, (body,), (name,)))
            elif mesh.is_watertight and mesh.volume < 0:
                out.append(Finding("mesh_health", f"{label} is inside out (negative volume)",
                                   True, (body,), (name,)))
            degenerate = int(np.sum(mesh.area_faces < area_limit))
            if degenerate:
                out.append(Finding("mesh_health", f"{label} has {degenerate} degenerate "
                                   "(zero-area) triangles", False, (body,), (name,)))
            if np.any(counts == 1):
                out.append(Finding("mesh_health", f"{label} is open ({int(np.sum(counts == 1))} "
                                   "boundary edges); it cannot have a volume", False, (body,),
                                   (name,)))
            out += self._debris(mesh, body, name)
        return out

    def _debris(self, mesh, body: str, name: str) -> list[Finding]:
        pieces = mesh.split(only_watertight=False)
        if len(pieces) < 2:
            return []
        overall = float(np.linalg.norm(mesh.extents))
        small = [p for p in pieces if np.linalg.norm(p.extents) < self.config.debris_fraction
                 * overall]
        if not small:
            return []
        return [Finding("mesh_health", f"{body!r}/{name} has {len(small)} tiny debris piece(s) "
                        f"(< {self.config.debris_fraction:.0%} of the shape)", False, (body,),
                        (name,))]

    def dimensions(self) -> list[Finding]:
        self.scene.place(self.rest.transforms)
        spans = {key: float(np.max(high - low)) for key, (low, high) in self.scene.aabb.items()}
        median = float(np.median([s for s in spans.values() if s > 0])) if spans else 0.0
        out = []
        for (body, name), span in spans.items():
            if span > self.config.max_dimension:
                out.append(Finding("dimensions", f"{body!r}/{name} spans {span:.3g} m; check the "
                                   "units (the SDK uses meters)", False, (body,), (name,), amount=span))
            elif median > 0 and span > self.config.outlier_ratio * median:
                out.append(Finding("dimensions", f"{body!r}/{name} spans {span:.3g} m, "
                                   f"{span / median:.0f}x the median shape ({median:.3g} m)",
                                   False, (body,), (name,), amount=span))
        return out

    def islands(self) -> list[Finding]:
        out = []
        limit = self.tol["contact"]
        for body, keys in self.scene.body_of.items():
            if len(keys) < 2:
                continue
            groups = _components(keys, lambda a, b: self.scene.shape_distance(a, b) <= limit)
            if len(groups) > 1:
                names = [sorted(k[1] for k in g) for g in groups]
                out.append(Finding("islands", f"body {body!r} has {len(groups)} separate pieces "
                                   f"{names}; shapes of one body should touch", False, (body,)))
        return out

    # -- rest pose --

    def overlap_at_rest(self) -> list[Finding]:
        self.scene.place(self.rest.transforms)
        out = []
        for a, b, depth, volume in self.scene.collisions(self.scene.candidate_pairs(0.0)):
            if depth <= self.tol["overlap"]:
                continue
            self.rest_overlaps.add(frozenset((a, b)))
            reason = self._overlap_allowance(a, b)
            message = (f"{a[0]!r}/{a[1]} and {b[0]!r}/{b[1]} overlap by {depth * 1000:.2f} mm "
                       f"({volume * 1e6:.3g} cm3) in the rest pose")
            out.append(Finding("overlap", message, reason is None, (a[0], b[0]), (a[1], b[1]),
                               pose={}, amount=depth, allowed=reason))
        return out

    def floating(self) -> list[Finding]:
        self.scene.place(self.rest.transforms)
        names = [b.name for b in self.model.bodies]
        gaps = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                gaps[frozenset((a, b))] = self.scene.distance(a, b)
        touching = {pair for pair, gap in gaps.items() if gap <= self.tol["contact"]}
        self.rest_contacts = touching
        root = self.model.validate().root
        reached = _reach(root, names, touching)
        out = []
        for group in _components([n for n in names if n not in reached],
                                 lambda a, b: frozenset((a, b)) in touching):
            nearest = min(((gaps[frozenset((g, r))], g, r) for g in group for r in reached),
                          default=None)
            reasons = [self._isolated_allowance(body) for body in group]
            allowed = "; ".join(r for r in reasons if r) if all(reasons) else None
            detail = (f"; nearest connected body is {nearest[2]!r}, {nearest[0] * 1000:.2f} mm "
                      f"from {nearest[1]!r}") if nearest else ""
            out.append(Finding("floating", f"{sorted(group)} not connected to the root "
                               f"{root!r} by contact{detail}", allowed is None, tuple(sorted(group)),
                               pose={}, amount=nearest[0] if nearest else None, allowed=allowed))
        return out

    # -- sweeps --

    def _drivers(self) -> dict[str, Joint]:
        drivers, self.loop_driver = drivable_joints(self.model)
        return drivers

    def sweeps(self) -> list[Finding]:
        out = []
        art = self.model.validate()
        drivers = self.drivers = self._drivers()
        by_name = {j.name: j for j in self.model.joints}
        hinges_by_driver: dict[str, list[Joint]] = {}
        for joint in art.tree:
            if joint.type is not JointType.REVOLUTE:
                continue
            source = joint
            while source.mimic is not None:
                source = by_name[source.mimic.source]
            driver = self.loop_driver.get(source.name, source.name)
            if driver in drivers:
                hinges_by_driver.setdefault(driver, []).append(joint)
        for joint in drivers.values():
            try:
                low, high = _joint_range(joint, self.config, self.size)
            except ValidationError as exc:
                out.append(Finding("swept_collision", str(exc), False, joint=joint.name))
                continue
            out += self._sweep(joint, low, high, hinges_by_driver.get(joint.name, []))
        return out

    def _sweep(self, joint: Joint, low: float, high: float, hinges: list[Joint]) -> list[Finding]:
        step = (self.config.revolute_step if joint.type is JointType.REVOLUTE
                else self.config.prismatic_step_fraction * self.size)
        count = int(np.clip(math.ceil((high - low) / step) + 1, self.config.min_samples,
                            self.config.max_samples))
        values = sorted({*np.linspace(low, high, count).tolist(), 0.0})
        evals = {v: self._evaluate(joint, v, hinges) for v in values}
        for a, b in zip(values[:-1], values[1:], strict=True):
            self._refine(joint, hinges, evals, a, b, self.config.refine_depth, allow_near=True)
        return self._summarize(joint, hinges, [evals[v] for v in sorted(evals)])

    def _evaluate(self, joint: Joint, value: float, hinges: list[Joint]) -> _PoseEval:
        try:
            pose = self.model.solve({joint.name: float(value)})
        except PoseError as exc:
            return _PoseEval(value, None, loop_error=str(exc))
        self.scene.place(pose.transforms)
        moved = {name for name, matrix in pose.transforms.items()
                 if not np.allclose(matrix, self.rest.transforms[name], atol=1e-12)}
        result = _PoseEval(value, pose.transforms)
        for hinge in hinges:  # also at the rest pose: it is the reference gap
            result.gaps[hinge.name] = self.scene.distance(hinge.parent.body.name,
                                                          hinge.child.body.name)
        if not moved:
            return result
        margin = self.config.refine_fraction * self.size
        near_pairs = self.scene.candidate_pairs(margin, moved)
        touching_pairs = [p for p in near_pairs if all(
            low_a <= high_b and low_b <= high_a for low_a, high_a, low_b, high_b in zip(
                *self.scene.aabb[p[0]], *self.scene.aabb[p[1]], strict=True))]
        for a, b, depth, volume in self.scene.collisions(touching_pairs):
            pair = frozenset((a, b))
            if pair in self.rest_overlaps:
                continue
            if depth > 0.0:
                result.raw[pair] = depth
            if depth > self.tol["overlap"]:
                result.collisions.append((a, b, depth, volume, self._overlap_allowance(a, b)))
        for a, b in near_pairs:  # bodies apart at rest that come close: refine around here
            if frozenset((a[0], b[0])) in self.rest_contacts:
                continue
            if self.scene.shape_distance(a, b) < margin:
                result.near = True
                break
        return result

    def _first_contact(self, joint: Joint, hinges: list[Joint], evals: list[_PoseEval],
                       pair: frozenset, hits: list[float]) -> float:
        """Joint value where the pair first interpenetrates at all, coming from the rest
        pose: bisect between the last clear sample and the first touching one."""
        by_value = {e.value: e for e in evals}
        onset = min(hits, key=abs)
        toward_rest = sorted((v for v in by_value if v * onset >= 0 and abs(v) < abs(onset)),
                             key=abs, reverse=True)
        touching = onset
        for value in toward_rest:  # walk back while the pair still (shallowly) touches
            if by_value[value].raw.get(pair, 0.0) > 0.0:
                touching = value
            else:
                clear = value
                break
        else:
            return touching
        for _ in range(self.config.refine_depth + 2):
            middle = (clear + touching) / 2
            if self._evaluate(joint, middle, hinges).raw.get(pair, 0.0) > 0.0:
                touching = middle
            else:
                clear = middle
        return touching

    @staticmethod
    def _blocked(evaluation: _PoseEval) -> bool:
        return evaluation.loop_error is not None or any(c[4] is None for c in evaluation.collisions)

    def _refine(self, joint, hinges, evals, a, b, depth, allow_near) -> None:
        if depth <= 0:
            return
        ea, eb = evals[a], evals[b]
        if self._blocked(ea) != self._blocked(eb):  # bisect to where the trouble starts
            middle = (a + b) / 2
            evals[middle] = self._evaluate(joint, middle, hinges)
            same_as_a = self._blocked(evals[middle]) == self._blocked(ea)
            self._refine(joint, hinges, evals, middle if same_as_a else a,
                         b if same_as_a else middle, depth - 1, False)
        elif allow_near and not self._blocked(ea) and (ea.near or eb.near):
            points = [a + (b - a) * k / 4 for k in (1, 2, 3)]
            for point in points:
                evals[point] = self._evaluate(joint, point, hinges)
            for left, right in zip([a, *points], [*points, b], strict=True):
                self._refine(joint, hinges, evals, left, right, depth - 1, False)

    def _summarize(self, joint: Joint, hinges: list[Joint], evals: list[_PoseEval]) -> list[Finding]:
        return [*self._loop_findings(joint, evals),
                *self._collision_findings(joint, hinges, evals),
                *self._separation_findings(joint, hinges, evals)]

    def _loop_findings(self, joint: Joint, evals: list[_PoseEval]) -> list[Finding]:
        runs: list[list[_PoseEval]] = []  # consecutive unreachable samples
        previous = None
        for evaluation in evals:
            if evaluation.loop_error is not None:
                if runs and runs[-1][-1] is previous:
                    runs[-1].append(evaluation)
                else:
                    runs.append([evaluation])
            previous = evaluation
        if not runs:
            return []
        spans = ", ".join(
            _format_value(joint, run[0].value) if len(run) == 1 else
            f"{_format_value(joint, run[0].value)} to {_format_value(joint, run[-1].value)}"
            for run in runs)
        first = min((run[0] for run in runs), key=lambda e: abs(e.value))
        return [Finding("loop", f"the loop cannot close for {joint.name} = {spans} (inside its "
                        f"limits): {first.loop_error}", True, joint=joint.name,
                        pose={joint.name: first.value})]

    def _collision_findings(self, joint: Joint, hinges: list[Joint],
                            evals: list[_PoseEval]) -> list[Finding]:
        out = []
        worst: dict[frozenset, dict] = {}
        for evaluation in evals:
            for a, b, depth, volume, allowed in evaluation.collisions:
                entry = worst.setdefault(frozenset((a, b)), {"a": a, "b": b, "values": [],
                                                             "depth": 0.0, "at": None,
                                                             "allowed": allowed})
                entry["values"].append(evaluation.value)
                if depth > entry["depth"]:
                    entry.update(depth=depth, at=evaluation.value, volume=volume)
        for pair, entry in worst.items():
            self.reported_pairs.add(pair)
            a, b, hits = entry["a"], entry["b"], sorted(entry["values"])
            contact = self._first_contact(joint, hinges, evals, pair, hits)
            span = _format_value(joint, hits[0]) if hits[0] == hits[-1] else \
                f"{_format_value(joint, hits[0])} to {_format_value(joint, hits[-1])}"
            message = (f"{a[0]!r}/{a[1]} hits {b[0]!r}/{b[1]} while {joint.name!r} moves: "
                       f"first contact at {_format_value(joint, contact)}, deeper than "
                       f"{self.tol['overlap'] * 1000:.2f} mm at {span} (deepest "
                       f"{entry['depth'] * 1000:.2f} mm at {_format_value(joint, entry['at'])})")
            out.append(Finding("swept_collision", message, entry["allowed"] is None,
                               (a[0], b[0]), (a[1], b[1]), joint.name, {joint.name: contact},
                               entry["depth"], entry["allowed"]))
        return out

    def _separation_findings(self, joint: Joint, hinges: list[Joint],
                             evals: list[_PoseEval]) -> list[Finding]:
        out = []
        for hinge in hinges:
            gaps = [(e.gaps[hinge.name], e.value) for e in evals if hinge.name in e.gaps]
            rest_gap = next((g for g, v in gaps if abs(v) < 1e-15), None)
            if rest_gap is None:
                continue
            widest, at = max(gaps)
            growth = widest - rest_gap
            if growth > self.tol["separation"]:
                parent, child = hinge.parent.body.name, hinge.child.body.name
                out.append(Finding(
                    "separation", f"hinge {hinge.name!r} pulls {child!r} away from {parent!r}: "
                    f"the gap grows from {rest_gap * 1000:.2f} mm to {widest * 1000:.2f} mm at "
                    f"{joint.name}={_format_value(joint, at)}; the axis or origin is misplaced",
                    True, (parent, child), joint=hinge.name, pose={joint.name: at},
                    amount=growth))
        return out

    # -- joints moving together --

    def combinations(self) -> list[Finding]:
        """Range corners (every driver at its min or max together, at most ``max_corners``)
        plus ``random_poses`` seeded random poses. Reports only pairs that the single-joint
        sweeps did not already report, with the full pose."""
        drivers = []
        for joint in self.drivers.values():
            try:
                drivers.append((joint, _joint_range(joint, self.config, self.size)))
            except ValidationError:
                continue  # an unlimited slide: already warned about
        if len(drivers) < 2:
            return []  # one joint: the single sweep already covered every pose
        rng = np.random.default_rng(self.config.random_seed)
        poses = [{j.name: rng_value for (j, _), rng_value in zip(drivers, corner, strict=True)}
                 for corner in _corners([r for _, r in drivers], self.config.max_corners, rng)]
        for _ in range(self.config.random_poses):
            poses.append({j.name: float(rng.uniform(low, high)) for j, (low, high) in drivers})
        found: dict[frozenset, tuple[float, dict, tuple, tuple, float]] = {}
        for pose in poses:
            for a, b, depth in self._pose_collisions(pose):
                pair = frozenset((a, b))
                if pair not in found or depth > found[pair][0]:
                    found[pair] = (depth, pose, a, b, 0.0)
        out = []
        for depth, pose, a, b, _ in found.values():
            out.append(Finding(
                "combined_collision", f"{a[0]!r}/{a[1]} hits {b[0]!r}/{b[1]} only when joints "
                f"move together: {depth * 1000:.2f} mm deep at {_format_pose(pose, self.model)}",
                True, (a[0], b[0]), (a[1], b[1]), pose=dict(pose), amount=depth))
        return out

    def _pose_collisions(self, pose: dict[str, float]):
        try:
            solved = self.model.solve(pose)
        except PoseError:
            return []  # loops: the driver's own sweep reports what cannot be reached
        self.scene.place(solved.transforms)
        moved = {name for name, matrix in solved.transforms.items()
                 if not np.allclose(matrix, self.rest.transforms[name], atol=1e-12)}
        hits = []
        for a, b, depth, _volume in self.scene.collisions(self.scene.candidate_pairs(0.0, moved)):
            pair = frozenset((a, b))
            if depth <= self.tol["overlap"] or pair in self.rest_overlaps \
                    or pair in self.reported_pairs or self._overlap_allowance(a, b):
                continue
            hits.append((a, b, depth))
        return hits


def _corners(ranges: list[tuple[float, float]], cap: int, rng) -> list[tuple[float, ...]]:
    """Every min/max combination in Gray-code order (one joint changes per step, which
    keeps the walking solver happy); a seeded random sample when there are more than
    ``cap``."""
    count = len(ranges)
    codes = [i ^ (i >> 1) for i in range(2 ** count)] if count <= 20 else []
    if len(codes) > cap or not codes:
        picked = set()
        while len(picked) < cap:
            picked.add(int(rng.integers(0, 2 ** count)))
        codes = sorted(picked)
    return [tuple(ranges[k][(code >> k) & 1] for k in range(count)) for code in codes]


def _components(items: list, linked: Callable[[Any, Any], bool]) -> list[set]:
    groups: list[set] = []
    remaining = list(items)
    while remaining:
        group = {remaining.pop(0)}
        frontier = deque(group)
        while frontier:
            current = frontier.popleft()
            for other in list(remaining):
                if linked(current, other):
                    remaining.remove(other)
                    group.add(other)
                    frontier.append(other)
        groups.append(group)
    return groups


def _reach(root: str, names: list[str], touching: set[frozenset]) -> set[str]:
    reached, frontier = {root}, deque([root])
    while frontier:
        current = frontier.popleft()
        for other in names:
            if other not in reached and frozenset((current, other)) in touching:
                reached.add(other)
                frontier.append(other)
    return reached


def call_run_tests(run_tests: Callable[..., Any], ctx: TestContext) -> Any:
    """Call ``run_tests(ctx)``, or ``run_tests()`` for the step-3 style without a context."""
    try:
        takes_context = bool(inspect.signature(run_tests).parameters)
    except (TypeError, ValueError):
        takes_context = True
    return run_tests(ctx) if takes_context else run_tests()
