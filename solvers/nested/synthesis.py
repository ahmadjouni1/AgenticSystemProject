#!/usr/bin/env python3
"""General nested dimensional synthesis for sampled kinematic trajectories.

NumPy + SciPy; no dynamics. A model implements batched FK and joint limits.
The included URDF adapter supports arbitrary fixed/revolute/prismatic serial
chains. It parameterizes joint-origin translations and rotations, not meshes.
See README.md for the mathematics, assumptions, and JSON input format.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Protocol
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

Array = NDArray[np.float64]


class KinematicModel(Protocol):
    joint_names: list[str]
    lower: Array
    upper: Array

    def fk(self, design: Array, q: Array) -> tuple[Array, Array, Array]:
        """Return positions (N,3), rotations (N,3,3), chain nodes (N,K,3).

        q is (N,DOF). All positions use metres; angles use radians.
        Nodes are optional: return (N,0,3) if no capsule obstacles are used.
        """
        ...


def vector(text: str | None, default: list[float]) -> Array:
    return np.asarray(default if text is None else list(map(float, text.split())), float)


@dataclass
class URDFJoint:
    name: str
    kind: str
    xyz: Array
    rpy: Array
    axis: Array
    coordinate: int | None


class SerialURDF:
    """Kinematic chain from base_link to tip_link, in the base-link frame.

    Design parameter bindings are affine increments about the original URDF:
        xyz_j(d) = xyz_j(original) + sum b_xyz * (d[k] - d_initial[k])
        rpy_j(d) = rpy_j(original) + sum b_rpy * (d[k] - d_initial[k]).
    A parameter can bind several joints; several parameters can bind one joint.
    Mimic joints and unsupported joint types are explicitly rejected.
    """

    def __init__(self, path: str | Path, base_link: str, tip_link: str,
                 parameters: list[dict], joint_bounds: dict | None = None):
        self.tree = ET.parse(path)
        root = self.tree.getroot()
        links = {x.attrib['name'] for x in root.findall('link')}
        if base_link not in links or tip_link not in links:
            raise ValueError('base_link and tip_link must exist in the URDF')
        parent_of = {}
        for element in root.findall('joint'):
            child = element.find('child').attrib['link']
            if child in parent_of:
                raise ValueError('URDF must be a tree; multiple parent joints found')
            parent_of[child] = element
        chain, visited = [], set()
        link = tip_link
        while link != base_link:
            if link in visited or link not in parent_of:
                raise ValueError('No acyclic chain from base_link to tip_link')
            visited.add(link)
            element = parent_of[link]
            chain.append(element)
            link = element.find('parent').attrib['link']
        chain.reverse()
        if not chain:
            raise ValueError('Choose different base and tip links')
        self.joints, self.joint_names = [], []
        lo, hi = [], []
        joint_bounds = joint_bounds or {}
        for element in chain:
            name, kind = element.attrib['name'], element.attrib['type']
            if kind not in {'fixed', 'revolute', 'continuous', 'prismatic'}:
                raise ValueError(f'Unsupported joint type: {name}: {kind}')
            if element.find('mimic') is not None:
                raise ValueError(f'Mimic joint {name}: supply a custom model instead')
            origin, axis = element.find('origin'), element.find('axis')
            xyz = vector(None if origin is None else origin.get('xyz'), [0, 0, 0])
            rpy = vector(None if origin is None else origin.get('rpy'), [0, 0, 0])
            axis_v = vector(None if axis is None else axis.get('xyz'), [1, 0, 0])
            coordinate = None
            if kind != 'fixed':
                if np.linalg.norm(axis_v) == 0:
                    raise ValueError(f'Zero joint axis: {name}')
                axis_v /= np.linalg.norm(axis_v)
                coordinate = len(lo)
                self.joint_names.append(name)
                limit = element.find('limit')
                if name in joint_bounds:
                    lower, upper = joint_bounds[name]
                elif kind == 'continuous':
                    raise ValueError(f'Supply finite unwrapped joint_bounds for {name}')
                elif limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
                    lower, upper = float(limit.get('lower')), float(limit.get('upper'))
                else:
                    raise ValueError(f'Missing joint bounds: {name}')
                lo.append(lower)
                hi.append(upper)
            self.joints.append(URDFJoint(name, kind, xyz, rpy, axis_v, coordinate))
        self.lower, self.upper = np.array(lo, float), np.array(hi, float)
        if not len(lo) or np.any(self.lower >= self.upper):
            raise ValueError('Need movable joints with finite lower < upper bounds')
        if not np.all(np.isfinite([self.lower, self.upper])):
            raise ValueError('Joint limits must be finite')
        self.reference = np.array([p['initial'] for p in parameters], float)
        lookup = {j.name: i for i, j in enumerate(self.joints)}
        self.xyz_basis = np.zeros((len(chain), len(parameters), 3))
        self.rpy_basis = np.zeros_like(self.xyz_basis)
        for k, p in enumerate(parameters):
            if not p.get('bindings'):
                raise ValueError(f'Design parameter {p["name"]} has no bindings')
            for b in p['bindings']:
                j = lookup[b['joint']]
                self.xyz_basis[j, k] += np.asarray(b.get('xyz', [0, 0, 0]), float)
                self.rpy_basis[j, k] += np.asarray(b.get('rpy', [0, 0, 0]), float)

    def fk(self, design: Array, q: Array) -> tuple[Array, Array, Array]:
        q = np.atleast_2d(q)
        n = len(q)
        rotations = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        positions = np.zeros((n, 3))
        nodes = [positions.copy()]
        delta = np.asarray(design) - self.reference
        for i, j in enumerate(self.joints):
            xyz = j.xyz + delta @ self.xyz_basis[i]
            rpy = j.rpy + delta @ self.rpy_basis[i]
            positions += np.einsum('nij,j->ni', rotations, xyz)
            rotations = rotations @ Rotation.from_euler('xyz', rpy).as_matrix()
            nodes.append(positions.copy())
            if j.kind in {'revolute', 'continuous'}:
                motion = Rotation.from_rotvec(q[:, j.coordinate, None] * j.axis).as_matrix()
                rotations = rotations @ motion
            elif j.kind == 'prismatic':
                positions += np.einsum('nij,nj->ni', rotations,
                                       q[:, j.coordinate, None] * j.axis)
                nodes.append(positions.copy())
        return positions, rotations, np.stack(nodes, axis=1)

    def write_kinematic_urdf(self, design: Array, path: str | Path) -> None:
        """Update joint origins ONLY. Not a mesh/collision/inertial redesign."""
        delta = np.asarray(design) - self.reference
        by_name = {x.attrib['name']: x for x in self.tree.getroot().findall('joint')}
        for i, j in enumerate(self.joints):
            element = by_name[j.name]
            origin = element.find('origin')
            if origin is None:
                origin = ET.SubElement(element, 'origin')
            for key, value in [('xyz', j.xyz + delta @ self.xyz_basis[i]),
                               ('rpy', j.rpy + delta @ self.rpy_basis[i])]:
                origin.set(key, ' '.join(f'{v:.12g}' for v in value))
        ET.indent(self.tree, space='  ')
        self.tree.write(path, encoding='utf-8', xml_declaration=True)


@dataclass
class Task:
    positions: Array
    q_start: Array
    q_end: Array
    max_step: Array
    position_tolerance: float = 1e-3
    length_scale: float = 1.0
    rotations: Array | None = None
    orientation_tolerance: float = np.deg2rad(5)
    # Each sphere: [x, y, z, radius]. Robot links are capsule approximations.
    obstacles: Array = field(default_factory=lambda: np.empty((0, 4)))
    link_radius: float = 0.0
    clearance: float = 0.0
    # Return dimensionless margins >= 0. Can implement richer collisions,
    # coupled design restrictions, or loop-closure tolerances here.
    extra_margins: Callable[[Array, Array], Array] | None = None


@dataclass
class Settings:
    inner_maxiter: int = 250
    outer_maxeval: int = 250
    inner_ftol: float = 1e-10
    feasibility_tolerance: float = 2e-6
    outer_tol: float = 2e-5
    multistarts: int = 1
    seed: int = 7


@dataclass
class InnerResult:
    design: Array
    q: Array
    max_error: float
    min_margin: float
    feasible: bool
    solver_success: bool
    message: str


class NestedSynthesis:
    """General solver; no URDF-specific or fixed-DOF logic in either loop.

    Internal: SLSQP, min z s.t. z >= normalized waypoint error norms.
    External: COBYLA over normalized design coordinates, rerunning internal
    optimization at every new trial. Infeasible inner solves return finite
    error + violations; they never silently count as successful designs.
    """

    def __init__(self, model: KinematicModel, task: Task, initial: Array,
                 lower: Array, upper: Array, settings: Settings | None = None):
        self.model, self.task = model, task
        self.settings = settings or Settings()
        self.initial, self.lower, self.upper = map(lambda x: np.asarray(x, float),
                                                  (initial, lower, upper))
        self.range = self.upper - self.lower
        if not len(initial) or np.any(self.range <= 0):
            raise ValueError('Need design parameters with lower < upper')
        if not np.all(np.isfinite([self.initial, self.lower, self.upper])):
            raise ValueError('Design values must be finite')
        if np.any(self.initial < self.lower) or np.any(self.initial > self.upper):
            raise ValueError('Initial design lies outside bounds')
        t = self.task
        t.positions = np.asarray(t.positions, float)
        t.q_start, t.q_end, t.max_step = map(lambda x: np.asarray(x, float),
                                           (t.q_start, t.q_end, t.max_step))
        self.n, self.dof = len(t.positions), len(model.lower)
        if t.positions.shape != (self.n, 3) or self.n < 2:
            raise ValueError('Need at least two 3D target positions')
        if not np.all(np.isfinite(t.positions)):
            raise ValueError('Target positions must be finite')
        if any(x.shape != (self.dof,) for x in (t.q_start, t.q_end, t.max_step)):
            raise ValueError('Endpoint configurations and max_step must have DOF entries')
        if not all(np.all(np.isfinite(x)) for x in (t.q_start, t.q_end, t.max_step)):
            raise ValueError('Endpoint configurations and max_step must be finite')
        if np.any(t.max_step <= 0) or min(t.position_tolerance, t.length_scale) <= 0:
            raise ValueError('Step limits and position/length scales must be positive')
        self.q_range = model.upper - model.lower
        if np.any(self.q_range <= 0) or not np.all(np.isfinite(self.q_range)):
            raise ValueError('Model needs finite, nondegenerate joint bounds')
        for q in (t.q_start, t.q_end):
            if np.any(q < model.lower) or np.any(q > model.upper):
                raise ValueError('Endpoint configuration violates joint bounds')
        if np.any(np.abs(t.q_end - t.q_start) > (self.n - 1) * t.max_step + 1e-12):
            raise ValueError('Endpoints cannot be connected under supplied step bounds')
        if t.rotations is not None:
            t.rotations = np.asarray(t.rotations, float)
            if t.rotations.shape != (self.n, 3, 3):
                raise ValueError('Target rotations must be (N,3,3)')
            if not 0 < t.orientation_tolerance <= np.pi:
                raise ValueError('Orientation tolerance must lie in (0, pi]')
            if not np.allclose(t.rotations @ t.rotations.transpose(0, 2, 1),
                               np.eye(3), atol=1e-7) or not np.allclose(
                                   np.linalg.det(t.rotations), 1, atol=1e-7):
                raise ValueError('Target rotations must be proper rotation matrices')
        t.obstacles = np.asarray(t.obstacles, float).reshape(-1, 4)
        if np.any(t.obstacles[:, 3] < 0) or min(t.link_radius, t.clearance) < 0:
            raise ValueError('Radii and clearance must be nonnegative')
        if self.settings.multistarts < 1:
            raise ValueError('multistarts must be >= 1')
        self.cache: dict[tuple, InnerResult] = {}
        self.history: list[dict] = []

    def decode_q(self, y: Array) -> Array:
        q = np.empty((self.n, self.dof))
        q[0], q[-1] = self.task.q_start, self.task.q_end
        q[1:-1] = self.model.lower + y.reshape(self.n - 2, self.dof) * self.q_range
        return q

    def hard_margins(self, design: Array, q: Array,
                     fk: tuple[Array, Array, Array] | None = None,
                     max_step: Array | None = None) -> Array:
        """Positive is feasible. Position tracking remains the objective."""
        t = self.task
        _, rotations, nodes = self.model.fk(design, q) if fk is None else fk
        step = t.max_step if max_step is None else max_step
        dq = np.diff(q, axis=0)
        margins = [(step - dq).ravel() / np.tile(self.q_range, len(dq)),
                   (step + dq).ravel() / np.tile(self.q_range, len(dq)),
                   ((q - self.model.lower) / self.q_range).ravel(),
                   ((self.model.upper - q) / self.q_range).ravel()]
        if t.rotations is not None:
            trace = np.einsum('nij,nij->n', t.rotations, rotations)
            cosine = (trace - 1) / 2
            c = np.cos(t.orientation_tolerance)
            margins.append((cosine - c) / max(1 - c, 1e-10))
        if len(t.obstacles):
            if nodes.shape[1] < 2:
                raise ValueError('Capsule obstacles need model chain nodes')
            a, v = nodes[:, :-1], np.diff(nodes, axis=1)
            vv = np.einsum('nkj,nkj->nk', v, v)
            for sphere in t.obstacles:
                w = sphere[:3] - a
                fraction = np.clip(np.einsum('nkj,nkj->nk', w, v)
                                   / np.maximum(vv, 1e-30), 0, 1)
                closest = a + fraction[:, :, None] * v
                distance = np.linalg.norm(closest - sphere[:3], axis=2)
                margins.append(((distance - sphere[3] - t.link_radius - t.clearance)
                                / t.length_scale).ravel())
        if t.extra_margins is not None:
            margins.append(np.asarray(t.extra_margins(design, q), float).ravel())
        result = np.concatenate(margins)
        if not np.all(np.isfinite(result)):
            raise ValueError('Constraint callback returned nonfinite values')
        return result

    def solve_internal(self, design: Array) -> InnerResult:
        """Fixed deterministic seeds at every geometry; no stateful warm-start bias."""
        cfg, t = self.settings, self.task
        q_seed = np.linspace(t.q_start, t.q_end, self.n)
        y_seed = ((q_seed[1:-1] - self.model.lower) / self.q_range).ravel()
        if not len(y_seed):
            return self._result(design, q_seed, True, 'Endpoints only; no free coordinates')
        rng = np.random.default_rng(cfg.seed)
        seeds = [y_seed]
        for _ in range(cfg.multistarts - 1):
            seeds.append(np.clip(y_seed + rng.normal(0, .12, len(y_seed)), 0, 1))
        candidates = []
        for seed in seeds:
            # One FK call per distinct y, shared by objective constraints.
            saved: dict = {}

            def evaluate(y):
                key = np.asarray(y).tobytes()
                if saved.get('key') != key:
                    q = self.decode_q(y)
                    fk = self.model.fk(design, q)
                    errors2 = np.sum(((fk[0] - t.positions) / t.length_scale) ** 2, axis=1)
                    saved.update(key=key, errors2=errors2,
                                 margins=self.hard_margins(design, q, fk))
                return saved['errors2'], saved['margins']

            def tracking_constraint(x):
                errors2, margins = evaluate(x[:-1])
                # A 1e-10 normalized-length smoothing floor avoids the norm's
                # derivative singularity at zero without a squared objective's
                # loss of resolution close to a high-precision solution.
                return np.r_[x[-1] - np.sqrt(errors2 + 1e-20), margins]

            def optimize_tracking(y):
                e2, _ = evaluate(y)
                x0 = np.r_[y, np.sqrt(max(e2) + 1e-20) + 1e-8]
                objective_jac = np.r_[np.zeros(len(y)), 1.]
                return minimize(lambda x: x[-1], x0, jac=lambda x: objective_jac,
                                method='SLSQP', bounds=[(0, 1)] * len(y) + [(0, None)],
                                constraints=[{'type': 'ineq', 'fun': tracking_constraint}],
                                options={'maxiter': cfg.inner_maxiter, 'ftol': cfg.inner_ftol})

            optimized = optimize_tracking(seed)
            q = self.decode_q(optimized.x[:-1])
            first = self._result(design, q, optimized.success, optimized.message)
            candidates.append(first)
            if not first.feasible:
                # Restore constraints with a common slack rho. This returns a
                # graded violation even when no feasible path was found.
                y0 = optimized.x[:-1]
                _, margins = evaluate(y0)
                x0 = np.r_[y0, max(0., -min(margins)) + 1e-7]
                restored = minimize(
                    lambda x: x[-1], x0,
                    jac=lambda x: np.r_[np.zeros(len(y0)), 1.], method='SLSQP',
                    bounds=[(0, 1)] * len(y0) + [(0, None)],
                    constraints=[{'type': 'ineq',
                                  'fun': lambda x: evaluate(x[:-1])[1] + x[-1]}],
                    options={'maxiter': cfg.inner_maxiter, 'ftol': cfg.inner_ftol})
                candidates.append(self._result(design, self.decode_q(restored.x[:-1]),
                                               False, 'Feasibility restoration: ' + restored.message))
                if min(evaluate(restored.x[:-1])[1]) >= -cfg.feasibility_tolerance:
                    retried = optimize_tracking(restored.x[:-1])
                    candidates.append(self._result(design, self.decode_q(retried.x[:-1]),
                                                   retried.success, retried.message))
        feasible = [r for r in candidates if r.feasible]
        if feasible:
            return min(feasible, key=lambda r: r.max_error)
        return min(candidates, key=lambda r: (-r.min_margin, r.max_error))

    def _result(self, design, q, success, message) -> InnerResult:
        fk = self.model.fk(design, q)
        error = float(np.max(np.linalg.norm(fk[0] - self.task.positions, axis=1)))
        margin = float(np.min(self.hard_margins(design, q, fk)))
        return InnerResult(np.array(design).copy(), q.copy(), error, margin,
                           margin >= -self.settings.feasibility_tolerance,
                           bool(success), str(message))

    def evaluate_geometry(self, u: Array) -> InnerResult:
        # COBYLA may evaluate outside bounds: clip for safe model evaluation,
        # while explicit outer bounds still reject such proposals.
        u = np.clip(np.asarray(u, float), 0, 1)
        key = tuple(u)  # exact cache; rounding would corrupt outer search steps
        if key not in self.cache:
            result = self.solve_internal(self.lower + self.range * u)
            self.cache[key] = result
            self.history.append({'evaluation': len(self.history) + 1,
                                 'design': result.design.tolist(),
                                 'max_error_m': result.max_error,
                                 'min_margin': result.min_margin,
                                 'feasible': result.feasible,
                                 'inner_solver_success': result.solver_success})
        return self.cache[key]

    def solve(self, verbose=True) -> tuple[InnerResult, object]:
        u0 = (self.initial - self.lower) / self.range
        initial = self.evaluate_geometry(u0)
        if verbose:
            print(f'Initial maximum error: {initial.max_error:.6g} m', flush=True)

        def objective(u):
            r = self.evaluate_geometry(u)
            # Finite scores preserve information in infeasible regions.
            return r.max_error / self.task.length_scale + 100 * max(0., -r.min_margin)

        def constraint(u):
            r = self.evaluate_geometry(u)
            return np.r_[u, 1 - u, r.min_margin + self.settings.feasibility_tolerance]

        outer = minimize(objective, u0, method='COBYLA',
                         constraints=[{'type': 'ineq', 'fun': constraint}],
                         options={'maxiter': self.settings.outer_maxeval,
                                  'rhobeg': .15, 'tol': self.settings.outer_tol,
                                  'catol': self.settings.feasibility_tolerance})
        self.evaluate_geometry(outer.x)
        # Return best actually checked feasible candidate, not blindly outer.x.
        feasible = [r for r in self.cache.values() if r.feasible]
        candidates = feasible or list(self.cache.values())
        best = min(candidates, key=(lambda r: r.max_error) if feasible
                   else (lambda r: (-r.min_margin, r.max_error)))
        if verbose:
            print(f'Best maximum error: {best.max_error:.6g} m; '
                  f'feasible={best.feasible}; geometry evaluations={len(self.cache)}', flush=True)
        return best, outer

    def validate_dense(self, result: InnerResult, subdivisions=10) -> dict:
        """Check interpolated q against target polyline, not an unknown true curve.

        This finite check is evidence, not a mathematical continuous-path proof.
        Custom task.extra_margins is evaluated on the dense path as supplied.
        """
        if subdivisions < 1:
            raise ValueError('subdivisions must be >= 1')
        s = np.arange(self.n)
        dense_s = np.linspace(0, self.n - 1, (self.n - 1) * subdivisions + 1)
        q = np.stack([np.interp(dense_s, s, result.q[:, j]) for j in range(self.dof)], axis=1)
        positions = np.stack([np.interp(dense_s, s, self.task.positions[:, j])
                              for j in range(3)], axis=1)
        t = self.task
        target_rotations = None if t.rotations is None else Slerp(
            s, Rotation.from_matrix(t.rotations))(dense_s).as_matrix()
        # Build a separate validation task so the optimization task is unchanged.
        dense_task = Task(positions, t.q_start, t.q_end, t.max_step / subdivisions,
                          t.position_tolerance, t.length_scale, target_rotations,
                          t.orientation_tolerance, t.obstacles, t.link_radius, t.clearance,
                          t.extra_margins)
        validator = NestedSynthesis(self.model, dense_task, self.initial,
                                   self.lower, self.upper, self.settings)
        fk = self.model.fk(result.design, q)
        error = float(np.max(np.linalg.norm(fk[0] - positions, axis=1)))
        margin = float(np.min(validator.hard_margins(result.design, q, fk)))
        return {'samples': len(q), 'target_interpolation': 'linear positions, SLERP rotations',
                'joint_interpolation': 'linear, in unwrapped coordinates',
                'max_error_m': error, 'min_margin': margin,
                'constraints_pass': margin >= -self.settings.feasibility_tolerance,
                'position_pass': error <= t.position_tolerance,
                'passes': bool(error <= t.position_tolerance and
                               margin >= -self.settings.feasibility_tolerance)}


def load_problem(path: str | Path) -> tuple[NestedSynthesis, dict]:
    path = Path(path)
    spec = json.loads(path.read_text())
    p = spec['design_parameters']
    model = SerialURDF(path.parent / spec['urdf'], spec['base_link'], spec['tip_link'],
                       p, spec.get('joint_bounds'))
    waypoints = spec['waypoints']
    rotations = None
    if 'quaternions_xyzw' in waypoints:
        rotations = Rotation.from_quat(waypoints['quaternions_xyzw']).as_matrix()
    task = Task(np.array(waypoints['positions_m'], float),
                np.array(spec['q_start'], float), np.array(spec['q_end'], float),
                np.array(spec['max_joint_step'], float),
                position_tolerance=spec.get('position_tolerance_m', .001),
                length_scale=spec.get('length_scale_m', 1.), rotations=rotations,
                orientation_tolerance=spec.get('orientation_tolerance_rad', np.deg2rad(5)),
                obstacles=np.array(spec.get('sphere_obstacles', []), float).reshape(-1, 4),
                link_radius=spec.get('capsule_radius_m', 0.),
                clearance=spec.get('clearance_m', 0.))
    settings = Settings(**spec.get('settings', {}))
    solver = NestedSynthesis(model, task, [x['initial'] for x in p],
                             [x['lower'] for x in p], [x['upper'] for x in p], settings)
    return solver, spec


def save_outputs(solver: NestedSynthesis, best: InnerResult, outer, spec: dict,
                 output: str | Path, subdivisions=10) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = solver.solve_internal(solver.initial)
    dense = solver.validate_dense(best, subdivisions)
    report = {'design': dict(zip([p['name'] for p in spec['design_parameters']],
                                best.design.tolist())),
              'joint_order': solver.model.joint_names,
              'initial_max_error_m': baseline.max_error,
              'waypoint_max_error_m': best.max_error,
              'waypoint_constraints_pass': best.feasible,
              'waypoint_position_pass': best.max_error <= solver.task.position_tolerance,
              'inner_solver_success': best.solver_success,
              'inner_message': best.message,
              'outer_solver_success': bool(outer.success),
              'outer_message': str(outer.message),
              'geometry_evaluations': len(solver.cache),
              'dense_validation': dense,
              'functional_success': bool(best.feasible and
                                         best.max_error <= solver.task.position_tolerance and
                                         dense['passes'])}
    (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'history.json').write_text(json.dumps(solver.history, indent=2, allow_nan=False) + '\n')
    header = ','.join(['waypoint'] + solver.model.joint_names)
    np.savetxt(output / 'joint_path.csv', np.c_[np.arange(solver.n), best.q],
               delimiter=',', header=header, comments='')
    if isinstance(solver.model, SerialURDF):
        solver.model.write_kinematic_urdf(best.design, output / 'optimized_kinematics.urdf')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(121, projection='3d')
    target = solver.task.positions
    ax.plot(*target.T, 'ko--', label='Target waypoints')
    for result, label in [(baseline, 'Initial geometry'), (best, 'Optimized geometry')]:
        actual = solver.model.fk(result.design, result.q)[0]
        ax.plot(*actual.T, '.-', label=label)
    ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title='End-effector path')
    ax.legend(fontsize=8)
    ax2 = fig.add_subplot(122)
    ax2.plot(best.q)
    ax2.set(xlabel='Waypoint', ylabel='Joint coordinate (rad or m)', title='Optimized joint path')
    ax2.legend(solver.model.joint_names, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'trajectory.png', dpi=170)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problem', help='JSON task and parameter bindings')
    parser.add_argument('--output', default='results')
    parser.add_argument('--subdivisions', type=int, default=10)
    args = parser.parse_args()
    solver, spec = load_problem(args.problem)
    best, outer = solver.solve()
    report = save_outputs(solver, best, outer, spec, args.output, args.subdivisions)
    print(json.dumps(report, indent=2))
    # Solver convergence is not the same as task success.
    raise SystemExit(0 if report['functional_success'] else 2)


if __name__ == '__main__':
    main()
