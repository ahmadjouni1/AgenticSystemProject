# General nested kinematic dimensional synthesis in Python

This is a runnable reconstruction of the nested optimization formulation we
discussed, not the RCIM authors' released implementation. Both optimization
loops work with any number of joint coordinates, design parameters, and target
waypoints. There is no two-link model or two-link IK formula in the solver.

The included URDF adapter handles an arbitrary serial chain of fixed,
revolute, continuous, and prismatic joints. A Python model interface permits
other kinematic representations. Topology remains fixed during optimization.

## Install and run

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python synthesis.py examples/spatial_3r.json --output results
python -m unittest discover -s tests -v
```

To run your arm after creating its task JSON:

```bash
python synthesis.py my_arm_task.json --output my_arm_results --subdivisions 20
```

`examples/spatial_3r.json` is only an execution example. The same command and
solver accept a different serial URDF with a different DOF count. The tests also
solve a six-DOF chain with mixed prismatic/revolute joints and orientation
constraints, and a custom model with no URDF dependency.

The exit code is 0 when the requested task passes both waypoint and dense checks,
2 when optimization completed but the task did not pass. Invalid inputs raise
an error. Read `result.json` rather than interpreting optimizer convergence as
functional success.

## Python equivalents to the MATLAB tools

| Role | Python implementation used here | Comparable MATLAB capability |
| --- | --- | --- |
| Numerical arrays and transformations | NumPy | Arrays and matrix operations |
| Constrained joint-path optimization | `scipy.optimize.minimize(method="SLSQP")` | `fmincon` with SQP |
| Geometry optimization | `scipy.optimize.minimize(method="COBYLA")` | Constrained derivative-free optimization; not an exact `fmincon` algorithm equivalent |
| Maximum waypoint error | An auxiliary-variable minimax formulation | A problem that can be formulated for `fminimax` |
| Orientation operations | `scipy.spatial.transform.Rotation` and `Slerp` | Rotation and pose utilities |
| Result plots | Matplotlib | Plotting tools |
| URDF kinematics | Included `SerialURDF` adapter | Serial robot forward kinematics |

SLSQP and MATLAB SQP are related numerical approaches, not identical
implementations. Numerical results can differ.

## Mathematical problem

Let d be the single design vector shared by the entire path. Let Q contain the
joint configurations at all N waypoints. FK(d, q_i) returns the end-effector
position and orientation. The task is expressed in the chosen base-link frame.

For fixed d, the inner problem is:

\[
\begin{aligned}
\min_{Q,z}\quad &z\\
\text{subject to}\quad
&\sqrt{\|p(d,q_i)-p_i^{target}\|^2/\ell^2+10^{-20}}\leq z,\\
&q_{min}\leq q_i\leq q_{max},\\
&q_1=q_{start},\qquad q_N=q_{end},\\
&-\Delta q_{max}\leq q_{i+1}-q_i\leq\Delta q_{max},\\
&\text{optional orientation, obstacle, and custom constraints.}
\end{aligned}
\]

Here ell is `length_scale_m`. The tiny smoothing floor makes the norm
differentiable at zero. Position accuracy is evaluated afterward using the true
Euclidean distance, not the smoothed expression. The start/end coordinates are
eliminated from the optimization variables, so they remain fixed exactly.

The outer optimizer changes d and reruns the inner solve for every new geometry.
It seeks to reduce the optimized maximum position error while satisfying the
design bounds and the inner constraints. Both d and the free joint coordinates
are normalized internally to reduce differences in numerical scale.

Pseudo-code:

```python
for trial_design in constrained_geometry_optimizer:
    for deterministic_initial_path in initial_paths:
        path = solve_constrained_minimax_IK(trial_design, initial_path)
        if no_constraint_feasible_path_was_found:
            path = restore_feasibility_with_a_common_slack()
            if restored_path_is_feasible:
                path = solve_constrained_minimax_IK(trial_design, path)
    best_inner = choose_best_feasible_path_or_least_violating_path()
    return_tracking_error_and_constraint_margins_to_outer_optimizer(best_inner)
return_best_checked_feasible_design_and_its_path()
```

SLSQP uses numerical constraint derivatives. The outer search is derivative-free,
which avoids differentiating through an approximate IK solve. COBYLA receives
explicit geometry bounds and the worst inner constraint margin. The outer score
also includes a finite violation penalty so infeasible trial geometries provide
useful information. An unsuccessful inner solve means **no feasible solution
was found**, not that infeasibility has been mathematically proven.

For repeatability, every geometry uses the same deterministic path seeds.
Exact-repeat geometry evaluations are cached. Results do not depend on a
changing warm-start history. `multistarts` controls the number of internal
initializations. These remain local optimization methods: no global optimum or
guaranteed success is claimed.

## Supply your robot and target trajectory

Copy the example JSON and replace the URDF, base/tip link names, design
parameters, targets, joint endpoints, and joint step bounds.

Joint coordinates are ordered exactly as printed under `joint_order` in the
result: movable joints along the base-to-tip chain. Use metres for lengths and
prismatic coordinates, radians for angular coordinates, and right-handed frames.
Continuous joints require explicit finite unwrapped `joint_bounds`, such as
`{"turntable": [-6.283185307, 6.283185307]}`. Interpolation does not wrap angles.

Each design parameter has:

```json
{
  "name": "upper_arm_length_m",
  "initial": 0.4,
  "lower": 0.2,
  "upper": 0.7,
  "bindings": [
    {"joint": "elbow_joint", "xyz": [1, 0, 0]}
  ]
}
```

This says that increasing this length by 0.1 m moves `elbow_joint`'s origin
by 0.1 m along its parent-frame x axis. The input URDF represents the geometry
at the parameter's `initial` value. For each binding:

\[
xyz_j(d)=xyz_j^{URDF}+\sum_k b_{jk}^{xyz}(d_k-d_k^{initial}).
\]

Bindings can also contain `"rpy": [0, 1, 0]`. An angular parameter with that
binding changes the origin's pitch by the parameter increment in radians.
The adapter applies an affine change to roll/pitch/yaw coordinates, then builds
the corresponding rotation. It is not an axis-angle increment.

One parameter can affect several joint origins; several parameters can affect
one origin. This supports coupled lengths, offsets, mounting angles, and joint
frame orientations without hard-coding a particular robot. Arbitrary nonlinear
design-to-geometry relationships belong in a custom model's `fk` function.

`waypoints.positions_m` must contain N rows of `[x,y,z]`. For a mathematical
curve, sample the curve yourself in the same base frame. Optionally add
`waypoints.quaternions_xyzw`, with N quaternion rows, and set
`orientation_tolerance_rad`. The built-in orientation constraint compares full
3D rotations; a tool-axis-only constraint should instead use a custom margin.

`q_start`, `q_end`, and `max_joint_step` each have DOF entries. Step bounds apply
between consecutive samples; they are kinematic continuity restrictions, not
motor or dynamics limits. If you change the sampling density, adjust the bounds
consistently.

Setting `position_tolerance_m` specifies the acceptance threshold. Position
error remains the minimized objective so inaccurate initial geometries can
still improve. If an endpoint must be satisfied as a separate strict constraint,
add an appropriate endpoint margin through the Python API.

Optional spherical obstacles are supplied as:

```json
{
  "sphere_obstacles": [[0.3, 0.1, 0.2, 0.05]],
  "capsule_radius_m": 0.015,
  "clearance_m": 0.005
}
```

This uses capsule approximations between consecutive kinematic chain nodes.
It does not load URDF collision meshes and does not check self-collision.
Use a custom collision callback if that fidelity is needed.

## Use the solver with a different kinematics implementation

Only this model interface is required; DOF is inferred from the bound arrays:

```python
class MyRobot:
    joint_names = [...]             # length DOF
    lower = np.array([...])         # finite bounds, rad or m
    upper = np.array([...])

    def fk(self, design, joint_path):
        # joint_path: (N, DOF), for any N, including dense validation samples
        # Implement your FK here: products of exponentials, DH, a robotics
        # package, or a differentiable/articulated geometry representation.
        return positions, rotations, nodes
        # positions: (N,3); rotations: (N,3,3)
        # nodes: (N,K,3), or (N,0,3) if capsule obstacles are not used
```

Instantiate it through the Python API:

```python
import numpy as np
from synthesis import NestedSynthesis, Settings, Task

task = Task(
    positions=target_positions,         # (N,3)
    q_start=start_configuration,        # (DOF,)
    q_end=end_configuration,            # (DOF,)
    max_step=maximum_joint_steps,       # (DOF,)
    position_tolerance=0.001,
    length_scale=1.0,
)

# Optional: any dimensionless design/path feasibility margins >= 0.
# This callback must work for both waypoint and dense validation sample counts.
# task.extra_margins = lambda design, path: your_constraints(design, path)

solver = NestedSynthesis(
    model=my_robot,
    task=task,
    initial=initial_design,
    lower=design_lower_bounds,
    upper=design_upper_bounds,
    settings=Settings(multistarts=3, outer_maxeval=500),
)
best, optimizer_result = solver.solve()
validation = solver.validate_dense(best, subdivisions=20)
print(best.design, best.q, validation)
```

For a loop-closure equality h(d,Q)=0 with a numerical tolerance delta, a
callback can return both `delta - h` and `delta + h`, with consistent scaling.
The solver does not derive loop-closure equations or solve arbitrary mechanism
topology automatically; your model must provide the corresponding kinematics
and constraints.

## Outputs and validation

The CLI saves:

- `result.json`: optimized design, maximum errors, feasibility, solver status,
  dense validation, and overall task success.
- `joint_path.csv`: the optimized waypoint joint configurations.
- `history.json`: every unique trial geometry and its measured result.
- `trajectory.png`: target and achieved paths plus joint configurations.
- `optimized_kinematics.urdf`: a URDF with optimized joint origins.

The exported URDF updates **joint origins only**. It does not resize visual
meshes, collision geometry, or inertial properties. It is a kinematic result,
not a completed manufacturable redesign. An Articraft integration must rebuild
the link geometry consistently from the optimized design parameters.

Dense validation linearly interpolates Q and the target positions, and uses
SLERP for target orientations. It tests the resulting joint motion against that
target polyline. It does not know the original analytic curve between samples,
and finite sampling is not a proof of continuous collision freedom or tracking.
For an analytic target, evaluate that function on the validation grid, or refine
the input waypoints and rerun optimization when the dense check fails.

## Verified run

Tested using Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8.
The eight automated tests cover arbitrary-model use, fixed endpoint behavior,
failed feasibility, repeatability, inconsistent step limits, spatial URDF FK,
six-DOF synthesis with orientation constraints, and URDF export/reload.

For the supplied spatial trajectory, one run reduced maximum waypoint error
from 0.231216 m to 0.00000404 m. A 101-sample interpolation check had maximum
error 0.0007084 m, below the chosen 0.002 m threshold. The run used 73 unique
geometry evaluations. These are a small controlled test, not a benchmark or a
performance guarantee for another arm.

## References and attribution

- Russo et al., *Task-oriented optimal dimensional synthesis of robotic
  manipulators with limited mobility*, RCIM (2021),
  https://doi.org/10.1016/j.rcim.2020.102096
- SciPy SLSQP:
  https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
- SciPy COBYLA:
  https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html
- SciPy rotations:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

The model interface, URDF bindings, SQP/COBYLA selection, smoothed minimax
implementation, feasibility restoration, endpoint/step constraints, caching,
and validation here are explicit implementation choices. They are not claimed
to reproduce unspecified details of the RCIM code.
