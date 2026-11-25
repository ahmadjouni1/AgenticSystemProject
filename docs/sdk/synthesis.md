# Dimensional synthesis

When a point must follow a path (a wiper tip sweeping an arc, a piston stroke, a tool tracing
a line), do not work the lengths out in your head: `synthesize` computes them. It returns
the dimensions, the path they predict and the error. Each call runs up to
`time_limit_s` (default 180 s).

The target is plain data: `{"points": [[x, y, z], ...], "tolerance": 0.002}`, in meters,
in order of motion (3 to 400 points, the more the better).

## Four-bar

```text
{"mechanism": "four_bar",
 "target": {"points": [[0.20, 0, 0.30], [0.25, 0, 0.34], ...], "tolerance": 0.002},
 "constraints": {"plane": "xz", "crank_span": 6.2832}}
```

`crank_span` is how far the crank turns from the first point to the last (default a full
turn; negative turns the other way). The answer gives `rest_points`: world positions at
q = 0 of `crank_joint` (crank pivot), `coupler_joint` (crank pin), `rocker_joint` (rocker
pivot), `pin` (coupler-rocker pin) and `output_point`, plus `parameters` and
`derived.rocker_length`. Build bars between those points, the crank driving, the
coupler-rocker pin as the loop closure, and the output point on the coupler.

## Slider-crank

```text
{"mechanism": "slider_crank",
 "target": {"points": [[0.10, 0, 0.05], [0.12, 0, 0.05], ...], "tolerance": 0.001},
 "constraints": {"plane": "xz", "output": "slider", "fixed": {"slider_angle": 0.0}}}
```

`output: "slider"` tracks the slider pin (a piston); the default tracks a point on the rod
(`point_x`, `point_y`). `rest_points` gives `crank_joint`, `rod_joint`, `slide` (the rail,
through the slider pin) and `output_point`. `derived.offset` is the rail's offset from the
crank pivot.

Both linkages take `initial`, `bounds` and `fixed` ({name: value}) for any parameter in
`parameters`. A result with `within_tolerance: false` is still the best found: call again
with `initial` from it, a different `crank_span`, or fewer free parameters.

## Serial arm

Works on your newest build: compile first.

```text
{"mechanism": "serial_arm",
 "target": {"points": [[0.5, 0.0, 0.3], [0.45, 0.2, 0.35], ...], "tolerance": 0.002},
 "constraints": {"tip_body": "forearm", "tip_point": [0.3, 0, 0],
   "parameters": [{"name": "upper_arm", "initial": 0.35, "lower": 0.2, "upper": 0.6,
                   "bindings": [{"joint": "elbow", "xyz": [1, 0, 0]}]}]}}
```

`initial` is the length as built now; a binding moves that joint's pivot by (value -
initial) x `xyz`, in its parent body's frame. Optional: `q_start`, `q_end` (joint values
at the first and last point; default from inverse kinematics), `max_evaluations` (default
500). The answer gives `joint_origins` (new `parent.at(xyz=..., rpy=...)` per joint) and
the joint values at each target point.

## Check the path at compile

Declare the target next to `object_model`; compile then drives `joint` over its range and
fails if the output point misses the path by more than `tolerance`:

```text
target_path = {"body": "coupler", "point": [0.05, 0.03, 0.0], "joint": "crank_joint",
               "points": [[0.20, 0, 0.30], ...], "tolerance": 0.002}
```

`point` is in the body's own frame. Optional: `values` (one driver value per point, to
compare point by point) and `rms_tolerance`.
