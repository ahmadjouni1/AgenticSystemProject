# Dimensional synthesis

`synthesize` computes a linkage's dimensions from the motion you want, instead of working
them out in your head. It returns the dimensions, the predicted motion and its error. Each
call runs up to `time_limit_s` (default 180 s).

**Supported, and nothing else:**

| `mechanism` | Targets |
| --- | --- |
| `four_bar` | `sweep` (the rocker's swing, radians), a coupler-point `points` path, or `crank` + `output` |
| `slider_crank` | `stroke` (the slider's travel, meters), a rod-point `points` path, or `crank` + `output` |
| `serial_arm` | a tip `points` path, on your newest build |

Any other mechanism (scissor lift, pantograph, gears) answers `unsupported`: design it
normally. Targets are plain data with an optional `tolerance` in the same unit.

## Wipers: a rocker sweep

```text
{"mechanism": "four_bar", "target": {"sweep": 1.5708, "tolerance": 0.01},
 "constraints": {"plane": "xz", "initial": {"ground_length": 0.25}}}
```

The crank (the motor) turns a full circle and the rocker swings 90 degrees back and forth.
`initial.ground_length` sets the size; `min_transmission_deg` (default 35) keeps the linkage
from jamming. Mount each wiper arm on a rocker.

## Pump: a slider stroke

```text
{"mechanism": "slider_crank", "target": {"stroke": 0.06, "tolerance": 0.001}}
```

## A coupler-point path

```text
{"mechanism": "four_bar",
 "target": {"points": [[0.20, 0, 0.30], [0.25, 0, 0.34], ...], "tolerance": 0.002},
 "constraints": {"plane": "xz", "crank_span": 6.2832}}
```

Points in order of motion; `crank_span` is how far the crank turns from the first to the last.

## What comes back

**`skeleton_main_py`**: a complete main.py of simple bars at the solved pivots, with the loop
closure and `target_path`, already compiled (`skeleton_compile` says it passed). Write it as
main.py, compile, then add detail in small edits; keep its pivots, lengths, joint names and
`target_path`.

Also `parameters` (lengths in meters, angles in radians), `derived` (the rocker's length,
`extent`, `min_transmission_deg`, the slider's `offset`), and `rest_points`: the world
position at q = 0 of `crank_joint` (crank pivot), `coupler_joint` or `rod_joint` (crank pin),
`rocker_joint` (rocker pivot) or `slide` (the rail), and `pin` (the loop's closing pin).
Build bars between those points with the crank driving and the pin as the loop closure.
Constraints for any parameter: `initial`, `bounds`, `fixed` ({name: value}); `plane`
("xy", "xz", "yz") places the linkage, at `height` for a sweep or stroke (a path gives its
own). If `within_tolerance` is false, call
again with `initial` from the result or fewer free parameters.

## Serial arm

```text
{"mechanism": "serial_arm", "target": {"points": [[0.5, 0, 0.3], ...]},
 "constraints": {"tip_body": "forearm", "tip_point": [0.3, 0, 0],
   "parameters": [{"name": "upper_arm", "initial": 0.35, "lower": 0.2, "upper": 0.6,
                   "bindings": [{"joint": "elbow", "xyz": [1, 0, 0]}]}]}}
```

It returns `joint_origins` (new `parent.at(xyz=..., rpy=...)` per joint).

## Check it at compile

Declare the target next to `object_model`; a miss blocks the compile:

```text
target_path = {"joint": "crank_joint", "output_joint": "wiper_rocker",
               "sweep": 1.5708, "tolerance": 0.01}
target_path = {"joint": "crank_joint", "output_joint": "piston_slide",
               "stroke": 0.06, "tolerance": 0.001}
target_path = {"body": "coupler", "point": [0.05, 0.03, 0.0], "joint": "crank_joint",
               "points": [[0.20, 0, 0.30], ...], "tolerance": 0.002}
```

`joint` is the driver; `output_joint` is your rocker hinge or slider joint; `point` is in the
body's own frame.
