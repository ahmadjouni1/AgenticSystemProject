# Solvers in the replica: your nested synthesis, and what we built around it

## Your code

`solvers/nested/synthesis.py` and `README.md` are **byte-identical copies** of
`Project_extra_tools/` (commit `03f4c6c`). `solvers/nested/ATTRIBUTION.md` and NOTICE credit
you. **Never edit them:**

- `test_the_teammates_files_are_unmodified_copies` checks their SHA-256.
- `.gitattributes` marks them `-text`, so line endings stay as you wrote them.

A change of yours means a new copy, with the new hashes in the test. Everything the replica
needs is in separate files that import your classes.

Your folder had no `examples/spatial_3r.json` and no `tests/`, so we wrote both, following
your README: `solvers/nested/examples/` and the first section of `tests/test_solvers.py`.

## The interface (`solvers/base.py`)

```text
solve(mechanism, target, constraints) -> SolverResult
solve_isolated(mechanism, target, constraints, timeout_s)   # a fresh process
SolverResult: parameters, trajectory, error {max, rms}, feasible, message, details
```

- **Targets are JSON data only, never executed.** There are four forms:
  - a path, `{"points": [[x, y, z], ...]}`;
  - a rocker sweep, `{"sweep": rad}`;
  - a slider stroke, `{"stroke": m}`;
  - a function, `{"crank": [...], "output": [...]}`.
  
  Each takes an optional `tolerance`.
- **One process per solve.** `solve_isolated` sends the request on stdin to
  `python -m solvers.worker` and reads one JSON line back. It kills the process at the timeout
  (`Popen.kill`: TerminateProcess on Windows, SIGKILL elsewhere), and removes credential
  variables from its environment.
- **Supported:** `four_bar`, `slider_crank`, `serial_arm`. Any other name raises
  `Unsupported` at once, with a hint to design the mechanism normally.

## How we run your solver (`solvers/search.py`)

`Synthesis` subclasses `NestedSynthesis` and changes none of your maths. It adds three
things:

- **A deadline.** The best design checked so far is kept, by your own end rule.
- **An early stop** once a feasible design is within a tenth of the tolerance.
- **A polish.** After your search, one SLSQP solve over the design and the joint path together, in your
  minimax form with your `hard_margins`.

**Why the polish.** On the spatial 3R example your outer COBYLA search stops 12-17 mm from
the known design, from every start we tried. There several waypoint errors are equal, so the
maximum has a corner: a step along any single design axis raises it (measured), and COBYLA's
trust region shrinks to its lower bound. The error falls smoothly along the straight line to
the answer. Restarting COBYLA and relaxing its margin constraint did not help. The polish
reaches the known design (< 1e-6 m) in 1-2 s from four starts. Results report the error after
your search (`nested_max_error`) and after the polish.

## Plug-in A: serial arms (`solvers/serial_arm.py`)

- **What it wraps:** your `SerialURDF` on URDF text.
- **Its options:**
  - an optional tip point, added as a fixed link;
  - endpoints by inverse kinematics when `q_start` / `q_end` are not given;
  - `max_evaluations` (your `outer_maxeval`, default 500).
- **In the agent:** the `synthesize` tool takes the URDF the compiler exported for the
  newest build, and converts the model's body frames to link frames and back. It returns
  new `parent.at(xyz, rpy)` origins.
- **Cost:** 1-2 s per solve.

## Plug-in B: closed loops (`solvers/planar.py`, `solvers/function.py`)

**Linkages:** a four-bar and a slider-crank, built as real replica `Assembly` objects (closed
at q = 0 with `frame_in`) and posed by our loop solver, with its walking warm start.

**Path targets** go through your custom-model interface (`fk(design, q)`):

- **The curve:** each new design is posed every 5 degrees of crank, and `fk` interpolates
  that curve.
- **Reach:** a crank angle where the loop cannot close, or flips branch, ends the curve, and
  `extra_margins` makes target points beyond it infeasible.
- **The reported error:** the final design is posed again directly at every solved crank
  angle, and those poses give the error.
- **Cost:** about 2 ms per pose, 0.15 s per design, 37-118 s per solve.

**Function targets** (a rocker sweep, a slider stroke, output against crank angle):

- **Why not your solver:** the crank angles are given, so your inner joint-path loop has
  nothing to do. The design is fitted directly with `least_squares` on the posed outputs.
- **Penalties:** for a crank that cannot make the turn, and for a four-bar transmission
  angle under 35 degrees.
- **The check:** the final design is posed again every degree.
- **Cost:** 1-5 s per solve.

## The compile check (`sdk/target_path.py`)

`main.py` may declare `target_path`: a point path, or a sweep, stroke or function of an
output joint against the driver. Compile drives the joint over its range and blocks the
build if the output misses by more than the tolerance.

## What synth_v1 showed (2026-11-25)

- **The setup:** the first agent test of the tool, on 4 closed-loop prompts, with
  path-only targets and a prompt rule listing every linkage.
- **Wipers and scissor lift:** the model spent two 32k-token replies trying to make the tool
  fit. Both ended with no build.
  - A wiper needs a rocker *sweep*, which path-only synthesis could not express.
  - A scissor lift is neither a four-bar nor a slider-crank.
- **The pump**, which the tool fits, used it well: three solves within 0.05 mm, a build
  around them, and a declared `target_path`.
- **Since then:** function targets, an explicit supported list, and a rule that names where
  the tool applies (synth_v2). Results are in BUILD_LOG, step 13.

## Known gaps

- **Plug-in B is local:**
  - Paths from a poor start can fail.
  - Several linkages may fit, so dimensions are not unique (only the motion is).
- **No coverage for** five-bars, Stephenson and Watt six-bars, scissors, pantographs, gears
  or cams.
- **The function fit is the replica's own.** It does not use your solver at all.
- **Plug-in A:** no closed loops and no mimic joints (your adapter rejects them).
- **The polish needs many crank sweeps on closed loops,** which is the slow part.

## Testing offline (no network, no API keys)

```text
uv run pytest tests/test_solvers.py -q              # about 5 minutes
uv run python scripts/synthesis_bench.py            # times and accuracy on known answers
```

Every test generates a mechanism with known dimensions, records its motion, solves from
another start, and checks the motion again through the compile's own check.
