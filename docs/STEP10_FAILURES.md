# Step 10 failure analysis (B and C runs)

From the existing step 10 runs only (no new API calls). The data comes from
`scripts/step10_failures.py`, which reads every run's `record.json`, `conversation.jsonl` and
`result/compile_log.jsonl` and writes `runs/step10/failures.json`.

**Scope:** 24 runs (B: v1 prompt and docs; C: B plus low effort before the first compile).
14 of them are analysed one by one: the 10 failed runs, and the 4 successful runs that cost
$0.13 or more (the median B/C run costs $0.138; A's costs $0.077).

**How time is charged.** Each run is cut at its compile calls:
- **planning:** from the start to the end of the turn that first calls compile;
- **fix &lt;kind&gt;:** after a failed compile, up to and including the next compile call,
  charged to that compile's primary failure;
- **verify:** after a passing compile;
- **unfinished &lt;kind&gt;:** after the last, failed compile, until the run stopped.

Minutes come from the message timestamps. Model calls are 98% of all time; tool time is at
most 39 s per run.

## 1. Run by run

Each timeline step reads: segment, minutes / turns; "pass" marks a passing compile.

| Run | End state | Turns, min, cost | Timeline | Failure kinds, bodies and joints |
| --- | --- | --- | --- | --- |
| B cabinet 1 | success | 12, 10.7, $0.135 | planning 4.4/3 → overlap 2.3/2 → overlap 2.8/2 → test_failure 1.0/2 → pass → verify 0.2/3 | overlap carcass/drawer; swept_collision carcass/left_door, carcass/right_door (at their hinges); floating drawer; door contact gap 1.66 mm in its own `expect_contact` |
| B pump 1 | time_exceeded, **with a passing build** | 13, 16.5, $0.213 | planning 9.9/6 (one 380 s turn) → runtime 0.2/2 → floating 4.4/3 → pass at 14.5 min → verify 2.1/2 | `'list' object has no attribute 'translate'`; floating crank, rod, piston; swept_collision crank/rod |
| B pump 2 | time_exceeded | 10, 15.2, $0.170 | planning 10.8/4 (one 32k cut-off, 374 s) → floating 1.3/2 → overlap 2.2/3 → unfinished floating 0.9/1 | floating crank, rod, piston; loop cannot close at crank_joint; overlap/swept ground/piston, rod/piston; floating rod, piston |
| B pliers 2 | time_exceeded, no build | 9, 15.4, $0.188 | planning 6.7/4 → validation 1.9/2 → validation 3.1/2 → unfinished validation 3.7/1 | "profile polygon intersects itself", 3 compiles in a row (the jaw outline) |
| B pump 3 | time_exceeded, never compiled | 4, 17.3, $0.157 | planning 17.3/4: two 32k cut-offs (444 s, 407 s), then a write at 17 min | none: never compiled |
| B table 3 | success | 13, 7.5, $0.141 | planning 6.6/4 (one 296 s turn) → validation 0.1/2 → validation 0.3/2 → pass → verify 0.5/5 | duplicate shape names from a loop (`'bracket'`, `'foot'`) |
| B cabinet 3 | time_exceeded | 7, 15.6, $0.237 | planning 5.0/2 → overlap 4.6/2 → separation 2.7/2 → unfinished swept 3.3/1 | overlap door_left/door_right; swept_collision carcass/door_left, carcass/door_right; separation: both hinges pull the doors away (axis off the door edge) |
| C cabinet 1 | success | 14, 9.3, $0.143 | planning 4.2/2 → pass → verify 0.5/2 → (an edit) overlap 3.8/5 → pass → verify 0.9/5 | the model edited the carcass sides after the pass and created an overlap carcass/drawer |
| C table 1 | time_exceeded | 9, 18.1, $0.194 | planning 4.0/3 → swept 7.4/3 → floating 0.7/2 → unfinished swept 6.0/1 (a 358 s rewrite) | swept_collision tabletop/leg frames and leg frame/leg frame at left_fold, right_fold; floating both leg frames |
| C pump 1 | success | 17, 14.9, $0.179 | planning 6.4/6 → validation 0.2/2 → swept 5.9/2 (a 342 s rewrite) → pass → verify 2.4/7 | rounded-box radius too large; swept_collision frame/rod, crank/rod at crank_joint |
| C table 2 | time_exceeded | 11, 15.8, $0.187 | planning 4.9/4 → floating 6.3/4 → floating 1.6/2 → unfinished swept 3.0/1 | floating leg_a, leg_b (gap at the hinges), twice; swept_collision leg_a/leg_b, top/leg_a, top/leg_b |
| C pump 2 | time_exceeded, **with a passing build** | 9, 16.0, $0.199 | planning 9.0/5 (one 32k cut-off) → overlap 5.5/2 → pass at 14.5 min → verify 1.5/2 | overlap at rest: ground/crank boss inside the crank hub, ground/rod (9 failures) |
| C pliers 2 | no_progress | 8, 9.3, $0.100 | planning 9.1/4 (one 32k cut-off, 360 s) → runtime ×4 in 0.2 min | **compiler bug:** `PermissionError: [WinError 5] Access is denied: ...result\.compile-progress...`; then 4 identical compiles returned the cached error |
| C pliers 3 | time_exceeded, no build | 7, 16.3, $0.178 | planning 6.7/3 → validation 7.9/3 (one 309 s turn reading docs) → unfinished runtime 1.8/1 | rounded-box radius too large; then `StdFail_NotDone: BRep_API: command not done` (a build123d operation) |

**Where all B/C time went** (24 runs, 275 minutes):

| Segment | Turns | Minutes | Share |
| --- | --- | --- | --- |
| planning (up to the first compile) | 87 | 157.8 | **57%** |
| fix overlap | 23 | 31.0 | 11% |
| fix swept_collision | 12 | 27.9 | 10% |
| fix validation | 22 | 18.1 | 7% |
| fix floating | 16 | 16.4 | 6% |
| verify | 67 | 15.4 | 6% |
| fix separation, runtime_error, test_failure | 15 | 8.0 | 3% |

- **Slow turns:** 31 model calls took 200 s or more. Together they are 155 minutes, 56% of all
  time.
- **Cut-offs:** 5 of them hit the 32k output cap and returned nothing, wasting 31.6 minutes.
- **Fix cycles:** the median one takes 1.3 minutes; it takes 5-8 minutes when the fixing turn
  is a full rewrite with long reasoning.

## 2. The time limit

**9 of the 10 B/C failures are `time_exceeded`** (B 5 of 5, C 4 of 5). The tenth is the
`no_progress` run caused by the compiler bug. A's 3 failures are also `time_exceeded`, all on
the pump.

The cap is checked before each model call, so runs stop at 15.2-18.1 minutes: the last call
started before 15 minutes. How close each run was:

| Run | Last compile (minute) | Failures left | Trend of failure counts | Passing build? | With a 25-minute cap |
| --- | --- | --- | --- | --- | --- |
| B pump 1 | pass (14.5) | 0 | 1 → 4 → 0 | yes | **certain**: it was probing a passing build and only needed its final message |
| C pump 2 | pass (14.5) | 0 | 9 → 0 | yes | **certain**: same |
| B pump 2 | floating (14.3) | 2 | 4 → 3 → 2 | no | **likely**: falling, cycles of 1.3-2.2 min |
| C table 2 | swept_collision (12.8) | 2 | 3 → 2 → 2 | no | **likely** |
| C table 1 | swept_collision (12.1) | 2 | 6 → 2 → 2 | no | **likely**; the last turn was already a 6-minute rewrite |
| B cabinet 3 | swept_collision (12.3) | 6 | 7 → 2 → 6 | no | uncertain: it was oscillating between collision and separation (the hinge placement) |
| B pliers 2 | validation (11.7) | 1 | 1 → 1 → 1 | no build | uncertain: the same profile error 3 times, then a rewrite at 15 min that was never compiled |
| B pump 3 | none | - | - | no build | unlikely: its first main.py was written at 17 min; first compiles fail in 19 of 23 B/C runs |
| C pliers 3 | runtime_error (14.5) | 1 | 1 → 1 | no build | unlikely: a different error (build123d) after 8 minutes on the first one |

**Estimate with a 25-minute cap: 5 of the 9 succeed, perhaps 6** (2 certain, 3 likely, 1 of
the 3 uncertain or unlikely).
- **B:** 7/12 → 9-10/12.
- **C:** 7/12 → 10/12.
- **A:** it would gain 0-1. Its 3 pump runs had 0 passing compiles, and 1 never compiled.
- **Cost:** each of these runs would cost about $0.05-0.10 more.

**Two of the "failures" had a passing build and lost only to the clock while verifying.**
That is a loop rule, not a time-limit question. When time runs out and the latest compile
passed with no edit after it, the run should end with a final summary turn and count as a
success.

## 3. Causes, grouped

**Primary cause per run** (the one that cost the most time or blocked finishing), over the 14
analysed runs, and how many of those runs show the cause at all:

| Group | Primary in | Present in | Time (all 24 B/C runs) | Evidence |
| --- | --- | --- | --- | --- |
| **Slow reasoning turns** | **5**: B pump 1, B pump 2, B pump 3, B table 3, C pump 2 | 14 | planning 158 min (57%); turns of 200 s or more: 155 min | median planning is 6.4 min (A: 3.8 min); 5 cut-offs at 32k in 4 of the 14 runs; B pump 3 never compiled |
| **Geometry colliding at rest or in motion** | **4**: B cabinet 1, C cabinet 1, C table 1, C pump 1 | 9 | 59 min (21%) | 75 collision failures. **30 involve a detail shape** (handle, hinge, boss, pin, foot, rail...); **45 are main parts**: doors against the carcass sides, folding leg frames against the top and each other, the rod against the frame |
| **Joint frames, axes and contact at joints** | **2**: B cabinet 3, C table 2 | 6 | floating, separation, loop, contact tests: 22 min (8%) | 18 floating failures. **9 have a gap of 0.95-2.0 mm** (7 of 9 in B), exactly the "1-3 mm clearance" the v1 prompt asks for, while the floating check needs joined parts to touch. Separation: door hinge axes not on the door edge |
| **Helper misuse (wrong arguments)** | **2**: B pliers 2, C pliers 3 | 4 | validation and runtime: 20 min (7%) | 10 failed compiles: rounded-box radius over half the smallest size (3), self-intersecting profile (4), a hole outside its profile (1), a profile list treated as a mesh (1), a build123d operation failing (1) |
| **Other** | **1**: C pliers 2 (compiler bug) | 4 | - | `PermissionError` on the progress file, then a cached error repeated until `no_progress`; duplicate shape names from a loop (3 compiles); 2 runs that lost a passing build to the clock |

**"Detailed geometry colliding" is a smaller cause than it looks.** Only 30 of the 75
collision failures involve a detail shape. Most are main parts moving through each other:
- doors swinging into carcass sides that now have real thickness;
- folding leg frames that do not nest;
- a connecting rod that sweeps through the frame.

These are joint placement and range choices more than detail.

## 4. Fixes, ranked by expected impact

**1. Loop: the time limit and the end state.**
- **The change:**
  - when time runs out and the latest compile passed with no edit after it, give the model
    one final no-tool turn (or end it as a success);
  - raise the cap to 25 minutes;
  - check the clock again before a fix cycle starts, not only before each call.
- **Evidence:** 9 of 10 failures are `time_exceeded`, 2 of them with a passing build. A fix
  cycle takes a median of 1.3 minutes. 3 more runs were 2 failures from passing with a falling
  trend.
- **Expected:** +5-6 of 24 B/C runs (§2), about 1 more for A. It is a few lines in
  `agent/loop.py`.
- **Cost:** more spend on runs that would fail anyway. B pump 3 and C pliers 3 would have
  used about $0.08 more each.

**2. Docs and prompt: contact at joints, and recipes for the two motions that failed most.**
- **The change:**
  - the prompt says "leave 1-3 mm clearance between parts that move past each other" and,
    separately, "joined bodies touch at the joint". It should say plainly that the parts a
    joint connects touch (gap 0, or `allow_overlap` for a pin in a barrel), and that
    clearance is only for parts sliding past each other;
  - `joints_and_loops.md` gets two short recipes:
    - a door on a carcass with real panel thickness: the hinge axis on the door's outer back
      edge, and a range that clears the side panel;
    - folding legs that nest: one hinge line inset from the other, and the legs fold toward
      different sides.
- **Evidence:**
  - 9 of 18 floating failures are the 1-2 mm clearance;
  - 4 of the 14 runs fight door/carcass collisions or separation, and 3 fight folding legs;
  - floating, separation and swept collisions of main parts account for about 25% of all
    B/C time.
- **Expected:** fewer failed compiles on cabinets and tables (2 runs lost, 3 high-cost
  runs). The docs pages stay under 2k tokens.

**3. SDK: helpers that fail softly and say how to fix the input.**
- **The change:**
  - `RoundedBoxGeometry` clamps the radius to just under half the smallest size, with a
    warning signal instead of an error;
  - a self-intersecting profile names the first crossing edges and their points;
  - `ExtrudeWithHolesGeometry` names the hole that leaves the outline;
  - duplicate shape names in one body get a hint ("use f'foot_{i}'").
- **Evidence:** 10 failed compiles; 2 runs lost; 20 minutes. B pliers 2 got the same
  self-intersection error 3 times without learning where the crossing was.
- **Expected:** 1-2 runs, and fewer rewrites.

**A bug to fix regardless (1 run, C pliers 2).**
- **What happens:** the compile worker writes `.compile-progress.json` with
  `os.replace`. On Windows, `os.replace` fails while another process (an indexer, the
  antivirus or the runner) has the file open, and it killed `run_tests()`. The failed result
  was then cached by `main.py`'s hash, so every later compile returned the same error.
- **The fix:**
  - progress writes are best effort: retry, then skip;
  - results whose failure is internal to the compiler are not cached.
- **Not done here:** this step was analysis only.

## Would a two-phase build (skeleton first, then detail) address the main causes?

**Partly, and it targets the two largest causes.**

**What it addresses:**
- **Slow planning** (57% of the time).
  - Planning the first main.py with every handle, lathe profile and rounded edge is what the
    model deliberates over.
  - A skeleton, with plain primitives at real sizes and the real joints, is a much smaller
    first program.
  - Evidence: A, whose objects are mostly boxes, plans for a median 3.8 minutes (B/C: 6.4)
    with a median 15.2k reasoning tokens before the first compile (B: 24.5k). Its first
    compile passes in 4 of 11 runs (B: 2 of 11, C: 2 of 12).
- **Motion failures of main parts and joint frames.** 45 of 75 collision failures and all
  floating, separation and loop failures are about main parts and joint placement. A
  skeleton exposes them while the program is small, so each fix is quick and cheap.

**What it does not address:**
- **Helper misuse.** It happens when details are added, so it moves to phase 2 unchanged
  (fix 3 still applies).
- **Detail collisions** (30 of 75) also move to phase 2. They are then easier to find,
  because only the new detail changed. But C cabinet 1 shows the risk: an edit after a pass
  created a new overlap and cost 5 turns.
- **The clock.** Two phases add at least one write and one compile, so it needs fix 1 (the
  time limit and the end state) to pay off.

**How to try it:**
- **The loop:** phase 1 must pass the motion checks (floating, separation, swept collision,
  loop closure) before the prompt asks for detail.
- **Phase 2:** the passing skeleton is the fallback best build.
- **Measure** against B on the same 4 prompts: planning reasoning, success, and the quality
  metrics. Detail must not fall back to A's level.
