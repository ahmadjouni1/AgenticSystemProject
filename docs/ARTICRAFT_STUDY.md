# Articraft study: how the original works, step by step

Source: `Previous work/Agentic_Systems_Project` at `f3853d3`. Paths below are relative to that
folder. The teammate's commit `03f4c6c` (DeepSeek provider, `synthesize_trajectory`,
`sdk/trajectory.py`, `sdk/_synthesis.py`, trajectory viewer, stderr prints in the harness) is
ignored except in step 13. Token counts use tiktoken `o200k_base`.

**Core idea:** `prompt -> LLM -> tools (read / edit / compile) -> record`. The model writes
`main.py` against an SDK. A subprocess compiles it, checks it, and exports USDZ. The run ends
when the latest workspace compiles and the model gives a visible answer.

## 1. Geometry SDK

**Files:** `sdk/assembly.py` (`RigidBodyAssembly`, `Joint`, `JointDOF`, `JointAxis`,
`Articulation`, `_propagate_transforms`, `_solve_closed_loops`), `sdk/bodies.py`
(`RigidBody.add/.at`), `sdk/frames.py` (`JointFrame`, `BodyFrame`, `WORLD`), and `sdk/_mesh/*`
(about 6k lines).

**How it works.**
- **Bodies and shapes.** An assembly holds rigid bodies, and each body holds named shapes. A
  shape is either a build123d `Shape`, tessellated at 1 mm, or a `MeshGeometry` (vertices and
  faces in meters). Mesh primitives include box, rounded box, cylinder, cone, sphere, capsule,
  torus, lathe, extrude, loft, pipe and sweep. Manifold3d does the booleans.
- **Joints.** Every joint is a D6 joint between two frames bound with `body.at(point or
  feature)`. The DOFs you list are free and the rest are locked. No DOF means fixed, one
  rotation means a hinge, one translation means a slide, and three rotations mean a ball.
- **Limits.** The two frames coincide at rest, so every limit range must contain 0.
- **Loops.** You author every physical joint. The articulation names a spanning tree, and
  joints left out of it are loop closures. `forward_kinematics` solves the free DOFs with
  `scipy.least_squares` (dogbox, bounded), stepping in 5 increments from zero. It raises
  `LoopClosureError` when the residual is above 1e-6.
- **Mimic joints: none.** Coupled motion is only possible through real loops.

```python
model.joint("body_to_lid", body.at((0.0, 0.05, 0.05)), lid.at(),
            dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.8)),))
model.articulation("main", root="body", joints=["body_to_lid"])
# _propagate_transforms: child = parent @ frame0 @ motion @ inv(frame1)
if known0 and not known1:
    transforms[joint.body1] = transforms[joint.body0] @ frame0 @ motion @ np.linalg.inv(frame1)
```

**Data:** `JointFrame(xyz, rpy)` in meters and radians (sxyz order). DOF ids look like
`"joint.rotX"`. `PhysicsState` holds `{body: 4x4}` plus `{dof_id: value}`.

**Weak points / ideas:**
- No mimic or gear joints. A `mimic=(joint, ratio, offset)` option would be cheap to add.
- Requiring 0 inside every limit range is awkward, for example for a lid authored closed.
- build123d uses degrees and the SDK uses radians, which is a trap.
- The mesh library is very large. Start the replica with build123d plus about 5 primitives.

## 2. Export (meshes, URDF/USDZ/GLB)

**Files:** `sdk/export.py` (`export_assembly`, `_write_body_geometry`, `_graph_joint_schema`,
`_assembly_to_payload`, `_audit_assembly_usdz`).

**How it works.**
- **Resolve and mesh.** The assembly is resolved. Each shape is turned into a trimesh mesh with
  crease-angle normals, a collider and a mass.
- **USD stage.** The stage is Z-up and in meters. It holds a physics scene, one Xform per body,
  typed joints and an `ArticulationRootAPI`.
- **Package and check.** The stage is packed to USDZ, validated with `UsdValidation`, read back
  and audited.
- **Formats.** USDZ is the only real output. There is no URDF or GLB in the core. URDF exists
  only in the teammate's kinematic-only exporter, and MJCF only inside `simulate.py`.

```python
if joint.is_fixed:
    schema = UsdPhysics.FixedJoint.Define(stage, path)
elif joint.is_revolute:
    schema = UsdPhysics.RevoluteJoint.Define(stage, path)
    schema.CreateAxisAttr(axis.value[-1].upper()); _write_specialized_limits(schema, joint)
elif joint.is_prismatic:
    schema = UsdPhysics.PrismaticJoint.Define(stage, path) ...
else:  # D6: one LimitAPI per axis; a locked axis is low=1, high=-1
    schema = UsdPhysics.Joint.Define(stage, path)
```

**Data:**
- USDZ files: `result/usdz/NNNN.usdz`, with the index incremented on every export.
- Manifest: `result/model.json` (schema v2) with `scene`, `rigid_bodies[shapes, mass,
  material]`, `joints[type, body0/1, frame0/1, dofs, exclude_from_articulation]`,
  `articulations` and `reference_state`.

**Weak points / ideas:**
- `usd-core` is heavy and pinned below 26.8.
- `model.json` is overwritten on each export instead of versioned.
- A compile with failed checks can still leave a USDZ behind.
- For the replica, make URDF + GLB (via trimesh) the main outputs and USDZ optional.

## 3. Compiler (running model code, sandboxing, timeouts, errors)

**Files:** `agent/workspace/local.py` (`compile_path`, `_run_isolated_process`),
`compiler/worker.py` (`_compile_workspace`, `_CompileTracker`), `compiler/feedback.py`
(`render_compile_signals`, `compile_failure_signature`), `agent/_child_process.py`.

**How it works.**
- **Process.** Each compile runs `python -m articraft.compiler.worker --raw <run_dir>` in a new
  process group with a 900 s timeout. On timeout it sends SIGTERM, then SIGKILL.
- **Worker.** It loads `main.py` with `runpy`, runs `run_tests()`, runs the baseline checks
  (step 4), and exports. It prints one JSON payload.
- **Timeout diagnosis.** Phase progress goes to `result/.compile-progress.json`, so a timeout
  names the phase that hung.
- **Sandboxing.** It only strips `*_API_KEY` variables from the environment. There is no
  filesystem or network isolation.
- **Error feedback.** Errors become typed `CompileSignal`s, rendered into a `<compile_signals>`
  block with fix advice for each failure kind.
- **Repeat detection.** A SHA-1 signature of the failures detects a repeated failure. After 3
  failures in a row, the block advises one `exec_command` inspection before editing again.

```python
with tracker.phase("loading main.py and building the model"):
    globals_dict = runpy.run_path(str(workspace / "main.py"), run_name="__main__")
object_model = globals_dict.get("object_model")
if not isinstance(object_model, RigidBodyAssembly):
    raise TypeError("main.py must define object_model as a RigidBodyAssembly")
with tracker.phase("running authored tests"):
    authored_report = _run_required_tests(globals_dict)
baseline_report = _run_baseline_tests(object_model, authored_report, tracker=tracker, ...)
test_report = _merge_test_reports(authored_report, baseline_report)
with tracker.phase("exporting and validating the USDZ file"):
    export_result = export_assembly(object_model, export_dir)
_raise_for_failed_test_report(test_report)
```

**Data:** the model sees only `{"status", "compile_signals": "<compile_signals><summary>
status=error failures=1 ...</summary><failures>...</failures><response_rules>...</...>"}`.
Stdout, traceback and the full report go to the record and the TUI.

**Weak points / ideas:**
- **It does not run on Windows.** `os.killpg`, `start_new_session`, `-lc` shells and the docs
  symlink are all POSIX-only. The replica needs `CREATE_NEW_PROCESS_GROUP` plus `taskkill /T`.
- A cold start costs about 3 s per compile.
- There is no real sandbox.
- The signal format is excellent. Keep it.

## 4. Checks (structure, collisions, poses, floating parts)

**Files:** `sdk/testing.py` (`TestContext`, `FailureKind`, `fail_if_*`, `expect_*`, `allow_*`),
`sdk/_collision.py` (`MeshCollisionKernel` on python-fcl), `sdk/_mesh/health.py`,
`worker._run_baseline_tests`.

**How it works.**
- **Authored checks.** `run_tests()` adds prompt-specific checks such as `expect_distance`,
  `expect_contact`, `expect_gap`, `expect_no_collision_at_poses`, `sample_dof`, and
  `track_point`.
- **Baseline checks.** These always run:
  - structure is valid;
  - mesh health (degenerate triangles, bad edges, winding, debris);
  - no floating parts: a distance graph from the root body, with a 1e-6 m contact tolerance;
  - no disconnected islands within a body;
  - no scale outliers;
  - no overlap at rest (more than 5 mm deep and 5e-7 m³);
  - hinges don't separate: 4 samples per rotational joint, failing if the gap grows by more
    than 3 mm.
- **Allowances.** The author's `allow_*` exceptions carry over into the baseline checks.
- **Blocking.** Only validity, mesh health and missing mass block a compile. Overlap, floating
  and separation are only diagnostics, despite their `fail_` names.

```python
ctx.fail_if_mesh_unhealthy(); ctx.fail_if_isolated_parts()
ctx.warn_if_part_contains_disconnected_geometry_islands(); ctx.warn_if_absurd_dimensions()
ctx.fail_if_parts_overlap_in_current_pose(); ctx.fail_if_articulation_separates_child()
blocking_kinds = {FailureKind.MODEL_VALIDITY, FailureKind.MESH_HEALTH, FailureKind.MISSING_MASS}
blocking = tuple(f for f in report.failures if f.kind in blocking_kinds)
diagnostics = tuple(f for f in report.failures if f.kind not in blocking_kinds)
```

**Data:** `TestReport{passed, checks, failures[name, details, kind], warnings, allowances,
metrics, artifacts, diagnostics}`.

**Weak points / ideas:**
- Collisions are checked only at the rest pose.
- Prismatic joints are never swept, and 4 samples is coarse.
- Tolerances are absolute (5 mm is wrong for tiny objects).
- The pairwise checks are O(n²).
- Add a swept collision check over each joint's range and scale-relative tolerances.

## 5. Viewer

**Files:** `viewer.py` (`load_viewer_run`, `kinematics_from_usdz`, `_solve_pose`, `_handler`),
`viewer.html` (three.js 0.185.1 from a CDN).

**How it works.**
- **Server.** `articraft view <run>` starts a stdlib `ThreadingHTTPServer` on 127.0.0.1 and
  lists each numbered USDZ as a version.
- **Sliders.** A slider move POSTs to `/api/pose/N`. The server rebuilds the kinematics from
  the USDZ (cached per version), runs `forward_kinematics` (which solves loops), and returns a
  4x4 per body.
- **Unreachable poses** return HTTP 409, and the page keeps the last good pose.

```python
for item in model.joints:
    if item.exclude_from_articulation:
        continue  # a closure's value is derived, never driven
    for dof in item.joint.dofs:
        if item.joint.name in values:
            positions[item.joint.dof_id(dof)] = float(values[item.joint.name])
state = model.forward_kinematics(positions)
return {"bodies": {name: flattened_4x4, ...}, "dofs": state.dof_positions}
```

**Data:** `/api/bootstrap` returns versions, joints and limits. `POST {joint: value}` returns
`{bodies: {name: 16 floats}, dofs}`.

**Weak points / ideas:**
- The CDN means it needs internet.
- Every slider tick costs an HTTP round trip plus a Python solve.
- Replica: load GLB, run tree FK in JavaScript, and call the server only for loops.

## 6. LLM client (providers, token/cost tracking, caching)

**Files:** `agent/protocols.py` (`Model.query`, `ContextSummarizer`), `agent/provider/`
(`openai.py`, `anthropic.py`, `gemini.py`, `openrouter.py`, `_chat.py`), `api.py`,
`settings.py`.

**How it works.**
- **Interface.** Each adapter converts a neutral message list and returns `{text, tool_calls,
  token_usage, cost, provider_content}`.
- **OpenAI (default).** Uses the Responses API over WebSocket with `store=False`. A
  `previous_response_id` chain means only new items are sent, with a fallback to the full
  input.
- **Anthropic.** Uses automatic prompt caching (`ephemeral`, 1h) and replays thinking blocks
  through `provider_content`.
- **Retries.** Up to 4 attempts with exponential backoff (0.5–20 s), inside each adapter only.
- **Cost.** Computed from hard-coded price tables, including cache read and write rates. The
  running total of tokens and cost is saved to `record.json` after every call.

```python
request = {"model": self.config.anthropic_model, "max_tokens": _MAX_OUTPUT_TOKENS,
           "messages": _messages(messages),
           "cache_control": {"type": "ephemeral", "ttl": "1h"}}
cost = (usage["input_tokens"] * p.input_price
        + cache_5m * p.input_price * 1.25 + cache_1h * p.input_price * 2.0
        + usage["cache_read_input_tokens"] * p.input_price * p.cache_read_multiplier
        + usage["output_tokens"] * p.output_price) / 1_000_000
```

**Data:** messages are OpenAI-Responses-shaped:
- `{"role", "content"}`
- an assistant message with `tool_calls: [{"id", "name", "arguments": "<json>"}]`
- `{"type": "function_call_output", "call_id", "output"}`

**Weak points / ideas:**
- Prices go stale, and an unknown model is recorded as costing $0.
- There are 4 format converters.
- There is no cost cap.
- Replica: support 1–2 providers behind the same small protocol.

## 7. Agent tools

**Files:** `agent/tools/__init__.py` (registry), `_core.py` (`Tool`, `ToolContext`,
`workspace_digest`, path scoping), one module per tool, `_exec.py` (`ExecSessions`).

| Tool | Input | Output | Used when |
| --- | --- | --- | --- |
| `read` ‖ | `path`, `offset?`, `limit?` | `{path, text}` with `L12: …` line numbers | reading `main.py` and docs under `docs/sdk/` |
| `view_image` ‖ | `path`, `detail` | metadata + image content | inspecting preview PNGs, the reference image, doc figures |
| `edit` | `path`, `edits[{old_text, new_text}]` | `{path, replaced}`; each old text must match once; all or nothing | targeted repairs after compile |
| `write` | `path`, `content` | `{path, bytes}` | first `main.py`, `previews.py`, helpers |
| `exec_command` | `command`, `cwd?`, `timeout?`, `yield_time_ms` (10 s, max 30 s), `max_output_tokens` (10k) | `{stdout, stderr, returncode, session_id, running, timed_out}` | rendering previews, quick inspections via `$ARTICRAFT_PYTHON` |
| `write_stdin` | `session_id`, `chars` | same as `exec_command` | polling a long command; every other tool is blocked meanwhile |
| `compile` | none | `{status, compile_signals}`; cached if the digest is unchanged | after each real change and before finishing |
| `synthesize_trajectory` (teammate) | `task_path`, … | dimensions, TCP, joint path | explicit end-effector path prompts only |

‖ = may run in parallel. All other tools run one at a time, in order.

```python
@dataclass(frozen=True)
class Tool(Generic[ToolOutput]):
    name: str
    schema: dict[str, Any]
    run: Callable[[ToolContext, dict[str, Any]], Awaitable[ToolOutput]]
    supports_parallel: bool = False   # read/view_image are batched with asyncio.gather
```

**Weak points / ideas:**
- Errors come back as a bare `{"error": str}`.
- `exec_command` gives the model a full login shell.
- Replica: 5 tools (`read`, `write`, `edit`, `compile`, `render_preview`).

## 8. Agent loop (turn structure, retries, stop condition)

**Files:** `agent/harness.py` (`Agent._run_created`, `_run_tool_calls`, `_run_tool`,
reminders).

**How it works.** Each of up to 100 turns:
1. Compact the context if needed (step 9).
2. Call `model.query(messages, tools)`.
3. Append the reply to memory and to `conversation.jsonl`.
4. Run the tool calls and append their results.

**Stopping.**
- **Success** needs visible text with no tool calls, and a workspace SHA-256 digest equal to
  the one from the last successful compile.
- **Reminders.** Otherwise a `<compile_required>` or `<final_response_required>` user reminder
  is injected.
- **Empty replies.** Three empty replies in a row end the run.
- **Errors.** There are no retries at the loop level. A failed query ends the run with
  `status=error`, and so do running out of turns, a stale compile, and a missing USDZ.

```python
if not tool_calls:
    workspace_is_compiled = _latest_workspace_is_compiled(context)
    if text.strip() and workspace_is_compiled:
        final_text = text
        break
    if consecutive_empty_responses >= MAX_CONSECUTIVE_EMPTY_RESPONSES:
        termination_error = "agent returned three consecutive responses ..."; break
    _append_reminder(self.messages, conversation_path,
        _final_response_required_reminder() if workspace_is_compiled
        else _compile_required_reminder(context))
    continue
await self._run_tool_calls(context, tool_calls, conversation_path)
```

**Weak points / ideas:**
- There is no cost, time or repair budget.
- "Compiled" is not the same as "good", because diagnostics never block.
- Replica: add budgets and explicit stages (plan → build → check → fix).

## 9. Context and memory

**Files:** `agent/compaction.py` (`prepare_compaction`, `Compaction.apply`),
`prompts/compaction.md`, provider `summarize_context` (Anthropic uses native compaction beta
`compact-2026-01-12`), `compile._compact_result`, `_exec.py` caps,
`LocalWorkspace._link_sdk_docs`.

**How it works.**
- **What stays.** History is append-only and verbatim. Messages 0–2 (system prompt,
  preloaded quickstart, task) stay forever.
- **Trimming.** Nothing is trimmed per turn. Only tool outputs are kept small: compile returns
  only its signals, and exec output is capped at 10k tokens (head + tail).
- **Compaction.** It starts once estimated tokens exceed the window minus 16,384, which is
  about 256k for 272k models. It keeps the newest 20k or more tokens and summarizes the rest in
  a separate call (at most 8,192 output tokens, fixed headings). In that summary, tool outputs
  are clipped to 2,000 characters and edit bodies are dropped. The checkpoint lists the
  modified files and inspected images.
- **Caching.** Anthropic 1h cache, OpenAI response chain plus server cache, Gemini implicit.
- **Docs on demand.** The docs tree is symlinked to `workspace/docs/sdk` and read with `read`.
- **Cross-run memory:** none.

```python
tokens_before = _context_tokens(messages)   # last usage.total_tokens + chars/4 for the tail
if tokens_before <= context_window_tokens - RESERVE_TOKENS:
    return None
cut = _recent_cut(messages)                  # keep >= KEEP_RECENT_TOKENS (20k)
...
return [*messages[:STATIC_MESSAGE_COUNT], checkpoint, *self.recent_messages]
```

**Weak points / ideas:**
- Superseded `main.py` reads, old images (about 1.2k tokens each) and old compile reports are
  billed again every turn.
- Typical runs never reach 256k, so they never compact.
- Replica: pin the latest `main.py` and the latest compile signals, drop stale reads and images
  early, and add a cross-run "lessons" file.

## 10. System prompt and SDK docs given to the model

**Files:** `prompts/system.md`, `task.md`, `compaction.md`, `sdk/docs/**`,
`harness._read_prompt` (strips `<image_prompt>` blocks for text-only models),
`_read_sdk_quickstart`.

**Structure.** The system prompt has seven sections: `<role>`, `<quality_requirements>`,
`<workflow>`, `<authoring_contract>`, `<testing>`, `<tools>`, `<final_response>`.
- **Four rules:** realistic geometry, intentional motion, intentional connections, no
  unintended overlap.
- **Workflow:** survey the docs → brief → build → preview and view → compile → repair.

```text
Count the pivots before writing joints. A body pinned in two places takes two
joints, and that makes the mechanism a ring rather than a chain ... Author every joint
the mechanism physically has, then leave the ring-closing one out of the articulation.
```

| Always sent | Tokens | | Docs on demand | Files | Tokens |
| --- | --- | --- | --- | --- | --- |
| `system.md` (image / text-only) | 3,402 / 2,861 | | `common/` (testing 5.0k, joints 1.6k, …) | 11 | 15,804 |
| `task.md` | 256 | | `mesh/` (geometry 5.2k, sweeps 5.2k, …) | 6 | 20,401 |
| quickstart (user message) | 1,210 | | `examples/*.py` | 7 | 4,852 |
| 8 tool schemas | ~1,150 | | build123d md / examples / snippets | 32/67/25 | 65.8k / 49.4k / 35.8k |

The docs total about 192k tokens. A run reads about 10–20k of them.

**Weak points / ideas:**
- The text-only prompt keeps a duplicated "Finish only when…" line.
- The trajectory text (about 250 tokens) is sent on every run.
- About 150k tokens are generic build123d docs.
- Replica: a ~2k-token prompt, an API cheat sheet, and 3–5 curated examples.

## 11. Run storage (folders, versions, resume)

**Files:** `agent/workspace/local.py::create_run`, `agent/record.py`, `harness._create_run`,
`app.py` (`replay`, `view`, `texture`).

```text
runs/<YYYYMMDD-HHMMSS>-<prompt-slug>[-2]/
  record.json          {run_id, status, attempts, error, result, cost, token_usage}
  conversation.jsonl   one JSON event per message / tool result / reminder / compaction
  input/reference.png  (image runs)
  workspace/  main.py, previews.py, *.png, docs/sdk -> packaged docs (symlink)
  result/     model.json, usdz/0000.usdz, 0001.usdz ..., trajectories/NNNN.json
```

**How it works.**
- **Run ids.** A timestamp plus a prompt slug, with a `-2` suffix on collision.
- **Record.** `record.json` is rewritten after every model call and every compile.
- **Versions.** A version is simply the next USDZ index.
- **Resume: none.** `replay` only redraws the log in the TUI.

**Weak points / ideas:**
- Writes are not atomic.
- `conversation.jsonl` already holds everything a `resume` command would need.
- Save a snapshot of `main.py` next to each USDZ.

## 12. Evaluation / tests

**Files:** `tests/` (662 test functions), `tests/harness.py` (`ScriptedModel`, `run_scenario`,
`WarmEnvironment`), `test_live_generation.py`, `benchmarks/sdk_benchmark.py`, `ci.yml`.

**How it works.** Tests fall into four lanes:
- **unit;**
- **warm compile:** a kept-alive worker, about 0.1 s per compile;
- **scripted agent:** fake model replies drive the real loop;
- **live:** one test, "a simple box" must reach `success`.

CI runs ruff, basedpyright and the non-live pytest lanes. The SDK benchmark tracks mesh speed
and quality.

```python
artifacts = run_scenario("a box", [
    calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
    calls(tool_call("compile")),
    text("done"),
], env=WarmEnvironment(output_dir=tmp_path))
assert artifacts.record.status == "success"
```

**Weak points / ideas:**
- There is no evaluation of output quality: no prompt suite, no scores, no model comparison.
- Replica: 10–20 prompts, recording success rate, turns, dollars, failed compiles and the
  diagnostics left at the end. Later, a VLM judge looking at the previews.

## 13. Extras

- **Image input** (`agent/images.py`).
  - The image is resized to a cap ("high" = 2,048 px or 2,500 patches; "original" =
    6,000 px).
  - It is saved to `input/` and the workspace, and attached to the task message.
  - Previews come from `sdk/visual.py::render_view`, a CPU rasterizer built on trimesh and PIL
    that also does sections and motion strips. The model opens them with `view_image` and
    registers them with `attach_artifact`.
- **Simulation** (`simulate.py`). `articraft simulate` translates the USDZ into a MuJoCo
  `MjSpec` and runs a drop test, a sliding-friction test and a released-joints test. Only
  dynamic friction is carried over. `--physics` makes missing mass a blocking failure.
  Textures come from ambientCG.
- **Teammate's trajectory feature** (`03f4c6c`).
  1. The model exports a kinematic URDF and writes a task JSON.
  2. `synthesize_trajectory` runs a nested SciPy optimization (`sdk/_synthesis.py`, in
     `synthesis_worker.py`) over joint-origin dimensions and the joint path.
  3. The model rebuilds the geometry. Compile then verifies it when `trajectory_request.json`
     exists, and saves `result/trajectories/NNNN.json` for the viewer.
  - Limits: open serial chains only, no loops, no mimic joints, no dynamics.

---

## One full run

```text
CLI / api.generate(prompt) -> Model adapter + LocalWorkspace -> Agent._run
  create runs/<id>/ (starter main.py, docs symlink, record.json)
  messages = [system.md, <sdk_quickstart>, task (+image)]
  loop turn 1..100:
    near window-16k? -> summarize_context -> [static 3, checkpoint, recent 20k]
    model.query(messages, tool schemas) -> {text, tool_calls, usage, cost} -> record + jsonl
    tool calls?  yes: read/view_image in parallel; write/edit/exec in order
                      compile -> subprocess worker: runpy main.py -> run_tests()
                              -> baseline checks -> export USDZ + model.json -> JSON
                      <- {status, <compile_signals>}; save workspace digest on success
                 no:  visible text and digest fresh?  -> DONE
                      else inject <compile_required> / <final_response_required>
  record.json {status: success, result: result/usdz/NNNN.usdz} -> `view` / `simulate`
```

Typical sequence: read `main.py` and 3–6 docs (in parallel) → write `main.py` → write
`previews.py` → exec (render) → view_image ×N → compile → edit → compile … → final text.

## Keep / simplify / improve

| Step | Articraft approach | Verdict | Why |
| --- | --- | --- | --- |
| 1 SDK | D6 joints on bound frames, tree + loop closures, build123d + big mesh library | Keep joints; simplify mesh; add mimic | The frame/closure model is clean. The mesh library is 6k+ lines. |
| 2 Export | Validated USDZ + manifest v2 | Improve | URDF + GLB are simpler and widely read. Version the manifest. |
| 3 Compiler | Fresh subprocess, runpy, phase tracker, 900 s timeout | Keep; improve portability | The best part of the design, but it is POSIX-only and has no real sandbox. |
| 4 Checks | Authored TestContext + baseline fcl checks; 3 kinds block | Keep; improve | Add swept collisions and scale-relative tolerances. |
| 5 Viewer | stdlib HTTP, three.js from a CDN, server FK per slider tick | Simplify | GLB with FK in the browser. |
| 6 LLM client | 5 adapters, neutral format, hard-coded prices | Simplify | 1–2 providers, one price table. |
| 7 Tools | 7 core tools + interactive shell | Simplify | 5 tools, with `render_preview` instead of the shell. |
| 8 Loop | Turn loop, freshness digest, reminders | Keep; improve | The freshness gate is valuable. Add budgets and stages. |
| 9 Context | Append-only, compaction near 256k, caching | Improve | Drop stale reads and images early. Add cross-run lessons. |
| 10 Prompt/docs | 3.4k prompt, preloaded quickstart, ~192k docs | Simplify | A shorter prompt and curated docs. |
| 11 Storage | `runs/<id>/`, jsonl log, numbered USDZ | Keep; improve | Add resume and a `main.py` snapshot per version. |
| 12 Eval | 662 tests in 4 lanes, 1 live smoke test | Keep harness; improve eval | There is no quality benchmark. |
| 13 Extras | Image input, previews, MuJoCo, trajectory synthesis | Later | Not part of the core loop. |

## Token budget of a typical run (estimate)

There are no recorded runs in either repo, so this is an estimate built from the measured
static sizes above. **Assumed run:** an image-capable model, a hinged object, about 30 turns,
5 compiles, 6 doc reads, 6 image views and 4 exec calls.

| Bucket | Contents | Tokens at last turn | Share |
| --- | --- | --- | --- |
| System prompt | `system.md` 3.4k + tool schemas 1.15k + `task.md` 0.26k | ~4.8k | 9% |
| Docs | quickstart 1.3k + reads (joints 1.6k, assembly 1.5k, testing 5.0k, visual 1.5k, mesh page ~4k, example ~0.7k) | ~15k | 29% |
| Code | `main.py` write ~4k + `previews.py` ~1k + ~8 edits ~2.5k + 2 re-reads ~4k | ~11k | 21% |
| Tool outputs | compile signals 5×0.6k + exec 4×0.6k + images 6×1.2k + acknowledgements ~1k | ~13k | 25% |
| History | assistant text, replayed reasoning, reminders | ~8k | 16% |
| **Total** | | **~52k** | 100% |

- **Billed input.** The whole context is sent again on every turn. Billed input is therefore
  about 30 turns × 29k average ≈ **0.85M tokens**, about 95% of it cache reads. Output is about
  45k tokens.
- **Cost, from the repo's price tables:** about $1.2 per run on Claude Sonnet 5 ($3 / $15) and
  about $3.7 on GPT-6 Astra ($10 / $1 cached / $50).
- **Compaction never triggers.** The run stays far below 256k.
- **Easiest saving for the replica:** dropping stale reads and images, which are 25–30% of the
  context and are billed again every turn.
