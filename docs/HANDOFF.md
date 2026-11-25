# Handoff: what to do next, in order

**Where things stand:** the agent works on open mechanisms and is weak on closed loops. The
synthesis tool answers correctly, but has not raised success yet.

**Read first:**
- `docs/RESULTS.md`: every experiment, both metrics;
- `docs/SOLVERS.md`: your solver, the plug-ins and the skeleton;
- `BUILD_LOG.md`, step 13: the full story.

**Budget:** the DeepSeek balance is about $1.3, so ask before any live call, and run
off-peak only (not 01-04 or 06-10 UTC on weekdays).

## 1. The 32k output cut-off during planning (the main failure cause)

**Evidence:**
- **synth_v2:** 5 of 6 runs never wrote main.py; each spent 2-4 replies of 32k reasoning
  tokens and was cut off.
- **Articraft baseline:** 5 of its 12 runs ended on a cut-off reply.
- **Step 10b:** the pump runs had 2-3 cut-offs each.
- **synth_v3:** the wipers planned the whole car (56-85k tokens) before any compile.

**What has not worked:**
- Low planning effort (step 10, variant C: 7/12, no gain).
- Telling the model to keep it short (the `REPLY_CUT_OFF` reminder).

**Ideas, cheapest first:**
- **Make the first turn a tool call.** Force a `write` of a skeleton (`tool_choice`), so
  planning happens in main.py, where the compiler can check it.
- **Stop a long reply early.** Stream the reply, and cut it at about 12k reasoning tokens
  with a reminder (the reasoning is lost anyway at 32k).
- **Smaller tasks.** Split the object into parts, one compile per part.

**Where:**
- `agent/loop.py`: `REPLY_CUT_OFF`, `EffortPolicy`, `timed()` (the output cap), and the
  task text in `PROMPT_SETS`;
- `llm/openai_chat.py`: streaming, and `tool_choice`.

## 2. A live re-test of the loop fixes and the solver, with a no-tool control

**Evidence:**
- synth_v3's two pump runs held a passing, stroke-checked skeleton (holding 2/4), then lost
  it to a whole-file rewrite and the time limit.
- The two loop fixes added since, `LoopFixes` (finish on the best build at the time limit;
  refuse whole-file writes after a pass), are tested only with scripted models
  (`tests/test_loop_fixes.py`).

**What to run** (when there is budget; about $0.19 per run):

```text
uv run python scripts/eval.py --name fixes_tool --runs 2 --jobs 2 --benchmark prompts/benchmark_synthesis.json --max-total 1.6 --off-peak-only --agent-args "--prompt-set v2"
uv run python scripts/eval.py --name fixes_control --runs 2 --jobs 2 --benchmark prompts/benchmark_synthesis.json --max-total 1.6 --off-peak-only --agent-args "--prompt-set v1"
uv run python scripts/synthesis_report.py
```

- **The fixes are on by default** for both arms. `--no-finish-on-best` and
  `--allow-rewrites` switch them off.
- **Compare with both metrics.** A run that "finished on best build" is not a strict
  success.
- **Add the new evaluations** to `SETUPS` in `scripts/synthesis_report.py`, and to `EVALS`
  in `scripts/metrics_summary.py`.

## 3. Wipers need a two-rocker skeleton (or one rocker plus a mimic)

**Evidence:** both synth_v3 wipers runs got a compiled single four-bar skeleton. Both then
redesigned the whole object to add the second wiper, and failed (one never compiled, one
overlapped). "A pair of wipers driven by one motor" is not one four-bar.

**Two options:**
- **Easier:** one rocker, plus a second wiper pivot on a mimic joint of the rocker. This
  needs a `twin` option in the skeleton. The mimic is kinematically exact for symmetric
  wipers.
- **The real mechanism:** a second coupler from the first rocker to a second rocker (a
  six-bar). This needs a new linkage in `solvers/planar.py` (`LINKAGES`), with its sweep
  in `solvers/function.py` (`OUTPUT_JOINT`, `FREE`).

**Where:**
- `solvers/skeleton.py` (`_four_bar`; layers as documented there);
- `agent/tools/synthesize.py` (constraints passthrough);
- tests in `tests/test_solvers.py` ("the skeleton" section). Every skeleton must compile
  with 0 failures and 0 warnings.

## 4. OpenAI Responses API adapter (needed before any OpenAI run)

**Evidence:** `llm/openai_chat.py` speaks Chat Completions only. OpenAI's reasoning models
keep their reasoning across tool calls through the Responses API, the way DeepSeek's
`reasoning_content` is echoed now. No OpenAI run has been made.

**Where:**
- **The adapter:** a new `llm/openai_responses.py` implementing `llm.protocol.Model`
  (`query(messages, tools) -> Reply`), with the same usage, cost, budget, retry and call-log
  hooks as `OpenAIChatModel`.
- **The role:** in `llm/config.py`.
- **The tests:** with a fake transport, as in `tests/test_llm.py`.
- **Model IDs:** check OpenAI's current docs for model names and prices. Never guess them.

## 5. Plug-in B speed (0.5-2 minutes per path solve)

**Evidence:**
- Path solves take 37-118 s (`scripts/synthesis_bench.py`), about 4 times as long with a
  busy CPU.
- Most of it is the polish: its finite differences need a new 73-pose crank sweep per free
  parameter. One pose of our loop solver costs about 2 ms.
- Sweep and stroke targets are fast (1-5 s).

**Ideas:**
- **A closed-form inner model.** Use the four-bar's and slider-crank's position equations
  (circle intersection) inside the optimizer, and keep the replica `Assembly` only for the
  final verification. That is likely 100 times faster.
- **Cheaper finite differences.** Fewer grid poses during the polish, or a coarser step.

**Where:** `solvers/planar.py` (`LoopModel.curve`, `sweep`) and `solvers/search.py`
(`polish`).

## 6. Plug-ins for scissor lifts and pantographs

**Evidence:**
- Both are now "unsupported" by design. Without the tool they went 0/3 (scissor) and 1/2
  (pantograph) in eval_v1, and 0/3 and 1/2 across synth_v1-v2.
- Both are parallelogram linkages with simple closed forms:
  - **Scissor:** lift height against actuator stroke or pin spacing; n stages multiply the
    rise.
  - **Pantograph:** the scale is the ratio of the pivot distances.

**Where:**
- **The plug-ins:** new modules next to `solvers/planar.py`, registered in
  `solvers/base.py` (`SUPPORTED`, `plugins()`).
- **The skeletons:** in `solvers/skeleton.py`.
- **Targets:** function targets (`stroke`, a scale, or `crank` + `output`) fit both.

## Running everything offline (no network, no keys)

```text
uv sync
uv run pytest -q                                   # about 15-20 minutes; 1 test skipped
uv run pytest tests/test_solvers.py -q             # solver, plug-ins, tool, skeleton
uv run python scripts/synthesis_bench.py           # solve times and accuracy
uv run python scripts/metrics_summary.py           # both metrics for every setup
uv run python scripts/synthesis_report.py          # closed-loop comparison
uv run python scripts/eval_report.py --name v1 --no-log
uv run python -m viewer runs/<run folder>          # look at any recorded build
```

**Rules:**
- Never edit `solvers/nested/` (wrap it instead).
- Keep Windows support.
- Run the tests before committing.
- Never commit `.env`.
- Keep the docs pages within their token limits, which `tests/test_docs.py` checks.
