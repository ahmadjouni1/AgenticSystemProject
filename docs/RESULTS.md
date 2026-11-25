# Results

Every live experiment in the replica so far.

- **Model:** DeepSeek `deepseek-v4-pro` for all of them, with a $0.30-1.00 cap per run.
- **Time limit:** 15 minutes per run until step 10, then 25.
- **Details:** BUILD_LOG.md has the full tables for each step; `runs/` holds every record.

**Two metrics.**
- **Strict success** (used since step 8): the run ended with its latest compile passing every
  blocking check and a final message.
- **Holding a passing build** (added in step 13): the run's best build passed every check,
  the trajectory check included when declared, whatever the end state.

Both come from `scripts/eval_report.py` and `scripts/metrics_summary.py`. Intervals are
Wilson 95%.

## Steps 8-10b: building the agent (development prompts)

| Experiment | Setup | Runs | Strict success | Holding a passing build | Cost / run | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Step 8, first live runs | the loop, prompt v0 | 3 | 3/3 | 3/3 | $0.050 | the laptop copied the docs example (94% identical): not evidence of skill |
| Step 9, context policy | before / after (rules a-e) | 8 / 8 | 8/8 → 7/8 | 8/8 → 7/8 | $0.056 → $0.056 | input tokens -57%, peak input -32%; cost unchanged (90% of it is output reasoning); the failure was a pre-existing bug, fixed |
| Step 10, prompts and docs | A: step 9 prompt | 12 | 9/12 | 9/12 | $0.106 | |
| | B: v1 prompt + doc pages | 12 | 7/12 | 8/12 | $0.136 | |
| | C: B + low planning effort | 12 | 7/12 | 8/12 | $0.140 | |
| Step 10b, fixes | **B2**: B + fixes | 12 | **10/12** | 10/12 | $0.114 | $0.137 per success; became the default |
| | D: B2 + two-phase | 12 | 9/12 | 10/12 | $0.146 | more detail (89% non-box shapes); every failure in both variants is the crank-slider pump |

Steps 10 and 10b used 4 prompts: a cabinet, a folding table, a crank-slider pump and
pliers. **Spend:** step 10 $4.73, step 10b $3.12.

## Step 12: the benchmark (eval_v1) and the Articraft baseline

**The benchmark:** 12 new prompts in 5 categories; none is a docs example.

- **eval_v1:** our agent (B2), 3 runs per prompt.
- **Articraft:** run once per prompt in WSL, with the same model and caps, and judged by its
  own checks.

| Category | eval_v1 strict | Articraft (its own checks) | eval_v1 cost / run | Articraft cost / run |
| --- | --- | --- | --- | --- |
| single hinge or slide | 6/6 | 1/2 | $0.049 | $0.085 |
| multi-joint chains | 5/8 | 1/3 | $0.238 | $0.259 |
| independent joints | 6/6 | 1/2 | $0.136 | $0.197 |
| closed loops | **3/8** | 1/3 | $0.236 | $0.150 |
| everyday tools | 6/6 | 2/2 | $0.121 | $0.204 |
| **overall** | **26/34 (76%, 60-88%)**; holding 27/34 | **6/12 (50%, 25-75%)**; holding 6/12 | $0.165 | $0.183 |

- **Cost per success:** $0.216 for ours, $0.367 for Articraft.
- **Spend:** eval_v1 $5.89 (one run cut off by HTTP 402 is counted apart); Articraft $2.29.
- **Why Articraft failed:** in 5 of its 12 runs, a reply hit the 32k output limit and ended
  the run. Of the other 7 runs, 6 succeeded.

## Step 13: dimensional synthesis on closed loops

**The tool:** our teammate's nested solver, wrapped as plug-ins and offered as a
`synthesize` tool (prompt set v2). It was tried on the closed-loop prompts, the weak spot
in every earlier evaluation. There were three versions:

- **synth_v1:** path targets only. The run was stopped after 5 runs, when the PC had to
  sleep.
- **synth_v2:** added rocker sweep and slider stroke targets, a supported list, and a
  precise rule.
- **synth_v3:** added a skeleton main.py, already compiled, returned by the tool.

| Wipers + pump | Runs | Strict success | Holding a passing build | Cost / run | Reasoning before 1st compile (median) | Tool used |
| --- | --- | --- | --- | --- | --- | --- |
| no tool (eval_v1, step 10b B2) | 6 | **3/6** | 3/6 | $0.229 | 37k | - |
| synth_v1 | 1 (+1 interrupted) | 0/1 | 0/1 | $0.165 | 75k | 0/1 |
| synth_v2 | 3 | 0/3 | 0/3 | $0.191 | 90k | 3/3 |
| synth_v3 | 4 | 0/4 | **2/4** | $0.190 | 51k | 4/4 |

Over the four closed-loop prompts (adding the scissor lift and pantograph), no tool scored
4/11, synth_v1 1/3 and synth_v2 0/6. **Spend:** $0.65, $1.13 and $0.76.

**Offline accuracy, on known answers:**
- **Serial arm:** reaches the known design (0.004% off) in 2 s.
- **Path synthesis:** four-bar and slider-crank reproduce the path within 0.07-0.16 mm, in
  37-118 s.
- **Sweep and stroke:** a 90° wiper sweep comes within 0.07° and a 60 mm stroke within
  0.0001 mm, in 1-5 s.
- **The skeleton:** compiles with no warnings in every test.

## Main findings, in plain words

1. **The agent is reliable on open mechanisms and weak on closed loops.**
   - Single joints, independent joints and everyday tools: 18/18.
   - Chains: 5/8.
   - Closed loops: 3/8 in eval_v1, and the pump fails in most setups.
2. **Most failures are planning that runs out of output.**
   - The model reasons for 32k tokens (the provider's output cap) before writing anything.
     The reply is cut off and contains nothing usable.
   - This is the main failure cause for us and for Articraft (5 of its 12 runs).
3. **Cutting the context saved tokens but not money.** Step 9: input -57%, cost unchanged.
   The cost is the reasoning output.
4. **The step 10b fixes were the best change measured:** 10/12 at $0.137 per success.
5. **On the benchmark, our agent beat Articraft:** 76% against 50%, and $0.216 against
   $0.367 per success. The intervals overlap, so this is a strong hint, not a proven
   difference.
6. **The synthesis tool answers correctly but did not raise success** in three tries.
   - **synth_v1:** a tool that cannot express the task (a wiper's sweep) made the model
     plan longer.
   - **synth_v2:** correct numbers did not stop the model from planning the whole object in
     its head.
   - **synth_v3:** a ready skeleton gave passing, path-checked builds in 2/4 runs (the
     pumps). Each run then replaced its skeleton with one big rewrite that failed before
     the time limit.
   - The two loop fixes added since (finish on the best build; edits only after a pass)
     target exactly that. They are tested with scripted models and **not yet measured
     live**.

## Limitations

- **Small samples, wide intervals:**
  - 1-3 runs per prompt, and 12-34 runs per setup.
  - A 95% interval spans 30-60 percentage points; no difference reported here is
    statistically established.
  - The synthesis re-tests have 3-6 runs each.
- **One model, one provider:**
  - Every run used DeepSeek `deepseek-v4-pro`.
  - Results may not transfer to other models. No OpenAI run was made; the Responses API
    adapter does not exist yet.
- **Different checkers:**
  - Articraft's builds were judged by its own checks, not ours. Its USDZ outputs cannot be
    read without `usd-core`, which was not approved.
  - Its runs also used high reasoning effort on every turn, and had the teammate's two
    synthesis tools available (unused).
- **Prompts are not independent of development:** steps 8-10b tuned the agent on their own
  prompts. The benchmark prompts are new, but the closed-loop re-tests reuse the benchmark
  and the step 10 pump.
- **Conditions varied:**
  - Off-peak and peak pricing.
  - One evaluation was interrupted by out-of-credit (HTTP 402) and resumed.
  - One run batch overlapped a CPU-heavy game for its first minutes.
  - The time limit grew from 15 to 25 minutes at step 10b.
- **The strict metric and the time limit interact.** Some failures hold a passing build
  (the second metric), so they are failures of finishing, not of building.
