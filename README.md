# Replica

A step-by-step rebuild of [Articraft](https://github.com/mattzh72/articraft), a system
where a language model writes Python against a small 3D SDK to build **articulated
objects**: laptops, lamps, linkages, gantries. The goal is to understand every part
by building it again, measuring it, and writing down what we kept, simplified or
improved.

Each step is recorded in [BUILD_LOG.md](BUILD_LOG.md):
- what was built;
- how Articraft does it;
- what changed and why;
- open ideas.

The study of the original that guides the steps is
[docs/ARTICRAFT_STUDY.md](docs/ARTICRAFT_STUDY.md).

## Install and test

You need Windows, macOS or Linux and [uv](https://docs.astral.sh/uv/). Development
happens on Windows 11 with Python 3.13 (pinned in `.python-version`).

```shell
uv sync
uv run pytest -q
```

The suite has 337 tests, none of which use the network. It takes about 2 min, most of it in the compiler tests, which
start real subprocesses. The viewer's JavaScript tests run under Node.js if it is
installed, and are skipped otherwise.

Try the examples:

```shell
uv run python -m examples.four_bar      # a loop-closure linkage, crank turned in steps
uv run python -m examples.gantry        # mimic joints
uv run python scripts/measure_cold_start.py
```

Open models in the viewer (a local page; three.js is vendored, so it works offline):

```shell
uv run python scripts/make_demo_runs.py   # compiles runs/four_bar, runs/drawer, runs/desk_lamp
uv run python -m viewer runs/four_bar     # opens the browser; Ctrl+C to stop
```

Check the LLM providers with your own keys (a tiny tool-call round trip each; a few cents):

```powershell
$env:OPENAI_API_KEY = "..."; $env:DEEPSEEK_API_KEY = "..."
uv run python scripts/llm_smoke.py
```

Keys are read only from those environment variables, or from `replica/.env` for any that
are not set (git-ignored; `NAME=value` lines). They are never logged.

Run the agent on a prompt (DeepSeek by default, $1 and 100 calls per run; keys from the
environment or `replica/.env`):

```shell
uv run python -m agent "a drawer that slides out of a cabinet"
uv run python -m viewer runs/<the run folder it prints>
```

By default the agent uses the context policy (`--context v1`: checkpoints, pinned file
state, compaction by the cheap model), low reasoning effort once the latest compile passed
(`--effort adaptive`) and a 25-minute limit (`--max-time 1500`; at the limit, a passing,
fresh build gets one final turn and counts as a success; no fix cycle starts with less than
`--fix-reserve 120` seconds left). `--two-phase` asks for a skeleton that must pass every
check before the detail. `--context none --effort default` gives the step 8 behaviour. The model gets the step 10 system prompt and SDK docs
(`--prompt-set v1`; `v0` is step 9's); `--planning-effort low` lowers the reasoning effort
before the first compile. Compare batches of runs with
`uv run python scripts/context_report.py <runs folder> <runs folder>`, or rerun the step 10
experiment with `scripts/step10_experiment.py` and `scripts/step10_report.py` (step 10b:
`scripts/step10b_experiment.py`, then `step10_report.py --folder runs/step10b --variants B2 D`). Each run's
`record.json` has quality metrics of its best build (bodies, shapes, joints, share of
non-box shapes, helpers used, triangles), and `renders/final/` its 4 views.

Call the agent's tools by hand (write, compile, render, probe) on the laptop and the drawer:

```shell
uv run python scripts/tools_demo.py
```

## What is here

| Folder | What it holds |
| --- | --- |
| `sdk/` | Bodies, frames, joints (fixed / revolute / prismatic, limits, mimic), assemblies with loop closures, forward kinematics, Articraft's most useful mesh helpers, and `export` (URDF + GLB + manifest) |
| `compiler/` | Runs `workspace/main.py` in a fresh, killable process, validates it, runs its checks, exports it, and reports `<compile_signals>` |
| `examples/` | 8 articulated objects: laptop, four-bar, gantry, toaster, door hinge, desk lamp, computer mouse, folding chair |
| `viewer/` | `python -m viewer <run>`: a local web page with a version picker, joint sliders, Play, and compile signals shown at the pose where they happen |
| `llm/` | The LLM client: one `query(messages, tools) -> Reply` protocol, OpenAI and DeepSeek via the `openai` package, usage and cost per call, per-run budget, retries, a call log, and `ScriptedModel` for tests. Roles and prices are in `llm/config.py` |
| `prompts/` | The system prompt (v1), step 9's frozen prompt and docs (`v0/`), the starter `main.py`, and every evaluation prompt (`eval_prompts.json`) |
| `agent/` | The agent loop (`python -m agent`: turns, stop rule, end states, run folder with `record.json`) and its tools: `read`, `write`, `edit`, `compile`, `render` (CPU PNG views), `probe` (positions, gaps, collisions at a pose). Structured errors, workspace-only paths, no shell |
| `tests/` | pytest suite, with a small standalone URDF forward-kinematics reader |
| `scripts/` | Benchmarks and the generator for `docs/SHAPE_HELPERS.md` |
| `docs/` | The Articraft study, the ranked table of its shape helpers, and `docs/sdk/`: the agent's SDK docs (quickstart with a complete example, shapes, joints and loops, testing; every example is compiled by the tests) |

## Build status

The steps follow the parts of the original listed in the study.

| Step | Part | Status |
| --- | --- | --- |
| 1, 1b, 1c | Geometry SDK: bodies, frames, joints, mimic, loop closures, walking warm start, 26 shape helpers, `frame_in`, `body.at(feature)`, manifold3d booleans | Done |
| 2 | Export: URDF + GLB + OBJ + JSON manifest, versioned `result/vNNNN/` | Done |
| 3 | Compiler: subprocess worker, Windows and POSIX process-tree kill, typed compile signals, cache, repeat detection | Done |
| 4 | Checks: run_tests(ctx) API, mesh health, floating parts, overlap, swept collisions (revolute + prismatic, loops), separation, scale-relative tolerances, blocking policy | Done |
| 5 | Viewer: loader, local server, vendored three.js, tree FK in the browser, loop poses from the server, signal overlay | Done |
| 6 | LLM client: neutral messages, OpenAI + DeepSeek, usage, cost (price table dated), budget caps, retries, prompt-cache-friendly prefix, call log, ScriptedModel | Done |
| 7 | Agent tools: read, write, edit, compile, render, probe; structured errors; path scoping; output cap; fixed schema order | Done |
| 8 | Agent loop: prompts v0, stop rule (fresh passing compile), end states, best build, run folder with conversation and per-turn record, CLI | Done |
| 9 | Context and memory: checkpoints (old reasoning not re-sent), pinned file state, doc notes, old-output shortening, compaction, reasoning cap on routine turns, wall-clock budget, optional lessons file; measured before/after | Done |
| 10 | System prompt v1 (short written plan, realism rules, finish checklist) and SDK docs (quickstart, shapes, joints and loops, testing); quality metrics and final renders per run; A/B/C experiment (more detailed objects; planning reasoning not reduced) | Done |
| 10b | Fixes from the step 10 failure analysis: 25-minute limit with a final turn and a fix reserve, joint contact rule and recipes, soft-failing helpers, a compiler fix on Windows; `--two-phase` option; B2/D experiment prepared | Implemented; experiment not run |
| 11 | Run storage | Planned |
| 12 | Evaluation | Planned |
| 13 | Extras (images, simulation, trajectories) | Planned |

**Known risk: there is no sandbox.** As in Articraft, the compiler runs model-written
code with your own permissions. Only `*_API_KEY` / `*_TOKEN` environment variables are
removed. See BUILD_LOG.md, step 3.

## Attribution

This project is inspired by **Articraft**:
[github.com/mattzh72/articraft](https://github.com/mattzh72/articraft), and its
research paper, [arXiv:2605.15187](https://arxiv.org/abs/2605.15187).

It includes code adapted from Articraft, which is licensed under the Apache License 2.0.
Every file that contains adapted code says so at the top. [NOTICE](NOTICE) lists what was
adapted, file by file.

## License

[Apache License 2.0](LICENSE) for the whole repository. See [NOTICE](NOTICE) for
attribution.
