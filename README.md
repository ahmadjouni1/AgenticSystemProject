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

The suite has 273 tests, none of which use the network. It takes about 75 s, most of it in the compiler tests, which
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

Keys are read only from those environment variables and are never logged.

## What is here

| Folder | What it holds |
| --- | --- |
| `sdk/` | Bodies, frames, joints (fixed / revolute / prismatic, limits, mimic), assemblies with loop closures, forward kinematics, Articraft's most useful mesh helpers, and `export` (URDF + GLB + manifest) |
| `compiler/` | Runs `workspace/main.py` in a fresh, killable process, validates it, runs its checks, exports it, and reports `<compile_signals>` |
| `examples/` | 8 articulated objects: laptop, four-bar, gantry, toaster, door hinge, desk lamp, computer mouse, folding chair |
| `viewer/` | `python -m viewer <run>`: a local web page with a version picker, joint sliders, Play, and compile signals shown at the pose where they happen |
| `llm/` | The LLM client: one `query(messages, tools) -> Reply` protocol, OpenAI and DeepSeek via the `openai` package, usage and cost per call, per-run budget, retries, a call log, and `ScriptedModel` for tests. Roles and prices are in `llm/config.py` |
| `tests/` | pytest suite, with a small standalone URDF forward-kinematics reader |
| `scripts/` | Benchmarks and the generator for `docs/SHAPE_HELPERS.md` |
| `docs/` | The Articraft study and the ranked table of its shape helpers |

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
| 7 | Agent tools | Next |
| 8 | Agent loop | Planned |
| 9 | Context and memory | Planned |
| 10 | System prompt and SDK docs | Planned |
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
