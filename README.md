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

The suite has 187 tests. It takes about 45 s, most of it in the compiler tests, which
start real subprocesses.

Try the examples:

```shell
uv run python -m examples.four_bar      # a loop-closure linkage, crank turned in steps
uv run python -m examples.gantry        # mimic joints
uv run python scripts/measure_cold_start.py
```

## What is here

| Folder | What it holds |
| --- | --- |
| `sdk/` | Bodies, frames, joints (fixed / revolute / prismatic, limits, mimic), assemblies with loop closures, forward kinematics, Articraft's most useful mesh helpers, and `export` (URDF + GLB + manifest) |
| `compiler/` | Runs `workspace/main.py` in a fresh, killable process, validates it, runs its checks, exports it, and reports `<compile_signals>` |
| `examples/` | 8 articulated objects: laptop, four-bar, gantry, toaster, door hinge, desk lamp, computer mouse, folding chair |
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
| 4 | Checks: collisions, poses, floating parts, mesh health | Next |
| 5 | Viewer | Planned |
| 6 | LLM client | Planned |
| 7 | Agent tools | Planned |
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

It includes code adapted from Articraft, which is licensed under the
[Apache License 2.0](LICENSES/Apache-2.0.txt). The adapted portions stay under that
license, and every file that contains them says so at the top. [NOTICE](NOTICE) lists
what was adapted, file by file.
