# Articraft baseline (step 12): prepared, not run

`scripts/articraft_baseline.py` runs the step 12 benchmark (`prompts/benchmark_v1.json`) through
the original Articraft, with the same model as our builder (`deepseek-v4-pro`, Articraft's own
DeepSeek provider) and the same per-run caps. It has **not been run** (no budget). Its
command building, cost computation, stopping and record conversion are tested without
WSL or network (`tests/test_eval.py`).

## Why WSL

Articraft does not run on Windows. Its compiler uses `os.killpg`, `start_new_session` and
`-lc` shells, and its docs folder is a symlink, all POSIX-only (study §1 and §3). WSL 2 runs
it unchanged.

## Setup (once)

1. **Install WSL 2 with Ubuntu** (PowerShell as administrator, then reboot):
   `wsl --install -d Ubuntu`. Open Ubuntu once and create the Linux user.
2. **Install uv inside WSL:** `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open
   a new shell. uv installs the Python that Articraft asks for (3.11 or later) on the first
   `uv sync`.
3. **Make a copy of Articraft in the Linux file system.** "Previous work" stays read-only.
   Articraft writes `.venv/` and `runs/`, and its docs symlink needs a Linux file system:

   ```shell
   cp -r "/mnt/c/Users/<you>/Desktop/EECE503P/Main project/Previous work/Agentic_Systems_Project" ~/articraft
   cd ~/articraft && rm -rf .venv runs && uv sync
   ```

   This installs Articraft's dependencies inside WSL only, `usd-core` among them. The
   replica's environment does not change.
4. **Check it without a key:** `cd ~/articraft && uv run articraft --help`.
5. **Do not create a key file in WSL.** Keep `DEEPSEEK_API_KEY` in the Windows environment
   or `replica/.env`, as for our runs. The script forwards it by name through `WSLENV`
   (`DEEPSEEK_API_KEY/u`), so the key is never on a command line, in a log or in a second
   file.

## Running it (when there is budget)

```shell
uv run python scripts/articraft_baseline.py --dry-run                  # the 12 commands
uv run python scripts/articraft_baseline.py --only stapler             # smoke test, <= $0.30
uv run python scripts/articraft_baseline.py                            # 12 prompts x 1 run
uv run python scripts/eval_report.py --name articraft_v1 --no-log      # the same tables
```

- **Options:** `--distro` (default `Ubuntu`) and `--articraft-dir` (default `~/articraft`).
- **Where the output goes:** `runs/eval_articraft_v1/jobs/<id>-r1/`. It holds Articraft's
  own run folder (`articraft/<run-id>/`) and a replica-style `record.json`, so
  `eval_report.py` and `python -m agent --list` work on the baseline too.
- **How runs are taken:** one at a time. No new run starts once `--max-total` (default
  $1.50) could be passed, and a rerun skips the jobs that already have a record.

## What is the same, and what is not

| | Replica (B2 default) | Articraft baseline |
| --- | --- | --- |
| Model | `deepseek-v4-pro` | `deepseek-v4-pro` |
| Reasoning effort | provider default; `low` on routine turns | `high`, Articraft's DeepSeek default, on every turn |
| Output cap | 32,000 tokens, less near the time limit | 32,000 (`ARTICRAFT_DEEPSEEK_MAX_OUTPUT_TOKENS`, set by the script) |
| Turns | 60 | 60 (`ARTICRAFT_MAX_TURNS`, set by the script; its default is 100) |
| Time limit | 25 min inside the loop (final turn, fix reserve) | 25 min, enforced from outside: the process is stopped, with no final turn |
| Cost cap | $0.30, checked after every call | $0.30, enforced from outside: its `record.json` token counts are polled every 5 s and the process is stopped past the cap |
| Cost | recorded per call | **not recorded by Articraft for DeepSeek** ("dollar costs are unestimated"); computed by the script from its token counts with our price table |
| Prompt, docs, checks | ours | its own |
| Output | URDF + GLB + OBJ + `model.json` per version | USDZ per version (`result/usdz/NNNN.usdz`) + `result/model.json` |

**The cost-cap overshoot** is a little larger than ours. It is what Articraft generates in
the last few seconds before a poll, plus the call in flight. One 32k-token reply costs about
$0.07.

## Can our checks run on its outputs?

**Not without `usd-core`.** Our checks need each body's meshes: floating parts, overlap,
swept collisions, separation and loop closure, and also our renders and triangle counts.
Articraft writes meshes only into its USDZ files. Reading USDZ needs the `pxr` module from
`usd-core` (Articraft pins `usd-core>=26.5,<26.8`). `usd-core` is not installed in the
replica and has not been approved, so it was not added.

What works without it:
- **From `result/model.json`:** bodies, shapes and their geometry types, joints with their
  degrees of freedom, and the articulations. The script turns these into `bodies`,
  `shapes`, `joints`, `movable_joints` and `non_box_share` (`articraft_quality`), so the
  quality table can compare those columns. Triangles and helpers stay empty.
- **Articraft's own verdict** (`success` after its own checks), which is the success rate
  the baseline reports.

Two possible routes to run our checks later, each needing a decision first:
- **Approve `usd-core` for the replica,** and write a USDZ → trimesh loader.
- **Convert inside WSL, with Articraft's own environment** (which already has `usd-core`):
  a small script that exports each body's mesh from the USDZ to OBJ, plus a `model.json`
  our loader reads. The replica would get no new dependency, but it is new code in the WSL
  copy.

## Caveats for the comparison

- **One run per prompt** gives wide intervals. See the eval report's notes.
- **Different effort and a stricter outside time stop** both work against Articraft; an
  equal-effort run would pass `--effort` differently. The table above lists every
  difference, so a reader can weigh them.
- **The success rules differ.** Articraft's success is a fresh compile that passed its own
  checks, followed by a text reply. Ours is the same rule with our checks. Neither system's
  checks are run on the other's output.
