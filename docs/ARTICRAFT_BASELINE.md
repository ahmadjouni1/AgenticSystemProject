# Articraft baseline (step 12): set up, no paid runs yet

`scripts/articraft_baseline.py` runs the step 12 benchmark (`prompts/benchmark_v1.json`) through
the original Articraft, with the same model as our builder (`deepseek-v4-pro`, Articraft's own
DeepSeek provider) and the same per-run caps.
- **Set up:** WSL is installed and Articraft runs in it. A free end-to-end check with an
  invalid key went through the whole path (2026-11-24).
- **Not yet run with a real key.**
- **Tests** (`tests/test_eval.py`, no WSL or network): command building, cost computation,
  the start rules, stopping and record conversion.

## Why WSL

Articraft does not run on Windows. Its compiler uses `os.killpg`, `start_new_session` and
`-lc` shells, and its docs folder is a symlink, all POSIX-only (study §1 and §3). WSL 2 runs
it unchanged.

## Setup (once)

1. **Install WSL 2 and Ubuntu.**
   - **If WSL itself is missing:** it needs PowerShell as administrator and a reboot,
     `wsl --install`.
   - **If WSL is already there** (on this machine it was, for Docker Desktop, whose
     `docker-desktop` distribution is not for general use): adding a distribution needs
     neither admin nor a reboot, `wsl --install -d Ubuntu --no-launch`.
   - **The Linux user:** create one, either by opening Ubuntu once, or without the
     interactive prompt:
     `wsl -d Ubuntu -u root -- bash -c "useradd -m -s /bin/bash -G sudo user"`, then set
     `[user] default=user` in `/etc/wsl.conf` and restart with `wsl --terminate Ubuntu`.
     A user made that way has no password. To use `sudo`, set one with
     `wsl -d Ubuntu -u root passwd user`.
2. **Install uv inside WSL:** `curl -LsSf https://astral.sh/uv/install.sh | sh`. `bash -lc`
   shells then find it in `~/.local/bin`.
3. **Make a copy of Articraft in the Linux file system.** "Previous work" stays read-only.
   Articraft writes `.venv/` and `runs/`. Its run folders get a symlink
   (`workspace/docs/sdk`); WSL creates it fine on `/mnt/c`, where the script puts the runs.

   ```shell
   mkdir ~/articraft
   cd "/mnt/c/Users/<you>/Desktop/EECE503P/Main project/Previous work/Agentic_Systems_Project"
   tar --exclude=./runs --exclude=./.venv -cf - . | tar -xf - -C ~/articraft
   cd ~/articraft && uv python pin 3.13 && uv sync
   ```

   - **Why pin Python 3.13:** Ubuntu 26.04's Python is 3.14, which uv picks by default.
     `xatlas` has no 3.14 wheel and its source build fails, while it has wheels up to 3.13.
     uv downloads its own 3.13; the system Python is untouched.
   - **What `uv sync` installs:** Articraft's dependencies, inside WSL only, `usd-core`
     among them. The replica's environment does not change.
4. **Check it without a key:** `cd ~/articraft && uv run articraft generate --help`. The
   CLI's subcommand is `generate`.
5. **Optional, free end-to-end check:** run one prompt with an invalid key:
   `$env:DEEPSEEK_API_KEY = "sk-invalid"` in that PowerShell, then
   `uv run python scripts/articraft_baseline.py --name articraft_smoke --only stapler`.
   - **Expected:** DeepSeek answers HTTP 401 and nothing is charged.
   - **What it shows:** WSL starts Articraft, the key is forwarded, the run folder is
     written, and `record.json` becomes an `error` record with the reason.
   - **Afterwards:** delete `runs/eval_articraft_smoke`.
5. **Do not create a key file in WSL.** Keep `DEEPSEEK_API_KEY` in the Windows environment
   or `replica/.env`, as for our runs. The script forwards it by name through `WSLENV`
   (`DEEPSEEK_API_KEY/u`), so the key is never on a command line, in a log or in a second
   file.

## Running it (when there is budget)

```shell
uv run python scripts/articraft_baseline.py --max-total 3 --off-peak-only --dry-run
uv run python scripts/articraft_baseline.py --max-total 3 --off-peak-only
uv run python scripts/eval_report.py --name articraft_v1 --no-log      # the same tables
```

- **The dry run** prints the plan and the first command. It also gives the cost estimate,
  how many runs are sure to start under `--max-total`, the price period, whether a run may
  start now, and the DeepSeek balance (the free endpoint).
- **Options:** `--distro` (default `Ubuntu`) and `--articraft-dir` (default `~/articraft`).
- **Where the output goes:** `runs/eval_articraft_v1/jobs/<id>-r1/`. It holds Articraft's
  own run folder (`articraft/<run-id>/`) and a replica-style `record.json`, so
  `eval_report.py` and `python -m agent --list` work on the baseline too.
- **How runs are taken:** one at a time, in the benchmark's round-robin order by category.
- **The total cap.** A run starts only if `--max-total` (default $1.50) cannot be passed
  even if the run ends at its cap plus a full reply in flight. That is $0.30 + about $0.08
  off-peak, or about $0.16 at peak.
- **`--off-peak-only`:** a run starts only if it cannot reach DeepSeek's peak hours
  (01-04 and 06-10 UTC on weekdays) before its 25-minute limit.
- **Reruns.** A rerun skips finished jobs. Jobs that ended in `error` start again from
  scratch, since Articraft cannot resume. The earlier attempt's record is kept as
  `record.NN.json`, and its cost still counts toward `--max-total`.

## What is the same, and what is not

| | Replica (B2 default) | Articraft baseline |
| --- | --- | --- |
| Model | `deepseek-v4-pro` | `deepseek-v4-pro` |
| Reasoning effort | provider default; `low` on routine turns | `high`, Articraft's DeepSeek default, on every turn |
| Output cap | 32,000 tokens, less near the time limit; a cut-off reply is handled in the loop | 32,000 (`ARTICRAFT_DEEPSEEK_MAX_OUTPUT_TOKENS`, set by the script); a cut-off reply (`finish_reason: length`) ends Articraft's run as an error |
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
