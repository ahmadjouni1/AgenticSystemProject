"""Step 12: the Articraft baseline. PREPARED, NOT RUN (no budget yet). Setup and caveats:
docs/ARTICRAFT_BASELINE.md.

    uv run python scripts/articraft_baseline.py --name articraft_v1 --dry-run
    uv run python scripts/articraft_baseline.py --name articraft_v1 --distro Ubuntu \\
        --articraft-dir /home/<you>/articraft
    uv run python scripts/eval_report.py --name articraft_v1 --no-log

Runs the benchmark prompts (prompts/benchmark_v1.json) through the original Articraft, the
copy of "Previous work" installed inside WSL, with its own DeepSeek provider and the same
model as our builder (deepseek-v4-pro). One run at a time, by default.

**What it does per prompt:**
- starts ``uv run articraft --no-tui --provider deepseek --model deepseek-v4-pro --effort
  high --output-dir <job folder>/articraft "<prompt>"`` through ``wsl.exe``. The API key is
  passed through the environment (``WSLENV``), never on a command line or in a file;
- watches Articraft's ``record.json`` every few seconds. Articraft has no cost or time cap,
  and records no dollar cost for DeepSeek, so the cost is computed here from its token
  counts with our price table. The run is stopped (``pkill`` inside WSL) when it passes the
  per-run cap or the time limit;
- writes a replica-style ``record.json`` in the job folder (status, reason, cost, time,
  turns, the exported USDZ), so ``eval_report.py`` and ``python -m agent --list`` read the
  baseline like our own evaluations.

**End states:** Articraft ``success`` -> ``success``; stopped for cost -> ``budget_exceeded``;
stopped for time -> ``time_exceeded``; "agent hit max turns limit" -> ``max_turns``; any
other error -> ``error``. Articraft's own success rule is close to ours (a fresh compile
that passed, then a final text reply), but its checks are its own.

**Partly comparable:** bodies, shapes, joints and the non-box share come from Articraft's
``result/model.json``. Triangles, renders and our own checks need its USDZ meshes in our
loader, which needs ``usd-core`` (not installed or approved). See the doc.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import BENCHMARK, RUNS_ROOT, Job, load_benchmark, plan  # noqa: E402

from fileio import atomic_write_json  # noqa: E402
from llm import Usage  # noqa: E402
from llm.accounting import cost as price_of  # noqa: E402

MODEL = "deepseek-v4-pro"  # our builder's model (llm/config.py ROLES["builder"])
EFFORT = "high"  # Articraft's DeepSeek default; ours sends none (provider default) or low
PER_RUN_CAP = 0.30
MAX_TIME = 25 * 60.0
MAX_TURNS = 60  # ours; Articraft's default is 100 (ARTICRAFT_MAX_TURNS)
POLL_S = 5.0
FORWARDED_ENV = ("DEEPSEEK_API_KEY", "ARTICRAFT_MAX_TURNS",
                 "ARTICRAFT_DEEPSEEK_MAX_OUTPUT_TOKENS")


def wsl_path(path: Path) -> str:
    """C:\\Users\\x -> /mnt/c/Users/x (WSL's default automount)."""
    windows = PureWindowsPath(str(path))
    if not windows.drive:
        return Path(path).as_posix()
    return "/mnt/" + windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])


@dataclass
class Settings:
    name: str
    distro: str = "Ubuntu"
    articraft_dir: str = "~/articraft"  # inside WSL: a copy, never "Previous work" itself
    max_cost: float = PER_RUN_CAP
    max_time: float = MAX_TIME
    max_turns: int = MAX_TURNS
    runs: int = 1
    only: list[str] | None = None
    runs_root: Path = RUNS_ROOT
    benchmark: Path = BENCHMARK

    @property
    def eval_dir(self) -> Path:
        return Path(self.runs_root) / f"eval_{self.name}"


def command(job: Job, settings: Settings) -> list[str]:
    """The wsl.exe command line for one job. No key in it."""
    output = wsl_path(job.folder / "articraft")
    inner = (f"cd {settings.articraft_dir} && uv run articraft --no-tui --provider deepseek "
             f"--model {MODEL} --effort {EFFORT} --output-dir {shlex.quote(output)} "
             f"{shlex.quote(job.prompt)}")
    return ["wsl.exe", "-d", settings.distro, "--", "bash", "-lc", inner]


def environment(settings: Settings, base: dict[str, str] | None = None) -> dict[str, str]:
    """The Windows environment plus WSLENV, which forwards the key and Articraft's limits
    into WSL by name (the values stay in the environment)."""
    env = dict(os.environ if base is None else base)
    env["ARTICRAFT_MAX_TURNS"] = str(settings.max_turns)
    env["ARTICRAFT_DEEPSEEK_MAX_OUTPUT_TOKENS"] = "32000"  # our builder's cap
    names = [n for n in env.get("WSLENV", "").split(":") if n]
    names += [n + "/u" for n in FORWARDED_ENV if n + "/u" not in names]
    env["WSLENV"] = ":".join(names)
    return env


def stop_command(job: Job, settings: Settings, signal: str = "TERM") -> list[str]:
    """Stop the job's Articraft process inside WSL (matched by its unique output folder)."""
    return ["wsl.exe", "-d", settings.distro, "--", "pkill", f"-{signal}", "-f", "--",
            wsl_path(job.folder / "articraft")]


# ---- reading Articraft's run --------------------------------------------------------------------


def articraft_run(job: Job) -> Path | None:
    root = job.folder / "articraft"
    runs = sorted(p for p in root.iterdir() if (p / "record.json").is_file()) \
        if root.is_dir() else []
    return runs[-1] if runs else None


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def cost_of(token_usage: dict[str, Any], at: datetime | None = None) -> float:
    """USD for Articraft's summed DeepSeek token counts, with our price table."""
    usage = Usage(input_tokens=int(token_usage.get("input_tokens", 0)),
                  cached_input_tokens=int(token_usage.get("cached_input_tokens", 0)),
                  output_tokens=int(token_usage.get("output_tokens", 0)))
    value = price_of(MODEL, usage, at)
    return float(value) if not isinstance(value, str) else 0.0


def turns_of(run: Path) -> int:
    """Model replies in Articraft's conversation log (one line per event)."""
    count = 0
    log = run / "conversation.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("role") == "assistant" or event.get("type") in ("assistant",
                                                                          "model_response"):
                count += 1
    return count


def articraft_quality(model_json: Path) -> dict[str, Any] | None:
    """What Articraft's result/model.json tells without its USDZ: bodies, shapes, joints,
    the non-box share. Triangles and helpers need the meshes (USDZ, usd-core): None."""
    model = read_json(model_json)
    if not model.get("rigid_bodies"):
        return None
    shapes = [s for b in model["rigid_bodies"] for s in b.get("shapes", [])]
    boxes = sum(s.get("geometry_type") in ("Box", "BoxGeometry") for s in shapes)
    joints = model.get("joints") or []
    return {"bodies": len(model["rigid_bodies"]), "shapes": len(shapes),
            "joints": len(joints), "movable_joints": sum(bool(j.get("dofs")) for j in joints),
            "non_box_share": round(1 - boxes / len(shapes), 4) if shapes else 0.0,
            "helper_count": None, "triangles": None, "source": "articraft model.json"}


def convert(job: Job, run: Path | None, stopped: str | None, seconds: float,
            started: datetime, settings: Settings) -> dict[str, Any]:
    """A replica-style record of one Articraft run."""
    record = read_json(run / "record.json") if run else {}
    spent = cost_of(record.get("token_usage") or {}, started)
    status = record.get("status", "error" if run else "not_started")
    error = record.get("error") or ""
    if stopped == "cost":
        status, reason = "budget_exceeded", f"stopped at ${spent:.4f} (cap ${settings.max_cost})"
    elif stopped == "time":
        status, reason = "time_exceeded", f"stopped after {seconds:.0f} s"
    elif status == "success":
        reason = "articraft: success"
    elif "max turns" in error:
        status, reason = "max_turns", f"articraft: {error}"
    else:
        status, reason = "error", f"articraft: {error or 'no record'}"
    result = record.get("result") or ""
    best = None
    if run and result and (run / result).is_file():
        best = {"version": Path(result).stem, "passed_checks": status == "success",
                "path": (run / result).relative_to(job.folder).as_posix()}
    return {
        "status": status, "reason": reason, "prompt": job.prompt,
        "run_dir": job.folder.as_posix(), "system": "articraft",
        "model": {"provider": "deepseek", "model": MODEL, "role": "articraft"},
        "started": started.isoformat(timespec="seconds"),
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": round(seconds, 2), "turns": turns_of(run) if run else 0,
        "cost_usd": round(spent, 6), "cost_note": "computed from Articraft's token counts "
        "with llm/config.py PRICES (Articraft records no DeepSeek cost)",
        "tokens": record.get("token_usage") or {},
        "best": best, "articraft_run": run.relative_to(job.folder).as_posix() if run else None,
        "articraft_status": record.get("status"), "articraft_error": error,
        "sessions": [{"session": 1, "status": status}],
        "quality": articraft_quality(run / "result" / "model.json")
        if run and status == "success" else None,
        "per_turn": [],
        "compiles": {},
    }


# ---- running ------------------------------------------------------------------------------------


def watch(job: Job, process: Any, settings: Settings, stop: Callable[[str], None],
          clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep) -> tuple[str | None, float]:
    """Wait for the process; stop it past the cost cap or the time limit.
    Returns (why it was stopped: "cost", "time" or None, seconds)."""
    started = clock()
    while process.poll() is None:
        seconds = clock() - started
        run = articraft_run(job)
        spent = cost_of(read_json(run / "record.json").get("token_usage") or {}) if run else 0.0
        why = "cost" if spent > settings.max_cost else "time" if seconds > settings.max_time \
            else None
        if why:
            stop("TERM")
            for _ in range(int(10 / max(POLL_S, 0.01)) + 1):
                if process.poll() is not None:
                    break
                sleep(min(POLL_S, 10))
            if process.poll() is None:
                stop("KILL")
                process.wait()
            return why, clock() - started
        sleep(POLL_S)
    return None, clock() - started


def run_job(job: Job, settings: Settings, popen: Callable[..., Any] = subprocess.Popen,
            runner: Callable[..., Any] = subprocess.run, **watch_options) -> dict[str, Any]:
    (job.folder / "articraft").mkdir(parents=True, exist_ok=True)
    log = settings.eval_dir / "logs" / f"{job.key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    with log.open("a", encoding="utf-8") as out:
        process = popen(command(job, settings), stdout=out, stderr=subprocess.STDOUT,
                        env=environment(settings))
        stopped, seconds = watch(job, process, settings,
                                 lambda signal: runner(stop_command(job, settings, signal),
                                                       check=False),
                                 **watch_options)
    record = convert(job, articraft_run(job), stopped, seconds, started, settings)
    atomic_write_json(job.folder / "record.json", record, indent=2)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="articraft_v1")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--articraft-dir", default="~/articraft",
                        help="the Articraft copy inside WSL (not 'Previous work' itself)")
    parser.add_argument("--max-cost", type=float, default=PER_RUN_CAP)
    parser.add_argument("--max-total", type=float, default=1.5,
                        help="no new run once this much is spent")
    parser.add_argument("--max-time", type=float, default=MAX_TIME)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args(argv)
    settings = Settings(args.name, args.distro, args.articraft_dir, args.max_cost,
                        args.max_time, MAX_TURNS, args.runs, args.only,
                        Path(args.runs_root).resolve())
    benchmark = load_benchmark(settings.benchmark)
    jobs = plan(benchmark, settings.eval_dir, settings.runs, settings.only)
    if args.dry_run:
        for job in jobs:
            print(f"{job.key}: {subprocess.list2cmdline(command(job, settings))}")
        print(f"{len(jobs)} runs; worst case ${len(jobs) * settings.max_cost:.2f} "
              f"(+ one call's overshoot each); --max-total ${args.max_total:.2f}")
        return 0
    from llm import load_env

    load_env()  # DEEPSEEK_API_KEY from the environment or replica/.env
    if not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("DEEPSEEK_API_KEY is not set")
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    (settings.eval_dir / "benchmark.json").write_text(
        Path(settings.benchmark).read_text(encoding="utf-8"), encoding="utf-8")
    spent = 0.0
    for job in jobs:
        done = read_json(job.folder / "record.json")
        if done.get("status") not in (None, "error", "not_started"):
            spent += float(done.get("cost_usd") or 0)
            continue  # finished in an earlier invocation
        if spent + settings.max_cost > args.max_total:
            print(f"STOPPED: ${spent:.2f} spent; another run could pass --max-total")
            break
        record = run_job(job, settings)
        spent += record["cost_usd"]
        print(f"{job.key}: {record['status']} ${record['cost_usd']:.4f} "
              f"{record['duration_s']:.0f} s; total ${spent:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
