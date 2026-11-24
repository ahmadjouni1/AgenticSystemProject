"""Step 12: run the benchmark (prompts x runs) with the default agent (B2), within a budget.

    uv run python scripts/eval.py --name v1 --dry-run     # the plan and the worst-case cost
    uv run python scripts/eval.py --name v1               # 12 prompts x 1 run, 2 at a time
    uv run python scripts/eval_report.py --name v1        # tables, render sheets, BUILD_LOG

Each run is ``python -m agent`` in its own process. Run folders go to
``runs/eval_<name>/jobs/<prompt id>-r<rep>/``, console logs to ``runs/eval_<name>/logs/``;
every finished session is added to ``runs/index.jsonl`` (step 11), which this script reads to
know which runs are finished.

**Money.** ``--max-cost`` is each run's cap ($0.30). ``--max-total`` ($1.50) caps the whole
evaluation: a run starts only if the spend so far plus a full per-run cap for every run in
flight, the new one included, stays within it (otherwise it waits for a run in flight to end
and frees its reservation, and stops when none is left). So no run is started that could
take the total past the cap; the total can only go over by what one model call goes past a run's own
cap (the budget is checked after each call: at most about $0.07 per run for a 32k-token
reply). Spend counts every run of this evaluation, earlier invocations included.

**A rerun resumes.** Finished runs (any end state: success, time_exceeded, budget_exceeded,
...) are kept. Unfinished ones (a crash, the machine asleep, a connection error, a stop on
HTTP 402) are resumed with ``--resume`` and what is left of their allowance (cost and time),
at most ``MAX_SESSIONS`` sessions per run. Runs not started yet are started.

**Stops.** No run starts after one gets HTTP 402 (out of credit). With ``--off-peak-only``
no run starts at DeepSeek's peak price (01-04 and 06-10 UTC, Monday to Friday: twice the
off-peak rate); runs already going finish, and a rerun later starts the rest. Windows is kept awake while
the script runs. ``--jobs 2`` by default: DeepSeek lowers concurrency on a low balance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import storage  # noqa: E402
from fileio import atomic_write_json, read_jsonl  # noqa: E402

RUNS_ROOT = ROOT / "runs"
BENCHMARK = ROOT / "prompts" / "benchmark_v1.json"
FINISHED = {"success", "failed_checks", "time_exceeded", "budget_exceeded", "max_turns",
            "no_progress"}
MAX_SESSIONS = 3  # a run that keeps crashing is left as it is after this many sessions
MIN_COST_LEFT = 0.02  # USD: less than this left of a run's cap is not worth a resume
MIN_TIME_LEFT = 120.0  # s
PER_RUN_CAP = 0.30
MAX_TOTAL = 1.50
MAX_TIME = 25 * 60.0
OUT_OF_CREDIT = "HTTP 402"

Launcher = Callable[[list[str], Path], int]


# ---- the plan -----------------------------------------------------------------------------------


def load_benchmark(path: Path = BENCHMARK) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [p["id"] for p in data["prompts"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate prompt ids in {path}")
    for p in data["prompts"]:
        if p["category"] not in data["categories"]:
            raise ValueError(f"{p['id']}: unknown category {p['category']!r}")
    return data


@dataclass
class Job:
    id: str
    category: str
    prompt: str
    rep: int
    folder: Path  # holds the one run folder of this job

    @property
    def key(self) -> str:
        return f"{self.id}-r{self.rep}"


def plan(benchmark: dict[str, Any], eval_dir: Path, runs: int,
         only: list[str] | None = None) -> list[Job]:
    """Repetition first; within one, the categories in turn (one prompt of each, then the
    next of each, ...). An evaluation stopped by the spend cap has then covered every
    category, and every prompt once before any is repeated."""
    chosen = [p for p in benchmark["prompts"] if not only or p["id"] in only]
    unknown = set(only or []) - {p["id"] for p in benchmark["prompts"]}
    if unknown:
        raise ValueError(f"unknown prompt ids: {sorted(unknown)}")
    queues = [[p for p in chosen if p["category"] == c] for c in benchmark["categories"]]
    prompts = [q[i] for i in range(max(map(len, queues), default=0)) for q in queues
               if i < len(q)]
    return [Job(p["id"], p["category"], p["prompt"], rep,
                eval_dir / "jobs" / f"{p['id']}-r{rep}")
            for rep in range(1, runs + 1) for p in prompts]


def run_dir_of(job: Job) -> Path | None:
    found = storage.find_runs(job.folder) if job.folder.is_dir() else []
    return found[0] if found else None


# ---- the state of a job -------------------------------------------------------------------------


@dataclass
class State:
    action: str  # "start", "resume", "done", "busy"
    run_dir: Path | None = None
    status: str | None = None
    spent: float = 0.0
    duration: float = 0.0
    sessions: int = 0
    why: str = ""


def latest_entries(index: Path) -> dict[str, dict[str, Any]]:
    """The last index line of each run (by its path relative to the index's folder)."""
    latest: dict[str, dict[str, Any]] = {}
    for entry in read_jsonl(index):
        latest[entry["run"]] = entry
    return latest


def job_state(job: Job, latest: dict[str, dict[str, Any]], index: Path, cap: float,
              max_time: float) -> State:
    run_dir = run_dir_of(job)
    if run_dir is None:
        return State("start")
    record = storage.read_record(run_dir) or {}
    status = storage.status_of(run_dir, record) if record else "crashed"
    spent = float(record.get("cost_usd") or 0.0)
    duration = float(record.get("duration_s") or 0.0)
    sessions = len(record.get("sessions") or [None])
    state = State("done", run_dir, status, spent, duration, sessions)
    try:
        key = run_dir.resolve().relative_to(index.parent.resolve()).as_posix()
    except ValueError:
        key = run_dir.as_posix()
    entry = latest.get(key)
    if status == "running":
        state.action, state.why = "busy", "another process is running it"
    elif entry is not None and entry.get("status") in FINISHED and status in FINISHED:
        state.why = f"finished: {status}"
    elif status == "success":
        state.why = "finished: success"
    elif sessions >= MAX_SESSIONS:
        state.why = f"left as {status} after {sessions} sessions"
    elif cap - spent < MIN_COST_LEFT or max_time - duration < MIN_TIME_LEFT:
        state.why = f"left as {status}: its allowance is used up"
    else:
        state.action, state.why = "resume", f"unfinished ({status})"
    return state


# ---- running ------------------------------------------------------------------------------------


def launch_subprocess(argv: list[str], log: Path) -> int:
    """``python -m agent <argv>`` in its own process, its output appended to ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as out:
        return subprocess.call([sys.executable, "-m", "agent", *argv], cwd=ROOT, stdout=out,
                               stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (released when it exits)."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


@dataclass
class Settings:
    name: str
    runs: int = 1
    jobs: int = 2
    max_cost: float = PER_RUN_CAP
    max_total: float = MAX_TOTAL
    max_time: float = MAX_TIME
    only: list[str] | None = None
    agent_args: list[str] = field(default_factory=list)
    runs_root: Path = RUNS_ROOT
    benchmark: Path = BENCHMARK
    off_peak_only: bool = False

    @property
    def eval_dir(self) -> Path:
        return Path(self.runs_root) / f"eval_{self.name}"

    @property
    def index(self) -> Path:
        return Path(self.runs_root) / storage.INDEX


class Evaluation:
    """Hands out jobs under the total-spend cap; one lock guards the books."""

    def __init__(self, settings: Settings, jobs: list[Job], launcher: Launcher,
                 log: Callable[[str], None] = print):
        self.settings, self.jobs, self.launcher, self.say = settings, jobs, launcher, log
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)  # a run finished: books changed
        self.queue = list(jobs)
        self.spent: dict[str, float] = {}  # job key -> its run's cost so far
        self.reserved: dict[str, float] = {}  # job key -> the allowance of a run in flight
        self.outcome: dict[str, dict[str, Any]] = {}
        self.stop: str | None = None
        latest = latest_entries(settings.index)
        self.states = {j.key: job_state(j, latest, settings.index, settings.max_cost,
                                        settings.max_time) for j in jobs}
        for job in jobs:
            self.spent[job.key] = self.states[job.key].spent

    def committed(self) -> float:
        return sum(self.spent.values()) + sum(self.reserved.values())

    def claim(self) -> tuple[Job, list[str], float] | None:
        """The next job to run, its agent arguments and its cost allowance; None when no
        more runs may start."""
        s = self.settings
        with self.lock:
            while self.queue and self.stop is None:
                job = self.queue.pop(0)
                state = self.states[job.key]
                if state.action in ("done", "busy"):
                    self.outcome[job.key] = {"action": "kept", "why": state.why}
                    continue
                if state.action == "resume":
                    allowance = round(s.max_cost - state.spent, 4)
                    seconds = round(s.max_time - state.duration, 1)
                    argv = ["--resume", str(state.run_dir), "--index", str(s.index),
                            "--max-cost", str(allowance), "--max-time", str(seconds)]
                else:
                    allowance = s.max_cost
                    argv = [job.prompt, "--runs", str(job.folder), "--index", str(s.index),
                            "--max-cost", str(allowance), "--max-time", str(s.max_time),
                            *s.agent_args]
                if s.off_peak_only and peak_now():
                    self.queue.insert(0, job)
                    self.stop = (f"peak price: DeepSeek's peak hours began "
                                 f"({price_note()}); rerun later to start the rest")
                    break
                if self.committed() + allowance > s.max_total + 1e-9:
                    self.queue.insert(0, job)
                    if self.reserved:  # a run in flight may end under its reservation
                        self.changed.wait()
                        continue
                    self.stop = (f"total-spend cap: ${self.committed():.2f} spent + "
                                 f"${allowance:.2f} for {job.key} > ${s.max_total:.2f}")
                    break
                self.reserved[job.key] = allowance
                return job, argv, allowance
            return None

    def finish(self, job: Job, code: int, seconds: float, log: Path) -> None:
        run_dir = run_dir_of(job)
        record = storage.read_record(run_dir) if run_dir else None
        with self.lock:
            self.reserved.pop(job.key, None)
            if record is not None:
                self.spent[job.key] = float(record.get("cost_usd") or 0.0)
            text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            if OUT_OF_CREDIT in text and self.stop is None:
                self.stop = "the provider returned HTTP 402 (out of credit)"
            self.outcome[job.key] = {
                "action": self.states[job.key].action, "exit": code,
                "seconds": round(seconds, 1),
                "status": (record or {}).get("status"), "cost_usd": self.spent[job.key]}
            self.changed.notify_all()
        self.say(f"{job.key}: {self.outcome[job.key]['status']} (exit {code}) after "
                 f"{seconds:.0f} s, ${self.spent[job.key]:.4f}; total ${self.total():.4f}")

    def total(self) -> float:
        return sum(self.spent.values())

    def worker(self) -> None:
        while True:
            claimed = self.claim()
            if claimed is None:
                return
            job, argv, allowance = claimed
            log = self.settings.eval_dir / "logs" / f"{job.key}.log"
            self.say(f"{job.key}: {self.states[job.key].action} (allowance ${allowance:.2f})")
            started = time.time()
            try:
                code = self.launcher(argv, log)
            except Exception as exc:  # noqa: BLE001 - one broken launch must not stop the rest
                code = -1
                self.say(f"{job.key}: launch failed: {type(exc).__name__}: {exc}")
            self.finish(job, code, time.time() - started, log)

    def run(self) -> dict[str, Any]:
        workers = max(1, self.settings.jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(self.worker) for _ in range(workers)]:
                future.result()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        s = self.settings
        rows = []
        for job in self.jobs:
            run_dir = run_dir_of(job)
            record = storage.read_record(run_dir) if run_dir else None
            rows.append({"key": job.key, "id": job.id, "category": job.category,
                         "rep": job.rep,
                         "run": run_dir.relative_to(s.eval_dir).as_posix() if run_dir else None,
                         "status": storage.status_of(run_dir, record) if record else
                         "not_started",
                         "cost_usd": float((record or {}).get("cost_usd") or 0.0),
                         "this_invocation": self.outcome.get(job.key)})
        return {"name": s.name, "benchmark": Path(s.benchmark).name, "runs": s.runs,
                "max_cost": s.max_cost, "max_total": s.max_total, "max_time": s.max_time,
                "agent_args": s.agent_args, "spent_usd": round(self.total(), 6),
                "stopped": self.stop, "jobs": rows}


def run_eval(settings: Settings, launcher: Launcher = launch_subprocess,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    benchmark = load_benchmark(settings.benchmark)
    eval_dir = settings.eval_dir
    (eval_dir / "logs").mkdir(parents=True, exist_ok=True)
    snapshot = eval_dir / "benchmark.json"
    if not snapshot.is_file():  # the prompts this evaluation was run with
        shutil.copyfile(settings.benchmark, snapshot)
    jobs = plan(benchmark, eval_dir, settings.runs, settings.only)
    evaluation = Evaluation(settings, jobs, launcher, log)
    todo = [j.key for j in jobs if evaluation.states[j.key].action in ("start", "resume")]
    log(f"eval_{settings.name}: {len(jobs)} runs planned, {len(todo)} to start or resume, "
        f"{settings.jobs} at a time; ${evaluation.total():.4f} spent so far; caps "
        f"${settings.max_cost:.2f} per run, ${settings.max_total:.2f} in all")
    summary = evaluation.run()
    atomic_write_json(eval_dir / "eval.json", summary, indent=2)
    if summary["stopped"]:
        log(f"STOPPED: {summary['stopped']}")
    log(f"spent ${summary['spent_usd']:.4f}; report: uv run python scripts/eval_report.py "
        f"--name {settings.name}")
    return summary


def dry_run(settings: Settings) -> str:
    benchmark = load_benchmark(settings.benchmark)
    jobs = plan(benchmark, settings.eval_dir, settings.runs, settings.only)
    latest = latest_entries(settings.index)
    lines = [f"eval_{settings.name}: {len(jobs)} runs, {settings.jobs} at a time"]
    worst = 0.0
    for job in jobs:
        state = job_state(job, latest, settings.index, settings.max_cost, settings.max_time)
        if state.action == "start":
            worst += settings.max_cost
        elif state.action == "resume":
            worst += settings.max_cost - state.spent
        lines.append(f"  {job.key:<22} {job.category:<12} {state.action:<7} {state.why}")
    lines.append(f"worst case if every run used its whole cap: ${worst:.2f}; the total-spend "
                 f"cap stops new runs at ${settings.max_total:.2f}")
    lines.append(price_note())
    return "\n".join(lines)


def peak_now(at=None) -> bool:
    """Whether deepseek-v4-pro is at its full (peak) price now."""
    from datetime import UTC, datetime

    from llm import config

    off_peak = config.PRICES["deepseek-v4-pro"].off_peak
    return off_peak is not None and off_peak.factor(at or datetime.now(UTC)) >= 1


def price_note(at=None) -> str:
    """DeepSeek's price right now: off-peak rates are half (llm/config.py)."""
    from datetime import UTC, datetime

    from llm import config

    at = at or datetime.now(UTC)
    off_peak = config.PRICES["deepseek-v4-pro"].off_peak
    if off_peak is None:
        return "deepseek-v4-pro: no time-of-day pricing"
    hours = ", ".join(f"{a:02d}-{b:02d}" for a, b in off_peak.peak_hours_utc)
    now = "off-peak (half price)" if off_peak.factor(at) < 1 else "PEAK (full price)"
    return (f"now ({at:%a %H:%M} UTC) is {now}; DeepSeek peak hours: {hours} UTC"
            + (", Monday to Friday" if off_peak.weekdays_only else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="the evaluation: runs/eval_<name>/")
    parser.add_argument("--runs", type=int, default=1, help="runs per prompt")
    parser.add_argument("--jobs", type=int, default=2, help="runs at the same time")
    parser.add_argument("--max-cost", type=float, default=PER_RUN_CAP, help="USD per run")
    parser.add_argument("--max-total", type=float, default=MAX_TOTAL,
                        help="USD for the whole evaluation (no new run past it)")
    parser.add_argument("--max-time", type=float, default=MAX_TIME, help="seconds per run")
    parser.add_argument("--only", nargs="*", default=None, metavar="ID", help="prompt ids")
    parser.add_argument("--benchmark", default=str(BENCHMARK))
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--agent-args", default="",
                        help='extra arguments for new runs, e.g. "--two-phase"')
    parser.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")
    parser.add_argument("--off-peak-only", action="store_true",
                        help="start no run at DeepSeek's peak price (runs going finish)")
    args = parser.parse_args(argv)
    settings = Settings(args.name, args.runs, args.jobs, args.max_cost, args.max_total,
                        args.max_time, args.only, args.agent_args.split(),
                        Path(args.runs_root).resolve(), Path(args.benchmark).resolve(),
                        args.off_peak_only)
    if args.dry_run:
        print(dry_run(settings))
        return 0
    keep_awake()
    summary = run_eval(settings, launch_subprocess,
                       log=lambda text: print(text, flush=True))
    return 2 if summary["stopped"] and "402" in summary["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
