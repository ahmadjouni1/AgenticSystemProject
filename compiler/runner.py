# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Run the compile worker in a fresh, killable subprocess and turn its payload into a
``CompileResult``.

    from compiler import compile
    result = compile(run_dir, timeout=300)
    print(result.signals_text)   # the only thing the agent sees

``run_dir`` contains ``workspace/main.py``. Results go to ``run_dir/result``:

| File | What it holds |
| --- | --- |
| ``vNNNN/`` | exports from successful compiles |
| ``compile_log.jsonl`` | one line per compile: full stdout, stderr, traceback |
| ``.compile-cache.json`` | the last result, keyed by the workspace digest |
| ``.compile-history.json`` | failure signatures, used for repeat detection |

**No sandbox.** The worker runs model-written code with the user's own permissions. Only
``*_API_KEY`` / ``*_TOKEN`` variables are removed from its environment. Articraft is the
same; see BUILD_LOG.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compiler import feedback
from fileio import append_line, atomic_write_text
from compiler.feedback import CompileSignal
from compiler.worker import HANG_FILE, PROGRESS_FILE

DEFAULT_TIMEOUT = 300.0
REPLICA_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = "compile_log.jsonl"
CACHE_FILE = ".compile-cache.json"
HISTORY_FILE = ".compile-history.json"
_SECRET = re.compile(r".*(_API_KEY|_TOKEN)$", re.IGNORECASE)
_VERSION = re.compile(r"v\d{4,}|\.v\d{4,}-[0-9a-f]+\.tmp")
_IGNORED_DIRS = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache"}


@dataclass(frozen=True)
class CompileResult:
    status: str
    """"success" or "error"."""
    signals_text: str
    """The ``<compile_signals>`` block: the only thing the agent sees."""
    signals: tuple[CompileSignal, ...]
    version: int | None
    """Export version written by this compile (None if it failed)."""
    export_dir: Path | None
    phase: str | None
    """Last phase reached: where it stopped, for a failure."""
    phases: dict[str, float] = field(default_factory=dict)
    """Seconds spent in each finished phase."""
    duration: float = 0.0
    """Wall time of the compile call, including interpreter start-up."""
    startup: float | None = None
    """Interpreter start + imports: wall time minus the time inside phases."""
    cached: bool = False
    digest: str = ""
    returncode: int | None = None
    repeat_count: int = 0
    """How many compiles in a row (this one included) failed in exactly this way."""
    log_path: Path | None = None
    target_path: dict | None = None
    """The trajectory check's summary when main.py declares target_path (step 13)."""

    @property
    def ok(self) -> bool:
        return self.status == "success"


def workspace_digest(workspace: Path) -> str:
    """SHA-256 over every file's relative path and bytes (caches and bytecode ignored)."""
    digest = hashlib.sha256()
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if any(part in _IGNORED_DIRS for part in relative.parts) or path.suffix == ".pyc":
            continue
        if path.is_file():
            digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


_TOOLCHAIN_DIGEST: str | None = None


def toolchain_digest() -> str:
    """SHA-256 of the replica's own sdk/ and compiler/ sources. Part of the cache key, so
    a cached result is never reused after the SDK or the compiler itself changed."""
    global _TOOLCHAIN_DIGEST
    if _TOOLCHAIN_DIGEST is None:
        digest = hashlib.sha256()
        for package in ("sdk", "compiler"):
            for path in sorted((REPLICA_ROOT / package).rglob("*.py")):
                digest.update(path.relative_to(REPLICA_ROOT).as_posix().encode())
                digest.update(path.read_bytes())
        _TOOLCHAIN_DIGEST = digest.hexdigest()
    return _TOOLCHAIN_DIGEST


def child_environment() -> dict[str, str]:
    """The current environment without credentials, with the replica importable."""
    env = {k: v for k, v in os.environ.items() if not _SECRET.match(k)}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPLICA_ROOT), env.get("PYTHONPATH", "")) if p
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # keep __pycache__ out of the workspace
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def compile(run_dir: str | os.PathLike[str], timeout: float = DEFAULT_TIMEOUT, *,  # noqa: A001
            force: bool = False) -> CompileResult:
    """Compile ``run_dir/workspace/main.py``. Returns the cached result when the
    workspace is unchanged since the last compile (``force=True`` to recompile)."""
    run_dir = Path(run_dir).resolve()
    workspace, result_dir = run_dir / "workspace", run_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / LOG_FILE
    if not (workspace / "main.py").is_file():
        signals = (feedback.signal_missing_main(),)
        return CompileResult("error", feedback.render_signals("error", signals), signals,
                             None, None, None, log_path=log_path)

    digest = workspace_digest(workspace)
    cached = None if force else _cached(result_dir, digest, timeout)
    if cached is not None:
        _log(log_path, {"digest": digest, "cached": True, "status": cached.status})
        return cached

    before = _versions(result_dir)
    started = time.perf_counter()
    run = _run_worker(run_dir, workspace, timeout)
    duration = time.perf_counter() - started
    payload, signals = _interpret(run, workspace, result_dir, timeout)
    status = payload.get("status", "error")
    if status != "success":
        _remove_new_versions(result_dir, before)  # a failed compile leaves no version
    repeat = _record_history(result_dir, signals)
    export = payload.get("export") or {}
    phases = payload.get("phases") or {}
    summary = ""
    if status == "success":
        model = payload.get("model") or {}
        summary = (f"Compile passed: exported v{export['version']:04d} "
                   f"(bodies={model.get('bodies')}, joints={model.get('joints')}, "
                   f"closures={model.get('closures')}, mimics={model.get('mimics')}).")
        path = model.get("target_path")
        if path and path.get("kind") in ("sweep", "stroke"):
            summary += (f" target_path: {path['output_joint']} {path['kind']} "
                        f"{path['achieved']} {path['unit']} (target {path['target']:.4g}, "
                        f"tolerance {path['tolerance']:.3g} {path['unit']}).")
        elif path:
            unit = path.get("unit", "mm")
            summary += (f" target_path: max error {path['max_error_mm']} {unit}, RMS "
                        f"{path['rms_error_mm']} {unit} (tolerance {path['tolerance_mm']:.3g} "
                        f"{unit}).")
    result = CompileResult(
        status=status,
        signals_text=feedback.render_signals(status, signals, success_summary=summary,
                                             repeat_count=repeat),
        signals=tuple(signals),
        version=export.get("version"),
        export_dir=Path(export["directory"]) if export.get("directory") else None,
        phase=payload.get("phase"),
        phases=phases,
        duration=round(duration, 4),
        startup=round(duration - sum(phases.values()), 4) if phases else None,
        digest=digest,
        returncode=run["returncode"],
        repeat_count=repeat,
        log_path=log_path,
        target_path=(payload.get("model") or {}).get("target_path"),
    )
    _log(log_path, {
        "digest": digest, "cached": False, "status": status, "duration": result.duration,
        "startup": result.startup, "phase": result.phase, "phases": phases,
        "returncode": run["returncode"], "timed_out": run["timed_out"],
        "signals": [s.to_dict() for s in signals], "export": export or None,
        "target_path": (payload.get("model") or {}).get("target_path"),
        "stdout": payload.get("stdout", ""), "stderr": payload.get("stderr", ""),
        "traceback": payload.get("traceback", ""), "worker_stdout": run["stray_stdout"],
        "worker_stderr": run["stderr"], "hang_dump": run.get("hang_dump", ""),
    })
    if not internal_failure(payload, signals):
        _write_cache(result_dir, result, timeout, summary)
    return result


def internal_failure(payload: dict[str, Any], signals: list[CompileSignal]) -> bool:
    """A failure of the compiler itself (an exception outside main.py, or a worker that
    died without a result). Never cached: the same main.py deserves a fresh try."""
    return bool(payload.get("internal")) or any(
        s.kind == feedback.SignalKind.CRASH.value for s in signals)


# ---- the subprocess ------------------------------------------------------------------------


def _run_worker(run_dir: Path, workspace: Path, timeout: float) -> dict[str, Any]:
    command = [sys.executable, "-m", "compiler.worker", str(run_dir), "--timeout", str(timeout)]
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    for name in (PROGRESS_FILE, HANG_FILE):
        (run_dir / "result" / name).unlink(missing_ok=True)
    process = subprocess.Popen(command, cwd=workspace, env=child_environment(),
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, **options)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:  # something still holds the pipes open
            stdout, stderr = b"", b""
    text = stdout.decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    payload_line = lines[-1] if lines and lines[-1].lstrip().startswith("{") else ""
    return {"returncode": process.returncode, "timed_out": timed_out,
            "payload_line": payload_line,
            "stray_stdout": "\n".join(lines[:-1] if payload_line else lines),
            "stderr": stderr.decode("utf-8", errors="replace")}


def kill_process_tree(process: subprocess.Popen) -> None:
    """Kill the worker and everything it started (grandchildren included)."""
    if process.poll() is not None and os.name != "nt":
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _interpret(run: dict[str, Any], workspace: Path, result_dir: Path,
               timeout: float) -> tuple[dict[str, Any], list[CompileSignal]]:
    progress = _read_json(result_dir / PROGRESS_FILE)
    hang = (result_dir / HANG_FILE)
    hang_dump = hang.read_text(encoding="utf-8", errors="replace") if hang.is_file() else ""
    for path in (result_dir / PROGRESS_FILE, hang):
        path.unlink(missing_ok=True)
    run["hang_dump"] = hang_dump
    if run["timed_out"]:
        started = progress.get("phase_started_unix")
        seconds = time.time() - started if isinstance(started, (int, float)) else None
        signal_ = feedback.signal_timeout(timeout, progress.get("phase"), seconds,
                                          feedback.locate_hang(hang_dump, workspace), workspace)
        return {"status": "error", "phase": progress.get("phase"),
                "phases": progress.get("phases", {})}, [signal_]
    try:
        payload = json.loads(run["payload_line"])
    except json.JSONDecodeError:
        return ({"status": "error", "phase": progress.get("phase"),
                 "phases": progress.get("phases", {})},
                [feedback.signal_crash(run["returncode"], run["stderr"])])
    return payload, [CompileSignal.from_dict(s) for s in payload.get("signals", [])]


# ---- versions, cache, history, log -----------------------------------------------------------


def _versions(result_dir: Path) -> set[str]:
    return {p.name for p in result_dir.iterdir() if _VERSION.fullmatch(p.name)}


def _remove_new_versions(result_dir: Path, before: set[str]) -> None:
    for name in _versions(result_dir) - before:
        shutil.rmtree(result_dir / name, ignore_errors=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _cached(result_dir: Path, digest: str, timeout: float) -> CompileResult | None:
    data = _read_json(result_dir / CACHE_FILE)
    if data.get("digest") != digest or data.get("toolchain") != toolchain_digest():
        return None
    signals = tuple(CompileSignal.from_dict(s) for s in data["signals"])
    if any(s.kind == feedback.SignalKind.TIMEOUT for s in signals) and data["timeout"] != timeout:
        return None  # a different time limit deserves a fresh try
    export_dir = Path(data["export_dir"]) if data.get("export_dir") else None
    if data["status"] == "success" and (export_dir is None or not export_dir.is_dir()):
        return None  # the export was removed since; build it again
    text = feedback.render_signals(data["status"], signals, success_summary=data["summary"],
                                   repeat_count=data.get("repeat_count", 0), cached=True)
    return CompileResult(
        status=data["status"], signals_text=text, signals=signals, version=data.get("version"),
        export_dir=export_dir, phase=data.get("phase"), phases=data.get("phases", {}),
        duration=0.0, startup=None, cached=True, digest=digest,
        returncode=data.get("returncode"), repeat_count=data.get("repeat_count", 0),
        log_path=result_dir / LOG_FILE, target_path=data.get("target_path"),
    )


def _write_cache(result_dir: Path, result: CompileResult, timeout: float, summary: str) -> None:
    atomic_write_text(result_dir / CACHE_FILE, json.dumps({
        "digest": result.digest, "toolchain": toolchain_digest(), "timeout": timeout,
        "status": result.status, "summary": summary,
        "signals": [s.to_dict() for s in result.signals], "signals_text": result.signals_text,
        "version": result.version,
        "export_dir": str(result.export_dir) if result.export_dir else None,
        "phase": result.phase, "phases": result.phases, "returncode": result.returncode,
        "repeat_count": result.repeat_count, "target_path": result.target_path,
    }, indent=2))


def _record_history(result_dir: Path, signals: list[CompileSignal]) -> int:
    """Append this compile's failure signature; return the streak of identical failures."""
    path = result_dir / HISTORY_FILE
    history = _read_json(path).get("signatures", [])
    signature = feedback.failure_signature(signals)
    history = [*history, signature][-50:]
    atomic_write_text(path, json.dumps({"signatures": history}))
    if signature is None:
        return 0
    streak = 0
    for previous in reversed(history):
        if previous != signature:
            break
        streak += 1
    return streak


def _log(path: Path, entry: dict[str, Any]) -> None:
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    append_line(path, json.dumps(entry))
