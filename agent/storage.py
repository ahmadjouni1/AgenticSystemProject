"""Run storage (step 11): read a run folder back, resume it, list runs, the run index.

A run folder (agent/loop.py) holds ``record.json`` (rewritten after every turn),
``conversation.jsonl`` (one line per message, ``llm.messages.to_dict`` plus ``turn``,
``time`` and loop notes), ``workspace/`` and ``result/`` (each version has its ``main.py``).
That is enough to continue a run that stopped:

- ``load_conversation`` reads the messages back. A last line cut short by a crash is
  skipped. Tool calls at the end that never got a result (the process died while a tool
  ran, or the run stopped before running them) get an ``interrupted`` / ``not_run`` result,
  so the conversation is valid again; these results are appended to the log too.
- ``check_resumable`` refuses a run that succeeded, or one another process is still running
  (``.lock`` holds the running process's id).
- ``clean_after_crash`` deletes half-written temporary files and export staging folders.

``runs/index.jsonl`` gets one line per finished session (``append_index``); step 12 reads
it. ``list_runs`` reads every ``record.json`` under a folder (``python -m agent --list``).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fileio import (append_line, atomic_write_json, drop_cut_off_line, read_jsonl,
                    remove_stale_temps)
from llm.messages import Assistant, Message, ToolResult, from_dict, to_dict

RECORD = "record.json"
CONVERSATION = "conversation.jsonl"
LOCK = ".lock"
INDEX = "index.jsonl"
UNFINISHED = "running"  # a record still saying this after its process ended: a crash
SKIP_DIRS = {"workspace", "result", "renders", "__pycache__"}


class ResumeError(RuntimeError):
    """The run cannot be resumed (it succeeded, it is still running, files are missing)."""


# ---- reading a run back -----------------------------------------------------------------------


@dataclass
class Saved:
    """A run folder as it was left."""
    run_dir: Path
    record: dict[str, Any]
    messages: list[Message]
    turn_of: list[int]
    repaired: list[dict[str, Any]] = field(default_factory=list)  # entries added on load


def read_record(run_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(run_dir) / RECORD).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_conversation(run_dir: Path) -> tuple[list[Message], list[int], list[dict[str, Any]]]:
    """(messages, the turn of each, the log entries added to close open tool calls).

    Only the end of the log can hold open calls (the loop answers every call before it
    logs anything else); open calls anywhere else mean the log is damaged. A last line cut
    short by a crash is removed from the file first, so the next session can append."""
    drop_cut_off_line(Path(run_dir) / CONVERSATION)
    entries = read_jsonl(Path(run_dir) / CONVERSATION)
    if not entries:
        raise ResumeError(f"{run_dir} has no {CONVERSATION}")
    messages = [from_dict(e) for e in entries]
    turn_of = [int(e.get("turn", 0)) for e in entries]
    open_calls: dict[str, tuple[str, str]] = {}  # id -> (name, why it may be open)
    for i, (entry, message) in enumerate(zip(entries, messages, strict=True)):
        if isinstance(message, ToolResult):
            open_calls.pop(message.call_id, None)
            continue
        if open_calls:
            raise ResumeError(f"{CONVERSATION} line {i + 1}: tool calls "
                              f"{sorted(open_calls)} have no result before it")
        if isinstance(message, Assistant):
            why = "not_run" if entry.get("stopped") or entry.get("not_run") else "interrupted"
            open_calls = {c.id: (c.name, why) for c in message.tool_calls}
    repaired = []
    last_turn = turn_of[-1]
    for call_id, (name, why) in open_calls.items():
        if why == "not_run":
            data = {"error_kind": "not_run",
                    "message": f"the run stopped before this {name} call ran; nothing was "
                               "changed by it",
                    "hint": "call it again if you still need it"}
        else:
            data = {"error_kind": "interrupted",
                    "message": f"the run stopped while this {name} call was running; it may "
                               "or may not have taken effect",
                    "hint": "read the file or compile to see the current state"}
        result = ToolResult(call_id, name, json.dumps(data), True)
        messages.append(result)
        turn_of.append(last_turn)
        repaired.append({"turn": last_turn, **to_dict(result), "repaired": why})
    return messages, turn_of, repaired


def load(run_dir: str | Path) -> Saved:
    run_dir = Path(run_dir).resolve()
    record = read_record(run_dir)
    if record is None:
        raise ResumeError(f"{run_dir} has no readable {RECORD}")
    messages, turn_of, repaired = load_conversation(run_dir)
    return Saved(run_dir, record, messages, turn_of, repaired)


# ---- the lock: is another process running this run? -----------------------------------------


def pid_alive(pid: int) -> bool:
    """Whether a process with this id exists. (On Windows ``os.kill(pid, 0)`` would
    terminate it, so the Win32 API is asked instead.)"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(run_dir: Path) -> None:
    atomic_write_json(Path(run_dir) / LOCK, {"pid": os.getpid(), "since": _now()})


def release_lock(run_dir: Path) -> None:
    (Path(run_dir) / LOCK).unlink(missing_ok=True)


def lock_holder(run_dir: Path) -> int | None:
    """The id of another live process that holds the run, else None."""
    try:
        pid = int(json.loads((Path(run_dir) / LOCK).read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return pid if pid != os.getpid() and pid_alive(pid) else None


def status_of(run_dir: Path, record: dict[str, Any]) -> str:
    """The record's status, with ``crashed`` for a run whose process ended mid-session."""
    status = record.get("status", "?")
    if status == UNFINISHED and lock_holder(run_dir) is None:
        return "crashed"
    return status


def check_resumable(saved: Saved) -> str:
    """The status being resumed from, or ResumeError."""
    status = status_of(saved.run_dir, saved.record)
    if status == "success":
        raise ResumeError(f"{saved.run_dir.name} already succeeded; nothing to resume")
    if status == UNFINISHED:
        raise ResumeError(f"{saved.run_dir.name} is running in process "
                          f"{lock_holder(saved.run_dir)}; stop it first")
    return status


def clean_after_crash(run_dir: Path) -> list[str]:
    """Delete what an interrupted write or export leaves behind. Returns what was removed."""
    run_dir = Path(run_dir)
    removed = [p.name for folder in (run_dir, run_dir / "workspace", run_dir / "result")
               for p in remove_stale_temps(folder)]
    old_temp = run_dir / "record.json.tmp"  # before step 11
    if old_temp.is_file():
        old_temp.unlink()
        removed.append(old_temp.name)
    result = run_dir / "result"
    if result.is_dir():
        for path in result.iterdir():
            if path.is_dir() and (path.name.endswith(".tmp") or path.name == ".last_failed_build"):
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path.name)
    return removed


def append_conversation(run_dir: Path, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        append_line(Path(run_dir) / CONVERSATION,
                    json.dumps({"time": _now(), **entry}, ensure_ascii=False))


# ---- the index and the list -------------------------------------------------------------------


def default_index(run_dir: Path, runs_root: Path) -> Path:
    """``runs/index.jsonl`` for a run anywhere under ``runs/`` (experiments use subfolders);
    else ``index.jsonl`` next to the run folder."""
    run_dir, runs_root = Path(run_dir).resolve(), Path(runs_root).resolve()
    if run_dir.is_relative_to(runs_root):
        return runs_root / INDEX
    return run_dir.parent / INDEX


def index_entry(record: dict[str, Any], index_path: Path) -> dict[str, Any]:
    run_dir = Path(record["run_dir"])
    try:
        where = run_dir.resolve().relative_to(index_path.parent.resolve()).as_posix()
    except ValueError:
        where = run_dir.as_posix()
    best = record.get("best") or {}
    sessions = record.get("sessions") or [{}]
    return {
        "run": where, "prompt": record.get("prompt"), "status": record.get("status"),
        "reason": (record.get("reason") or "").splitlines()[0][:200] if record.get("reason")
        else "",
        "session": sessions[-1].get("session", 1), "ended": record.get("updated"),
        "turns": record.get("turns"), "cost_usd": record.get("cost_usd"),
        "duration_s": record.get("duration_s"),
        "best": best.get("version"), "best_passed": best.get("passed_checks"),
        "model": (record.get("model") or {}).get("model"),
        "prompt_set": record.get("prompt_set"),
        "two_phase": bool(record.get("two_phase")),
        "quality": {k: (record.get("quality") or {}).get(k)
                    for k in ("bodies", "shapes", "joints", "non_box_share")}
        if record.get("quality") else None,
    }


def append_index(record: dict[str, Any], index_path: Path) -> dict[str, Any]:
    entry = index_entry(record, index_path)
    append_line(index_path, json.dumps(entry, ensure_ascii=False))
    return entry


def find_runs(root: str | Path) -> list[Path]:
    """Every run folder (one with a record.json) under ``root``, ``root`` itself included."""
    root = Path(root)
    found = []
    for folder, dirs, files in os.walk(root):
        if RECORD in files:
            found.append(Path(folder))
            dirs[:] = []  # a run folder holds no runs
            continue
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
    return sorted(found)


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    """One row per run under ``root``: status, turns, cost, best version, sessions."""
    root = Path(root).resolve()
    rows = []
    for run_dir in find_runs(root):
        record = read_record(run_dir)
        if record is None:
            rows.append({"run": _relative(run_dir, root), "status": "unreadable"})
            continue
        best = record.get("best") or {}
        rows.append({
            "run": _relative(run_dir, root),
            "status": status_of(run_dir, record),
            "turns": record.get("turns"),
            "cost_usd": record.get("cost_usd"),
            "best": best.get("version") if best.get("passed_checks") else
            ("last_failed" if best else None),
            "sessions": len(record.get("sessions") or [None]),
            "prompt": record.get("prompt", ""),
        })
    return rows


def format_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no runs found"
    head = ("run", "status", "turns", "cost", "best", "sessions", "prompt")
    table = [head] + [(r["run"], r["status"], _text(r.get("turns")),
                       f"${r['cost_usd']:.4f}" if isinstance(r.get("cost_usd"), (int, float))
                       else "-", r.get("best") or "-", _text(r.get("sessions")),
                       (r.get("prompt") or "")[:40]) for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(head) - 1)]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row[:-1], widths, strict=True))
             + "  " + row[-1] for row in table]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    spent = sum(r["cost_usd"] for r in rows if isinstance(r.get("cost_usd"), (int, float)))
    lines.append(f"\n{len(rows)} runs: " + ", ".join(f"{n} {s}" for s, n in sorted(
        counts.items())) + f"; ${spent:.4f} spent")
    return "\n".join(lines)


def _text(value: Any) -> str:
    return "-" if value is None else str(value)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix() or "."
    except ValueError:
        return path.as_posix()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


__all__ = ["CONVERSATION", "INDEX", "LOCK", "RECORD", "ResumeError", "Saved", "acquire_lock",
           "append_conversation", "append_index", "check_resumable",
           "clean_after_crash", "default_index", "find_runs", "format_list", "list_runs",
           "load", "load_conversation", "lock_holder", "pid_alive", "read_record",
           "release_lock", "status_of"]
