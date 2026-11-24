"""Crash-safe file writes (step 11).

A crash (the process killed, the laptop asleep and out of power, Ctrl+C) in the middle of a
plain ``write_text`` leaves half a file. Every file a later step reads back goes through one
of these instead:

- ``atomic_write_text`` / ``atomic_write_bytes`` / ``atomic_copy``: write a temporary file
  next to the target, flush it to disk, then ``os.replace`` it over the target. Readers see
  the old file or the new one, never half of one. On Windows ``os.replace`` fails while
  another process has the target open (an indexer, an antivirus scan, the viewer), so it is
  retried a few times.
- ``append_line``: one JSON line per ``write`` call, flushed to disk. A crash can still cut
  the last line short; ``read_jsonl`` skips a last line that does not parse.

Temporary files are named ``.<name>.<8 hex>.tmp``; ``remove_stale_temps`` deletes the ones a
crash left behind (resume calls it).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

REPLACE_ATTEMPTS = 6
STALE_TEMP = re.compile(r"^\..+\.[0-9a-f]{8}\.tmp$")


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")


def _replace(source: Path, target: Path) -> None:
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == REPLACE_ATTEMPTS:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8",
                      newline: str | None = None) -> None:
    """``Path.write_text`` with the same ``newline`` meaning, written atomically."""
    if newline not in (None, "", "\n"):
        text = text.replace("\n", newline)
    elif newline is None and os.linesep != "\n":
        text = text.replace("\n", os.linesep)
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | os.PathLike[str], data: Any, **dumps: Any) -> None:
    atomic_write_text(path, json.dumps(data, **dumps), newline="\n")


def atomic_copy(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(target)
    try:
        shutil.copyfile(source, temporary)
        _replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def append_line(path: str | os.PathLike[str], line: str) -> None:
    """Append one line (a newline is added) in a single write, flushed to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line.rstrip("\n") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Every JSON line of ``path`` ([] if it does not exist). A last line that does not
    parse (cut short by a crash) is skipped; a bad line anywhere else is an error."""
    path = Path(path)
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
    return out


def drop_cut_off_line(path: str | os.PathLike[str]) -> bool:
    """Remove a last line that a crash cut short, so that appending can go on (an append
    after it would join the two lines). True if one was removed."""
    path = Path(path)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    body, _, last = text.rstrip("\n").rpartition("\n")
    if not last.strip():
        return False
    try:
        json.loads(last)
    except json.JSONDecodeError:
        atomic_write_text(path, body + "\n" if body else "", newline="\n")
        return True
    if not text.endswith("\n"):  # complete, but its newline was not written
        append_line(path, "")
    return False


def remove_stale_temps(folder: str | os.PathLike[str]) -> list[Path]:
    """Delete the temporary files a crash left in ``folder`` (not recursive)."""
    folder = Path(folder)
    removed = []
    if folder.is_dir():
        for path in folder.iterdir():
            if path.is_file() and STALE_TEMP.match(path.name):
                path.unlink(missing_ok=True)
                removed.append(path)
    return removed
