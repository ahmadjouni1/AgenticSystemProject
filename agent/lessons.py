"""Cross-run lessons (optional, off by default): SDK pitfalls learned from failed compiles.

``runs/lessons.md`` is a short bullet list. With ``lessons=True`` the loop appends it to the
system prompt, and after the run adds what that run's failed compiles taught:
exceptions in main.py (``AttributeError: 'Box' object has no attribute 'translate'``) and
model-structure errors. Check failures (collisions, floating parts) are design problems of
one object, not SDK pitfalls, so they are not kept.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fileio import atomic_write_text, read_jsonl

LESSONS_FILE = Path(__file__).resolve().parents[1] / "runs" / "lessons.md"
MAX_LESSONS = 15
HEADER = "# SDK pitfalls from earlier runs\n\n"
_KINDS = {"runtime_error", "validation", "import_error", "syntax_error"}
_PRIMARY = re.compile(r"Primary issue: \[(\w+)\](?: at \S+?:\d+)?: (.+)")  # skip "at main.py:11"
_HINTS = {
    "has no attribute": "check the SDK quickstart for the right method or argument",
    "unexpected keyword argument": "check the function's arguments in the SDK quickstart",
    "must contain 0": "joint limits must contain 0, the authored rest pose",
}


def load(path: Path = LESSONS_FILE) -> list[str]:
    if not path.is_file():
        return []
    return [line[2:].strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")]


def prompt_section(path: Path = LESSONS_FILE) -> str:
    lessons = load(path)
    if not lessons:
        return ""
    return ("\n\n## Pitfalls seen in earlier runs\n\n"
            + "\n".join(f"- {lesson}" for lesson in lessons) + "\n")


def lessons_from_run(run_dir: Path) -> list[str]:
    """One line per distinct SDK error in the run's failed compiles."""
    found: list[str] = []
    conversation = run_dir / "conversation.jsonl"
    if not conversation.is_file():
        return found
    for entry in read_jsonl(conversation):
        content = entry.get("content") or ""
        if "Primary issue:" not in content:
            continue
        if entry.get("role") == "tool":
            try:
                content = json.loads(content).get("compile_signals", "")
            except json.JSONDecodeError:
                pass
        match = _PRIMARY.search(content)
        if not match or match.group(1) not in _KINDS:
            continue
        message = match.group(2).split("\\n")[0].strip()[:160]
        hint = next((h for key, h in _HINTS.items() if key in message), "")
        lesson = message + (f" ({hint})" if hint else "")
        if lesson not in found:
            found.append(lesson)
    return found


def learn(run_dir: Path, path: Path = LESSONS_FILE) -> list[str]:
    """Add this run's lessons (newest last, at most MAX_LESSONS kept). Returns the new ones."""
    old = load(path)
    new = [lesson for lesson in lessons_from_run(run_dir) if lesson not in old]
    if new:
        kept = (old + new)[-MAX_LESSONS:]
        atomic_write_text(path, HEADER + "".join(f"- {lesson}\n" for lesson in kept))
    return new
