"""read, write, edit: the workspace files (and the SDK docs, read-only)."""

from __future__ import annotations

import re
from typing import Any

from agent.tools.core import AgentTool, ToolContext, ToolError, resolve

DEFAULT_LINES = 400
MAX_LINES = 2_000
MAX_LINE_CHARS = 1_000
MAX_WRITE_CHARS = 400_000


def _text(path, display: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError("not_text", f"{display} is not a UTF-8 text file",
                        "only text files can be read; renders are returned by the render tool"
                        ) from None


def _missing(display: str, in_docs: bool) -> ToolError:
    where = "docs/sdk" if in_docs else "."
    return ToolError("not_found", f"{display} does not exist",
                     f"read {where!r} to list the files there")


# ---- read -------------------------------------------------------------------------------------


def run_read(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    target = resolve(ctx, args["path"])
    if target.path.is_dir():
        entries = sorted(
            (p.name + ("/" if p.is_dir() else "") for p in target.path.iterdir()
             if p.name not in ("__pycache__",) and not p.name.startswith(".")),
            key=lambda n: (not n.endswith("/"), n))
        return {"path": target.display, "entries": entries}
    if not target.path.is_file():
        raise _missing(target.display, target.in_docs)
    lines = _text(target.path, target.display).splitlines()
    offset = args.get("offset", 1)
    limit = args.get("limit", DEFAULT_LINES)
    if offset < 1 or limit < 1:
        raise ToolError("bad_arguments", "offset and limit must be at least 1",
                        "offset is the first line number (1-based)")
    limit = min(limit, MAX_LINES)
    if lines and offset > len(lines):
        raise ToolError("bad_arguments", f"offset {offset} is past the end "
                        f"({len(lines)} lines)", f"use an offset from 1 to {len(lines)}")
    shown = lines[offset - 1: offset - 1 + limit]
    numbered = "\n".join(
        f"L{offset + i}: " + (line if len(line) <= MAX_LINE_CHARS
                              else line[:MAX_LINE_CHARS] + f" [... {len(line)} chars]")
        for i, line in enumerate(shown))
    last = offset + len(shown) - 1
    result = {"path": target.display, "lines": f"{offset}-{last}" if shown else "0",
              "total_lines": len(lines), "text": numbered}
    if last < len(lines):
        result["next_offset"] = last + 1
    return result


READ = AgentTool(
    "read",
    "Read a workspace file or an SDK doc (docs/sdk/...) as numbered lines; a folder path "
    "lists its files. Use before editing.",
    {"type": "object", "properties": {
        "path": {"type": "string"},
        "offset": {"type": "integer", "description": "first line, 1-based"},
        "limit": {"type": "integer", "description": f"lines to return (default {DEFAULT_LINES})"},
    }, "required": ["path"], "additionalProperties": False},
    run_read, parallel_ok=True)


# ---- write ------------------------------------------------------------------------------------


def run_write(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    target = resolve(ctx, args["path"], write=True)
    content = args["content"]
    if len(content) > MAX_WRITE_CHARS:
        raise ToolError("too_large", f"content is {len(content)} characters "
                        f"(max {MAX_WRITE_CHARS})", "split the code into several files")
    if target.path.is_dir():
        raise ToolError("bad_arguments", f"{target.display} is a folder", "give a file path")
    created = not target.path.exists()
    target.path.parent.mkdir(parents=True, exist_ok=True)
    target.path.write_text(content, encoding="utf-8", newline="\n")
    return {"path": target.display, "created": created,
            "bytes": len(content.encode("utf-8")), "lines": len(content.splitlines())}


WRITE = AgentTool(
    "write",
    "Create or replace a workspace file with the full content. Use for new files; prefer "
    "edit for changes.",
    {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"},
    }, "required": ["path", "content"], "additionalProperties": False},
    run_write)


# ---- edit -------------------------------------------------------------------------------------


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _near_miss(text: str, old: str) -> str:
    """Where ``old`` would match if whitespace were ignored, as a hint."""
    words = old.split()
    if not words:
        return ""
    pattern = r"\s+".join(re.escape(w) for w in words)
    match = re.search(pattern, text)
    if match:
        return (f"it matches at line {_line_of(text, match.start())} if whitespace is ignored; "
                "copy the exact indentation and line breaks from read")
    first = old.strip().splitlines()[0].strip() if old.strip() else ""
    for number, line in enumerate(text.splitlines(), 1):
        if first and first in line:
            return f"its first line appears at line {number}; the lines after it differ"
    return "read the file again and copy old_text exactly"


def run_edit(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    target = resolve(ctx, args["path"], write=True)
    if not target.path.is_file():
        raise _missing(target.display, target.in_docs)
    edits = args["edits"]
    if not edits:
        raise ToolError("bad_arguments", "edits is empty", "give at least one edit")
    original = text = _text(target.path, target.display)
    for number, item in enumerate(edits, 1):
        old, new = item["old_text"], item["new_text"]
        label = f"edit {number} of {len(edits)}"
        if not old:
            raise ToolError("edit_failed", f"{label}: old_text is empty; nothing was changed",
                            "to add text, include a nearby line in old_text and new_text")
        count = text.count(old)
        if count == 0:
            raise ToolError("edit_failed",
                            f"{label}: old_text was not found in {target.display}; "
                            "nothing was changed", _near_miss(text, old))
        if count > 1:
            lines = []
            start = text.find(old)
            while start != -1:
                lines.append(_line_of(text, start))
                start = text.find(old, start + 1)
            raise ToolError("edit_failed",
                            f"{label}: old_text matches {count} times (lines "
                            f"{', '.join(map(str, lines))}); nothing was changed",
                            "include more surrounding lines so it matches exactly once")
        text = text.replace(old, new, 1)
    if text != original:
        target.path.write_text(text, encoding="utf-8", newline="\n")
    return {"path": target.display, "replaced": len(edits),
            "bytes": len(text.encode("utf-8")), "lines": len(text.splitlines())}


EDIT = AgentTool(
    "edit",
    "Replace exact text in a workspace file; each old_text must match exactly once, and all "
    "edits apply or none. Use for targeted fixes.",
    {"type": "object", "properties": {
        "path": {"type": "string"},
        "edits": {"type": "array", "items": {
            "type": "object", "properties": {
                "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["old_text", "new_text"], "additionalProperties": False}},
    }, "required": ["path", "edits"], "additionalProperties": False},
    run_edit)
