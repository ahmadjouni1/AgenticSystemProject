"""What every tool shares: the Tool type, the context, structured errors, path scoping,
argument checking and the output size cap.

A tool's ``run(ctx, args)`` returns a JSON-ready dict, or raises ``ToolError``. The
registry (``agent.tools.run_tool``) turns either into a ``ToolOutput`` whose ``text`` is
JSON, capped to ``MAX_OUTPUT_CHARS``. Errors always have the same shape:

    {"error_kind": "path_refused", "message": "...", "hint": "..."}
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import llm
from compiler import CompileResult
from compiler import compile as compile_run

REPLICA_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPLICA_ROOT / "docs" / "sdk"
DOCS_PREFIX = "docs/sdk"
MAX_OUTPUT_CHARS = 16_000  # about 4k tokens per tool result


class ToolError(Exception):
    """A tool failed in a way the model can act on."""

    def __init__(self, kind: str, message: str, hint: str = ""):
        super().__init__(message)
        self.kind, self.message, self.hint = kind, message, hint

    def to_dict(self) -> dict[str, str]:
        return {"error_kind": self.kind, "message": self.message, "hint": self.hint}


@dataclass
class ToolContext:
    """What tools may touch: one run's workspace, its run folder, and the compiler."""
    run_dir: Path
    compiler: Callable[[Path], CompileResult] = compile_run
    docs_dir: Path = DOCS_DIR
    last_compile: CompileResult | None = None
    cache: dict[str, Any] = field(default_factory=dict)  # loaded builds, keyed by folder

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    @property
    def workspace(self) -> Path:
        return self.run_dir / "workspace"  # the compiler's layout


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str  # one line: what it does and when to use it
    parameters: dict[str, Any]  # JSON Schema (object)
    run: Callable[[ToolContext, dict[str, Any]], dict[str, Any]]
    parallel_ok: bool = False  # safe to run alongside other parallel_ok calls

    def schema(self) -> llm.Tool:
        return llm.Tool(self.name, self.description, self.parameters)


@dataclass
class ToolOutput:
    name: str
    data: dict[str, Any]
    text: str  # what the model sees: JSON, capped
    images: list[Path] = field(default_factory=list)  # for models that accept images
    is_error: bool = False

    def message(self, call_id: str) -> llm.ToolResult:
        return llm.ToolResult(call_id, self.name, self.text, self.is_error)


# ---- paths ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    path: Path
    display: str  # as the model should write it next time
    in_docs: bool


def resolve(ctx: ToolContext, raw: object, *, write: bool = False) -> Resolved:
    """A workspace path, or ``docs/sdk/...`` (read-only). Refuses absolute paths, ``..``,
    and anything that resolves outside (symlinks included)."""
    if not isinstance(raw, str) or not raw.strip():
        raise ToolError("bad_arguments", "path must be a non-empty string",
                        "use a path relative to the workspace, like main.py")
    text = raw.strip().replace("\\", "/")
    if (PurePosixPath(text).is_absolute() or PureWindowsPath(raw.strip()).is_absolute()
            or PureWindowsPath(raw.strip()).drive):
        raise ToolError("path_refused", f"absolute paths are not allowed: {raw!r}",
                        "use a path relative to the workspace, like main.py")
    parts = [p for p in PurePosixPath(text).parts if p not in ("", ".")]
    if ".." in parts:
        raise ToolError("path_refused", f"'..' is not allowed in paths: {raw!r}",
                        "stay inside the workspace; SDK docs are under docs/sdk/")
    in_docs = parts[:2] == ["docs", "sdk"]
    if in_docs and write:
        raise ToolError("path_refused", f"{DOCS_PREFIX}/ is read-only: {raw!r}",
                        "write your files in the workspace, like main.py")
    base = ctx.docs_dir if in_docs else ctx.workspace
    rest = parts[2:] if in_docs else parts
    full = base.joinpath(*rest).resolve()
    if not full.is_relative_to(base.resolve()):
        raise ToolError("path_refused", f"{raw!r} resolves outside the allowed folders",
                        "stay inside the workspace; SDK docs are under docs/sdk/")
    display = "/".join([DOCS_PREFIX, *rest] if in_docs else rest) or "."
    return Resolved(full, display, in_docs)


# ---- arguments --------------------------------------------------------------------------------

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "object": (dict,), "array": (list,)}


def check_arguments(schema: dict[str, Any], args: object, where: str = "arguments") -> None:
    """Enough JSON Schema for our tools: types, required, enum, items, properties."""
    kind = schema.get("type")
    if kind is not None:
        ok = isinstance(args, _TYPES[kind]) and not (
            kind in ("integer", "number") and isinstance(args, bool))
        if not ok:
            raise ToolError("bad_arguments", f"{where} must be {kind}, got "
                            f"{type(args).__name__}", f"see the {where.split('.')[0]} schema")
    if "enum" in schema and args not in schema["enum"]:
        raise ToolError("bad_arguments", f"{where} must be one of {schema['enum']}, got {args!r}")
    if kind == "object":
        assert isinstance(args, dict)
        for name in schema.get("required", []):
            if name not in args:
                raise ToolError("bad_arguments", f"{where}: missing required {name!r}",
                                f"required: {schema.get('required')}")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for name, value in args.items():
            if name in properties:
                check_arguments(properties[name], value, f"{where}.{name}")
            elif extra is False:
                raise ToolError("bad_arguments", f"{where}: unknown field {name!r}",
                                f"fields: {sorted(properties)}")
            elif isinstance(extra, dict):
                check_arguments(extra, value, f"{where}.{name}")
    if kind == "array" and "items" in schema:
        assert isinstance(args, list)
        for i, item in enumerate(args):
            check_arguments(schema["items"], item, f"{where}[{i}]")


# ---- output size ------------------------------------------------------------------------------


def cap_text(text: str, limit: int, what: str = "text") -> str:
    """Head + tail of ``text`` within ``limit`` characters, with a note about the cut."""
    if len(text) <= limit:
        return text
    keep = max(0, limit - 160)
    head, tail = text[: keep * 2 // 3], text[len(text) - keep // 3:] if keep // 3 else ""
    note = (f"\n[... {len(text) - len(head) - len(tail)} characters of {what} omitted "
            "(head and tail shown); read a smaller range with offset/limit ...]\n")
    return head + note + tail


def to_text(data: dict[str, Any], limit: int = MAX_OUTPUT_CHARS) -> str:
    """JSON for the model, within ``limit``: the longest string fields are cut first."""
    text = json.dumps(data, ensure_ascii=False)
    for _ in range(8):
        if len(text) <= limit:
            return text
        path, longest = _longest_string(data)
        if path is None or len(longest) < 200:
            break
        excess = len(text) - limit
        _set(data, path, cap_text(longest, max(200, len(longest) - excess - 200), path[-1]))
        text = json.dumps(data, ensure_ascii=False)
    if len(text) > limit:  # many small fields: fall back to cutting the JSON itself
        text = json.dumps({"truncated_json": cap_text(text, limit - 40, "output")})
    return text


def _longest_string(data: Any, path: tuple = ()) -> tuple[tuple | None, str]:
    best: tuple[tuple | None, str] = (None, "")
    items = data.items() if isinstance(data, dict) else enumerate(data) \
        if isinstance(data, list) else ()
    for key, value in items:
        if isinstance(value, str):
            candidate = (path + (key,), value)
        else:
            candidate = _longest_string(value, path + (key,))
        if candidate[0] is not None and len(candidate[1]) > len(best[1]):
            best = candidate
    return best


def _set(data: Any, path: tuple, value: str) -> None:
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value
