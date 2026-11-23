"""The agent's tools and their registry.

    ctx = ToolContext(run_dir)
    out = run_tool(ctx, "read", {"path": "main.py"})   # ToolOutput: JSON text, images
    messages.append(out.message(call.id))              # an llm.ToolResult

``schemas()`` lists the tools in one fixed order, so the tool-schema part of the prompt
prefix never changes between calls (prompt caching). There is no shell tool.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import llm
from agent.tools.compile_tool import COMPILE
from agent.tools.core import (MAX_OUTPUT_CHARS, AgentTool, ToolContext, ToolError, ToolOutput,
                              check_arguments, to_text)
from agent.tools.files import EDIT, READ, WRITE
from agent.tools.probe import PROBE
from agent.tools.render import RENDER

TOOLS: tuple[AgentTool, ...] = (READ, WRITE, EDIT, COMPILE, RENDER, PROBE)
BY_NAME = {tool.name: tool for tool in TOOLS}


def schemas() -> list[llm.Tool]:
    """The tool schemas, always in this order: read, write, edit, compile, render, probe."""
    return [tool.schema() for tool in TOOLS]


def schema_size() -> dict[str, int]:
    """Size of the schemas as sent (Chat Completions JSON): characters and a token estimate
    (about 4 characters per token; no tokenizer is installed)."""
    text = json.dumps(llm.messages.tools_to_chat(schemas()), ensure_ascii=False)
    return {"chars": len(text), "tokens_estimate": round(len(text) / 4)}


def run_tool(ctx: ToolContext, name: str, args: Any,
             limit: int = MAX_OUTPUT_CHARS) -> ToolOutput:
    """Run one tool call. Never raises: failures come back as structured JSON errors."""
    tool = BY_NAME.get(name)
    images: list[Path] = []
    try:
        if tool is None:
            raise ToolError("unknown_tool", f"there is no tool named {name!r}",
                            f"tools: {[t.name for t in TOOLS]}")
        if args is None:
            args = {}
        check_arguments(tool.parameters, args)
        data = tool.run(ctx, args)
        images = [Path(p) for p in data.pop("_images", [])]
        is_error = False
    except ToolError as exc:
        data, is_error = exc.to_dict(), True
    except Exception as exc:  # a tool bug: still structured, with where it happened
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        data = {"error_kind": "internal_error",
                "message": f"{type(exc).__name__}: {exc}",
                "hint": f"this is a tool bug ({Path(frame.filename).name}:{frame.lineno}); "
                        "try another way"}
        is_error = True
    return ToolOutput(name, data, to_text(data, limit), images, is_error)


__all__ = ["BY_NAME", "TOOLS", "AgentTool", "ToolContext", "ToolError", "ToolOutput",
           "run_tool", "schema_size", "schemas"]
