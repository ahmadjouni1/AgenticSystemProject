"""compile: run step 3's compiler on the workspace."""

from __future__ import annotations

from typing import Any

from agent.tools.core import AgentTool, ToolContext


def run_compile(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    result = ctx.compiler(ctx.run_dir)  # cached by the compiler when nothing changed
    ctx.last_compile = result
    return {"status": result.status,
            "version": f"v{result.version:04d}" if result.version is not None else None,
            "cached": result.cached,
            "compile_signals": result.signals_text}


COMPILE = AgentTool(
    "compile",
    "Build, check and export workspace/main.py; returns the compile signals. Use after each "
    "real change and before finishing.",
    {"type": "object", "properties": {}, "additionalProperties": False},
    run_compile)
