"""The compiler: run workspace/main.py in a fresh process, check it, export it, and tell
the agent what happened as a compact <compile_signals> block."""

from __future__ import annotations

from compiler.feedback import CompileSignal, SignalKind, render_signals
from compiler.runner import CompileResult, compile, workspace_digest  # noqa: A004

__all__ = ["CompileResult", "CompileSignal", "SignalKind", "compile", "render_signals",
           "workspace_digest"]
