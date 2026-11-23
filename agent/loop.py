"""The agent loop: prompt -> model -> tools -> ... -> a checked, exported object.

    result = run("a laptop that opens and closes", model, budget)

Messages are ``[System(system.md), System(quickstart.md), User(task)]``; both system
messages are stable, so they form the cached prefix together with the tool schemas.

Each turn:
1. query the model (the budget is checked inside ``model.query``);
2. append its reply;
3. run its tool calls in order, consecutive ``parallel_ok`` calls (reads) concurrently;
4. append the results (and, for a model that accepts images, the rendered PNGs).

**Stop rule.** The run succeeds only when the model replies with text and no tool calls,
the latest compile passed every blocking check, and the workspace has not changed since
that compile (its digest equals the compile's). Otherwise a short ``<compile_required>``
or ``<final_response_required>`` reminder is added and the loop goes on.

**End states:** ``success``, ``failed_checks`` (the model keeps finishing while its latest
compile fails its checks), ``budget_exceeded``, ``max_turns``, ``no_progress`` (3 empty
replies in a row, 5 identical compile failures in a row, or 3 refused finishes without a
failing build to blame), ``error``. Every run keeps its best build: the latest version
that passed the checks, else the last build that failed them.

The run folder holds ``workspace/``, ``result/``, ``renders/``, ``llm_calls.jsonl``,
``conversation.jsonl`` (every message, tool call and result) and ``record.json``
(rewritten after every turn).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.tools import BY_NAME, ToolContext, ToolOutput, run_tool, schemas
from compiler import compile as compile_run
from compiler import workspace_digest
from compiler.feedback import failure_signature
from llm import (Budget, BudgetExceeded, CallLog, LLMError, Message, System, ToolResult, Usage,
                 User)
from llm.messages import to_chat, tools_to_chat
from llm.protocol import UNKNOWN, Model, Reply

REPLICA_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = REPLICA_ROOT / "prompts" / "system.md"
QUICKSTART = REPLICA_ROOT / "docs" / "sdk" / "quickstart.md"
TEMPLATE = REPLICA_ROOT / "prompts" / "main_template.py"
RUNS_ROOT = REPLICA_ROOT / "runs"

MAX_TURNS = 60
MAX_EMPTY_REPLIES = 3
MAX_SAME_FAILURES = 5
MAX_REFUSED_FINISHES = 3
PARALLEL_WORKERS = 4

Event = dict[str, Any]


@dataclass
class RunResult:
    status: str
    reason: str
    run_dir: Path
    turns: int
    final_text: str | None
    best: dict[str, Any] | None
    record: dict[str, Any] = field(repr=False)


# ---- the run folder ---------------------------------------------------------------------------


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "run"


def new_run_dir(prompt: str, root: str | Path = RUNS_ROOT) -> Path:
    """``runs/<UTC timestamp>-<slug>/``, with workspace/, result/ and renders/."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = Path(root) / f"{stamp}-{slugify(prompt)}"
    run_dir, n = base, 1
    while run_dir.exists():
        n += 1
        run_dir = base.with_name(f"{base.name}-{n}")
    for sub in ("workspace", "result", "renders"):
        (run_dir / sub).mkdir(parents=True)
    return run_dir


def best_build(run_dir: Path) -> dict[str, Any] | None:
    """The latest version that passed the checks, else the last build that failed them."""
    result = run_dir / "result"
    versions = sorted(p.name for p in result.iterdir()
                      if p.is_dir() and re.fullmatch(r"v\d{4,}", p.name)) if result.is_dir() else []
    if versions:
        return {"version": versions[-1], "passed_checks": True,
                "path": (result / versions[-1]).relative_to(run_dir).as_posix()}
    if (result / "last_failed" / "model.json").is_file():
        return {"version": "last_failed", "passed_checks": False,
                "path": "result/last_failed"}
    return None


def describe_best(best: dict[str, Any] | None) -> str:
    if best is None:
        return "no build was produced"
    if best["passed_checks"]:
        return f"best build: {best['version']} (passed the checks)"
    return "best build: result/last_failed (built, but failed its checks)"


def _message_dict(message: Message) -> dict[str, Any]:
    if isinstance(message, System):
        return {"role": "system", "content": message.content}
    if isinstance(message, User):
        out: dict[str, Any] = {"role": "user", "content": message.content}
        if message.images:
            out["images"] = list(message.images)
        return out
    if isinstance(message, ToolResult):
        return {"role": "tool", "call_id": message.call_id, "name": message.name,
                "content": message.content, "is_error": message.is_error}
    return {"role": "assistant", "content": message.content, "reasoning": message.reasoning,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments_json()}
                           for c in message.tool_calls]}


class _Files:
    def __init__(self, run_dir: Path):
        self.conversation = run_dir / "conversation.jsonl"
        self.record = run_dir / "record.json"

    def log(self, turn: int, message: Message, **extra: Any) -> None:
        entry = {"turn": turn, "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
                 **_message_dict(message), **extra}
        with self.conversation.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save(self, record: dict[str, Any]) -> None:
        temporary = self.record.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.record)


# ---- reminders --------------------------------------------------------------------------------


def compile_required(reason: str) -> str:
    return (f"<compile_required>\nYou cannot finish yet: {reason}. Call compile, fix every "
            "blocking failure, then finish with a short final message and no tool calls.\n"
            "</compile_required>")


FINAL_RESPONSE_REQUIRED = (
    "<final_response_required>\nThe latest compile passed and nothing changed since. Reply "
    "with a short final message (bodies, joints and their ranges) and no tool calls.\n"
    "</final_response_required>")


# ---- the loop ---------------------------------------------------------------------------------


class _Run:
    def __init__(self, prompt: str, model: Model, budget: Budget, max_turns: int, run_dir: Path,
                 compiler: Callable, on_event: Callable[[Event], None] | None):
        self.prompt, self.model, self.budget, self.max_turns = prompt, model, budget, max_turns
        self.run_dir = run_dir
        self.on_event = on_event or (lambda event: None)
        self.files = _Files(run_dir)
        self.ctx = ToolContext(run_dir, compiler=compiler)
        self.tools = schemas()
        self.tools_chars = len(json.dumps(tools_to_chat(self.tools), ensure_ascii=False))
        self.messages: list[Message] = []
        self.turn = 0
        self.started = time.time()
        self.usage = Usage()
        self.cost = 0.0
        self.unknown_cost_calls = 0
        self.per_turn: list[dict[str, Any]] = []
        self.tool_counts: Counter[str] = Counter()
        self.compiles = {"calls": 0, "passed": 0, "failed": 0, "cached": 0}
        self.empty_replies = 0
        self.refused_finishes = 0
        self.same_failures = 0
        self.last_signature: str | None = None
        self.final_text: str | None = None

    # -- bookkeeping --

    def add(self, message: Message, **extra: Any) -> None:
        self.messages.append(message)
        self.files.log(self.turn, message, **extra)

    def context_tokens(self) -> int:
        """What the next request sends, about 4 characters per token: messages as the
        provider receives them (DeepSeek also gets the current tool loop's reasoning back)
        plus the tool schemas."""
        settings = getattr(self.model, "settings", None)
        echo = getattr(settings, "reasoning_echo", "never")
        chars = len(json.dumps(to_chat(self.messages, echo), ensure_ascii=False))
        return round((chars + self.tools_chars) / 4)

    def record(self, status: str, reason: str = "") -> dict[str, Any]:
        last = self.ctx.last_compile
        best = best_build(self.run_dir)
        record = {
            "status": status, "reason": reason, "prompt": self.prompt,
            "run_dir": self.run_dir.as_posix(),
            "model": {"provider": getattr(self.model, "provider", "?"),
                      "model": getattr(self.model, "model", "?"),
                      "role": getattr(self.model, "role", None)},
            "started": datetime.fromtimestamp(self.started, UTC).isoformat(timespec="seconds"),
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self.started, 2),
            "turns": self.turn, "max_turns": self.max_turns,
            "compiles": {**self.compiles,
                         "last_status": last.status if last else None,
                         "last_version": f"v{last.version:04d}" if last and last.version
                         is not None else None},
            "tool_calls": dict(self.tool_counts),
            "tokens": self.usage.to_dict(),
            "cost_usd": round(self.cost, 6), "unknown_cost_calls": self.unknown_cost_calls,
            "budget": self.budget.to_dict(),
            "best": best, "best_description": describe_best(best),
            "final_text": self.final_text,
            "per_turn": self.per_turn,
        }
        self.files.save(record)
        return record

    def account(self, reply: Reply) -> dict[str, Any]:
        self.usage = self.usage + reply.usage
        if reply.cost == UNKNOWN:
            self.unknown_cost_calls += 1
        else:
            self.cost += float(reply.cost)
        return {"input_tokens": reply.usage.input_tokens,
                "cached_input_tokens": reply.usage.cached_input_tokens,
                "cached_share": round(reply.usage.cached_share, 4),
                "output_tokens": reply.usage.output_tokens,
                "reasoning_tokens": reply.usage.reasoning_tokens,
                "cost": reply.cost if reply.cost == UNKNOWN else round(float(reply.cost), 6),
                "latency_s": reply.latency_s}

    # -- the stop rule --

    def finish_blocker(self) -> str | None:
        """Why the model cannot finish now, or None if it can."""
        last = self.ctx.last_compile
        if last is None:
            return "you have not compiled yet"
        if not last.ok:
            primary = next((s for s in last.signals if s.severity == "failure"), None)
            what = f" ({primary.kind}: {primary.message[:160]})" if primary else ""
            return f"the latest compile failed{what}"
        if workspace_digest(self.ctx.workspace) != last.digest:
            return "the workspace changed after the last compile"
        return None

    # -- tools --

    def run_calls(self, calls, cut_off: bool = False) -> list[ToolOutput]:
        outputs: list[ToolOutput | None] = [None] * len(calls)
        i = 0
        while i < len(calls):
            j = i
            while (j < len(calls) and calls[j].error is None and calls[j].name in BY_NAME
                   and BY_NAME[calls[j].name].parallel_ok):
                j += 1
            if j - i > 1:  # a run of parallel-safe calls
                with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
                    batch = list(pool.map(lambda c: run_tool(self.ctx, c.name, c.arguments),
                                          calls[i:j]))
                outputs[i:j] = batch
                i = j
                continue
            call = calls[i]
            if call.error is not None and cut_off:
                data = {"error_kind": "reply_cut_off",
                        "message": "your reply reached the output token limit and was cut off, "
                                   f"so this {call.name} call's arguments are incomplete "
                                   f"({call.error})",
                        "hint": "nothing was run; reason less before calling tools, and write "
                                "large files in parts: a short first version, then edit"}
                outputs[i] = ToolOutput(call.name, data, json.dumps(data), [], True)
            elif call.error is not None:
                data = {"error_kind": "bad_arguments", "message": call.error,
                        "hint": "send the arguments as one JSON object"}
                outputs[i] = ToolOutput(call.name, data, json.dumps(data), [], True)
            else:
                outputs[i] = run_tool(self.ctx, call.name, call.arguments)
            i += 1
        return [o for o in outputs if o is not None]

    def after_compile(self) -> dict[str, Any]:
        result = self.ctx.last_compile
        self.compiles["calls"] += 1
        self.compiles["cached"] += int(result.cached)
        if result.ok:
            self.compiles["passed"] += 1
            self.same_failures, self.last_signature = 0, None
        else:
            self.compiles["failed"] += 1
            signature = failure_signature(result.signals) or result.signals_text
            self.same_failures = self.same_failures + 1 if signature == self.last_signature else 1
            self.last_signature = signature
        primary = next((s.kind for s in result.signals if s.severity == "failure"), None)
        return {"status": result.status, "cached": result.cached, "primary": primary,
                "version": result.version}

    # -- one run --

    def run(self) -> RunResult:
        self.add(System(SYSTEM_PROMPT.read_text(encoding="utf-8")))
        self.add(System(QUICKSTART.read_text(encoding="utf-8")))
        self.add(User(f"Build this object: {self.prompt}\n\nThe workspace has a starter "
                      "main.py. Plan the bodies and joints first, then write main.py."))
        status, reason = "max_turns", f"stopped after {self.max_turns} turns"
        try:
            status, reason = self.loop()
        except BudgetExceeded as exc:
            status, reason = "budget_exceeded", str(exc)
        except LLMError as exc:
            status, reason = "error", f"model call failed: {exc}"
        except Exception as exc:  # a loop bug: keep the run folder usable
            status = "error"
            reason = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
        if status != "success":
            reason = f"{reason}; {describe_best(best_build(self.run_dir))}"
        record = self.record(status, reason)
        self.on_event({"kind": "end", "status": status, "reason": reason,
                       "run_dir": self.run_dir, "record": record})
        return RunResult(status, reason, self.run_dir, self.turn, self.final_text,
                         record["best"], record)

    def loop(self) -> tuple[str, str]:
        while self.turn < self.max_turns:
            self.turn += 1
            metrics: dict[str, Any] = {"turn": self.turn,
                                       "context_tokens_est": self.context_tokens()}
            self.per_turn.append(metrics)
            try:
                reply = self.model.query(self.messages, self.tools)
            except BudgetExceeded as exc:
                if exc.reply is not None:  # paid for: keep it in the record
                    metrics.update(self.account(exc.reply))
                    self.add(exc.reply.message(), stopped="budget_exceeded")
                self.record("running")
                raise
            metrics.update(self.account(reply))
            self.add(reply.message(), finish_reason=reply.finish_reason)
            event: Event = {"kind": "turn", "turn": self.turn, "metrics": metrics, "tools": [],
                            "compile": None, "cost_total": self.cost}

            if reply.tool_calls:
                self.empty_replies = self.refused_finishes = 0
                outputs = self.run_calls(reply.tool_calls, reply.finish_reason == "length")
                images: list[str] = []
                compile_info = None
                for call, out in zip(reply.tool_calls, outputs, strict=True):
                    self.tool_counts[call.name] += 1
                    self.add(out.message(call.id))
                    event["tools"].append({"name": call.name, "args": call.arguments,
                                           "error": out.data.get("error_kind")
                                           if out.is_error else None})
                    if call.name == "compile" and not out.is_error:
                        compile_info = self.after_compile()
                    images += [str(p) for p in out.images]
                if images and getattr(self.model, "supports_images", False):
                    self.add(User("Rendered views (front, side, top, iso).", tuple(images)))
                metrics["tools"] = [c.name for c in reply.tool_calls]
                metrics["compile"] = compile_info
                event["compile"] = compile_info
                self.record("running")
                self.on_event(event)
                if self.same_failures >= MAX_SAME_FAILURES:
                    return "no_progress", (f"{self.same_failures} identical compile failures "
                                           "in a row")
                continue

            text = reply.text.strip()
            blocker = self.finish_blocker()
            if text and blocker is None:
                self.final_text = text
                metrics["finished"] = True
                self.record("running")
                self.on_event(event)
                return "success", "finished after a passing, fresh compile"
            if not text:
                self.empty_replies += 1
                if self.empty_replies >= MAX_EMPTY_REPLIES:
                    return "no_progress", f"{self.empty_replies} empty replies in a row"
            else:
                self.empty_replies = 0
                self.refused_finishes += 1
                if self.refused_finishes >= MAX_REFUSED_FINISHES:
                    last = self.ctx.last_compile
                    if last is not None and not last.ok:
                        return "failed_checks", (f"the model tried to finish "
                                                 f"{self.refused_finishes} times while the "
                                                 f"latest compile failed: {blocker}")
                    return "no_progress", (f"the model tried to finish {self.refused_finishes} "
                                           f"times without a fresh passing compile: {blocker}")
            reminder = FINAL_RESPONSE_REQUIRED if blocker is None else compile_required(blocker)
            kind = "final_response_required" if blocker is None else "compile_required"
            self.add(User(reminder), reminder=kind)
            metrics["reminder"] = kind
            event["reminder"] = kind
            self.record("running")
            self.on_event(event)
        return "max_turns", f"stopped after {self.max_turns} turns"


def run(prompt: str, model: Model, budget: Budget, max_turns: int = MAX_TURNS, *,
        run_dir: str | Path | None = None, runs_root: str | Path = RUNS_ROOT,
        compiler: Callable = compile_run,
        on_event: Callable[[Event], None] | None = None) -> RunResult:
    """Run the agent on ``prompt`` until it finishes with a checked build, or stops."""
    run_dir = Path(run_dir).resolve() if run_dir is not None else new_run_dir(prompt, runs_root)
    for sub in ("workspace", "result", "renders"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    main = run_dir / "workspace" / "main.py"
    if not main.exists():
        shutil.copyfile(TEMPLATE, main)
    # One budget and one call log for the run.
    if getattr(model, "budget", None) is None and hasattr(model, "budget"):
        model.budget = budget
    if getattr(model, "log", None) is None and hasattr(model, "log"):
        model.log = CallLog(run_dir)
    if getattr(model, "budget", budget) is not budget:
        raise ValueError("the model has a different Budget than the one given to run()")
    return _Run(prompt, model, budget, max_turns, run_dir, compiler, on_event).run()


__all__ = ["MAX_TURNS", "RunResult", "best_build", "new_run_dir", "run"]
