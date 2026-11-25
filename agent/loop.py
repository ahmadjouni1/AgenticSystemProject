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

**Time limit** (25 min by default, step 10b). When it runs out and the latest compile passed
with no edit after it, the model gets one final turn for its summary and the run ends as a
success; otherwise ``time_exceeded``. A fix cycle (the turn after a failed compile) does not
start with less than ``fix_reserve_s`` (2 min) left. Every model call gets the time left as
its timeout, retries included, and an output cap the model can write in that time (step 10b
follow-up: one cut-off reply took up to 9.5 min and carried two runs 91 s and 351 s past
the limit).

**Two-phase build** (``two_phase=True``, step 10b). Phase 1 asks for a skeleton (plain
primitives at real sizes, the real joints); its first passing compile (every check, the
motion checks included) starts phase 2, which asks for the detail. The skeleton's version
stays the best build unless a phase 2 compile passes.

**End states:** ``success``, ``failed_checks`` (the model keeps finishing while its latest
compile fails its checks), ``budget_exceeded``, ``time_exceeded`` (wall clock),
``max_turns``, ``no_progress`` (3 empty
replies in a row, 5 identical compile failures in a row, or 3 refused finishes without a
failing build to blame), ``error``. Every run keeps its best build: the latest version
that passed the checks, else the last build that failed them.

The run folder holds ``workspace/``, ``result/``, ``renders/``, ``llm_calls.jsonl``,
``conversation.jsonl`` (every message, tool call and result) and ``record.json``
(rewritten after every turn). **Resume** (step 11, ``resume``): a stopped run continues from
its conversation and workspace with a new allowance; the record keeps one entry per session,
and each finished session adds a line to ``runs/index.jsonl`` (agent/storage.py).
"""

from __future__ import annotations

import json
import re
import time
import traceback
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent import lessons as lessons_file
from agent import storage
from agent.context import ContextPolicy, checkpoint_text
from agent.tools import BY_NAME, ToolContext, ToolOutput, run_tool, schemas
from agent.tools.synthesize import SYSTEM_LINE as SYNTHESIS_LINE
from compiler import compile as compile_run
from compiler import workspace_digest
from compiler.feedback import failure_signature
from fileio import append_line, atomic_copy, atomic_write_json
from llm import (Budget, BudgetExceeded, CallLog, CallTimeLimit, LLMError, Message, System,
                 ToolResult, Usage, User)
from llm.messages import to_chat, to_dict, tools_to_chat
from llm.protocol import UNKNOWN, Model, Reply

REPLICA_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = REPLICA_ROOT / "prompts" / "system.md"
QUICKSTART = REPLICA_ROOT / "docs" / "sdk" / "quickstart.md"


@dataclass(frozen=True)
class PromptSet:
    """What the model is told: the system prompt, the quickstart (the second system message),
    the folder the agent sees as ``docs/sdk/``, and the task message."""
    name: str
    system: Path
    quickstart: Path
    docs_dir: Path
    task: str  # formatted with {prompt}
    synthesis: bool = False
    """Step 13: offer the synthesize tool, and list docs/sdk/synthesis.md in the system prompt
    (one line after the testing.md line)."""

    def system_text(self) -> str:
        text = self.system.read_text(encoding="utf-8")
        if not self.synthesis:
            return text
        lines = text.splitlines(keepends=True)
        at = next(i for i, line in enumerate(lines) if line.startswith("- `testing.md`"))
        while at + 1 < len(lines) and lines[at + 1].startswith("  "):  # wrapped bullet
            at += 1
        return "".join([*lines[:at + 1], SYNTHESIS_LINE + "\n", *lines[at + 1:]])


PROMPT_SETS = {
    # Step 9's prompt and docs, frozen for comparison (step 10 experiment, variant A).
    "v0": PromptSet("v0", REPLICA_ROOT / "prompts" / "v0" / "system.md",
                    REPLICA_ROOT / "prompts" / "v0" / "docs" / "quickstart.md",
                    REPLICA_ROOT / "prompts" / "v0" / "docs",
                    "Build this object: {prompt}\n\nThe workspace has a starter main.py. Plan "
                    "the bodies and joints first, then write main.py."),
    # Step 10: short written plan, realism rules, on-demand doc pages.
    "v1": PromptSet("v1", SYSTEM_PROMPT, QUICKSTART, QUICKSTART.parent,
                    "Build this object: {prompt}\n\nThe workspace has a starter main.py. Write "
                    "a short plan (at most about 15 lines) in your reply, then call write with "
                    "the whole main.py in the same reply."),
}
# Step 13: v1 plus the synthesize tool and one system-prompt line about it.
PROMPT_SETS["v2"] = replace(PROMPT_SETS["v1"], name="v2", synthesis=True)
DEFAULT_PROMPT_SET = "v1"
TEMPLATE = REPLICA_ROOT / "prompts" / "main_template.py"
RUNS_ROOT = REPLICA_ROOT / "runs"

MAX_TURNS = 60
MAX_TIME_S = 25 * 60  # wall clock per run (step 10b; 15 min before)
FIX_RESERVE_S = 120  # no fix cycle starts with less time left (median fix cycle: 1.3 min)
FINAL_TURN_S = 120  # the final turn's own allowance, after the time is up
DEFAULT_TOKENS_PER_S = 60  # output speed before the run has measured its own (DeepSeek, step
# 10/10b builder calls with >= 2k output tokens: 10th percentile 60, median 75 tokens/s)
CALL_OVERHEAD_S = 15  # reading the prompt before the first output token, roughly
MIN_OUTPUT_TOKENS = 2_000  # the output cap never goes lower; the timeout still holds
MAX_EMPTY_REPLIES = 3
MAX_SAME_FAILURES = 5
MAX_REFUSED_FINISHES = 3
PARALLEL_WORKERS = 4

Event = dict[str, Any]


@dataclass(frozen=True)
class EffortPolicy:
    """Less reasoning on routine turns. A turn is routine once the latest compile passed
    (verifying, probing, finishing); fixing turns keep the model's default. Planning turns
    (every call before the first compile) use ``planning_effort``, or the default if None."""
    routine_effort: str = "low"
    routine_max_output_tokens: int = 16_000
    planning_effort: str | None = None


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


class _Files:
    def __init__(self, run_dir: Path):
        self.conversation = run_dir / "conversation.jsonl"
        self.record = run_dir / "record.json"

    def log(self, turn: int, message: Message, **extra: Any) -> None:
        entry = {"turn": turn, "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
                 **to_dict(message), **extra}
        append_line(self.conversation, json.dumps(entry, ensure_ascii=False))

    def save(self, record: dict[str, Any]) -> None:
        atomic_write_json(self.record, record, indent=2, ensure_ascii=False)


# ---- reminders --------------------------------------------------------------------------------


def compile_required(reason: str) -> str:
    return (f"<compile_required>\nYou cannot finish yet: {reason}. Call compile, fix every "
            "blocking failure, then finish with a short final message and no tool calls.\n"
            "</compile_required>")


REPLY_CUT_OFF = (
    "<reply_cut_off>\nYour last reply reached the output token limit while reasoning and "
    "produced no answer and no tool call. Keep your reasoning short: plan the bodies and joints "
    "in a few lines, then call write with a first version of main.py, and refine it with edit "
    "after compiling.\n</reply_cut_off>")


TIME_UP_FINAL = (
    "<time_up>\nThe time for this run is up. The latest compile passed and nothing changed "
    "since, so that build is the result. Do not call any tool: reply now with a short final "
    "message (bodies, joints and their ranges).\n</time_up>")

SKELETON_TASK = (
    "Build this object: {prompt}\n\nThis is a two-phase build.\n\nPhase 1 (now): write a "
    "skeleton main.py. Every real body as plain primitives (Box, Cylinder, Sphere) at real "
    "sizes, and every real joint with its real type, pivot, axis and limits. No details yet: no "
    "handles, knobs, rounded edges or shape helpers. Write a short plan (at most about 10 lines) "
    "in your reply, then call write with the whole main.py in the same reply, then compile. "
    "Phase 1 ends when the skeleton passes every check, the motion checks included (floating, "
    "separation, swept collision, loop closure); then you get the phase 2 instructions.")


def detail_phase(version: str) -> str:
    return (
        f"<phase_2>\nThe skeleton passed every check; it is kept as the fallback build "
        f"({version}). Phase 2: add the detail that makes the object realistic and recognizable "
        "(rule 1 of \"What a good object is\"): shaped parts with the shape helpers, handles, "
        "knobs, rims, feet, rounded edges, real wall and panel thickness. Keep the skeleton's "
        "bodies, joints, pivots, axes and limits unless a detail needs a change. Add the details "
        "in a few groups with edit, and compile after each group, so a failure is in the newest "
        "detail. Finish as usual: a passing compile, the joints probed, a short final "
        "message.\n</phase_2>")


def resumed_message(status: str, reason: str, allowance: dict[str, Any]) -> str:
    what = reason.splitlines()[0][:200] if reason else ""
    parts = []
    if allowance.get("max_time_s"):
        parts.append(f"{allowance['max_time_s'] / 60:.0f} min")
    if allowance.get("max_cost_usd") is not None:
        parts.append(f"${allowance['max_cost_usd']:.2f}")
    parts.append(f"{allowance['max_turns']} turns")
    return (f"<resumed>\nThis run stopped ({status}{': ' + what if what else ''}) and now "
            f"continues with a new allowance ({', '.join(parts)}). The workspace is as it was "
            "left. Compile first to see where the build stands, then go on and finish as "
            "usual.\n</resumed>")


FINAL_RESPONSE_REQUIRED = (
    "<final_response_required>\nThe latest compile passed and nothing changed since. Reply "
    "with a short final message (bodies, joints and their ranges) and no tool calls.\n"
    "</final_response_required>")


# ---- the loop ---------------------------------------------------------------------------------


class _Run:
    def __init__(self, prompt: str, model: Model, budget: Budget, max_turns: int, run_dir: Path,
                 compiler: Callable, on_event: Callable[[Event], None] | None,
                 context: ContextPolicy | None, effort: EffortPolicy | None,
                 max_time_s: float | None, lessons: bool, prompt_set: PromptSet,
                 fix_reserve_s: float = FIX_RESERVE_S, two_phase: bool = False,
                 saved: storage.Saved | None = None, index: Path | None = None):
        self.prompt, self.model, self.budget, self.max_turns = prompt, model, budget, max_turns
        self.context, self.effort, self.max_time_s, self.lessons = (context, effort, max_time_s,
                                                                    lessons)
        self.prompt_set = prompt_set
        self.fix_reserve_s = fix_reserve_s
        self.two_phase = two_phase
        self.phase = 1 if two_phase else None
        self.skeleton: dict[str, Any] | None = None  # the passing skeleton (two-phase)
        self.detail_compiles = 0  # compiles in phase 2
        self.time_limit: dict[str, Any] | None = None  # how the time limit ended the run
        self.effort_turns: Counter[str] = Counter()
        self.doc_reads: Counter[str] = Counter()
        self.quality: dict[str, Any] | None = None
        self.final_renders: list[str] = []
        self.run_dir = run_dir
        self.on_event = on_event or (lambda event: None)
        self.files = _Files(run_dir)
        self.ctx = ToolContext(run_dir, compiler=compiler, docs_dir=prompt_set.docs_dir)
        self.tools = schemas(synthesis=prompt_set.synthesis)
        self.tools_chars = len(json.dumps(tools_to_chat(self.tools), ensure_ascii=False))
        self.messages: list[Message] = []
        self.turn_of: list[int] = []  # the turn each message was added in
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
        # sessions (step 11): a resumed run continues the counters of the earlier ones
        self.index = index
        self.saved = saved
        self.first_turn = 0  # the last turn before this session
        self.prior_cost = 0.0
        self.prior_duration = 0.0
        self.first_started = self.started
        self.sessions: list[dict[str, Any]] = []
        self.resumed_from: str | None = None
        if saved is not None:
            self.restore(saved)

    def restore(self, saved: storage.Saved) -> None:
        """Continue from a stopped run: its messages, counters and earlier sessions. The
        latest compile is not restored: the model compiles again (cached if unchanged)."""
        record = saved.record
        self.messages, self.turn_of = list(saved.messages), list(saved.turn_of)
        # the record also counts a turn whose call was refused (budget): no message has it
        self.turn = self.first_turn = max([*self.turn_of, int(record.get("turns") or 0)])
        self.per_turn = list(record.get("per_turn") or [])
        tokens = record.get("tokens") or {}
        self.usage = Usage(**{k: tokens.get(k, 0) for k in Usage.__dataclass_fields__})
        self.cost = self.prior_cost = float(record.get("cost_usd") or 0.0)
        self.prior_duration = float(record.get("duration_s") or 0.0)
        self.unknown_cost_calls = int(record.get("unknown_cost_calls") or 0)
        compiles = record.get("compiles") or {}
        self.compiles = {k: int(compiles.get(k, 0)) for k in self.compiles}
        self.tool_counts = Counter(record.get("tool_calls") or {})
        self.doc_reads = Counter(record.get("doc_reads") or {})
        self.effort_turns = Counter(record.get("effort_turns") or {})
        two = record.get("two_phase")
        if self.two_phase and isinstance(two, dict):
            self.phase, self.skeleton = two.get("phase", 1), two.get("skeleton")
            self.detail_compiles = int(two.get("detail_compiles", 0))
        try:
            self.first_started = datetime.fromisoformat(record["started"]).timestamp()
        except (KeyError, TypeError, ValueError):
            pass
        status = storage.status_of(saved.run_dir, record)
        self.resumed_from = status
        sessions = [dict(s) for s in record.get("sessions") or []]
        if not sessions:  # a record from before step 11: one session
            sessions = [{"session": 1, "started": record.get("started"),
                         "turns": [1, record.get("turns")],
                         "cost_usd": record.get("cost_usd"),
                         "duration_s": record.get("duration_s")}]
        sessions[-1].update({"status": status, "ended": record.get("updated"),
                             "reason": (record.get("reason") or "").split("\n")[0][:300]})
        self.sessions = sessions
        if self.context is not None:
            policy = record.get("context_policy") or {}
            self.context.hits.update(policy.get("hits") or {})
            self.context.summaries.extend(policy.get("summaries") or [])
            self.replay_doc_reads()

    def replay_doc_reads(self) -> None:
        """Rule c after a resume: the docs pages read in full, with the turn of the read."""
        calls: dict[str, str] = {}
        for message, turn in zip(self.messages, self.turn_of, strict=True):
            if hasattr(message, "tool_calls"):
                calls.update({c.id: c.arguments.get("path") for c in message.tool_calls
                              if c.name == "read" and isinstance(c.arguments.get("path"), str)})
            elif isinstance(message, ToolResult) and message.call_id in calls \
                    and not message.is_error:
                try:
                    data = json.loads(message.content)
                except json.JSONDecodeError:
                    continue
                path = calls[message.call_id].replace("\\", "/").removeprefix("./")
                if isinstance(data, dict) and "text" in data and path.startswith("docs/"):
                    self.context.record_read(path, turn)

    # -- bookkeeping --

    def add(self, message: Message, **extra: Any) -> None:
        self.messages.append(message)
        self.turn_of.append(self.turn)
        self.files.log(self.turn, message, **extra)

    def context_tokens(self, messages: list[Message] | None = None) -> int:
        """What a request sends, about 4 characters per token: messages as the provider
        receives them (DeepSeek also gets the current tool loop's reasoning back) plus the
        tool schemas. Default: the full history."""
        settings = getattr(self.model, "settings", None)
        echo = getattr(settings, "reasoning_echo", "never")
        messages = self.messages if messages is None else messages
        chars = len(json.dumps(to_chat(messages, echo), ensure_ascii=False))
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
            "started": datetime.fromtimestamp(self.first_started, UTC).isoformat(
                timespec="seconds"),
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
            "duration_s": round(self.prior_duration + time.time() - self.started, 2),
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
            "peak_context_tokens_est": max((t.get("context_tokens_est", 0)
                                            for t in self.per_turn), default=0),
            "peak_input_tokens": max((t.get("input_tokens", 0) for t in self.per_turn),
                                     default=0),
            "context_policy": None if self.context is None else {
                "config": asdict(self.context.config), "hits": dict(self.context.hits),
                "summaries": self.context.summaries},
            "effort_turns": dict(self.effort_turns),
            "max_time_s": self.max_time_s,
            "fix_reserve_s": self.fix_reserve_s,
            "time_limit": self.time_limit,
            "two_phase": None if not self.two_phase else {
                "phase": self.phase, "skeleton": self.skeleton,
                "detail_compiles": self.detail_compiles},
            "lessons": self.lessons,
            "prompt_set": self.prompt_set.name,
            "options": self.options(),
            "sessions": self.sessions + [self.session_entry(status, reason)],
            **self.planning_metrics(),
            "doc_reads": dict(self.doc_reads),
            "reminders": dict(Counter(t["reminder"] for t in self.per_turn
                                      if t.get("reminder"))),
            "quality": self.quality,
            "final_renders": self.final_renders,
            "per_turn": self.per_turn,
        }
        self.files.save(record)
        return record

    def options(self) -> dict[str, Any]:
        """How the run was set up: ``resume`` sets the next session up the same way."""
        return {"role": getattr(self.model, "role", None),
                "context": "none" if self.context is None else "v1",
                "effort": None if self.effort is None else asdict(self.effort),
                "prompt_set": self.prompt_set.name, "two_phase": self.two_phase,
                "lessons": self.lessons, "fix_reserve_s": self.fix_reserve_s}

    def session_entry(self, status: str, reason: str) -> dict[str, Any]:
        now = time.time()
        return {"session": len(self.sessions) + 1,
                "started": datetime.fromtimestamp(self.started, UTC).isoformat(
                    timespec="seconds"),
                "ended": datetime.fromtimestamp(now, UTC).isoformat(timespec="seconds"),
                "status": status, "reason": reason.split("\n")[0][:300],
                "resumed_from": self.resumed_from,
                "turns": [self.first_turn + 1, self.turn],
                "cost_usd": round(self.cost - self.prior_cost, 6),
                "duration_s": round(now - self.started, 2),
                "allowance": {"max_time_s": self.max_time_s, "max_turns": self.max_turns,
                              "max_cost_usd": self.budget.max_cost_usd,
                              "max_calls": self.budget.max_calls}}

    def planning_metrics(self) -> dict[str, Any]:
        """Tokens spent up to and including the turn that first calls compile (all of the
        run's tokens if it never compiles)."""
        first = next((t["turn"] for t in self.per_turn if "compile" in t.get("tools", [])),
                     None)
        turns = [t for t in self.per_turn if first is None or t["turn"] <= first]
        return {"first_compile_turn": first,
                "reasoning_before_first_compile": sum(t.get("reasoning_tokens", 0)
                                                      for t in turns),
                "output_before_first_compile": sum(t.get("output_tokens", 0) for t in turns)}

    def finish_quality(self) -> None:
        """Quality metrics and the 4 final renders of the best build (never fails the run)."""
        from agent.quality import quality, render_final

        best = best_build(self.run_dir)
        if best is None:
            return
        try:
            self.quality = quality(self.run_dir / best["path"])
            self.final_renders = render_final(self.run_dir, best)
        except Exception as exc:  # metrics are for reports; keep the run's outcome
            self.quality = {**(self.quality or {}),
                            "error": f"{type(exc).__name__}: {exc}"}

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
                    batch = list(pool.map(self._run_one, calls[i:j]))
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
                outputs[i] = self._run_one(call)
            i += 1
        return [o for o in outputs if o is not None]

    def _run_one(self, call) -> ToolOutput:
        if call.name == "read" and self.context is not None:
            note = self.context.doc_note(call.arguments.get("path"))
            if note is not None:  # rule c: a doc that is already in the context
                return ToolOutput("read", note, json.dumps(note), [], False)
        if call.name not in {tool.name for tool in self.tools}:  # synthesize off: unknown
            return run_tool(self.ctx, f"{call.name} (not offered)", call.arguments)
        return run_tool(self.ctx, call.name, call.arguments)

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
        if self.phase == 2:
            self.detail_compiles += 1
        primary = next((s.kind for s in result.signals if s.severity == "failure"), None)
        return {"status": result.status, "cached": result.cached, "primary": primary,
                "version": result.version}

    def skeleton_done(self) -> str | None:
        """Two-phase: the phase 2 message, once phase 1 has a passing, fresh compile."""
        if self.phase != 1 or self.finish_blocker() is not None:
            return None
        last = self.ctx.last_compile
        version = f"v{last.version:04d}" if last.version is not None else "the latest build"
        self.phase = 2
        self.skeleton = {"version": version, "turn": self.turn,
                         "minutes": round((time.time() - self.started) / 60, 2)}
        return detail_phase(version)

    def routine(self) -> bool:
        """A routine turn: the latest compile passed (verifying, probing, finishing). Not the
        turns that write phase 2's detail before its first compile."""
        last = self.ctx.last_compile
        if last is None or not last.ok:
            return False
        return not (self.phase == 2 and self.detail_compiles == 0)

    def fix_cycle_starts(self) -> bool:
        """The previous turn ended with a failed compile: the next turn starts a fix."""
        if not self.per_turn:
            return False
        compiled = self.per_turn[-1].get("compile")
        return bool(compiled) and compiled["status"] != "success"

    def time_left(self) -> float | None:
        if self.max_time_s is None:
            return None
        return self.max_time_s - (time.time() - self.started)

    def output_rate(self) -> float:
        """Output tokens per second: the median over this run's long replies so far."""
        rates = sorted(t["output_tokens"] / t["latency_s"] for t in self.per_turn
                       if t.get("output_tokens", 0) >= 2_000 and t.get("latency_s"))
        return rates[len(rates) // 2] if rates else DEFAULT_TOKENS_PER_S

    def timed(self, options: dict[str, Any], metrics: dict[str, Any],
              allowance: float | None = None) -> dict[str, Any]:
        """Cap a call at the time left: its timeout, and an output cap the model can write
        in that time (so it ends with a reply rather than being cut off)."""
        left = self.time_left() if allowance is None else allowance
        if left is None:
            return options
        left = max(left, 1.0)
        options = {**options, "timeout_s": round(left, 1)}
        metrics["timeout_s"] = options["timeout_s"]
        default = getattr(self.model, "max_output_tokens", None)
        if isinstance(default, int):
            current = options.get("max_output_tokens", default)
            cap = max(MIN_OUTPUT_TOKENS,
                      int(self.output_rate() * max(0.0, left - CALL_OVERHEAD_S) * 0.9))
            if cap < current:
                options["max_output_tokens"] = cap
                metrics["output_cap"] = cap
        return options

    def time_check(self) -> tuple[str, str] | None:
        """The end of the run if the clock says so; None to go on."""
        if self.max_time_s is None:
            return None
        elapsed = time.time() - self.started
        if elapsed > self.max_time_s:
            if self.finish_blocker() is None:
                return self.final_turn(elapsed)
            self.time_limit = {"elapsed_s": round(elapsed, 1), "outcome": "stopped"}
            return "time_exceeded", (f"stopped after {elapsed:.0f} s (the limit is "
                                     f"{self.max_time_s:.0f} s)")
        left = self.max_time_s - elapsed
        if self.fix_reserve_s and left < self.fix_reserve_s and self.fix_cycle_starts():
            self.time_limit = {"elapsed_s": round(elapsed, 1), "outcome": "no_fix_cycle"}
            return "time_exceeded", (f"stopped after {elapsed:.0f} s: {left:.0f} s left, less "
                                     f"than a fix cycle needs ({self.fix_reserve_s:.0f} s; the "
                                     f"limit is {self.max_time_s:.0f} s)")
        return None

    def final_turn(self, elapsed: float) -> tuple[str, str]:
        """Time is up with a passing, fresh compile: one last call for the summary. Tool calls
        in it are not run; the passing build is the result either way."""
        self.turn += 1
        self.add(User(TIME_UP_FINAL), reminder="time_up")
        sent = self.messages
        if self.context is not None:
            sent = self.context.view(self.messages, self.turn_of, self.turn,
                                     extra_chars=self.tools_chars)
        metrics: dict[str, Any] = {"turn": self.turn, "reminder": "time_up",
                                   "context_tokens_est": self.context_tokens(sent),
                                   "history_tokens_est": self.context_tokens(),
                                   "sent_messages": len(sent)}
        self.per_turn.append(metrics)
        options: dict[str, Any] = {}
        if self.effort is not None:
            options = {"effort": self.effort.routine_effort,
                       "max_output_tokens": self.effort.routine_max_output_tokens}
            self.effort_turns["routine"] += 1
            metrics["effort"] = options["effort"]
        options = self.timed(options, metrics, allowance=FINAL_TURN_S)
        text = ""
        try:
            reply = self.model.query(sent, self.tools, **options)
        except CallTimeLimit:  # the passing build is the result anyway
            metrics["timed_out"] = True
        else:
            metrics.update(self.account(reply))
            self.add(reply.message(), finish_reason=reply.finish_reason, not_run=bool(
                reply.tool_calls))
            metrics["tools_not_run"] = [c.name for c in reply.tool_calls]
            text = reply.text.strip()
        if text:
            self.final_text = text
        metrics["finished"] = True
        self.time_limit = {"elapsed_s": round(elapsed, 1), "outcome": "final_turn",
                           "final_text": bool(text)}
        self.record("running")
        return "success", (f"time limit reached ({elapsed:.0f} s of {self.max_time_s:.0f} s) "
                           "with a passing, fresh compile; "
                           + ("final message given" if text else "no final message"))

    # -- one run --

    def run(self) -> RunResult:
        storage.acquire_lock(self.run_dir)
        try:
            return self._run()
        finally:
            storage.release_lock(self.run_dir)

    def start(self) -> None:
        """The first messages: the prompts and task, or (resumed) the resume note."""
        if self.saved is not None:
            if self.saved.repaired:  # results for the calls the stop left open
                storage.append_conversation(self.run_dir, self.saved.repaired)
            previous = self.sessions[-1]
            allowance = self.session_entry("running", "")["allowance"]
            self.add(User(resumed_message(previous["status"], previous.get("reason", ""),
                                          allowance)), reminder="resumed")
            return
        system = self.prompt_set.system_text()
        if self.lessons:
            system += lessons_file.prompt_section()
        self.add(System(system))
        self.add(System(self.prompt_set.quickstart.read_text(encoding="utf-8")))
        task = SKELETON_TASK if self.two_phase else self.prompt_set.task
        self.add(User(task.format(prompt=self.prompt)))

    def _run(self) -> RunResult:
        self.start()
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
        if self.lessons:
            lessons_file.learn(self.run_dir)
        self.finish_quality()
        record = self.record(status, reason)
        if self.index is not None:
            try:
                storage.append_index(record, self.index)
            except OSError as exc:  # the index is for reports; keep the run's outcome
                record["index_error"] = str(exc)
        self.on_event({"kind": "end", "status": status, "reason": reason,
                       "run_dir": self.run_dir, "record": record})
        return RunResult(status, reason, self.run_dir, self.turn, self.final_text,
                         record["best"], record)

    def loop(self) -> tuple[str, str]:
        while self.turn - self.first_turn < self.max_turns:
            stop = self.time_check()
            if stop is not None:
                return stop
            self.turn += 1
            sent = self.messages
            if self.context is not None:
                sent = self.context.view(self.messages, self.turn_of, self.turn,
                                         extra_chars=self.tools_chars)
            metrics: dict[str, Any] = {"turn": self.turn,
                                       "context_tokens_est": self.context_tokens(sent),
                                       "history_tokens_est": self.context_tokens(),
                                       "sent_messages": len(sent)}
            self.per_turn.append(metrics)
            options: dict[str, Any] = {}
            last = self.ctx.last_compile
            if self.effort is not None:
                if self.routine():
                    options = {"effort": self.effort.routine_effort,
                               "max_output_tokens": self.effort.routine_max_output_tokens}
                    self.effort_turns["routine"] += 1
                elif self.compiles["calls"] == 0 and self.effort.planning_effort is not None:
                    options = {"effort": self.effort.planning_effort}
                    self.effort_turns["planning"] += 1
                else:
                    self.effort_turns["default"] += 1
                metrics["effort"] = options.get("effort", "default")
            options = self.timed(options, metrics)
            try:
                reply = self.model.query(sent, self.tools, **options)
            except CallTimeLimit:
                metrics["timed_out"] = True
                elapsed = time.time() - self.started
                self.record("running")
                if self.finish_blocker() is None:
                    return self.final_turn(elapsed)
                self.time_limit = {"elapsed_s": round(elapsed, 1), "outcome": "call_cut"}
                return "time_exceeded", (f"the model call was cut off at the time limit after "
                                         f"{elapsed:.0f} s (the limit is "
                                         f"{self.max_time_s:.0f} s)")
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
                checkpoint = self.context is not None and self.context.config.checkpoints
                compiled = False
                for call, out in zip(reply.tool_calls, outputs, strict=True):
                    self.tool_counts[call.name] += 1
                    path = call.arguments.get("path") if call.name == "read" else None
                    if isinstance(path, str):
                        path = path.replace("\\", "/").removeprefix("./")
                        if path.startswith("docs/"):
                            self.doc_reads[path] += 1
                            if self.context is not None and not out.is_error \
                                    and "text" in out.data:
                                self.context.record_read(path, self.turn)  # rule c
                    if checkpoint and call.name == "compile" and not out.is_error:
                        compiled = True  # rule a: the signals go in the checkpoint message
                        data = {**out.data, "compile_signals": "[in the <checkpoint> message "
                                                               "after these tool results]"}
                        out = ToolOutput(out.name, data, json.dumps(data), out.images, False)
                    self.add(out.message(call.id))
                    event["tools"].append({"name": call.name, "args": call.arguments,
                                           "error": out.data.get("error_kind")
                                           if out.is_error else None})
                    if call.name == "compile" and not out.is_error:
                        compile_info = self.after_compile()
                    images += [str(p) for p in out.images]
                if images and getattr(self.model, "supports_images", False):
                    self.add(User("Rendered views (front, side, top, iso).", tuple(images)))
                if compiled:
                    self.add(User(checkpoint_text(self.turn, self.ctx.last_compile)),
                             checkpoint=True)
                    self.context.checkpoint()
                if compile_info is not None:
                    phase_message = self.skeleton_done()
                    if phase_message is not None:
                        self.add(User(phase_message), reminder="phase_2")
                        metrics["reminder"] = "phase_2"
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
            if not text and reply.finish_reason == "length":
                reminder, kind = REPLY_CUT_OFF, "reply_cut_off"
            self.add(User(reminder), reminder=kind)
            metrics["reminder"] = kind
            event["reminder"] = kind
            self.record("running")
            self.on_event(event)
        return "max_turns", f"stopped after {self.max_turns} turns"


def run(prompt: str, model: Model, budget: Budget, max_turns: int = MAX_TURNS, *,
        run_dir: str | Path | None = None, runs_root: str | Path = RUNS_ROOT,
        compiler: Callable = compile_run,
        on_event: Callable[[Event], None] | None = None,
        context: ContextPolicy | None = None, effort: EffortPolicy | None = None,
        max_time_s: float | None = MAX_TIME_S, lessons: bool = False,
        prompt_set: str | PromptSet = DEFAULT_PROMPT_SET,
        fix_reserve_s: float = FIX_RESERVE_S, two_phase: bool = False,
        index: str | Path | None = None) -> RunResult:
    """Run the agent on ``prompt`` until it finishes with a checked build, or stops.

    ``context``: the context policy (agent/context.py), or None to send the full history.
    ``effort``: less reasoning on routine (and optionally planning) turns, or None for the
    model's settings. ``max_time_s``: wall-clock budget (end state ``time_exceeded``).
    ``lessons``: use and update runs/lessons.md (off by default). ``prompt_set``: the system
    prompt, quickstart and docs (``PROMPT_SETS``; "v0" is step 9's). ``fix_reserve_s``: no fix
    cycle starts with less time left. ``two_phase``: skeleton first, then detail (an optional
    high-detail mode; off by default). ``index``: the run index the finished run is added to
    (default: ``runs/index.jsonl`` for a run under ``runs_root``, else next to the run)."""
    if isinstance(prompt_set, str):
        prompt_set = PROMPT_SETS[prompt_set]
    run_dir = Path(run_dir).resolve() if run_dir is not None else new_run_dir(prompt, runs_root)
    for sub in ("workspace", "result", "renders"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    main = run_dir / "workspace" / "main.py"
    if not main.exists():
        atomic_copy(TEMPLATE, main)
    _wire(model, budget, run_dir)
    index = Path(index) if index is not None else storage.default_index(run_dir, runs_root)
    return _Run(prompt, model, budget, max_turns, run_dir, compiler, on_event, context, effort,
                max_time_s, lessons, prompt_set, fix_reserve_s, two_phase, index=index).run()


def _wire(model: Model, budget: Budget, run_dir: Path) -> None:
    """One budget and one call log for the run."""
    if getattr(model, "budget", None) is None and hasattr(model, "budget"):
        model.budget = budget
    if getattr(model, "log", None) is None and hasattr(model, "log"):
        model.log = CallLog(run_dir)
    if getattr(model, "budget", budget) is not budget:
        raise ValueError("the model has a different Budget than the one given to run()")


def saved_options(record: dict[str, Any]) -> dict[str, Any]:
    """How a stopped run was set up (``record["options"]``; inferred for older records)."""
    options = dict(record.get("options") or {})
    options["role"] = (options.get("role") or (record.get("model") or {}).get("role")
                       or "builder")
    options.setdefault("context", "v1" if record.get("context_policy") else "none")
    options.setdefault("effort", asdict(EffortPolicy()))
    options.setdefault("prompt_set", record.get("prompt_set") or DEFAULT_PROMPT_SET)
    options.setdefault("two_phase", bool(record.get("two_phase")))
    options.setdefault("lessons", bool(record.get("lessons")))
    options.setdefault("fix_reserve_s", record.get("fix_reserve_s", FIX_RESERVE_S))
    return options


def resume(run_dir: str | Path, model: Model, budget: Budget, max_turns: int = MAX_TURNS, *,
           compiler: Callable = compile_run, on_event: Callable[[Event], None] | None = None,
           context: ContextPolicy | None = None, effort: EffortPolicy | None = None,
           max_time_s: float | None = MAX_TIME_S, runs_root: str | Path = RUNS_ROOT,
           index: str | Path | None = None) -> RunResult:
    """Continue a stopped run (``time_exceeded``, ``budget_exceeded``, a crash, ...) from its
    conversation.jsonl and workspace. ``budget``, ``max_turns`` and ``max_time_s`` are this
    session's new allowance; the prompt set, two-phase, lessons and fix reserve come from the
    record (``saved_options``); ``context`` and ``effort`` are the caller's (the CLI builds
    them from the saved options). Raises ``storage.ResumeError`` if the run succeeded or is
    still running."""
    saved = storage.load(run_dir)
    storage.check_resumable(saved)
    storage.clean_after_crash(saved.run_dir)
    options = saved_options(saved.record)
    _wire(model, budget, saved.run_dir)
    index = Path(index) if index is not None else storage.default_index(saved.run_dir, runs_root)
    return _Run(saved.record["prompt"], model, budget, max_turns, saved.run_dir, compiler,
                on_event, context, effort, max_time_s, bool(options["lessons"]),
                PROMPT_SETS[options["prompt_set"]], float(options["fix_reserve_s"]),
                bool(options["two_phase"]), saved=saved, index=index).run()


__all__ = ["DEFAULT_PROMPT_SET", "FINAL_TURN_S", "FIX_RESERVE_S", "MAX_TIME_S", "MAX_TURNS",
           "PROMPT_SETS", "EffortPolicy", "PromptSet", "RunResult", "best_build", "new_run_dir",
           "resume", "run", "saved_options"]
