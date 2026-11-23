"""What the model sees: a shorter view of the conversation, rebuilt only at checkpoints.

The full history is kept (and saved to conversation.jsonl); ``ContextPolicy.view`` turns it
into the messages actually sent. Rules:

a. **Checkpoints and reasoning.** A compile's signals are sent as a user message
   (``<checkpoint ...>``) after the tool results. That starts a new DeepSeek turn, so the
   reasoning of the finished tool loop is no longer sent back (``to_chat`` echoes only the
   reasoning after the last user message).
b. **Pinned state.** Per file, only the latest snapshot (a ``read`` result or a ``write``'s
   content) is kept in full; older ones become a one-line stub. Only the latest compile
   signals are kept in full.
c. **Docs already in context.** Reading ``docs/sdk/quickstart.md`` returns a short note: it is
   the second system message. (Applied by the loop when the tool runs.)
d. **Old tool outputs.** Results older than ``old_after_turns`` turns are cut to their first
   lines plus a stub. ``probe`` and ``render`` results are kept: they are short and numeric.
e. **Compaction.** Above ``compact_above_tokens``, the middle of the conversation is
   summarized by the cheap model into a checkpoint (tried, failed, works, current main.py).
   The system prompts, the task and the last ``keep_recent_turns`` turns are kept.

**Cache-friendliness.** The prefix (two system messages + the task) is never touched.
Rules b, d and e are applied only when the view is rebuilt, at a checkpoint; between
checkpoints the view only grows at the end, so the provider's prompt cache covers it.
No rule removes a tool call without its result or a result without its call.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from llm import Assistant, LLMError, Message, System, ToolCall, ToolResult, User
from llm.messages import to_chat

CHECKPOINT = "<checkpoint"
SUMMARY = "<checkpoint kind=summary"
DOCS_IN_CONTEXT = {"docs/sdk/quickstart.md": "the second system message"}
KEEP_FULL_TOOLS = {"probe", "render"}


@dataclass(frozen=True)
class ContextConfig:
    checkpoints: bool = True  # a
    pin_state: bool = True  # b
    doc_notes: bool = True  # c
    shorten_old_outputs: bool = True  # d
    old_after_turns: int = 6
    compaction: bool = True  # e
    compact_above_tokens: int = 40_000
    keep_recent_turns: int = 4


def estimate_tokens(messages: list[Message], echo: str = "current_turn",
                    extra_chars: int = 0) -> int:
    return round((len(json.dumps(to_chat(messages, echo), ensure_ascii=False))
                  + extra_chars) / 4)


def checkpoint_text(turn: int, result: Any) -> str:
    """The user message that carries a compile's signals (rule a)."""
    primary = next((s.kind for s in result.signals if s.severity == "failure"), None)
    version = f"v{result.version:04d}" if result.version is not None else "none"
    return (f"<checkpoint turn={turn} compile={result.status} primary={primary or '-'} "
            f"version={version}>\n{result.signals_text}\n</checkpoint>")


def _stub_checkpoint(message: User, superseded_at: int) -> User:
    head = message.content.splitlines()[0]
    return User(f"{head}\n[older compile signals, superseded at turn {superseded_at}]\n"
                "</checkpoint>")


def _loads(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


class ContextPolicy:
    """Builds the sent view from the full history. One instance per run."""

    def __init__(self, config: ContextConfig = ContextConfig(), summarizer: Any = None,
                 workspace: Path | None = None):
        self.config, self.summarizer, self.workspace = config, summarizer, workspace
        self.hits: Counter[str] = Counter()
        self._seen: set[tuple[str, Any]] = set()  # (rule, item key): count each item once
        self._frozen: list[Message] | None = None
        self._frozen_upto = 0  # history index the frozen view covers
        self._dirty = True
        self._summary: User | None = None
        self._summary_upto = 0
        self.summaries: list[dict[str, Any]] = []

    # -- events from the loop --

    def checkpoint(self) -> None:
        """A checkpoint message was added: rebuild the view before the next query."""
        self._dirty = True
        self.hits["a_checkpoints"] += 1

    def doc_note(self, path: object) -> dict[str, Any] | None:
        """Rule c: the tool output to return instead of a doc already in context."""
        if not self.config.doc_notes or not isinstance(path, str):
            return None
        key = path.strip().replace("\\", "/").lstrip("./")
        where = DOCS_IN_CONTEXT.get(key)
        if where is None:
            return None
        self.hits["c_doc_notes"] += 1
        return {"path": key, "note": f"{key} is already in your context ({where}); use it "
                                     "from there instead of reading it again."}

    def _hit(self, rule: str, key: Any) -> None:
        if (rule, key) not in self._seen:
            self._seen.add((rule, key))
            self.hits[rule] += 1

    # -- the view --

    def view(self, history: list[Message], turns: list[int], current_turn: int,
             extra_chars: int = 0) -> list[Message]:
        prefix_end = next(i for i, m in enumerate(history) if isinstance(m, User)) + 1
        if self._dirty or self._frozen is None:
            self._frozen = self._rebuild(history, turns, current_turn, prefix_end)
            self._frozen_upto = len(history)
            self._dirty = False
        else:
            self._frozen = self._frozen + history[self._frozen_upto:]
            self._frozen_upto = len(history)
        if (self.config.compaction and estimate_tokens(self._frozen, extra_chars=extra_chars)
                > self.config.compact_above_tokens):
            if self._compact(history, turns, current_turn, prefix_end):
                self._frozen = self._rebuild(history, turns, current_turn, prefix_end)
        self._count_dropped_reasoning(self._frozen)
        return list(self._frozen)

    def _count_dropped_reasoning(self, view: list[Message]) -> None:
        """Rule a's effect: reasoning before the last user message is not sent back."""
        last_user = max(i for i, m in enumerate(view) if isinstance(m, User))
        first_user = next(i for i, m in enumerate(view) if isinstance(m, User))
        for message in view[first_user + 1:last_user]:
            if isinstance(message, Assistant) and message.reasoning:
                self._hit("a_reasoning_dropped", hash((message.content, message.reasoning)))

    def _rebuild(self, history, turns, current_turn, prefix_end) -> list[Message]:
        start = max(prefix_end, self._summary_upto)
        body = list(enumerate(history[start:], start))
        out: list[Message] = list(history[:prefix_end])
        if self._summary is not None:
            out.append(self._summary)
        calls: dict[str, ToolCall] = {}
        for _i, message in body:
            if isinstance(message, Assistant):
                calls.update({c.id: c for c in message.tool_calls})

        # Rule b: the latest snapshot of each file, and the latest compile signals.
        latest_snapshot: dict[str, int] = {}
        latest_compile = -1
        for i, message in body:
            if isinstance(message, Assistant):
                for call in message.tool_calls:
                    if call.name == "write" and isinstance(call.arguments.get("path"), str):
                        latest_snapshot[_norm(call.arguments["path"])] = i
            elif isinstance(message, ToolResult):
                call = calls.get(message.call_id)
                if call and call.name == "read" and not message.is_error:
                    data = _loads(message.content)
                    if data and "text" in data:
                        latest_snapshot[_norm(data["path"])] = i
                if call and call.name == "compile" and not message.is_error:
                    latest_compile = max(latest_compile, i)
            elif isinstance(message, User) and message.content.startswith(CHECKPOINT) \
                    and not message.content.startswith(SUMMARY):
                latest_compile = max(latest_compile, i)

        for i, message in body:
            out.append(self._shorten(i, message, turns, current_turn, calls, latest_snapshot,
                                     latest_compile))
        return out

    def _shorten(self, i, message, turns, current_turn, calls, latest_snapshot,
                 latest_compile) -> Message:
        config = self.config
        if isinstance(message, Assistant) and config.pin_state:
            changed = []
            for call in message.tool_calls:
                path = call.arguments.get("path")
                if (call.name == "write" and isinstance(path, str)
                        and latest_snapshot.get(_norm(path), i) > i):
                    later = turns[latest_snapshot[_norm(path)]]
                    stub = {"path": path, "content": f"[older content of {path}, superseded "
                                                     f"at turn {later}]"}
                    changed.append(ToolCall(call.id, call.name, stub, json.dumps(stub)))
                    self._hit("b_pinned_stubs", ("write", call.id))
                else:
                    changed.append(call)
            if changed != list(message.tool_calls):
                return replace(message, tool_calls=tuple(changed))
            return message
        if isinstance(message, User) and message.content.startswith(CHECKPOINT) \
                and not message.content.startswith(SUMMARY):
            if config.pin_state and i < latest_compile:
                self._hit("b_pinned_stubs", ("checkpoint", i))
                return _stub_checkpoint(message, turns[latest_compile])
            return message
        if not isinstance(message, ToolResult) or message.is_error:
            return message
        call = calls.get(message.call_id)
        name = call.name if call else message.name
        data = _loads(message.content)
        if data is None:
            return message
        if config.pin_state and name == "read" and "text" in data:
            path = _norm(data.get("path", ""))
            if latest_snapshot.get(path, i) > i:
                self._hit("b_pinned_stubs", ("read", message.call_id))
                later = turns[latest_snapshot[path]]
                return replace(message, content=json.dumps(
                    {"path": data.get("path"), "note": f"[older read of {data.get('path')}, "
                                                       f"superseded at turn {later}]"}))
            if not path.startswith("docs/"):
                return message  # the latest snapshot of a workspace file stays whole
        if config.pin_state and name == "compile" and i < latest_compile \
                and len(data.get("compile_signals", "")) > 80:
            self._hit("b_pinned_stubs", ("compile", message.call_id))
            data["compile_signals"] = (f"[older compile signals, superseded at turn "
                                       f"{turns[latest_compile]}]")
            return replace(message, content=json.dumps(data, ensure_ascii=False))
        if (config.shorten_old_outputs and name not in KEEP_FULL_TOOLS
                and turns[i] <= current_turn - config.old_after_turns
                and len(message.content) > 400):
            self._hit("d_shortened", message.call_id)
            short: dict[str, Any] = {}
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 200:
                    lines = value.splitlines()
                    value = "\n".join(lines[:5])[:400] + (
                        f"\n[... shortened: output from turn {turns[i]}; call the tool again "
                        "for all of it]")
                short[key] = value
            return replace(message, content=json.dumps(short, ensure_ascii=False))
        return message

    # -- rule e --

    def _compact(self, history, turns, current_turn, prefix_end) -> bool:
        if self.summarizer is None:
            self.hits["e_compaction_skipped"] += 1
            return False
        keep_from_turn = current_turn - self.config.keep_recent_turns
        start = max(prefix_end, self._summary_upto)
        cut = None
        for i in range(start, len(history)):
            if turns[i] > keep_from_turn and isinstance(history[i], Assistant):
                cut = i
                break
        if cut is None or cut <= start:
            self.hits["e_compaction_skipped"] += 1
            return False
        middle = history[start:cut]
        transcript = _transcript(([self._summary] if self._summary else []) + middle)
        prompt = [System(SUMMARIZER_PROMPT),
                  User(f"Conversation turns {turns[start]}-{turns[cut - 1]}:\n\n{transcript}")]
        try:
            reply = self.summarizer.query(prompt, [])
        except LLMError as exc:
            self.hits["e_compaction_failed"] += 1
            self.summaries.append({"turn": current_turn, "error": str(exc)})
            return False
        main = ""
        if self.workspace is not None and (self.workspace / "main.py").is_file():
            main = (self.workspace / "main.py").read_text(encoding="utf-8")
        self._summary = User(
            f"<checkpoint kind=summary turns={turns[start]}-{turns[cut - 1]}>\n"
            f"{reply.text.strip()}\n\nCurrent main.py:\n```python\n{main}\n```\n</checkpoint>")
        self._summary_upto = cut
        self.hits["e_compactions"] += 1
        self.summaries.append({"turn": current_turn, "turns": [turns[start], turns[cut - 1]],
                               "cost": reply.cost, "chars": len(self._summary.content)})
        return True


SUMMARIZER_PROMPT = (
    "You summarize part of an agent's work on a 3D articulated-object model (Python main.py "
    "with an SDK; a compiler checks it). Write at most 250 words, in four short sections: "
    "Tried (design decisions and changes), Failed (compile failures and why, with the exact "
    "error kinds), Works (what passed or was verified), Next (what was about to be done). "
    "Keep numbers, names of bodies and joints, and exact error messages. No preamble.")


def _norm(path: str) -> str:
    return str(path).strip().replace("\\", "/").lstrip("./")


def _transcript(messages: list[Message], limit: int = 1_500) -> str:
    lines = []
    for message in messages:
        if isinstance(message, Assistant):
            calls = ", ".join(f"{c.name}({c.arguments_json()[:200]})" for c in message.tool_calls)
            lines.append(f"ASSISTANT: {message.content[:limit]}" + (f"\nCALLS: {calls}"
                                                                    if calls else ""))
        elif isinstance(message, ToolResult):
            lines.append(f"TOOL {message.name}: {message.content[:limit]}")
        elif isinstance(message, User):
            lines.append(f"USER: {message.content[:limit]}")
    return "\n\n".join(lines)
