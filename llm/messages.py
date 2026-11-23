"""The neutral message format. This is the one place it is defined.

A conversation is a list of messages, in order:

| Message | Fields | Meaning |
| --- | --- | --- |
| ``System(content)`` | text | Instructions. Only at the start, so the prefix stays stable and cacheable |
| ``User(content, images)`` | text, optional PNG paths | A user turn (the task, a reminder, ...). ``images`` become image parts (data URLs); the agent loop adds them only for models that accept images |
| ``Assistant(content, tool_calls, reasoning)`` | text, ``[ToolCall]``, optional text | A model turn. ``reasoning`` is the provider's visible reasoning (DeepSeek's ``reasoning_content``), kept so it can be sent back when the provider needs it |
| ``ToolResult(call_id, name, content, is_error)`` | the id of the call it answers, the tool name, text | The result of one tool call. ``is_error`` is for our own bookkeeping: Chat Completions has no such field, so ``content`` must say what failed |

``ToolCall(id, name, arguments, raw_arguments, error)``: ``arguments`` is the parsed JSON
object. ``raw_arguments`` is the exact string the model sent; it is sent back unchanged, so
the conversation prefix stays byte-identical for prompt caching. If the string is not a JSON
object, ``arguments`` is ``{}`` and ``error`` says why.

``Tool(name, description, parameters)``: one function the model may call; ``parameters``
is a JSON Schema object.

Rules (``validate``):
- system messages come first and only first;
- after an assistant message with tool calls, every call is answered by a ``ToolResult``
  before the next user or assistant message;
- a ``ToolResult`` answers a call that is still open.

Conversion to and from the Chat Completions format (OpenAI and DeepSeek):
``to_chat`` / ``from_chat`` for messages, ``tools_to_chat`` / ``tools_from_chat`` for tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

ReasoningEcho = Literal["never", "current_turn"]


@dataclass(frozen=True)
class System:
    content: str


@dataclass(frozen=True)
class User:
    content: str
    images: tuple[str, ...] = ()  # PNG paths; sent only to models that accept images

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", tuple(str(i) for i in self.images))


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    error: str | None = None

    def arguments_json(self) -> str:
        if self.raw_arguments is not None:
            return self.raw_arguments
        return json.dumps(self.arguments, ensure_ascii=False)

    @classmethod
    def parse(cls, id: str, name: str, raw: str | None) -> ToolCall:
        """A call as the model sent it, with its arguments parsed (or the parse error)."""
        raw = raw if raw is not None else ""
        try:
            arguments = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            return cls(id, name, {}, raw, f"arguments are not valid JSON: {exc}")
        if not isinstance(arguments, dict):
            return cls(id, name, {}, raw, "arguments must be a JSON object")
        return cls(id, name, arguments, raw)


@dataclass(frozen=True)
class Assistant:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}})


Message = System | User | Assistant | ToolResult


class MessageError(ValueError):
    """A conversation that breaks the rules above."""


def validate(messages: list[Message], tools: tuple[Tool, ...] | list[Tool] = ()) -> None:
    if not messages:
        raise MessageError("a conversation needs at least one message")
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise MessageError(f"tool names must be unique: {names}")
    seen_other = False
    open_calls: dict[str, str] = {}
    for i, message in enumerate(messages):
        if not isinstance(message, (System, User, Assistant, ToolResult)):
            raise MessageError(f"message {i} is not a neutral message: {type(message).__name__}")
        if isinstance(message, System):
            if seen_other:
                raise MessageError(f"message {i}: system messages must all come first "
                                   "(a stable prefix is what prompt caching reuses)")
            continue
        seen_other = True
        if isinstance(message, ToolResult):
            if message.call_id not in open_calls:
                raise MessageError(f"message {i}: tool result for {message.call_id!r}, "
                                   "which is not an open tool call")
            del open_calls[message.call_id]
            continue
        if open_calls:
            raise MessageError(f"message {i}: tool calls {sorted(open_calls)} were not answered")
        if isinstance(message, Assistant):
            for call in message.tool_calls:
                if call.id in open_calls:
                    raise MessageError(f"message {i}: duplicate tool call id {call.id!r}")
                open_calls[call.id] = call.name
    if not any(not isinstance(m, System) for m in messages):
        raise MessageError("a conversation needs a user message after the system prompt")


# ---- Chat Completions format ------------------------------------------------------------------


def to_chat(messages: list[Message], reasoning: ReasoningEcho = "never") -> list[dict[str, Any]]:
    """Neutral messages -> Chat Completions ``messages``.

    ``reasoning="current_turn"`` sends ``reasoning_content`` back on the assistant messages
    after the last user message (the tool loop in progress), as DeepSeek's thinking mode
    expects; older reasoning is dropped. ``"never"`` sends none (OpenAI)."""
    last_user = max((i for i, m in enumerate(messages) if isinstance(m, User)), default=-1)
    chat: list[dict[str, Any]] = []
    for i, message in enumerate(messages):
        if isinstance(message, System):
            chat.append({"role": "system", "content": message.content})
        elif isinstance(message, User):
            if message.images:
                chat.append({"role": "user", "content": [
                    {"type": "text", "text": message.content},
                    *({"type": "image_url", "image_url": {"url": _data_url(path)}}
                      for path in message.images)]})
            else:
                chat.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolResult):
            chat.append({"role": "tool", "tool_call_id": message.call_id,
                         "content": message.content})
        else:
            item: dict[str, Any] = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                item["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": c.arguments_json()}}
                    for c in message.tool_calls]
            if reasoning == "current_turn" and i > last_user and message.reasoning:
                item["reasoning_content"] = message.reasoning
            chat.append(item)
    return chat


def _data_url(path: str) -> str:
    import base64
    from pathlib import Path

    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def assistant_from_chat(message: dict[str, Any]) -> Assistant:
    calls = tuple(
        ToolCall.parse(c["id"], c["function"]["name"], c["function"].get("arguments"))
        for c in message.get("tool_calls") or [])
    return Assistant(message.get("content") or "", calls, message.get("reasoning_content"))


def from_chat(chat: list[dict[str, Any]]) -> list[Message]:
    """Chat Completions ``messages`` -> neutral messages (the inverse of ``to_chat``)."""
    names: dict[str, str] = {}
    messages: list[Message] = []
    for item in chat:
        role = item["role"]
        if role in ("system", "developer"):
            messages.append(System(item["content"]))
        elif role == "user":
            content = item["content"]
            if isinstance(content, list):  # image parts are not turned back into paths
                content = "".join(part.get("text", "") for part in content
                                  if part.get("type") == "text")
            messages.append(User(content))
        elif role == "assistant":
            assistant = assistant_from_chat(item)
            names.update({c.id: c.name for c in assistant.tool_calls})
            messages.append(assistant)
        elif role == "tool":
            call_id = item["tool_call_id"]
            messages.append(ToolResult(call_id, names.get(call_id, ""), item["content"]))
        else:
            raise MessageError(f"unknown chat role {role!r}")
    return messages


def tools_to_chat(tools: tuple[Tool, ...] | list[Tool]) -> list[dict[str, Any]]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": t.parameters}} for t in tools]


def tools_from_chat(chat_tools: list[dict[str, Any]]) -> list[Tool]:
    return [Tool(t["function"]["name"], t["function"].get("description", ""),
                 t["function"].get("parameters", {"type": "object", "properties": {}}))
            for t in chat_tools]
