"""What every model does around the provider call: validate, budget, cost, log.

A provider implements ``_complete(messages, tools) -> Completion``; ``query`` does the rest.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from llm import accounting
from llm.accounting import Budget, CallLog
from llm.messages import Message, System, Tool, ToolCall, tools_to_chat, validate
from llm.protocol import Cost, ProviderError, Reply, Usage


@dataclass
class Completion:
    """What a provider returns, before cost and accounting."""
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    raw: dict[str, Any]
    model: str
    finish_reason: str | None = None
    reasoning: str | None = None
    retries: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def prefix_digest(messages: list[Message], tools: tuple[Tool, ...] | list[Tool]) -> str:
    """A short hash of the cacheable prefix (system messages + tool schemas). If it changes
    between calls of one conversation, the provider's prompt cache cannot be reused."""
    system = [m.content for m in messages if isinstance(m, System)]
    blob = json.dumps([system, tools_to_chat(tools)], sort_keys=False, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


_KEY_LIKE = re.compile(r"\b(sk|ds)-[A-Za-z0-9_\-*.]{6,}")


def redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Remove keys from text that goes into logs (provider errors can quote a key)."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return _KEY_LIKE.sub("[redacted]", text)


class ChatModelBase:
    provider = "base"
    supports_images = False

    def __init__(self, model: str, *, budget: Budget | None = None, log: CallLog | None = None,
                 role: str | None = None):
        self.model, self.budget, self.log, self.role = model, budget, log, role
        self._secrets: tuple[str, ...] = ()

    def _complete(self, messages: list[Message], tools: tuple[Tool, ...],
                  options: dict[str, Any]) -> Completion:
        raise NotImplementedError

    def _cost(self, usage: Usage, at: datetime) -> Cost:
        return accounting.cost(self.model, usage, at)

    def query(self, messages: list[Message], tools: tuple[Tool, ...] | list[Tool] = (), *,
              effort: str | None = None, max_output_tokens: int | None = None) -> Reply:
        """One call. ``effort`` / ``max_output_tokens`` override the model's settings for
        this call only (the loop lowers reasoning on routine turns)."""
        tools = tuple(tools)
        options = {k: v for k, v in (("effort", effort),
                                     ("max_output_tokens", max_output_tokens)) if v is not None}
        validate(messages, tools)
        if self.budget is not None:
            self.budget.before_call()
        at = datetime.now(UTC)
        started = time.perf_counter()
        base = {"time": at.isoformat(timespec="milliseconds"), "role": self.role,
                "provider": self.provider, "model": self.model,
                "prefix": prefix_digest(messages, tools), "messages": len(messages),
                "tools": len(tools), **({"options": options} if options else {})}
        try:
            done = self._complete(messages, tools, options)
        except ProviderError as exc:
            self._log({**base, "status": "error", "latency_s": _since(started),
                       "retries": exc.attempts - 1, "http_status": exc.status,
                       "error": redact(str(exc), self._secrets), "cost": 0.0})
            raise
        reply = Reply(done.text, done.tool_calls, done.usage, self._cost(done.usage, at),
                      done.raw, self.provider, done.model, done.finish_reason, done.reasoning,
                      _since(started), done.retries, done.extra)
        try:
            if self.budget is not None:
                self.budget.record(reply)
        finally:
            self._log({**base, "status": "ok", "served_model": reply.model,
                       "usage": reply.usage.to_dict(), "cost": reply.cost,
                       "latency_s": reply.latency_s, "retries": reply.retries,
                       "finish_reason": reply.finish_reason,
                       "tool_calls": [c.name for c in reply.tool_calls],
                       "budget": self.budget.to_dict() if self.budget else None})
        return reply

    def _log(self, record: dict[str, Any]) -> None:
        if self.log is not None:
            self.log.append(record)


def _since(started: float) -> float:
    return round(time.perf_counter() - started, 4)
