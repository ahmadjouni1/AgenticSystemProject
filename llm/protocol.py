"""The protocol every model implements, and what a call returns.

    reply = model.query(messages, tools)      # messages: see llm/messages.py
    reply.text, reply.tool_calls, reply.usage, reply.cost, reply.raw
    messages.append(reply.message())          # continue the conversation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from llm.messages import Assistant, Message, Tool, ToolCall

UNKNOWN = "unknown"  # the cost of a call to a model with no price; never reported as $0


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0  # all prompt tokens, cached or not
    cached_input_tokens: int = 0  # of those, read from the provider's prompt cache
    cache_write_tokens: int = 0  # of those, written to the cache (some models bill this)
    output_tokens: int = 0  # all completion tokens, reasoning included
    reasoning_tokens: int = 0  # of those, the model's reasoning

    @property
    def cached_share(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(*(a + b for a, b in zip(asdict(self).values(), asdict(other).values(),
                                             strict=True)))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cached_share": round(self.cached_share, 4)}


Cost = float | str  # USD, or UNKNOWN


@dataclass
class Reply:
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    cost: Cost
    raw: dict[str, Any]  # the provider's response, as JSON-ready data
    provider: str = ""
    model: str = ""  # the model that answered, as the provider reports it
    finish_reason: str | None = None
    reasoning: str | None = None
    latency_s: float = 0.0
    retries: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def message(self) -> Assistant:
        """This reply as the next message of the conversation."""
        return Assistant(self.text, self.tool_calls, self.reasoning)


class Model(Protocol):
    provider: str
    model: str

    def query(self, messages: list[Message], tools: tuple[Tool, ...] | list[Tool] = (), *,
              effort: str | None = None, max_output_tokens: int | None = None) -> Reply:
        """One model call. Raises LLMError (ProviderError, BudgetExceeded) on failure.
        ``effort`` / ``max_output_tokens`` override the model's settings for this call."""


class LLMError(RuntimeError):
    """A model call failed or cannot be made (configuration, credentials, provider)."""


class ProviderError(LLMError):
    """The provider refused or failed the call, after any retries."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False,
                 attempts: int = 1):
        super().__init__(message)
        self.status, self.retryable, self.attempts = status, retryable, attempts


class BudgetExceeded(LLMError):
    """The run's cost or call cap is reached. ``reply`` is the call that crossed it, if any."""

    def __init__(self, message: str, reply: Reply | None = None):
        super().__init__(message)
        self.reply = reply
