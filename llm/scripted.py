"""ScriptedModel: replays a fixed list of replies. Same protocol, no network.

    model = ScriptedModel([
        {"tool_calls": [{"name": "compile", "arguments": {}}]},
        "Done: the hinge now clears the base.",
    ])

A reply is a string (text only) or a dict with ``text``, ``tool_calls`` (``name``,
``arguments`` or ``raw_arguments`` (a string, sent as is), optional ``id``), ``usage`` (Usage fields), ``reasoning`` and
``finish_reason``. Every request is kept in ``model.requests``. Budget and call log work as
for a real model; the cost uses config.PRICES if ``model`` names a priced model, else $0
(nothing is billed).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from llm import accounting, config
from llm.accounting import Budget, CallLog
from llm.base import ChatModelBase, Completion
from llm.messages import Message, Tool, ToolCall
from llm.protocol import Cost, LLMError, Usage


class ScriptedModel(ChatModelBase):
    provider = "scripted"

    def __init__(self, replies: list[str | dict[str, Any]], model: str = "scripted", *,
                 budget: Budget | None = None, log: CallLog | None = None,
                 role: str | None = None, supports_images: bool = False):
        super().__init__(model, budget=budget, log=log, role=role)
        self.supports_images = supports_images
        self.replies = list(replies)
        self.requests: list[tuple[list[Message], tuple[Tool, ...]]] = []
        self._ids = 0

    def _cost(self, usage: Usage, at: datetime) -> Cost:
        return accounting.cost(self.model, usage, at) if self.model in config.PRICES else 0.0

    def _complete(self, messages: list[Message], tools: tuple[Tool, ...]) -> Completion:
        self.requests.append((list(messages), tools))
        if not self.replies:
            raise LLMError(f"ScriptedModel has no reply left (call {len(self.requests)})")
        reply = self.replies.pop(0)
        spec = {"text": reply} if isinstance(reply, str) else dict(reply)
        calls = []
        for call in spec.get("tool_calls", []):
            self._ids += 1
            call_id = call.get("id", f"call_{self._ids}")
            if "raw_arguments" in call:  # exactly as a model sent it (maybe broken JSON)
                calls.append(ToolCall.parse(call_id, call["name"], call["raw_arguments"]))
                continue
            arguments = call.get("arguments", {})
            calls.append(ToolCall(call_id, call["name"], arguments, json.dumps(arguments)))
        known = [t.name for t in tools]
        for call in calls:
            if call.name not in known:
                raise LLMError(f"scripted tool call {call.name!r} is not one of the tools "
                               f"offered: {known}")
        finish = spec.get("finish_reason", "tool_calls" if calls else "stop")
        return Completion(spec.get("text", ""), tuple(calls), Usage(**spec.get("usage", {})),
                          {"scripted": spec}, self.model, finish, spec.get("reasoning"))
