"""Which ways of sending reasoning back does DeepSeek accept in a thinking-mode tool loop?

    uv run python scripts/deepseek_reasoning_probe.py [model]      # default deepseek-v4-pro

DeepSeek's thinking-mode guide, which holds the rule, could not be read (HTTP 525), so this
asks the API. One tool call (``add(2, 3)``), its result, then the follow-up request in four
variants:

1. ``echo``: the assistant tool-call message keeps its reasoning_content (current loop);
2. ``checkpoint_drop``: a user message (a checkpoint) follows the tool result, and the earlier
   reasoning is dropped: what the context policy does (rule a);
3. ``drop_in_loop``: reasoning dropped with no user message in between;
4. ``old_echo``: a user message follows, and the old reasoning is still sent.

Prints the HTTP status and the answer for each. About $0.005 in total. Keys from the
environment or replica/.env.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402

from llm import load_env  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "add", "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"},
                                                    "b": {"type": "integer"}},
                   "required": ["a", "b"]}}}]


def main() -> int:
    load_env()
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-pro"
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("set DEEPSEEK_API_KEY")
        return 2
    client = openai.OpenAI(api_key=key, base_url="https://api.deepseek.com", max_retries=2)
    base = [{"role": "system", "content": "Use the add tool for every addition."},
            {"role": "user", "content": "What is 2 + 3?"}]
    first = client.chat.completions.create(model=model, messages=base, tools=TOOLS,
                                           max_tokens=2000)
    message = first.choices[0].message
    reasoning = getattr(message, "reasoning_content", None) or \
        (message.model_extra or {}).get("reasoning_content")
    call = message.tool_calls[0]
    print(f"first call: tool={call.function.name}({call.function.arguments}), "
          f"reasoning={len(reasoning or '')} chars")
    assistant = {"role": "assistant", "content": message.content or "",
                 "tool_calls": [{"id": call.id, "type": "function",
                                 "function": {"name": call.function.name,
                                              "arguments": call.function.arguments}}]}
    result = {"role": "tool", "tool_call_id": call.id, "content": "5"}
    checkpoint = {"role": "user", "content": "<checkpoint>The add tool returned 5.</checkpoint>"}
    variants = {
        "echo": base + [{**assistant, "reasoning_content": reasoning}, result],
        "checkpoint_drop": base + [assistant, result, checkpoint],
        "drop_in_loop": base + [assistant, result],
        "old_echo": base + [{**assistant, "reasoning_content": reasoning}, result, checkpoint],
    }
    outcome = {}
    for name, messages in variants.items():
        try:
            reply = client.chat.completions.create(model=model, messages=messages, tools=TOOLS,
                                                   max_tokens=2000)
            text = (reply.choices[0].message.content or "").strip().replace("\n", " ")
            outcome[name] = {"status": 200, "answer": text[:80],
                             "prompt_tokens": reply.usage.prompt_tokens}
        except openai.APIStatusError as exc:
            outcome[name] = {"status": exc.status_code, "error": str(exc)[:200]}
        print(f"{name:16s} {json.dumps(outcome[name])}")
    return 0 if all(o["status"] == 200 for o in outcome.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
