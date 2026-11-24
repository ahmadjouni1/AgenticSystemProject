"""LLM client: one small protocol, OpenAI and DeepSeek behind it, cost and budget tracking.

    from llm import Budget, User, System, Tool, get_model
    model = get_model("builder", run_dir="runs/r1", budget=Budget(max_cost_usd=2, max_calls=50))
    reply = model.query([System("..."), User("...")], tools=[Tool(...)])

- Message format: llm/messages.py. Protocol and Reply: llm/protocol.py.
- Roles, providers, prices, budgets, retries: llm/config.py.
"""

from __future__ import annotations

from pathlib import Path

from llm import config
from llm.accounting import Budget, CallLog, cost
from llm.env import load_env
from llm.messages import Assistant, Message, System, Tool, ToolCall, ToolResult, User
from llm.protocol import (UNKNOWN, BudgetExceeded, CallTimeLimit, LLMError, Model,
                          ProviderError, Reply, Usage)
from llm.scripted import ScriptedModel


def get_model(role: str, *, run_dir: str | Path | None = None,
              budget: Budget | None = None) -> Model:
    """The model configured for ``role`` in config.ROLES, logging to run_dir/llm_calls.jsonl.
    Keys: the environment, then replica/.env for any that are not set."""
    from llm.openai_chat import OpenAIChatModel

    load_env()
    if role not in config.ROLES:
        raise LLMError(f"unknown role {role!r}; roles: {sorted(config.ROLES)}")
    choice = config.ROLES[role]
    if choice.provider not in config.PROVIDERS:
        raise LLMError(f"role {role!r} uses unknown provider {choice.provider!r}")
    return OpenAIChatModel(
        config.PROVIDERS[choice.provider], choice.model, effort=choice.effort,
        max_output_tokens=choice.max_output_tokens, timeout_s=choice.timeout_s,
        budget=budget, log=CallLog(run_dir) if run_dir is not None else None, role=role,
        supports_images=choice.images)


__all__ = ["UNKNOWN", "Assistant", "Budget", "BudgetExceeded", "CallLog", "CallTimeLimit",
           "LLMError", "Message",
           "Model", "ProviderError", "Reply", "ScriptedModel", "System", "Tool", "ToolCall",
           "ToolResult", "Usage", "User", "config", "cost", "get_model", "load_env"]
