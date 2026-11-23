"""One real tool-call round trip per provider, printing tokens and cost. Run from replica/:

    $env:OPENAI_API_KEY = "..."; $env:DEEPSEEK_API_KEY = "..."    # PowerShell
    uv run python scripts/llm_smoke.py                  # both providers
    uv run python scripts/llm_smoke.py deepseek         # one provider
    uv run python scripts/llm_smoke.py --cache          # also show prompt caching at work

Per provider (the model of the first role in llm/config.py ROLES that uses it):
1. the model is asked to add 2 + 3 with the `add` tool and must call it;
2. the tool result goes back and the model answers (this checks the tool loop, including
   DeepSeek's reasoning_content echo).
With --cache, two more calls repeat a long system prompt (about 2,000 tokens): the second
should report cached input tokens.

Budget: $0.50 and 10 calls per run. Calls are logged to runs/llm_smoke/<time>/llm_calls.jsonl
(no keys, no message text). A provider without its key is skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import (Budget, CallLog, LLMError, System, Tool, ToolResult, User,  # noqa: E402
                 config)
from llm.openai_chat import OpenAIChatModel  # noqa: E402

ADD = Tool("add", "Add two integers and return the sum.", {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"], "additionalProperties": False})
SYSTEM = "You are a careful calculator. Always use the add tool for additions."
LONG_SYSTEM = SYSTEM + "\n\nReference table (ignore unless asked):\n" + "\n".join(
    f"- item {i}: the sum of {i} and {i + 1} is {2 * i + 1}." for i in range(160))


def row(label: str, reply) -> None:
    u = reply.usage
    cost = reply.cost if isinstance(reply.cost, str) else f"${reply.cost:.6f}"
    calls = ", ".join(f"{c.name}({c.arguments_json()})" for c in reply.tool_calls) or "-"
    print(f"  {label:12s} in={u.input_tokens:>6} cached={u.cached_input_tokens:>6} "
          f"({u.cached_share:4.0%}) out={u.output_tokens:>5} reasoning={u.reasoning_tokens:>5} "
          f"cost={cost:>10} {reply.latency_s:5.2f}s retries={reply.retries} "
          f"finish={reply.finish_reason} tools={calls}")
    if reply.text:
        print(f"  {'':12s} text: {reply.text.strip()[:120]!r}")


def smoke(provider_name: str, run_dir: Path, budget: Budget, cache: bool) -> bool:
    provider = config.PROVIDERS[provider_name]
    role_name, role = next(((n, r) for n, r in config.ROLES.items()
                            if r.provider == provider_name), (None, None))
    if role is None:
        print(f"{provider_name}: no role in llm/config.py uses it; skipped")
        return True
    if not os.environ.get(provider.key_env):
        print(f"{provider_name}: skipped (set {provider.key_env})")
        return True
    print(f"{provider_name}: role {role_name!r}, model {role.model!r}")
    model = OpenAIChatModel(provider, role.model, effort=role.effort,
                            max_output_tokens=4_000, timeout_s=role.timeout_s,
                            budget=budget, log=CallLog(run_dir), role=role_name)
    try:
        messages = [System(SYSTEM), User("What is 2 + 3? Use the tool.")]
        first = model.query(messages, [ADD])
        row("tool call", first)
        if not first.tool_calls:
            print("  FAILED: the model did not call the tool")
            return False
        messages.append(first.message())
        for call in first.tool_calls:
            total = call.arguments.get("a", 0) + call.arguments.get("b", 0)
            messages.append(ToolResult(call.id, call.name, str(total)))
        second = model.query(messages, [ADD])
        row("answer", second)
        ok = "5" in second.text
        if cache:
            for label in ("cache 1/2", "cache 2/2"):
                row(label, model.query([System(LONG_SYSTEM), User("Say OK.")], [ADD]))
    except LLMError as exc:
        print(f"  FAILED: {exc}")
        return False
    print(f"  {'OK' if ok else 'FAILED: the answer does not contain 5'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("providers", nargs="*", metavar="provider",
                        help=f"any of {', '.join(config.PROVIDERS)} (default: all)")
    parser.add_argument("--cache", action="store_true", help="also show prompt caching")
    args = parser.parse_args()
    unknown = sorted(set(args.providers) - set(config.PROVIDERS))
    if unknown:
        parser.error(f"unknown provider(s) {unknown}; choose from {list(config.PROVIDERS)}")
    args.providers = args.providers or list(config.PROVIDERS)
    run_dir = Path("runs") / "llm_smoke" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    budget = Budget(max_cost_usd=0.50, max_calls=10)
    print(f"prices checked {config.PRICES_CHECKED}; log: {run_dir / CallLog.FILE}\n")
    results = [smoke(name, run_dir, budget, args.cache) for name in args.providers]
    summary = CallLog(run_dir).summary()
    print(f"\ntotal: {summary['calls']} calls, ${summary['cost_usd']:.6f}"
          f" (+{summary['unknown_cost_calls']} of unknown cost), "
          f"cached share {summary['usage']['cached_share']:.0%}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
