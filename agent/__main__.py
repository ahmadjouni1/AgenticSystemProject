"""Run the agent on one prompt:

    uv run python -m agent "a laptop that opens and closes"
    uv run python -m agent "a four-bar linkage" --role cheap --max-cost 0.5 --max-turns 40

Prints one line per turn (tools, compile status, cost) and the run folder. Open the result
with ``uv run python -m viewer <run folder>``. Keys come from the environment or
replica/.env.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.context import ContextConfig, ContextPolicy
from agent.loop import (DEFAULT_PROMPT_SET, FIX_RESERVE_S, MAX_TIME_S, MAX_TURNS, PROMPT_SETS,
                        RUNS_ROOT, EffortPolicy, new_run_dir, run)
from llm import Budget, config, get_model, load_env


def _args_brief(name: str, args: dict) -> str:
    if name in ("read", "write", "edit") and isinstance(args.get("path"), str):
        return args["path"]
    if name == "probe":
        return ",".join(q.get("type", "?") for q in args.get("queries", []) if isinstance(q, dict))
    if name == "render" and args.get("views"):
        return ",".join(args["views"])
    return ""


def print_event(event: dict) -> None:
    if event["kind"] == "end":
        return
    m = event["metrics"]
    tools = []
    for tool in event["tools"]:
        brief = _args_brief(tool["name"], tool["args"] if isinstance(tool["args"], dict) else {})
        text = tool["name"] + (f"({brief})" if brief else "")
        if tool["error"]:
            text += f"!{tool['error']}"
        tools.append(text)
    line = f"turn {event['turn']:>3}  " + (", ".join(tools) if tools else "(text)")
    compiled = event.get("compile")
    if compiled:
        line += f"  -> compile {compiled['status']}"
        line += f" [{compiled['primary']}]" if compiled["primary"] else ""
        line += f" v{compiled['version']:04d}" if compiled["version"] is not None else ""
        line += " (cached)" if compiled["cached"] else ""
    if event.get("reminder"):
        line += f"  -> <{event['reminder']}>"
    cost = event["cost_total"]
    line += (f"  | ctx~{m['context_tokens_est'] / 1000:.1f}k tok, cached "
             f"{m.get('cached_share', 0):.0%}, ${cost:.4f}")
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="the object to build, in words")
    parser.add_argument("--role", default="builder", choices=sorted(config.ROLES))
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-cost", type=float, default=config.DEFAULT_MAX_COST_USD,
                        help="USD per run")
    parser.add_argument("--max-calls", type=int, default=config.DEFAULT_MAX_CALLS)
    parser.add_argument("--runs", default=str(RUNS_ROOT), help="folder for run folders")
    parser.add_argument("--context", choices=["v1", "none"], default="v1",
                        help="context policy (agent/context.py); none sends the full history")
    parser.add_argument("--effort", choices=["adaptive", "default"], default="adaptive",
                        help="adaptive: low reasoning effort once the latest compile passed")
    parser.add_argument("--max-time", type=float, default=MAX_TIME_S,
                        help="wall-clock seconds per run")
    parser.add_argument("--lessons", action="store_true",
                        help="use and update runs/lessons.md (off by default)")
    parser.add_argument("--prompt-set", choices=sorted(PROMPT_SETS), default=DEFAULT_PROMPT_SET,
                        help="system prompt + quickstart + docs; v0 is step 9's")
    parser.add_argument("--planning-effort", choices=["default", "low", "medium", "high"],
                        default="default",
                        help="reasoning effort before the first compile (with --effort adaptive)")
    parser.add_argument("--fix-reserve", type=float, default=FIX_RESERVE_S,
                        help="no fix cycle starts with fewer seconds left (0: off)")
    parser.add_argument("--two-phase", action="store_true",
                        help="skeleton first (must pass every check), then detail")
    args = parser.parse_args(argv)
    load_env()
    run_dir = new_run_dir(args.prompt, args.runs)
    budget = Budget(max_cost_usd=args.max_cost, max_calls=args.max_calls)
    model = get_model(args.role, run_dir=run_dir, budget=budget)
    context = None
    if args.context == "v1":  # the cheap model writes compaction summaries, on the same budget
        context = ContextPolicy(ContextConfig(), get_model("cheap", run_dir=run_dir,
                                                           budget=budget),
                                workspace=run_dir / "workspace")
    planning = None if args.planning_effort == "default" else args.planning_effort
    if planning is not None and args.effort != "adaptive":
        parser.error("--planning-effort needs --effort adaptive")
    effort = EffortPolicy(planning_effort=planning) if args.effort == "adaptive" else None
    print(f"run folder: {run_dir}\nmodel: {model.provider}/{model.model} (role {args.role}), "
          f"budget ${args.max_cost:.2f} / {args.max_calls} calls / {args.max_time:.0f} s, "
          f"{args.max_turns} turns; context {args.context}, effort {args.effort}, prompt set "
          f"{args.prompt_set}, planning effort {args.planning_effort}"
          + (", lessons on" if args.lessons else "")
          + (", two-phase" if args.two_phase else ""), flush=True)
    result = run(args.prompt, model, budget, args.max_turns, run_dir=run_dir,
                 on_event=print_event, context=context, effort=effort,
                 max_time_s=args.max_time, lessons=args.lessons, prompt_set=args.prompt_set,
                 fix_reserve_s=args.fix_reserve, two_phase=args.two_phase)
    record = result.record
    print(f"\n{result.status}: {result.reason}")
    print(json.dumps({k: record[k] for k in ("turns", "compiles", "cost_usd", "duration_s")}))
    print(f"tokens: {record['tokens']}")
    if record["context_policy"]:
        print(f"context policy hits: {record['context_policy']['hits']}")
    if record["quality"]:
        print(f"quality: {record['quality']}")
    print(f"run folder: {result.run_dir}\nview it: uv run python -m viewer {result.run_dir}")
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
