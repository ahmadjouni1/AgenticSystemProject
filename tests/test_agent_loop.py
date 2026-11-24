"""The agent loop with scripted models (no network) and the real compiler."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

from agent.loop import PROMPT_SETS, run
from llm import Budget, ScriptedModel
from llm.config import PROVIDERS
from llm.openai_chat import OpenAIChatModel

sys.path.insert(0, os.path.dirname(__file__))
from check_fixtures import DRAWER_SOURCE  # noqa: E402

# A known-good program: step 9's laptop (the docs examples are compiled by tests/test_docs.py).
LAPTOP_DOC = PROMPT_SETS["v0"].quickstart
GOOD = re.findall(r"```python\n(.*?)```", LAPTOP_DOC.read_text(encoding="utf-8"), re.S)[-1]
BAD_LIMITS = GOOD.replace("limits=(0.0, math.radians(135))", "limits=(0.5, 1.0)")
assert BAD_LIMITS != GOOD


def write(content: str) -> dict:
    return {"tool_calls": [{"name": "write", "arguments": {"path": "main.py",
                                                           "content": content}}]}


def edit(old: str, new: str) -> dict:
    return {"tool_calls": [{"name": "edit", "arguments": {
        "path": "main.py", "edits": [{"old_text": old, "new_text": new}]}}]}


COMPILE = {"tool_calls": [{"name": "compile", "arguments": {}}]}
DONE = "Done: a base and a lid on a hinge from 0 to 135 degrees."


def go(tmp_path: Path, replies: list, budget: Budget | None = None, **kwargs):
    budget = budget or Budget(max_cost_usd=None, max_calls=None)
    model = kwargs.pop("model", None) or ScriptedModel(replies, budget=budget)
    result = run("a laptop that opens and closes", model, budget, run_dir=tmp_path / "run",
                 **kwargs)
    return result, model


def conversation(result) -> list[dict]:
    path = result.run_dir / "conversation.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def reminders(result) -> list[str]:
    return [m["reminder"] for m in conversation(result) if m.get("reminder")]


# ---- success paths ------------------------------------------------------------------------------


def test_clean_success(tmp_path) -> None:
    result, model = go(tmp_path, [write(GOOD), COMPILE, DONE])
    assert result.status == "success" and result.turns == 3 and result.final_text == DONE
    assert result.best == {"version": "v0000", "passed_checks": True, "path": "result/v0000"}
    record = json.loads((result.run_dir / "record.json").read_text(encoding="utf-8"))
    assert record["status"] == "success" and record["turns"] == 3
    assert record["compiles"] == {"calls": 1, "passed": 1, "failed": 0, "cached": 0,
                                  "last_status": "success", "last_version": "v0000"}
    assert record["tool_calls"] == {"write": 1, "compile": 1}
    assert [t["turn"] for t in record["per_turn"]] == [1, 2, 3]
    assert all(t["context_tokens_est"] > 1000 for t in record["per_turn"])  # prompts + tools
    assert record["per_turn"][0]["context_tokens_est"] < record["per_turn"][2]["context_tokens_est"]
    assert {"cached_share", "cost", "input_tokens"} <= set(record["per_turn"][0])
    # The first request: system prompt, quickstart, task; the tools in registry order.
    messages, tools = model.requests[0]
    assert [type(m).__name__ for m in messages] == ["System", "System", "User"]
    assert "What a good object is" in messages[0].content
    assert "SDK quickstart" in messages[1].content
    assert [t.name for t in tools] == ["read", "write", "edit", "compile", "render", "probe"]
    roles = [m["role"] for m in conversation(result)]
    assert roles == ["system", "system", "user", "assistant", "tool", "assistant", "tool",
                     "assistant"]
    for sub in ("workspace", "result", "renders"):
        assert (result.run_dir / sub).is_dir()
    assert (result.run_dir / "llm_calls.jsonl").is_file()


def test_compile_failure_then_fix_then_success(tmp_path) -> None:
    result, _ = go(tmp_path, [write(BAD_LIMITS), COMPILE,
                              edit("limits=(0.5, 1.0)", "limits=(0.0, math.radians(135))"),
                              COMPILE, DONE])
    assert result.status == "success"
    compiles = result.record["compiles"]
    assert (compiles["failed"], compiles["passed"]) == (1, 1)
    failed = [m for m in conversation(result) if m["role"] == "tool" and m["name"] == "compile"]
    assert '"status": "error"' in failed[0]["content"] and "must contain 0" in failed[0]["content"]


def test_final_text_without_compiling_gets_a_reminder(tmp_path) -> None:
    result, model = go(tmp_path, ["I have built the laptop.", write(GOOD), COMPILE, DONE])
    assert result.status == "success" and result.turns == 4
    assert reminders(result) == ["compile_required"]
    reminder = model.requests[1][0][-1]
    assert reminder.content.startswith("<compile_required>")
    assert "you have not compiled yet" in reminder.content
    assert result.record["per_turn"][0]["reminder"] == "compile_required"


def test_edit_after_a_good_compile_needs_a_new_compile(tmp_path) -> None:
    result, model = go(tmp_path, [
        write(GOOD), COMPILE,
        edit("DEPTH / 2, 0.001", "DEPTH / 2.2, 0.001"),  # a narrower keyboard
        DONE,  # refused: main.py changed after the compile
        COMPILE, DONE])
    assert result.status == "success" and result.record["compiles"]["passed"] == 2
    assert reminders(result) == ["compile_required"]
    assert "the workspace changed after the last compile" in model.requests[4][0][-1].content


def test_text_after_a_good_compile_is_the_final_answer_but_empty_is_not(tmp_path) -> None:
    result, model = go(tmp_path, [write(GOOD), COMPILE, "", DONE])
    assert result.status == "success"
    assert reminders(result) == ["final_response_required"]
    assert model.requests[3][0][-1].content.startswith("<final_response_required>")


# ---- other end states ---------------------------------------------------------------------------


def test_call_cap_keeps_the_best_build(tmp_path) -> None:
    budget = Budget(max_cost_usd=None, max_calls=3)
    read = {"tool_calls": [{"name": "read", "arguments": {"path": "main.py"}}]}
    result, _ = go(tmp_path, [write(GOOD), COMPILE, read, read, DONE], budget)
    assert result.status == "budget_exceeded" and result.turns == 4
    assert "call cap reached: 3 of 3" in result.reason
    assert result.best["version"] == "v0000" and result.best["passed_checks"]
    assert "best build: v0000 (passed the checks)" in result.reason
    record = json.loads((result.run_dir / "record.json").read_text(encoding="utf-8"))
    assert record["status"] == "budget_exceeded" and record["best"]["version"] == "v0000"


def test_cost_cap_stops_after_the_paid_reply(tmp_path) -> None:
    budget = Budget(max_cost_usd=0.01, max_calls=None)
    big = {"input_tokens": 20_000, "output_tokens": 1_000}  # ~$0.03 on deepseek-v4-pro
    model = ScriptedModel([{**write(GOOD), "usage": big}, COMPILE], model="deepseek-v4-pro",
                          budget=budget)
    result, _ = go(tmp_path, [], budget, model=model)
    assert result.status == "budget_exceeded" and "cost cap exceeded" in result.reason
    assert result.best is None and "no build was produced" in result.reason
    assert result.record["cost_usd"] > 0.01
    assert conversation(result)[-1]["stopped"] == "budget_exceeded"  # the paid reply is kept


def test_five_identical_failures_is_no_progress(tmp_path) -> None:
    result, _ = go(tmp_path, [write(DRAWER_SOURCE)] + [COMPILE] * 6)
    assert result.status == "no_progress" and result.turns == 6
    assert "5 identical compile failures in a row" in result.reason
    assert result.record["compiles"]["failed"] == 5 and result.record["compiles"]["cached"] == 4
    assert result.best == {"version": "last_failed", "passed_checks": False,
                           "path": "result/last_failed"}
    assert "best build: result/last_failed (built, but failed its checks)" in result.reason


def test_finishing_over_failed_checks_is_failed_checks(tmp_path) -> None:
    result, _ = go(tmp_path, [write(DRAWER_SOURCE), COMPILE, "Done.", "Done.", "Done."])
    assert result.status == "failed_checks"
    assert reminders(result) == ["compile_required", "compile_required"]
    assert "swept_collision" in result.reason and result.best["version"] == "last_failed"


def test_three_empty_replies_is_no_progress(tmp_path) -> None:
    result, _ = go(tmp_path, ["", "", ""])
    assert result.status == "no_progress" and "3 empty replies" in result.reason


def test_max_turns(tmp_path) -> None:
    read = {"tool_calls": [{"name": "read", "arguments": {"path": "main.py"}}]}
    result, _ = go(tmp_path, [read] * 5, max_turns=2)
    assert result.status == "max_turns" and result.turns == 2


def test_a_model_error_ends_the_run_cleanly(tmp_path) -> None:
    result, _ = go(tmp_path, [write(GOOD)])  # the script runs out: the model call fails
    assert result.status == "error" and "no reply left" in result.reason
    assert json.loads((result.run_dir / "record.json").read_text())["status"] == "error"


# ---- tool execution -----------------------------------------------------------------------------


def test_consecutive_reads_run_together_and_keep_their_order(tmp_path) -> None:
    reads = {"tool_calls": [{"name": "read", "arguments": {"path": p}}
                            for p in ("main.py", "docs/sdk", "docs/sdk/quickstart.md")]}
    result, model = go(tmp_path, [reads, write(GOOD), COMPILE, DONE])
    assert result.status == "success"
    results = [m for m in conversation(result) if m["role"] == "tool"][:3]
    assert [json.loads(m["content"])["path"] for m in results] == [
        "main.py", "docs/sdk", "docs/sdk/quickstart.md"]


def test_images_are_attached_only_for_models_that_accept_them(tmp_path) -> None:
    render = {"tool_calls": [{"name": "render", "arguments": {"views": ["iso"]}}]}
    for accepts in (False, True):
        budget = Budget(max_cost_usd=None, max_calls=None)
        model = ScriptedModel([write(GOOD), COMPILE, render, DONE], budget=budget,
                              supports_images=accepts)
        result = run("a laptop", model, budget, run_dir=tmp_path / f"run{accepts}")
        assert result.status == "success"
        with_images = [m for m in conversation(result) if m.get("images")]
        assert len(with_images) == int(accepts)
        if accepts:
            assert with_images[0]["images"][0].endswith("iso.png")


def test_bad_json_arguments_come_back_as_a_structured_error(tmp_path) -> None:
    budget = Budget(max_cost_usd=None, max_calls=None)
    calls = iter([
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "main.py"'}}]},
            "finish_reason": "tool_calls"}], "usage": {}},
        {"choices": [{"message": {"role": "assistant", "content": ""},
                      "finish_reason": "stop"}], "usage": {}},
    ] + [{"choices": [{"message": {"role": "assistant", "content": ""},
                       "finish_reason": "stop"}], "usage": {}}] * 2)
    model = OpenAIChatModel(PROVIDERS["deepseek"], "deepseek-flash",
                            create=lambda **body: next(calls), budget=budget)
    result = run("a laptop", model, budget, run_dir=tmp_path / "run")
    tool = next(m for m in conversation(result) if m["role"] == "tool")
    data = json.loads(tool["content"])
    assert tool["is_error"] and data["error_kind"] == "bad_arguments"
    assert "not valid JSON" in data["message"]


# ---- DeepSeek reasoning across a tool loop -------------------------------------------------------


def test_deepseek_reasoning_is_echoed_across_the_tool_loop(tmp_path) -> None:
    def response(reasoning, text="", tool=None):
        message = {"role": "assistant", "content": text or None, "reasoning_content": reasoning}
        if tool:
            message["tool_calls"] = [{"id": f"call_{reasoning}", "type": "function",
                                      "function": {"name": tool[0],
                                                   "arguments": json.dumps(tool[1])}}]
        return {"id": "r", "model": "deepseek-v4-pro", "usage": {
            "prompt_tokens": 3000, "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 2048},
            "completion_tokens_details": {"reasoning_tokens": 20}},
            "choices": [{"message": message, "finish_reason": "tool_calls" if tool else "stop"}]}

    replies = iter([response("plan", tool=("write", {"path": "main.py", "content": GOOD})),
                    response("check", tool=("compile", {})),
                    response("wrap", text=DONE)])
    requests: list[dict] = []

    def create(**body):
        requests.append(json.loads(json.dumps(body)))
        return next(replies)

    budget = Budget(max_cost_usd=1.0, max_calls=10)
    model = OpenAIChatModel(PROVIDERS["deepseek"], "deepseek-v4-pro", create=create,
                            budget=budget, role="builder")
    result = run("a laptop", model, budget, run_dir=tmp_path / "run")
    assert result.status == "success" and len(requests) == 3
    assistant = [[m for m in r["messages"] if m["role"] == "assistant"] for r in requests]
    assert assistant[0] == []
    assert [m.get("reasoning_content") for m in assistant[1]] == ["plan"]
    assert [m.get("reasoning_content") for m in assistant[2]] == ["plan", "check"]
    # The cacheable prefix (system prompt, quickstart, task, tools) is identical every call.
    assert all(r["messages"][:3] == requests[0]["messages"][:3] for r in requests)
    assert all(r["tools"] == requests[0]["tools"] for r in requests)
    record = result.record
    assert record["tokens"]["reasoning_tokens"] == 60
    assert record["per_turn"][0]["cached_share"] == pytest.approx(2048 / 3000, abs=1e-4)
    assert record["cost_usd"] > 0 and record["model"]["model"] == "deepseek-v4-pro"
    calls = (result.run_dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3


# ---- CLI ----------------------------------------------------------------------------------------


def test_cli_runs_and_prints_the_run_folder(tmp_path, monkeypatch, capsys) -> None:
    import agent.__main__ as cli

    def fake_get_model(role, *, run_dir, budget):
        return ScriptedModel([write(GOOD), COMPILE, DONE], budget=budget, role=role)

    monkeypatch.setattr(cli, "get_model", fake_get_model)
    code = cli.main(["a laptop that opens and closes", "--runs", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "turn   1  write(main.py)" in out and "-> compile success v0000" in out
    assert "success: finished after a passing, fresh compile" in out
    assert "python -m viewer" in out
    (folder,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert folder.name.endswith("-a-laptop-that-opens-and-closes")
    assert (tmp_path / "index.jsonl").is_file()  # step 11: the runs folder's index


def test_a_reply_cut_off_at_the_output_limit_says_so(tmp_path) -> None:
    cut = {"finish_reason": "length", "tool_calls": [
        {"name": "write", "raw_arguments": '{"path": "main.py", "content": "from sdk imp'}]}
    result, model = go(tmp_path, [cut, write(GOOD), COMPILE, DONE])
    assert result.status == "success"
    tool = next(m for m in conversation(result) if m["role"] == "tool")
    data = json.loads(tool["content"])
    assert data["error_kind"] == "reply_cut_off" and "output token limit" in data["message"]
    assert "write large files in parts" in data["hint"]
    assert (result.run_dir / "workspace" / "main.py").read_text(encoding="utf-8") == GOOD


def test_context_estimate_counts_the_reasoning_deepseek_gets_back(tmp_path) -> None:
    long_reasoning = "think " * 4000  # ~24k characters
    replies = iter([
        {"model": "deepseek-v4-pro", "usage": {}, "choices": [{"finish_reason": "tool_calls",
         "message": {"role": "assistant", "content": None, "reasoning_content": long_reasoning,
                     "tool_calls": [{"id": "c1", "type": "function", "function": {
                         "name": "read", "arguments": '{"path": "main.py"}'}}]}}]},
        {"model": "deepseek-v4-pro", "usage": {}, "choices": [{"finish_reason": "stop",
         "message": {"role": "assistant", "content": "", "reasoning_content": "x"}}]},
    ] * 2)
    budget = Budget(max_cost_usd=None, max_calls=3)
    model = OpenAIChatModel(PROVIDERS["deepseek"], "deepseek-v4-pro",
                            create=lambda **body: next(replies), budget=budget)
    result = run("a laptop", model, budget, run_dir=tmp_path / "run")
    first, second = result.record["per_turn"][:2]
    assert second["context_tokens_est"] - first["context_tokens_est"] > 5000  # the echo
