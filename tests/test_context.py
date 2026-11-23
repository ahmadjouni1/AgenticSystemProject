"""The context policy (agent/context.py), effort, wall-clock budget and lessons, with scripted
models and the real compiler."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent import lessons
from agent.context import ContextConfig, ContextPolicy
from agent.loop import QUICKSTART, EffortPolicy, run
from llm import Assistant, Budget, ScriptedModel, ToolResult, User
from llm.messages import to_chat, validate

GOOD = re.findall(r"```python\n(.*?)```", QUICKSTART.read_text(encoding="utf-8"), re.S)[-1]
BAD = GOOD.replace("limits=(0.0, math.radians(135))", "limits=(0.5, 1.0)")
FIX = {"tool_calls": [{"name": "edit", "arguments": {"path": "main.py", "edits": [
    {"old_text": "limits=(0.5, 1.0)", "new_text": "limits=(0.0, math.radians(135))"}]}}]}
COMPILE = {"tool_calls": [{"name": "compile", "arguments": {}}]}
READ_MAIN = {"tool_calls": [{"name": "read", "arguments": {"path": "main.py"}}]}
DONE = "Done: base and lid on a hinge, 0 to 135 degrees."


def write(content: str, reasoning: str | None = None) -> dict:
    reply = {"tool_calls": [{"name": "write", "arguments": {"path": "main.py",
                                                            "content": content}}]}
    if reasoning:
        reply["reasoning"] = reasoning
    return reply


def go(tmp_path, replies, config=ContextConfig(), summarizer=None, **kwargs):
    budget = kwargs.pop("budget", None) or Budget(max_cost_usd=None, max_calls=None)
    model = ScriptedModel(replies, budget=budget, model=kwargs.pop("model_name", "scripted"))
    policy = ContextPolicy(config, summarizer, workspace=tmp_path / "run" / "workspace")
    result = run("a laptop", model, budget, run_dir=tmp_path / "run", context=policy, **kwargs)
    return result, model, policy


def sent(model, k):
    return model.requests[k][0]


def history(result) -> list[dict]:
    lines = (result.run_dir / "conversation.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def assert_well_formed(model) -> None:
    """Every sent view keeps the tool-call/result pairing and the same prefix."""
    first = sent(model, 0)
    for messages, _tools in model.requests:
        validate(messages)
        assert messages[:3] == first[:3]
        assert to_chat(messages[:3]) == to_chat(first[:3])


# ---- a: checkpoints and reasoning ---------------------------------------------------------------


def test_compile_signals_go_in_a_checkpoint_and_old_reasoning_is_dropped(tmp_path) -> None:
    result, model, policy = go(tmp_path, [write(BAD, "plan the hinge"),
                                          {**COMPILE, "reasoning": "now compile"},
                                          {**FIX, "reasoning": "limits must contain 0"},
                                          COMPILE, DONE])
    assert result.status == "success"
    assert_well_formed(model)
    view = sent(model, 2)  # after the first compile
    checkpoint = view[-1]
    assert isinstance(checkpoint, User) and checkpoint.content.startswith(
        "<checkpoint turn=2 compile=error primary=validation version=none>")
    assert "must contain 0" in checkpoint.content
    compile_result = view[-2]
    assert isinstance(compile_result, ToolResult)
    assert "[in the <checkpoint> message" in json.loads(compile_result.content)["compile_signals"]
    chat = to_chat(view, "current_turn")
    assert not any("reasoning_content" in m for m in chat)  # both loops before it finished
    later = to_chat(sent(model, 3), "current_turn")  # the fix's reasoning is current again
    assert [m.get("reasoning_content") for m in later if m["role"] == "assistant"][-1] == \
        "limits must contain 0"
    assert policy.hits["a_checkpoints"] == 2 and policy.hits["a_reasoning_dropped"] == 3
    # The saved conversation is complete: the checkpoint and the full signals are in it.
    saved = history(result)
    assert any(m.get("checkpoint") and "must contain 0" in m["content"] for m in saved)
    assert sum(1 for m in saved if m["role"] == "assistant" and m["reasoning"]) == 3


# ---- b: pinned state ----------------------------------------------------------------------------


def test_only_the_latest_snapshot_of_main_and_compile_signals_stay_in_full(tmp_path) -> None:
    # Stubs are applied when the view is rebuilt, at a checkpoint (the final compile here).
    result, model, policy = go(tmp_path, [write(BAD), COMPILE, READ_MAIN, FIX, COMPILE,
                                          READ_MAIN, COMPILE, DONE])
    assert result.status == "success"
    assert_well_formed(model)
    view = sent(model, 7)
    reads = [json.loads(m.content) for m in view if isinstance(m, ToolResult)
             and json.loads(m.content).get("path") == "main.py"
             and ("text" in json.loads(m.content) or "note" in json.loads(m.content))]
    assert reads[0] == {"path": "main.py",
                        "note": "[older read of main.py, superseded at turn 6]"}
    assert reads[-1]["text"].startswith("L1: import math")  # the latest read, in full
    write_call = next(c for m in view if isinstance(m, Assistant) for c in m.tool_calls
                      if c.name == "write")
    assert write_call.arguments["content"] == "[older content of main.py, superseded at turn 6]"
    checkpoints = [m.content for m in view if isinstance(m, User)
                   and m.content.startswith("<checkpoint")]
    assert len(checkpoints) == 3
    assert all("[older compile signals, superseded at turn 7]" in c for c in checkpoints[:2])
    assert "Compile passed" in checkpoints[-1]
    assert policy.hits["b_pinned_stubs"] == 4  # the write, the first read, two old signals
    # Between checkpoints nothing is stubbed: at turn 6 the turn-3 read was still whole.
    assert sum("older read" in m.content for m in sent(model, 5)
               if isinstance(m, ToolResult)) == 0
    saved = history(result)  # nothing is lost in the saved conversation
    assert any(m["role"] == "assistant" and m["tool_calls"] and
               json.loads(m["tool_calls"][0]["arguments"]).get("content") == BAD for m in saved)


# ---- c: docs already in context -----------------------------------------------------------------


def test_reading_the_quickstart_returns_a_note(tmp_path) -> None:
    reads = {"tool_calls": [{"name": "read", "arguments": {"path": "docs/sdk/quickstart.md"}},
                            {"name": "read", "arguments": {"path": "docs/sdk/README.md"}}]}
    result, model, policy = go(tmp_path, [reads, write(GOOD), COMPILE, DONE])
    assert result.status == "success"
    results = [m for m in sent(model, 1) if isinstance(m, ToolResult)]
    note = json.loads(results[0].content)
    assert "already in your context (the second system message)" in note["note"]
    assert json.loads(results[1].content)["text"].startswith("L1: # SDK docs")  # not in context
    assert policy.hits["c_doc_notes"] == 1


# ---- d: old tool outputs ------------------------------------------------------------------------


def test_old_tool_outputs_are_shortened_but_probes_keep_their_numbers(tmp_path) -> None:
    probe = {"tool_calls": [{"name": "probe", "arguments": {
        "pose": {"hinge": 1.0}, "queries": [{"type": "position", "body": "lid"}]}}]}
    # Checkpoints and pinning off, so the old compile result keeps its long signal block.
    config = ContextConfig(checkpoints=False, pin_state=False, old_after_turns=3)
    replies = [write(BAD), COMPILE, FIX, COMPILE, probe, READ_MAIN, READ_MAIN, DONE]
    result, model, policy = go(tmp_path, replies, config)
    assert result.status == "success"
    assert_well_formed(model)
    # Without checkpoints the view is rebuilt only once: shortening needs a rebuild, so
    # check what a rebuild at the last turn gives.
    fresh = ContextPolicy(config)
    full = [m for m in sent(model, 7)]
    turns = [0, 0, 0] + [_turn_of(result, i) for i in range(3, len(full))]
    view = fresh.view(full, turns, current_turn=8)
    results = {m.call_id: json.loads(m.content) for m in view if isinstance(m, ToolResult)}
    assert "[... shortened: output from turn 2" in results["call_2"]["compile_signals"]
    assert "status=error" in results["call_2"]["compile_signals"]  # the first lines stay
    assert results["call_5"]["results"][0]["origin_m"]  # the probe is kept in full
    assert "shortened" not in results["call_4"]["compile_signals"]  # turn 4: not old yet
    assert fresh.hits["d_shortened"] == 1


def _turn_of(result, index: int) -> int:
    return history(result)[index]["turn"]


# ---- e: compaction ------------------------------------------------------------------------------


def test_compaction_summarizes_the_middle_with_the_cheap_model(tmp_path) -> None:
    budget = Budget(max_cost_usd=None, max_calls=None)
    summarizer = ScriptedModel(["Tried: a laptop with a hinge. Failed: none. Works: compile "
                                "passed. Next: verify the lid."], budget=budget, role="cheap")
    replies = [write(GOOD), COMPILE, READ_MAIN, READ_MAIN, READ_MAIN, READ_MAIN, DONE]
    config = ContextConfig(compact_above_tokens=4_000, keep_recent_turns=2)
    result, model, policy = go(tmp_path, replies, config, summarizer, budget=budget)
    assert result.status == "success"
    assert_well_formed(model)
    assert policy.hits["e_compactions"] >= 1
    compacted = next(messages for messages, _ in model.requests
                     if any(isinstance(m, User) and m.content.startswith(
                         "<checkpoint kind=summary") for m in messages))
    summary = compacted[3]
    assert summary.content.startswith("<checkpoint kind=summary turns=")
    assert "Works: compile passed" in summary.content and "Current main.py" in summary.content
    assert "object_model = laptop" in summary.content  # the file, from the workspace
    assert len(compacted) < len(model.requests[-1][0]) + 50
    assert budget.calls == len(model.requests) + 1  # the summary is on the run's budget
    assert result.record["context_policy"]["summaries"][0]["turns"][0] == 1


def test_no_summarizer_means_no_compaction(tmp_path) -> None:
    config = ContextConfig(compact_above_tokens=1_000)
    result, model, policy = go(tmp_path, [write(GOOD), COMPILE, DONE], config, None)
    assert result.status == "success" and policy.hits["e_compaction_skipped"] >= 1


# ---- cache-friendliness -------------------------------------------------------------------------


def test_between_checkpoints_the_view_only_grows(tmp_path) -> None:
    result, model, _ = go(tmp_path, [write(GOOD), COMPILE, READ_MAIN, READ_MAIN, READ_MAIN, DONE])
    assert result.status == "success"
    for k in range(2, 5):  # after the compile at turn 2: turns 3, 4, 5 only append
        before, after = to_chat(sent(model, k)), to_chat(sent(model, k + 1))
        assert after[:len(before)] == before


def test_record_reports_the_policy_and_both_context_sizes(tmp_path) -> None:
    result, _, _ = go(tmp_path, [write(GOOD), COMPILE, READ_MAIN, DONE])
    record = result.record
    assert record["context_policy"]["config"]["old_after_turns"] == 6
    assert record["context_policy"]["hits"]["a_checkpoints"] == 1
    turn = record["per_turn"][-1]
    assert turn["context_tokens_est"] <= turn["history_tokens_est"]
    assert record["peak_context_tokens_est"] >= turn["context_tokens_est"]


# ---- effort, wall clock, lessons ----------------------------------------------------------------


def test_routine_turns_get_low_effort(tmp_path) -> None:
    budget = Budget(max_cost_usd=None, max_calls=None)
    model = ScriptedModel([write(BAD), COMPILE, FIX, COMPILE, READ_MAIN, DONE], budget=budget)
    result = run("a laptop", model, budget, run_dir=tmp_path / "run", effort=EffortPolicy())
    assert result.status == "success"
    assert model.options == [{}, {}, {}, {},
                             {"effort": "low", "max_output_tokens": 16_000},
                             {"effort": "low", "max_output_tokens": 16_000}]
    assert result.record["effort_turns"] == {"default": 4, "routine": 2}


def test_time_exceeded(tmp_path, monkeypatch) -> None:
    import agent.loop as loop

    clock = iter(range(0, 10_000, 100))  # each reading of the clock is 100 s later
    monkeypatch.setattr(loop.time, "time", lambda: next(clock))
    budget = Budget(max_cost_usd=None, max_calls=None)
    model = ScriptedModel([write(GOOD), COMPILE, READ_MAIN, DONE], budget=budget)
    result = run("a laptop", model, budget, run_dir=tmp_path / "run", max_time_s=250)
    assert result.status == "time_exceeded" and "the limit is 250 s" in result.reason
    assert result.turns < 4 and result.record["max_time_s"] == 250


def test_lessons_are_off_by_default_and_learned_when_on(tmp_path, monkeypatch) -> None:
    path = tmp_path / "lessons.md"
    monkeypatch.setattr(lessons, "LESSONS_FILE", path)
    monkeypatch.setattr(lessons.load, "__defaults__", (path,))
    monkeypatch.setattr(lessons.prompt_section, "__defaults__", (path,))
    monkeypatch.setattr(lessons.learn, "__defaults__", (path,))
    broken = GOOD.replace('base.add("deck", Box((WIDTH, DEPTH, BASE_T)), xyz=(0, 0, BASE_T / 2))',
                          'base.add("deck", Box((WIDTH, DEPTH, BASE_T)).translate(0, 0, 0))')
    assert broken != GOOD
    budget = Budget(max_cost_usd=None, max_calls=None)
    script = [write(broken), COMPILE, write(GOOD), COMPILE, DONE]
    off = run("a laptop", ScriptedModel(list(script), budget=budget), budget,
              run_dir=tmp_path / "off")
    assert off.status == "success" and not path.exists()  # off: nothing learned
    on_model = ScriptedModel(list(script), budget=budget)
    on = run("a laptop", on_model, budget, run_dir=tmp_path / "on", lessons=True)
    assert on.status == "success"
    learned = lessons.load(path)
    assert learned == ["AttributeError: 'Box' object has no attribute 'translate' (check the "
                       "SDK quickstart for the right method or argument)"]
    again = ScriptedModel(list(script), budget=budget)
    run("a laptop", again, budget, run_dir=tmp_path / "again", lessons=True)
    system = again.requests[0][0][0].content
    assert "## Pitfalls seen in earlier runs" in system and "'translate'" in system
    assert "Pitfalls" not in on_model.requests[0][0][0].content  # learned after that run
    assert len(lessons.load(path)) == 1  # not duplicated


def test_without_a_policy_the_full_history_is_sent(tmp_path) -> None:
    budget = Budget(max_cost_usd=None, max_calls=None)
    model = ScriptedModel([write(GOOD), COMPILE, READ_MAIN, DONE], budget=budget)
    result = run("a laptop", model, budget, run_dir=tmp_path / "run")
    assert result.status == "success" and result.record["context_policy"] is None
    assert not any(isinstance(m, User) and m.content.startswith("<checkpoint")
                   for m in sent(model, 3))


@pytest.mark.parametrize("path", ["docs/sdk/quickstart.md", "./docs/sdk/quickstart.md",
                                  "docs\\sdk\\quickstart.md"])
def test_doc_note_path_forms(path) -> None:
    assert ContextPolicy().doc_note(path) is not None
    assert ContextPolicy(ContextConfig(doc_notes=False)).doc_note(path) is None
    assert ContextPolicy().doc_note("main.py") is None
    _ = Path  # keep the import for type hints in helpers


# ---- regression: an empty reply cut off while reasoning (live run, step 9) --------------------


def test_an_empty_cut_off_reply_is_not_sent_back_empty(tmp_path) -> None:
    cut = {"text": "", "finish_reason": "length", "reasoning": "a very long plan ..."}
    result, model, _ = go(tmp_path, [cut, write(GOOD), COMPILE, DONE])
    assert result.status == "success"
    assert [m["reminder"] for m in history(result) if m.get("reminder")] == ["reply_cut_off"]
    reminder = sent(model, 1)[-1]
    assert reminder.content.startswith("<reply_cut_off>")
    for messages, _tools in model.requests[1:]:
        for item in to_chat(messages, "current_turn"):
            if item["role"] == "assistant":  # never an assistant message with nothing set
                assert item.get("content") or item.get("tool_calls") or \
                    item.get("reasoning_content")
