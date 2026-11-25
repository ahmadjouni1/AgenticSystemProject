"""Step 13 wrap-up: two loop fixes from synth_v3, tested with scripted models only.

Implemented, not yet measured live. synth_v3's two pump runs each had a skeleton build that
passed every check (its stroke verified), then replaced main.py with one big rewrite that
failed, and ended ``time_exceeded`` with the passing build unused.

- **finish on best build:** when the time runs out without a fresh passing compile, end on the
  latest passing build as a success, labelled "finished on best build".
- **edits after a pass:** once a build has passed, a whole-file write of main.py is refused
  with a hint to use edit.
"""

from __future__ import annotations

import json

import pytest

import agent.loop as loop
from agent.loop import FINISHED_ON_BEST, LoopFixes, run, saved_options
from llm import Budget
from test_loop_10b import BAD, COMPILE, DETAIL, DONE, GOOD, TimedModel, write

BREAK = {"tool_calls": [{"name": "edit", "arguments": {"path": "main.py", "edits": [
    {"old_text": "limits=(0.0, math.radians(135))", "new_text": "limits=(0.5, 1.0)"}]}}]}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(loop.time, "time", fake)
    return fake


def go(tmp_path, replies, seconds, clock, **kwargs):
    budget = Budget(max_cost_usd=None, max_calls=None)
    model = TimedModel(replies, seconds, clock, budget=budget)
    result = run("a laptop", model, budget, run_dir=tmp_path / "run", **kwargs)
    return result, model


def tool_results(result) -> list[dict]:
    lines = (result.run_dir / "conversation.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(m["content"]) for m in map(json.loads, lines)
            if m.get("role") == "tool"]


# ---- finish on best build ----------------------------------------------------------------------


def test_time_up_after_a_failed_change_finishes_on_the_passing_build(tmp_path, clock) -> None:
    # 600 s per call: write, compile (passes), an edit that breaks it, compile (fails); the
    # time is up at the next turn.
    result, _model = go(tmp_path, [write(GOOD), COMPILE, BREAK, COMPILE, DONE], [600], clock,
                        max_time_s=1500, fix_reserve_s=0)
    assert result.status == "success", result.reason
    assert result.reason.startswith(f"{FINISHED_ON_BEST} v0000")
    assert result.record["finished_on_best_build"] is True
    assert result.record["time_limit"]["outcome"] == "finished_on_best_build"
    assert result.best == {"version": "v0000", "passed_checks": True, "path": "result/v0000"}
    assert result.record["options"]["fixes"] == {"finish_on_best": True,
                                                 "edits_after_pass": True}


def test_time_up_without_a_passing_build_is_still_time_exceeded(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(BAD), COMPILE, write(BAD), DONE], [600], clock,
                        max_time_s=1500, fix_reserve_s=0)
    assert result.status == "time_exceeded"
    assert result.record["finished_on_best_build"] is False


def test_a_fresh_passing_compile_still_gets_its_final_turn(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(GOOD), COMPILE, DONE, DONE], [800], clock,
                        max_time_s=1500)
    assert result.status == "success" and not result.reason.startswith(FINISHED_ON_BEST)
    assert result.record["finished_on_best_build"] is False


def test_with_the_fix_off_the_same_run_is_time_exceeded(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(GOOD), COMPILE, BREAK, COMPILE, DONE], [600], clock,
                        max_time_s=1500, fix_reserve_s=0,
                        fixes=LoopFixes(finish_on_best=False))
    assert result.status == "time_exceeded"


# ---- edits after a pass ----------------------------------------------------------------------


def test_a_whole_file_write_after_a_pass_is_refused(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(GOOD), COMPILE, write(DETAIL), DONE], [10], clock)
    refused = tool_results(result)[2]
    assert refused["error_kind"] == "write_refused" and "v0000" in refused["message"]
    assert "edit" in refused["hint"]
    assert (result.run_dir / "workspace" / "main.py").read_text(encoding="utf-8") == GOOD
    assert result.status == "success"  # the passing build is untouched and fresh


def test_rewrites_are_allowed_until_a_build_passes(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(BAD), COMPILE, write(GOOD), COMPILE, DONE], [10],
                        clock)
    assert result.status == "success"
    assert all(r.get("error_kind") != "write_refused" for r in tool_results(result))


def test_other_files_may_still_be_written_after_a_pass(tmp_path, clock) -> None:
    notes = {"tool_calls": [{"name": "write", "arguments": {"path": "notes.md",
                                                           "content": "sizes"}}]}
    result, _model = go(tmp_path, [write(GOOD), COMPILE, notes, DONE], [10], clock)
    assert all(r.get("error_kind") != "write_refused" for r in tool_results(result))


def test_rewrites_after_a_pass_are_allowed_with_the_fix_off(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(GOOD), COMPILE, write(DETAIL), COMPILE, DONE], [10],
                        clock, fixes=LoopFixes(edits_after_pass=False))
    assert all(r.get("error_kind") != "write_refused" for r in tool_results(result))
    assert result.best["version"] == "v0001"


def test_runs_recorded_before_the_fixes_resume_without_them() -> None:
    assert saved_options({"prompt": "x"})["fixes"] == {"finish_on_best": False,
                                                       "edits_after_pass": False}
