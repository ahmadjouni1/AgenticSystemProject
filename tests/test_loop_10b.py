"""Step 10b loop changes: the time limit (final turn, fix reserve) and the two-phase build.
All with ScriptedModel and a fake clock: no network."""

from __future__ import annotations

import re

import pytest

import agent.loop as loop
from agent.loop import FIX_RESERVE_S, MAX_TIME_S, PROMPT_SETS, EffortPolicy, run
from llm import Budget, ScriptedModel

LAPTOP_DOC = PROMPT_SETS["v0"].quickstart  # step 9's quickstart: a known-good laptop
GOOD = re.findall(r"```python\n(.*?)```", LAPTOP_DOC.read_text(encoding="utf-8"), re.S)[-1]
BAD = GOOD.replace("limits=(0.0, math.radians(135))", "limits=(0.5, 1.0)")  # fails validation
DETAIL = GOOD + "\n# phase 2: detail\n"
COMPILE = {"tool_calls": [{"name": "compile", "arguments": {}}]}
READ_MAIN = {"tool_calls": [{"name": "read", "arguments": {"path": "main.py"}}]}
DONE = "Done: a base and a lid on a hinge from 0 to 135 degrees."


def write(content: str) -> dict:
    return {"tool_calls": [{"name": "write", "arguments": {"path": "main.py",
                                                           "content": content}}]}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class TimedModel(ScriptedModel):
    """Each call takes ``seconds[i]`` of fake wall-clock time (the last value repeats)."""

    def __init__(self, replies, seconds, clock: Clock, **kwargs):
        super().__init__(replies, **kwargs)
        self.seconds, self.clock = list(seconds), clock

    def _complete(self, messages, tools, options):
        index = min(len(self.requests), len(self.seconds) - 1)
        self.clock.now += self.seconds[index]
        return super()._complete(messages, tools, options)


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


def last_user_text(model, k: int = -1) -> str:
    messages, _tools = model.requests[k]
    return messages[-1].content


# ---- the time limit -------------------------------------------------------------------------


def test_the_defaults_are_25_minutes_and_a_2_minute_fix_reserve() -> None:
    assert MAX_TIME_S == 25 * 60 and FIX_RESERVE_S == 120


def test_time_up_with_a_passing_build_gives_one_final_turn_and_succeeds(tmp_path, clock) -> None:
    result, model = go(tmp_path, [write(GOOD), COMPILE, READ_MAIN, DONE], [600], clock,
                       max_time_s=1500)
    assert result.status == "success", result.reason
    assert "time limit reached" in result.reason and "final message given" in result.reason
    assert "<time_up>" in last_user_text(model) and result.final_text == DONE
    assert result.record["time_limit"]["outcome"] == "final_turn"
    assert result.best["passed_checks"] and result.best["version"] == "v0000"


def test_tool_calls_in_the_final_turn_are_not_run(tmp_path, clock) -> None:
    result, model = go(tmp_path, [write(GOOD), COMPILE, READ_MAIN, READ_MAIN], [600], clock,
                       max_time_s=1500)
    assert result.status == "success" and "no final message" in result.reason
    assert result.record["tool_calls"]["read"] == 1  # the final turn's read was not run
    assert result.record["per_turn"][-1]["tools_not_run"] == ["read"]


def test_time_up_with_a_failing_compile_is_time_exceeded(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(BAD), COMPILE, READ_MAIN, DONE], [600], clock,
                        max_time_s=1500, fix_reserve_s=0)
    assert result.status == "time_exceeded" and "the limit is 1500 s" in result.reason
    assert result.record["time_limit"]["outcome"] == "stopped"


def test_time_up_after_an_edit_since_the_passing_compile_is_time_exceeded(tmp_path,
                                                                          clock) -> None:
    result, _model = go(tmp_path, [write(GOOD), COMPILE, write(DETAIL), DONE], [600], clock,
                        max_time_s=1500)
    assert result.status == "time_exceeded"  # a passing build exists, but main.py changed


def test_no_fix_cycle_starts_without_the_reserve(tmp_path, clock) -> None:
    # Calls take 700 s: the failed compile ends at 1400 s, 100 s before the limit.
    result, model = go(tmp_path, [write(BAD), COMPILE, write(GOOD), COMPILE, DONE], [700],
                       clock, max_time_s=1500)
    assert result.status == "time_exceeded" and "less than a fix cycle needs" in result.reason
    assert len(model.requests) == 2 and result.record["time_limit"]["outcome"] == "no_fix_cycle"


def test_a_fix_cycle_starts_with_enough_time_left(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(BAD), COMPILE, write(GOOD), COMPILE, DONE], [100],
                        clock, max_time_s=1500)
    assert result.status == "success" and result.record["time_limit"] is None


# ---- two-phase build ------------------------------------------------------------------------


def test_two_phase_asks_for_a_skeleton_then_for_detail(tmp_path, clock) -> None:
    result, model = go(tmp_path, [write(GOOD), COMPILE, write(DETAIL), COMPILE, DONE], [10],
                       clock, two_phase=True)
    assert result.status == "success", result.reason
    first_task = model.requests[0][0][2].content
    assert "Phase 1 (now): write a skeleton" in first_task and "a laptop" in first_task
    phase_2 = last_user_text(model, 2)  # the request after the skeleton's passing compile
    assert "<phase_2>" in phase_2 and "(v0000)" in phase_2
    two = result.record["two_phase"]
    assert two["phase"] == 2 and two["skeleton"]["version"] == "v0000"
    assert two["skeleton"]["turn"] == 2 and two["detail_compiles"] == 1
    assert result.best["version"] == "v0001"
    assert result.record["reminders"].get("phase_2") == 1


def test_two_phase_keeps_the_skeleton_as_the_fallback_build(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(GOOD), COMPILE, write(BAD), COMPILE, READ_MAIN, DONE],
                        [400], clock, two_phase=True, max_time_s=1500, fix_reserve_s=0)
    assert result.status == "time_exceeded"
    assert result.best == {"version": "v0000", "passed_checks": True, "path": "result/v0000"}
    assert result.record["two_phase"]["phase"] == 2


def test_a_failing_skeleton_stays_in_phase_1(tmp_path, clock) -> None:
    result, model = go(tmp_path, [write(BAD), COMPILE, write(GOOD), COMPILE, DONE], [10],
                       clock, two_phase=True)
    assert result.status == "success"
    assert not any("<phase_2>" in m.content for m in model.requests[2][0]
                   if hasattr(m, "content") and isinstance(m.content, str))
    assert result.record["two_phase"]["skeleton"]["turn"] == 4


def test_the_detail_turns_are_not_routine_until_phase_2_compiles(tmp_path, clock) -> None:
    result, model = go(tmp_path, [write(GOOD), COMPILE, write(DETAIL), COMPILE, DONE], [10],
                       clock, two_phase=True, effort=EffortPolicy())
    assert result.status == "success"
    routine = {"effort": "low", "max_output_tokens": 16_000}
    assert model.options == [{}, {}, {}, {}, routine]  # the detail write gets the default


def test_without_two_phase_nothing_changes(tmp_path, clock) -> None:
    result, model = go(tmp_path, [write(GOOD), COMPILE, DONE], [10], clock)
    assert result.status == "success" and result.record["two_phase"] is None
    assert "Phase 1" not in model.requests[0][0][2].content
