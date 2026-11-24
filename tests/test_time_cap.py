"""The time overrun fix (step 10b follow-up): every model call is capped at the time left in
the run, as its timeout (retries included) and as an output cap. In step 10b one reply could
take 9.5 minutes, and two pump runs ended 91 s and 351 s past the 1500 s limit."""

from __future__ import annotations

import json
import re

import httpx2
import openai
import pytest

import agent.loop as loop
from agent.loop import CALL_OVERHEAD_S, DEFAULT_TOKENS_PER_S, FINAL_TURN_S, PROMPT_SETS, run
from llm import Budget, CallLog, CallTimeLimit, ScriptedModel, System, User
from llm.config import PROVIDERS
from llm.openai_chat import MIN_ATTEMPT_S, OpenAIChatModel

LAPTOP_DOC = PROMPT_SETS["v0"].quickstart
GOOD = re.findall(r"```python\n(.*?)```", LAPTOP_DOC.read_text(encoding="utf-8"), re.S)[-1]
BAD = GOOD.replace("limits=(0.0, math.radians(135))", "limits=(0.5, 1.0)")
COMPILE = {"tool_calls": [{"name": "compile", "arguments": {}}]}
READ_MAIN = {"tool_calls": [{"name": "read", "arguments": {"path": "main.py"}}]}
DONE = "Done: a base and a lid on a hinge."
REQUEST = httpx2.Request("POST", "https://api.example/v1/chat/completions")


def write(content: str) -> dict:
    return {"tool_calls": [{"name": "write", "arguments": {"path": "main.py",
                                                           "content": content}}]}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class TimedModel(ScriptedModel):
    """Each call takes ``seconds[i]`` of fake time; ``CUT`` in the replies raises
    CallTimeLimit after using up the call's whole allowance."""

    CUT = object()

    def __init__(self, replies, seconds, clock: Clock, max_output_tokens=None, **kwargs):
        super().__init__(replies, **kwargs)
        self.seconds, self.clock = list(seconds), clock
        if max_output_tokens is not None:
            self.max_output_tokens = max_output_tokens

    def _complete(self, messages, tools, options):
        if self.replies and self.replies[0] is self.CUT:
            self.replies.pop(0)
            self.requests.append((list(messages), tools))
            self.timeouts.append(options.get("timeout_s"))
            self.options.append({k: v for k, v in options.items() if k != "timeout_s"})
            self.clock.now += options["timeout_s"]
            raise CallTimeLimit("scripted: the call's time ran out")
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
    model = TimedModel(replies, seconds, clock, budget=budget,
                       max_output_tokens=kwargs.pop("max_output_tokens", None))
    result = run("a laptop", model, budget, run_dir=tmp_path / "run", **kwargs)
    return result, model


# ---- the loop -------------------------------------------------------------------------------


def test_every_call_gets_the_time_left_as_its_timeout(tmp_path, clock) -> None:
    result, model = go(tmp_path, [write(GOOD), COMPILE, DONE], [100], clock, max_time_s=1500)
    assert result.status == "success"
    assert model.timeouts == [1500, 1400, 1300]
    assert [t["timeout_s"] for t in result.record["per_turn"]] == [1500, 1400, 1300]


def test_without_a_time_limit_calls_have_no_timeout(tmp_path, clock) -> None:
    _result, model = go(tmp_path, [write(GOOD), COMPILE, DONE], [100], clock, max_time_s=None)
    assert model.timeouts == [None, None, None]


def test_the_output_cap_shrinks_to_what_fits_in_the_time_left(tmp_path, clock) -> None:
    # The first call takes 1300 s: 200 s are left for the second one.
    result, model = go(tmp_path, [write(GOOD), COMPILE, DONE], [1300, 10], clock,
                       max_time_s=1500, max_output_tokens=32_000)
    cap = int(DEFAULT_TOKENS_PER_S * (200 - CALL_OVERHEAD_S) * 0.9)
    assert model.options[0] == {}  # 1500 s left: 32k tokens fit, the default stays
    assert model.options[1] == {"max_output_tokens": cap}
    assert result.record["per_turn"][1]["output_cap"] == cap


def test_a_call_cut_off_at_the_limit_ends_the_run_as_time_exceeded(tmp_path, clock) -> None:
    result, model = go(tmp_path, [write(BAD), COMPILE, TimedModel.CUT], [100], clock,
                       max_time_s=1500, fix_reserve_s=0)
    assert result.status == "time_exceeded"
    assert "cut off at the time limit after 1500 s" in result.reason
    assert result.record["time_limit"]["outcome"] == "call_cut"
    assert result.record["duration_s"] <= 1500  # no overrun
    assert result.record["per_turn"][-1]["timed_out"] is True
    assert model.timeouts[-1] == 1300


def test_a_call_cut_off_after_a_passing_build_gets_the_final_turn(tmp_path, clock) -> None:
    result, model = go(tmp_path, [write(GOOD), COMPILE, TimedModel.CUT, DONE], [100], clock,
                       max_time_s=1500)
    assert result.status == "success" and result.final_text == DONE
    assert result.record["time_limit"]["outcome"] == "final_turn"
    assert model.timeouts[-1] == FINAL_TURN_S  # the final turn has its own allowance
    assert "<time_up>" in model.requests[-1][0][-1].content


def test_a_final_turn_cut_off_still_succeeds_with_the_passing_build(tmp_path, clock) -> None:
    result, _model = go(tmp_path, [write(GOOD), COMPILE, TimedModel.CUT, TimedModel.CUT],
                        [100], clock, max_time_s=1500)
    assert result.status == "success" and "no final message" in result.reason
    assert result.best["passed_checks"]


# ---- the adapter: the timeout covers every attempt ------------------------------------------


class Transport:
    """client.chat.completions.create: each outcome may advance the fake clock first."""

    def __init__(self, clock: Clock, *outcomes):
        self.clock, self.outcomes, self.requests = clock, list(outcomes), []

    def __call__(self, **body):
        self.requests.append(body)
        seconds, outcome = self.outcomes.pop(0)
        self.clock.now += seconds
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def reply(text="ok") -> dict:
    return {"id": "x", "model": "deepseek-v4-pro", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": text},
         "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def adapter(transport, clock, tmp_path=None):
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    model = OpenAIChatModel(PROVIDERS["deepseek"], "deepseek-v4-pro", create=transport,
                            sleep=sleep, jitter=lambda: 1.0, clock=clock, timeout_s=300,
                            log=CallLog(tmp_path) if tmp_path else None)
    return model, sleeps


MESSAGES = [System("s"), User("u")]


def test_each_attempt_gets_at_most_the_time_left() -> None:
    clock = Clock()
    transport = Transport(clock, (0, reply()), (0, reply()))
    model, _ = adapter(transport, clock)
    model.query(MESSAGES, timeout_s=120)
    assert transport.requests[0]["timeout"] == 120  # min(300 per attempt, 120 left)
    model.query(MESSAGES)
    assert transport.requests[1]["timeout"] == 300  # no allowance: the per-attempt timeout


def test_a_retry_gets_what_is_left_and_backoff_never_passes_the_deadline() -> None:
    clock = Clock()
    transport = Transport(clock, (50, openai.APIConnectionError(request=REQUEST)), (0, reply()))
    model, sleeps = adapter(transport, clock)
    model.query(MESSAGES, timeout_s=60)
    assert transport.requests[0]["timeout"] == 60
    assert sleeps[0] <= 60 - 50 - MIN_ATTEMPT_S  # the backoff leaves time for the attempt
    assert transport.requests[1]["timeout"] == pytest.approx(60 - 50 - sleeps[0])


def test_an_attempt_timing_out_at_the_deadline_raises_call_time_limit(tmp_path) -> None:
    clock = Clock()
    transport = Transport(clock, (120, openai.APITimeoutError(request=REQUEST)), (0, reply()))
    model, _ = adapter(transport, clock, tmp_path)
    with pytest.raises(CallTimeLimit, match="ran out"):
        model.query(MESSAGES, timeout_s=120)
    assert len(transport.requests) == 1  # no retry after the deadline
    (entry,) = model.log.records()
    assert entry["status"] == "error" and entry["cost"] == "unknown"  # may still be billed


def test_a_timeout_with_time_left_is_retried_as_before() -> None:
    clock = Clock()
    transport = Transport(clock, (30, openai.APITimeoutError(request=REQUEST)), (0, reply("y")))
    model, _ = adapter(transport, clock)
    assert model.query(MESSAGES, timeout_s=120).text == "y"


def test_scripted_model_keeps_timeouts_apart_from_options() -> None:
    model = ScriptedModel(["a", "b"])
    model.query(MESSAGES, effort="low", timeout_s=30)
    model.query(MESSAGES)
    assert model.options == [{"effort": "low"}, {}]
    assert model.timeouts == [30, None]
    assert json.dumps(model.options)  # still plain data
