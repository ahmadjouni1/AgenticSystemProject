"""The LLM client, without network: messages, cost, budget, retries, logging, ScriptedModel."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx2
import openai
import pytest

from llm import (UNKNOWN, Assistant, Budget, BudgetExceeded, CallLog, LLMError, ProviderError,
                 ScriptedModel, System, Tool, ToolCall, ToolResult, Usage, User, config, cost,
                 get_model)
from llm.config import PROVIDERS, RetryPolicy
from llm.messages import (MessageError, from_chat, to_chat, tools_from_chat, tools_to_chat,
                          validate)
from llm.openai_chat import OpenAIChatModel, usage_from_chat

ADD = Tool("add", "Add two integers.", {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"], "additionalProperties": False})
WEEKDAY_PEAK = datetime(2026, 11, 23, 2, 30, tzinfo=UTC)  # a Monday, 02:30 UTC
WEEKDAY_OFF = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
SATURDAY_PEAK_HOUR = datetime(2026, 11, 28, 2, 30, tzinfo=UTC)


def tool_conversation() -> list:
    return [
        System("You are a calculator."),
        User("What is 2 + 3?"),
        Assistant("", (ToolCall("call_1", "add", {"a": 2, "b": 3}, '{"a": 2, "b": 3}'),),
                  reasoning="I should use the add tool."),
        ToolResult("call_1", "add", "5"),
        Assistant("2 + 3 = 5."),
        User("And 4 + 4?"),
        Assistant("", (ToolCall("call_2", "add", {"a": 4, "b": 4}, '{"a":4,"b":4}'),),
                  reasoning="Again the tool."),
        ToolResult("call_2", "add", "8"),
    ]


# ---- message conversion -------------------------------------------------------------------------


def test_messages_convert_to_chat_and_back_with_tool_calls() -> None:
    messages = tool_conversation()
    chat = to_chat(messages)
    assert chat[0] == {"role": "system", "content": "You are a calculator."}
    assert chat[2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}]}
    assert chat[3] == {"role": "tool", "tool_call_id": "call_1", "content": "5"}
    assert all("reasoning_content" not in m for m in chat)  # OpenAI: never sent back
    back = from_chat(chat)
    assert [type(m) for m in back] == [type(m) for m in messages]
    assert back[2].tool_calls == messages[2].tool_calls
    assert back[3] == messages[3] and back[4] == messages[4]
    # The model's exact argument string is sent back unchanged (a stable cache prefix).
    assert to_chat(back)[6]["tool_calls"][0]["function"]["arguments"] == '{"a":4,"b":4}'


def test_reasoning_is_sent_back_only_for_the_current_tool_loop() -> None:
    chat = to_chat(tool_conversation(), reasoning="current_turn")
    with_reasoning = [i for i, m in enumerate(chat) if "reasoning_content" in m]
    assert with_reasoning == [6]  # after the last user message only
    assert from_chat(chat)[6].reasoning == "Again the tool."


def test_tool_call_arguments_that_are_not_a_json_object() -> None:
    bad = ToolCall.parse("c", "add", '{"a": 2,')
    assert bad.arguments == {} and "not valid JSON" in bad.error
    assert bad.arguments_json() == '{"a": 2,'  # sent back as the model wrote it
    assert ToolCall.parse("c", "add", "[1, 2]").error == "arguments must be a JSON object"
    assert ToolCall.parse("c", "add", "").arguments == {}


def test_tool_schemas_convert_both_ways() -> None:
    chat = tools_to_chat([ADD])
    assert chat == [{"type": "function", "function": {
        "name": "add", "description": "Add two integers.", "parameters": ADD.parameters}}]
    assert tools_from_chat(chat) == [ADD]


@pytest.mark.parametrize(("messages", "error"), [
    ([], "at least one message"),
    ([System("a")], "needs a user message"),
    ([User("a"), System("b")], "must all come first"),
    ([System("s"), User("u"), ToolResult("x", "add", "1")], "not an open tool call"),
    ([System("s"), User("u"), Assistant("", (ToolCall("c", "add"),)), User("again")],
     "were not answered"),
])
def test_conversation_rules(messages, error) -> None:
    with pytest.raises(MessageError, match=error):
        validate(messages)


# ---- usage and cost -----------------------------------------------------------------------------


def test_usage_from_both_providers() -> None:
    openai_usage = usage_from_chat({
        "prompt_tokens": 2000, "completion_tokens": 300, "total_tokens": 2300,
        "prompt_tokens_details": {"cached_tokens": 1536, "cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 200}})
    assert openai_usage == Usage(2000, 1536, 0, 300, 200)
    assert openai_usage.cached_share == pytest.approx(0.768)
    deepseek_usage = usage_from_chat({  # DeepSeek: the hit count under its own name too
        "prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18,
        "prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 3})
    assert deepseek_usage == Usage(7, 4, 0, 11, 0)
    assert usage_from_chat(None) == Usage()


def test_cost_with_cached_tokens_and_cache_writes() -> None:
    # gpt-6.2-astra: $8 input, $0.40 cached, $40 output, $10 cache write per 1M tokens.
    usage = Usage(input_tokens=10_000, cached_input_tokens=8_000, output_tokens=1_000,
                  reasoning_tokens=600)
    expected = (2_000 * 8.00 + 8_000 * 0.40 + 1_000 * 40.00) / 1e6
    assert cost("gpt-6.2-astra", usage) == pytest.approx(expected)  # = $0.0592
    assert cost("gpt-6.2-astra", Usage(10_000, 0, 0, 1_000)) == pytest.approx(0.12)
    written = Usage(input_tokens=10_000, cache_write_tokens=6_000, output_tokens=0)
    assert cost("gpt-6.2-astra", written) == pytest.approx((4_000 * 8 + 6_000 * 10) / 1e6)
    # Reasoning tokens are part of the output tokens: they do not change the price.
    assert cost("gpt-6.2-astra", Usage(0, 0, 0, 1_000, 900)) == pytest.approx(0.04)


def test_long_context_price() -> None:
    long = Usage(input_tokens=300_000, output_tokens=0)
    assert cost("gpt-6.2-astra", long) == pytest.approx(300_000 * 16.00 / 1e6)
    assert cost("gpt-6.1-terra", long) == UNKNOWN  # its long-context rate is not in the table


def test_deepseek_off_peak_is_half_price() -> None:
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=100_000)
    peak = (500_000 * 0.30 + 500_000 * 0.006 + 100_000 * 1.20) / 1e6
    assert cost("deepseek-flash", usage, WEEKDAY_PEAK) == pytest.approx(peak)
    assert cost("deepseek-flash", usage, WEEKDAY_OFF) == pytest.approx(peak / 2)
    assert cost("deepseek-flash", usage, SATURDAY_PEAK_HOUR) == pytest.approx(peak / 2)


def test_unknown_model_costs_unknown_not_zero(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="llm"):
        assert cost("gpt-unreleased", Usage(1000, 0, 0, 100)) == UNKNOWN
    assert "no price for model 'gpt-unreleased'" in caplog.text


def test_every_role_has_a_provider_and_a_price() -> None:
    for name, role in config.ROLES.items():
        assert role.provider in config.PROVIDERS, name
        assert role.model in config.PRICES, name
    assert config.ROLES["builder"].model == "deepseek-v4-pro"  # development setting
    assert config.ROLES["cheap"].model == "deepseek-flash"
    assert config.DEFAULT_MAX_COST_USD == 1.00 and config.DEFAULT_MAX_CALLS == 100
    assert {"gpt-6.2-astra", "gpt-6.2-sol"} <= set(config.PRICES)  # kept for evaluation
    for price in config.PRICES.values():
        assert price.cached_input < price.input < price.output


# ---- a fake transport ---------------------------------------------------------------------------


def chat_response(text="", tool_calls=(), usage=None, model="gpt-6.2-astra",
                  finish="stop", reasoning=None) -> dict:
    message = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls, 1)]
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"id": "x", "model": model, "choices": [
        {"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20,
                           "prompt_tokens_details": {"cached_tokens": 0}}}


REQUEST = httpx2.Request("POST", "https://api.example/v1/chat/completions")


def status_error(cls, status: int, headers: dict | None = None):
    response = httpx2.Response(status, headers=headers or {}, request=REQUEST)
    return cls(f"error {status}", response=response, body=None)


class FakeTransport:
    """Stands in for client.chat.completions.create: raises or returns, in order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []

    def __call__(self, **body):
        self.requests.append(body)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_model(transport, provider="openai", model="gpt-6.2-astra", **kwargs):
    sleeps: list[float] = []
    instance = OpenAIChatModel(PROVIDERS[provider], model, create=transport,
                               sleep=sleeps.append, jitter=lambda: 1.0, **kwargs)
    return instance, sleeps


CONVERSATION = [System("You are a calculator."), User("What is 2 + 3?")]


# ---- the adapter --------------------------------------------------------------------------------


def test_openai_request_and_tool_call_reply() -> None:
    transport = FakeTransport(chat_response(tool_calls=[("add", {"a": 2, "b": 3})],
                                            finish="tool_calls", usage={
        "prompt_tokens": 1200, "completion_tokens": 50,
        "prompt_tokens_details": {"cached_tokens": 1024},
        "completion_tokens_details": {"reasoning_tokens": 30}}))
    model, _ = make_model(transport, effort="high", max_output_tokens=4000, timeout_s=42)
    reply = model.query(CONVERSATION, [ADD])
    body = transport.requests[0]
    assert body["model"] == "gpt-6.2-astra" and body["max_completion_tokens"] == 4000
    assert "max_tokens" not in body and body["reasoning_effort"] == "high"
    assert body["timeout"] == 42  # the per-call timeout reaches the transport
    assert body["messages"][0]["role"] == "system" and body["tools"] == tools_to_chat([ADD])
    assert reply.tool_calls == (ToolCall("call_1", "add", {"a": 2, "b": 3}, '{"a": 2, "b": 3}'),)
    assert reply.usage == Usage(1200, 1024, 0, 50, 30)
    assert reply.cost == pytest.approx((176 * 8 + 1024 * 0.4 + 50 * 40) / 1e6)
    assert reply.finish_reason == "tool_calls" and reply.raw["id"] == "x"
    assert reply.message() == Assistant("", reply.tool_calls)


def test_deepseek_uses_its_token_parameter_and_echoes_reasoning() -> None:
    transport = FakeTransport(
        chat_response(tool_calls=[("add", {"a": 2, "b": 3})], model="deepseek-flash",
                      reasoning="Use add."),
        chat_response("5", model="deepseek-flash"))
    model, _ = make_model(transport, provider="deepseek", model="deepseek-flash")
    messages = list(CONVERSATION)
    first = model.query(messages, [ADD])
    assert first.reasoning == "Use add."
    messages += [first.message(), ToolResult(first.tool_calls[0].id, "add", "5")]
    model.query(messages, [ADD])
    body = transport.requests[1]
    assert body["max_tokens"] == 32_000 and "max_completion_tokens" not in body
    assert body["messages"][2]["reasoning_content"] == "Use add."
    # The prefix (system + tools) is byte-identical between the two calls.
    assert body["messages"][0] == transport.requests[0]["messages"][0]
    assert body["tools"] is transport.requests[0]["tools"]


def test_keys_come_only_from_their_environment_variable(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key-1234")
    with pytest.raises(LLMError, match="set the DEEPSEEK_API_KEY environment variable"):
        OpenAIChatModel(PROVIDERS["deepseek"], "deepseek-flash")  # no OPENAI_API_KEY fallback
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key-5678")
    model = get_model("cheap")
    assert model.provider == "deepseek" and model.model == "deepseek-flash"


# ---- retries ------------------------------------------------------------------------------------


def test_rate_limits_and_5xx_are_retried_with_exponential_backoff() -> None:
    transport = FakeTransport(status_error(openai.RateLimitError, 429),
                              status_error(openai.InternalServerError, 500),
                              status_error(openai.InternalServerError, 503),
                              chat_response("ok"))
    model, sleeps = make_model(transport)
    reply = model.query(CONVERSATION)
    assert reply.text == "ok" and reply.retries == 3
    assert sleeps == [1.0, 2.0, 4.0]  # base 1 s, doubling (jitter fixed at its maximum)


def test_network_errors_and_timeouts_are_retried() -> None:
    transport = FakeTransport(openai.APIConnectionError(request=REQUEST),
                              openai.APITimeoutError(request=REQUEST), chat_response("ok"))
    model, sleeps = make_model(transport)
    assert model.query(CONVERSATION).retries == 2 and len(sleeps) == 2


def test_retry_after_is_honoured_and_capped() -> None:
    transport = FakeTransport(status_error(openai.RateLimitError, 429, {"retry-after": "7"}),
                              status_error(openai.RateLimitError, 429, {"retry-after": "999"}),
                              chat_response("ok"))
    model, sleeps = make_model(transport)
    model.query(CONVERSATION)
    assert sleeps == [7.0, 30.0]


def test_at_most_four_retries() -> None:
    transport = FakeTransport(*[status_error(openai.RateLimitError, 429) for _ in range(5)])
    model, sleeps = make_model(transport)
    with pytest.raises(ProviderError, match="failed after 5 attempts") as caught:
        model.query(CONVERSATION)
    assert caught.value.retryable and caught.value.status == 429 and caught.value.attempts == 5
    assert len(transport.requests) == 5 and sleeps == [1.0, 2.0, 4.0, 8.0]


@pytest.mark.parametrize(("cls", "status"), [
    (openai.BadRequestError, 400), (openai.AuthenticationError, 401),
    (openai.NotFoundError, 404), (openai.UnprocessableEntityError, 422)])
def test_bad_requests_are_not_retried(cls, status) -> None:
    transport = FakeTransport(status_error(cls, status), chat_response("never reached"))
    model, sleeps = make_model(transport)
    with pytest.raises(ProviderError, match=f"refused the request \\(HTTP {status}\\)"):
        model.query(CONVERSATION)
    assert len(transport.requests) == 1 and sleeps == []


def test_custom_retry_policy() -> None:
    transport = FakeTransport(status_error(openai.RateLimitError, 429), chat_response("ok"))
    model, sleeps = make_model(transport, retry=RetryPolicy(max_retries=1, base_delay_s=0.25))
    assert model.query(CONVERSATION).retries == 1 and sleeps == [0.25]


# ---- budget -------------------------------------------------------------------------------------


def test_call_cap_refuses_the_next_call_before_it_is_made() -> None:
    budget = Budget(max_cost_usd=None, max_calls=2)
    transport = FakeTransport(chat_response("1"), chat_response("2"), chat_response("3"))
    model, _ = make_model(transport, budget=budget)
    model.query(CONVERSATION)
    model.query(CONVERSATION)
    with pytest.raises(BudgetExceeded, match="call cap reached: 2 of 2"):
        model.query(CONVERSATION)
    assert len(transport.requests) == 2


def test_cost_cap_stops_the_run_after_the_call_that_crosses_it(tmp_path) -> None:
    budget = Budget(max_cost_usd=0.10, max_calls=None)
    big = {"prompt_tokens": 5_000, "completion_tokens": 1_000}  # $0.08 on gpt-6.2-astra
    transport = FakeTransport(chat_response("a", usage=big), chat_response("b", usage=big),
                              chat_response("c", usage=big))
    model, _ = make_model(transport, budget=budget, log=CallLog(tmp_path))
    assert model.query(CONVERSATION).cost == pytest.approx(0.08)
    with pytest.raises(BudgetExceeded, match=r"cost cap exceeded: \$0.1600 spent of \$0.10") as e:
        model.query(CONVERSATION)
    assert e.value.reply.text == "b"  # the paid reply is not lost
    with pytest.raises(BudgetExceeded, match="cost cap reached"):
        model.query(CONVERSATION)
    assert len(transport.requests) == 2
    assert [r["budget"]["spent_usd"] for r in CallLog(tmp_path).records()] == [0.08, 0.16]


def test_budget_is_shared_by_the_models_of_a_run() -> None:
    budget = Budget(max_cost_usd=None, max_calls=2)
    a = ScriptedModel(["a"], budget=budget)
    b = ScriptedModel(["b", "c"], budget=budget)
    a.query(CONVERSATION)
    b.query(CONVERSATION)
    with pytest.raises(BudgetExceeded):
        b.query(CONVERSATION)


def test_unknown_cost_counts_toward_calls_and_warns(caplog) -> None:
    budget = Budget(max_cost_usd=1.0, max_calls=10)
    model, _ = make_model(FakeTransport(chat_response("x")), model="gpt-unreleased",
                          budget=budget)
    with caplog.at_level(logging.WARNING, logger="llm"):
        reply = model.query(CONVERSATION)
    assert reply.cost == UNKNOWN
    assert budget.calls == 1 and budget.unknown_cost_calls == 1 and budget.spent_usd == 0
    assert "cost cap cannot count it" in caplog.text


# ---- call log -----------------------------------------------------------------------------------


def test_every_call_is_logged_without_keys(tmp_path, monkeypatch) -> None:
    secret = "sk-proj-FAKEKEYFORTESTS000000"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    transport = FakeTransport(
        status_error(openai.RateLimitError, 429),
        chat_response("ok", usage={"prompt_tokens": 2000, "completion_tokens": 10,
                                   "prompt_tokens_details": {"cached_tokens": 1500}}),
        status_error(openai.AuthenticationError, 401))
    transport.outcomes[2] = openai.AuthenticationError(
        f"Incorrect API key provided: {secret}",
        response=httpx2.Response(401, request=REQUEST), body=None)
    model = OpenAIChatModel(PROVIDERS["openai"], "gpt-6.2-astra", sleep=lambda s: None,
                            log=CallLog(tmp_path / "runs" / "r1"), role="builder")
    model._create = transport  # a real client was built from the env key; swap its transport
    model.query(CONVERSATION, [ADD])
    with pytest.raises(ProviderError) as caught:
        model.query(CONVERSATION, [ADD])
    assert secret not in str(caught.value)
    path = tmp_path / "runs" / "r1" / "llm_calls.jsonl"
    text = path.read_text(encoding="utf-8")
    assert secret not in text and "FAKEKEYFORTESTS" not in text
    ok, error = [json.loads(line) for line in text.splitlines()]
    assert ok["status"] == "ok" and ok["provider"] == "openai" and ok["model"] == "gpt-6.2-astra"
    assert ok["role"] == "builder" and ok["retries"] == 1 and ok["latency_s"] >= 0
    assert ok["usage"]["cached_input_tokens"] == 1500 and ok["usage"]["cached_share"] == 0.75
    assert ok["cost"] == pytest.approx((500 * 8 + 1500 * 0.4 + 10 * 40) / 1e6)
    assert ok["prefix"] == error["prefix"]  # same system prompt and tools: same cache prefix
    assert error["status"] == "error" and error["http_status"] == 401 and error["cost"] == 0.0
    summary = CallLog(tmp_path / "runs" / "r1").summary()
    assert summary["calls"] == 1 and summary["errors"] == 1
    assert summary["usage"]["cached_share"] == 0.75


# ---- ScriptedModel ------------------------------------------------------------------------------


def test_scripted_model_replays_a_tool_loop(tmp_path) -> None:
    model = ScriptedModel([
        {"tool_calls": [{"name": "add", "arguments": {"a": 2, "b": 3}}],
         "usage": {"input_tokens": 1000, "output_tokens": 20}},
        "2 + 3 = 5.",
    ], model="gpt-6.2-astra", log=CallLog(tmp_path))
    messages = list(CONVERSATION)
    first = model.query(messages, [ADD])
    assert first.tool_calls[0].name == "add" and first.tool_calls[0].arguments == {"a": 2, "b": 3}
    assert first.finish_reason == "tool_calls"
    assert first.cost == pytest.approx((1000 * 8 + 20 * 40) / 1e6)  # priced like the real model
    messages += [first.message(), ToolResult(first.tool_calls[0].id, "add", "5")]
    second = model.query(messages, [ADD])
    assert second.text == "2 + 3 = 5." and second.cost == 0.0
    assert len(model.requests) == 2 and model.requests[1][0][-1].content == "5"
    assert len(CallLog(tmp_path).records()) == 2
    with pytest.raises(LLMError, match="no reply left"):
        model.query(messages + [Assistant("ok"), User("more?")], [ADD])


def test_scripted_model_checks_its_script() -> None:
    with pytest.raises(LLMError, match="not one of the tools offered"):
        ScriptedModel([{"tool_calls": [{"name": "rm"}]}]).query(CONVERSATION, [ADD])
    with pytest.raises(MessageError):
        ScriptedModel(["x"]).query([User("u"), System("late")])


def test_scripted_model_is_a_model() -> None:
    model: object = ScriptedModel(["hi"])
    assert all(hasattr(model, name) for name in ("provider", "model", "query"))
    assert isinstance(SimpleNamespace(**vars(model.query(CONVERSATION))).usage, Usage)


# ---- through the real openai package, over a mock HTTP transport ---------------------------------


def test_round_trip_through_the_openai_sdk_without_network(tmp_path) -> None:
    """The real SDK serializes our request and parses the reply; only HTTP is mocked."""
    seen: list[dict] = []

    def handler(request):
        seen.append({"url": str(request.url), "auth": request.headers.get("authorization"),
                     "body": json.loads(request.content)})
        if len(seen) == 1:
            return httpx2.Response(429, headers={"retry-after": "0"},
                                   json={"error": {"message": "slow down"}})
        return httpx2.Response(200, json=chat_response(
            tool_calls=[("add", {"a": 2, "b": 3})], model="deepseek-flash", finish="tool_calls",
            reasoning="Use add.", usage={"prompt_tokens": 900, "completion_tokens": 40,
                                         "prompt_tokens_details": {"cached_tokens": 640},
                                         "prompt_cache_hit_tokens": 640,
                                         "prompt_cache_miss_tokens": 260,
                                         "completion_tokens_details": {"reasoning_tokens": 25}}))

    client = openai.OpenAI(api_key="ds-test-key-0000", base_url="https://api.deepseek.com",
                           max_retries=0, http_client=httpx2.Client(
                               transport=httpx2.MockTransport(handler)))
    model = OpenAIChatModel(PROVIDERS["deepseek"], "deepseek-flash",
                            create=client.chat.completions.create, sleep=lambda s: None,
                            log=CallLog(tmp_path))
    reply = model.query(CONVERSATION, [ADD])
    assert len(seen) == 2 and reply.retries == 1
    assert seen[1]["url"] == "https://api.deepseek.com/chat/completions"
    assert seen[1]["auth"] == "Bearer ds-test-key-0000"
    body = seen[1]["body"]
    assert body["model"] == "deepseek-flash" and body["max_tokens"] == 32_000
    assert body["tools"][0]["function"]["name"] == "add" and "timeout" not in body
    assert reply.tool_calls[0].arguments == {"a": 2, "b": 3} and reply.reasoning == "Use add."
    assert reply.usage == Usage(900, 640, 0, 40, 25)
    assert reply.raw["choices"][0]["message"]["reasoning_content"] == "Use add."
    assert "ds-test-key-0000" not in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8")


# ---- .env ---------------------------------------------------------------------------------------


def test_env_file_is_loaded_without_overriding(tmp_path, monkeypatch) -> None:
    from llm.env import load_env, parse_env

    env = tmp_path / ".env"
    env.write_text("\n".join([
        "# keys", "export DEEPSEEK_API_KEY='ds-from-file'", "OPENAI_API_KEY=sk-file",
        'QUOTED="a # not a comment"', "PLAIN=value  # comment", "not a line", "", ""]),
        encoding="utf-8")
    assert parse_env(env.read_text()) == {"DEEPSEEK_API_KEY": "ds-from-file",
                                          "OPENAI_API_KEY": "sk-file",
                                          "QUOTED": "a # not a comment", "PLAIN": "value"}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-already-set")
    for name in ("DEEPSEEK_API_KEY", "QUOTED", "PLAIN"):
        monkeypatch.delenv(name, raising=False)
    assert sorted(load_env(env)) == ["DEEPSEEK_API_KEY", "PLAIN", "QUOTED"]
    import os
    assert os.environ["OPENAI_API_KEY"] == "sk-already-set"  # the real environment wins
    assert os.environ["DEEPSEEK_API_KEY"] == "ds-from-file"
    assert load_env(tmp_path / "missing.env") == []


def test_env_file_is_git_ignored() -> None:
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(["git", "check-ignore", ".env"], cwd=root, capture_output=True,
                          text=True, check=False)
    assert done.returncode == 0 and done.stdout.strip() == ".env"


def test_empty_assistant_messages_are_not_sent() -> None:
    messages = [System("s"), User("u"), Assistant("", (), reasoning="long thought"),
                User("<reminder>"), Assistant("ok")]
    chat = to_chat(messages, reasoning="current_turn")
    assert [m["role"] for m in chat] == ["system", "user", "user", "assistant"]
    # In the current turn the reasoning is the content: it is kept.
    current = to_chat(messages[:3], reasoning="current_turn")
    assert current[-1] == {"role": "assistant", "content": "", "reasoning_content": "long thought"}
