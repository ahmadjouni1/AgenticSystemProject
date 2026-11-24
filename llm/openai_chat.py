"""OpenAI and DeepSeek through the official ``openai`` package (Chat Completions).

DeepSeek is the same adapter with its own base_url and DEEPSEEK_API_KEY (llm/config.py).
The transport is ``client.chat.completions.create``; tests pass a fake one.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import Any

import openai

from llm import config
from llm.accounting import Budget, CallLog
from llm.base import ChatModelBase, Completion, redact
from llm.config import Provider, RetryPolicy
from llm.messages import Message, Tool, assistant_from_chat, to_chat, tools_to_chat
from llm.protocol import CallTimeLimit, LLMError, ProviderError, Usage

Create = Callable[..., Any]


def usage_from_chat(usage: dict[str, Any] | None) -> Usage:
    """Chat Completions ``usage``, for both providers.

    Both report prompt_tokens_details.cached_tokens (DeepSeek also as
    prompt_cache_hit_tokens) and completion_tokens_details.reasoning_tokens."""
    usage = usage or {}
    prompt = usage.get("prompt_tokens_details") or {}
    completion = usage.get("completion_tokens_details") or {}
    return Usage(
        input_tokens=usage.get("prompt_tokens") or 0,
        cached_input_tokens=prompt.get("cached_tokens") or usage.get("prompt_cache_hit_tokens")
        or 0,
        cache_write_tokens=prompt.get("cache_write_tokens") or 0,
        output_tokens=usage.get("completion_tokens") or 0,
        reasoning_tokens=completion.get("reasoning_tokens") or 0,
    )


def classify(exc: Exception) -> tuple[bool, int | None]:
    """(retry?, HTTP status). Retry rate limits, 5xx and network errors; nothing else."""
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return True, None
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        return status == 429 or status >= 500, status
    return False, None


def retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


MIN_ATTEMPT_S = 5.0  # no attempt (first or retry) starts with less time left


def call_with_retries(call: Callable[..., Any], policy: RetryPolicy, provider: str,
                      sleep: Callable[[float], None] = time.sleep,
                      jitter: Callable[[], float] = random.random, *,
                      timeout_s: float | None = None,
                      clock: Callable[[], float] = time.monotonic) -> tuple[Any, int]:
    """(result, retries). Exponential backoff with jitter, honouring Retry-After.

    With ``timeout_s`` (the call's whole allowance), each attempt is ``call(left)`` with the
    seconds left, and no attempt or backoff runs past it: ``CallTimeLimit`` instead.
    Without it, ``call()`` takes no argument."""
    deadline = None if timeout_s is None else clock() + timeout_s
    for attempt in range(policy.max_retries + 1):
        left = None if deadline is None else deadline - clock()
        if left is not None and left < MIN_ATTEMPT_S:
            raise CallTimeLimit(f"{provider}: the call's {timeout_s:.0f} s ran out "
                                f"({attempt} attempts made)", attempts=max(attempt, 1))
        try:
            return (call() if left is None else call(left)), attempt
        except (openai.APIConnectionError, openai.APIStatusError) as exc:
            retryable, status = classify(exc)
            what = f"HTTP {status}" if status else type(exc).__name__
            if deadline is not None and isinstance(exc, openai.APITimeoutError) \
                    and deadline - clock() < MIN_ATTEMPT_S:
                raise CallTimeLimit(f"{provider}: the call's {timeout_s:.0f} s ran out "
                                    f"during attempt {attempt + 1}",
                                    attempts=attempt + 1) from exc
            if not retryable:
                raise ProviderError(f"{provider} refused the request ({what}): {exc}",
                                    status=status, attempts=attempt + 1) from exc
            if attempt == policy.max_retries:
                raise ProviderError(
                    f"{provider} failed after {attempt + 1} attempts ({what}): {exc}",
                    status=status, retryable=True, attempts=attempt + 1) from exc
            delay = min(policy.max_delay_s, policy.base_delay_s * 2 ** attempt)
            delay *= 0.5 + 0.5 * jitter()
            server = retry_after(exc)
            if server is not None:
                delay = max(delay, min(server, policy.max_delay_s))
            if deadline is not None:
                delay = max(0.0, min(delay, deadline - clock() - MIN_ATTEMPT_S))
            sleep(delay)
    raise AssertionError("unreachable")


class OpenAIChatModel(ChatModelBase):
    def __init__(self, provider: Provider, model: str, *, effort: str | None = None,
                 max_output_tokens: int = 32_000, timeout_s: float = 300.0,
                 create: Create | None = None, retry: RetryPolicy = config.RETRY,
                 sleep: Callable[[float], None] = time.sleep,
                 jitter: Callable[[], float] = random.random,
                 budget: Budget | None = None, log: CallLog | None = None,
                 role: str | None = None, supports_images: bool = False,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(model, budget=budget, log=log, role=role)
        self.provider = provider.name
        self._clock = clock
        self.supports_images = supports_images
        self.settings = provider
        self.effort, self.max_output_tokens, self.timeout_s = effort, max_output_tokens, timeout_s
        self.retry, self._sleep, self._jitter = retry, sleep, jitter
        if create is None:
            key = os.environ.get(provider.key_env, "").strip()
            if not key:  # checked here: the SDK would otherwise fall back to OPENAI_API_KEY
                raise LLMError(f"no {provider.name} key: set the {provider.key_env} "
                               "environment variable")
            self._secrets = (key,)
            # Our own retry loop and timeout; the SDK's retries are off.
            client = openai.OpenAI(api_key=key, base_url=provider.base_url, timeout=timeout_s,
                                   max_retries=0)
            create = client.chat.completions.create
        self._create = create
        self._tools_cache: tuple[tuple[Tool, ...], list[dict[str, Any]]] | None = None

    def request(self, messages: list[Message], tools: tuple[Tool, ...],
                options: dict[str, Any] | None = None) -> dict[str, Any]:
        """The Chat Completions request. System messages first, then the conversation; the
        same tools convert to the same schemas, so the cacheable prefix stays identical."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_chat(messages, self.settings.reasoning_echo),
            self.settings.token_param: (options or {}).get("max_output_tokens",
                                                           self.max_output_tokens),
        }
        if tools:
            if self._tools_cache is None or self._tools_cache[0] != tools:
                self._tools_cache = (tools, tools_to_chat(tools))
            body["tools"] = self._tools_cache[1]
        effort = (options or {}).get("effort", self.effort)
        if effort:
            body["reasoning_effort"] = effort
        body["timeout"] = self.timeout_s  # per attempt; _complete lowers it to the time left
        return body

    def _complete(self, messages: list[Message], tools: tuple[Tool, ...],
                  options: dict[str, Any]) -> Completion:
        body = self.request(messages, tools, options)
        allowance = (options or {}).get("timeout_s")

        def attempt(left: float | None = None) -> Any:
            if left is None:
                return self._create(**body)
            return self._create(**{**body, "timeout": min(self.timeout_s, left)})

        try:
            response, retries = call_with_retries(attempt, self.retry, self.provider,
                                                  self._sleep, self._jitter, timeout_s=allowance,
                                                  clock=self._clock)
        except ProviderError as exc:
            exc.args = (redact(str(exc), self._secrets),)
            raise
        raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        if not raw.get("choices"):
            raise ProviderError(f"{self.provider} returned no choices", attempts=retries + 1)
        choice = raw["choices"][0]
        message = assistant_from_chat(choice.get("message") or {})
        return Completion(message.content, message.tool_calls, usage_from_chat(raw.get("usage")),
                          raw, raw.get("model") or self.model, choice.get("finish_reason"),
                          message.reasoning, retries)
