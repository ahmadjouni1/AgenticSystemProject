"""LLM configuration in one file: providers, model roles, prices, budgets and retries.

Switch a role in one line, for example ``ROLES["builder"] = CHEAP``.

Model names and prices were checked on the providers' own pages on PRICES_CHECKED. Prices
change; re-check them on the PRICE_SOURCES pages and update the date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from llm.messages import ReasoningEcho

# ---- providers --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    name: str
    key_env: str  # the only place the key comes from
    base_url: str | None  # None: the openai package's default (OpenAI)
    token_param: str  # the request parameter that caps output tokens
    reasoning_echo: ReasoningEcho  # send the model's reasoning back in a tool loop?


PROVIDERS = {
    # Chat Completions: v1/chat/completions is a supported endpoint for the GPT-6.2 models;
    # max_tokens is deprecated there in favour of max_completion_tokens.
    "openai": Provider("openai", "OPENAI_API_KEY", None, "max_completion_tokens", "never"),
    # DeepSeek serves the same API shape at its own base_url; its output cap is max_tokens,
    # and its thinking mode returns reasoning_content.
    "deepseek": Provider("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com",
                         "max_tokens", "current_turn"),
}

# ---- roles ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    provider: str
    model: str
    effort: str | None = None  # reasoning_effort; None: the provider's default
    max_output_tokens: int = 32_000  # reasoning tokens count against this too
    timeout_s: float = 300.0  # per HTTP attempt


# OpenAI's docs: "If you're not sure where to start, use GPT-6.2 Astra, our flagship model
# for complex reasoning and coding."
STRONG = Role("openai", "gpt-6.2-astra")
# DeepSeek's docs: "Use deepseek-flash as the model name" (served by DeepSeek-V4.1-Flash).
CHEAP = Role("deepseek", "deepseek-flash")

ROLES: dict[str, Role] = {
    "builder": STRONG,
    "cheap": CHEAP,
}

# ---- prices -----------------------------------------------------------------------------------

PRICES_CHECKED = "2026-11-23"
PRICE_SOURCES = {
    "openai": "https://developers.openai.com/api/docs/pricing (Standard tier)",
    "deepseek": "https://api-docs.deepseek.com/quick_start/pricing",
}


@dataclass(frozen=True)
class OffPeak:
    """A time-of-day discount: ``multiplier`` outside the peak hours (UTC)."""
    multiplier: float
    peak_hours_utc: tuple[tuple[int, int], ...]  # [start, end) hours
    weekdays_only: bool  # peak hours apply Monday to Friday; weekends are off-peak

    def factor(self, at: datetime) -> float:
        if self.weekdays_only and at.weekday() >= 5:
            return self.multiplier
        peak = any(start <= at.hour < end for start, end in self.peak_hours_utc)
        return 1.0 if peak else self.multiplier


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""
    input: float
    cached_input: float
    output: float
    cache_write: float | None = None  # None: cache writes are billed as input
    long_context_above: int | None = None  # input tokens above which `long` applies
    long: Price | None = None  # None with a threshold: the long-context rate is unknown
    off_peak: OffPeak | None = None


_OPENAI_LONG = 272_000  # the pricing page's short/long context split
# DeepSeek: "Off-peak rates are half of the peak rates." Peak: 01:00-04:00 and 06:00-10:00
# UTC, Monday to Friday. Chinese public holidays are off-peak too; we do not know those
# dates, so a holiday call is costed at peak (over, never under).
_DEEPSEEK_OFF_PEAK = OffPeak(0.5, ((1, 4), (6, 10)), weekdays_only=True)

PRICES: dict[str, Price] = {
    # OpenAI, Standard tier, short context (<= 272K input tokens).
    "gpt-6.2-astra": Price(8.00, 0.40, 40.00, cache_write=10.00, long_context_above=_OPENAI_LONG,
                           long=Price(16.00, 0.80, 60.00, cache_write=10.00)),
    "gpt-6.2-sol": Price(2.00, 0.10, 10.00, cache_write=2.50, long_context_above=_OPENAI_LONG,
                         long=Price(4.00, 0.20, 15.00, cache_write=2.50)),
    "gpt-6.1-terra": Price(1.00, 0.05, 5.00, long_context_above=_OPENAI_LONG),
    "gpt-6-luna": Price(0.10, 0.01, 0.50, long_context_above=_OPENAI_LONG),
    # DeepSeek, peak rates (input = cache miss, cached input = cache hit).
    "deepseek-flash": Price(0.30, 0.006, 1.20, off_peak=_DEEPSEEK_OFF_PEAK),
    "deepseek-v4-pro": Price(1.32, 0.044, 3.96, off_peak=_DEEPSEEK_OFF_PEAK),
}

# ---- budget and retries -----------------------------------------------------------------------

DEFAULT_MAX_COST_USD = 5.00  # per run
DEFAULT_MAX_CALLS = 200  # per run


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 4  # after the first attempt; only rate limits, 5xx and network errors
    base_delay_s: float = 1.0  # 1, 2, 4, 8 s (times a jitter of 0.5-1.0)
    max_delay_s: float = 30.0  # also caps a server's Retry-After


RETRY = RetryPolicy()
