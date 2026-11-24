"""What calls cost, the per-run budget, and the call log (``runs/<id>/llm_calls.jsonl``)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fileio import append_line, read_jsonl
from llm import config
from llm.protocol import UNKNOWN, BudgetExceeded, Cost, Reply, Usage

log = logging.getLogger("llm")


def cost(model: str, usage: Usage, at: datetime | None = None) -> Cost:
    """USD for one call, from config.PRICES; UNKNOWN (with a warning) if there is no price.

    Uncached input, cached input, cache writes and output are billed separately; reasoning
    tokens are part of the output tokens."""
    price = config.PRICES.get(model)
    if price is None:
        log.warning("no price for model %r: its cost is recorded as %r (add it to "
                    "llm/config.py PRICES)", model, UNKNOWN)
        return UNKNOWN
    rates = price
    if price.long_context_above is not None and usage.input_tokens > price.long_context_above:
        if price.long is None:
            log.warning("no long-context price for %r (%d input tokens > %d): cost %r",
                        model, usage.input_tokens, price.long_context_above, UNKNOWN)
            return UNKNOWN
        rates = price.long
    uncached = max(0, usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens)
    write_rate = rates.cache_write if rates.cache_write is not None else rates.input
    usd = (uncached * rates.input + usage.cached_input_tokens * rates.cached_input
           + usage.cache_write_tokens * write_rate + usage.output_tokens * rates.output) / 1e6
    if price.off_peak is not None:
        usd *= price.off_peak.factor(at or datetime.now(UTC))
    return usd


class Budget:
    """Per-run caps on cost (USD) and number of calls. Shared by all models of a run.

    - Before a call: refused if the call cap is reached or the cost cap already spent.
    - After a call: if it took the total over the cost cap, BudgetExceeded is raised with
      the reply attached (the call was made and paid; the run must stop).
    Calls of unknown cost count toward the call cap but cannot count toward the cost cap;
    each one logs a warning when a cost cap is set."""

    def __init__(self, max_cost_usd: float | None = config.DEFAULT_MAX_COST_USD,
                 max_calls: int | None = config.DEFAULT_MAX_CALLS):
        self.max_cost_usd, self.max_calls = max_cost_usd, max_calls
        self.calls = 0
        self.spent_usd = 0.0
        self.unknown_cost_calls = 0
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExceeded(f"call cap reached: {self.calls} of {self.max_calls} "
                                     "calls made in this run")
            if self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd:
                raise BudgetExceeded(f"cost cap reached: ${self.spent_usd:.4f} spent of "
                                     f"${self.max_cost_usd:.2f}")

    def record(self, reply: Reply) -> None:
        with self._lock:
            self.calls += 1
            if reply.cost == UNKNOWN:
                self.unknown_cost_calls += 1
                if self.max_cost_usd is not None:
                    log.warning("a call to %r has unknown cost; the $%.2f cost cap cannot "
                                "count it", reply.model, self.max_cost_usd)
            else:
                self.spent_usd += float(reply.cost)
            over = self.max_cost_usd is not None and self.spent_usd > self.max_cost_usd
        if over:
            raise BudgetExceeded(f"cost cap exceeded: ${self.spent_usd:.4f} spent of "
                                 f"${self.max_cost_usd:.2f} after call {self.calls}", reply)

    def to_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "spent_usd": round(self.spent_usd, 6),
                "unknown_cost_calls": self.unknown_cost_calls,
                "max_cost_usd": self.max_cost_usd, "max_calls": self.max_calls}


class CallLog:
    """Appends one JSON line per call. Records hold metadata only: no messages, no keys."""

    FILE = "llm_calls.jsonl"

    def __init__(self, path: str | Path):
        path = Path(path)
        self.path = path / self.FILE if path.suffix != ".jsonl" else path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            append_line(self.path, line)

    def records(self) -> list[dict[str, Any]]:
        return read_jsonl(self.path)  # skips a last line cut short by a crash

    def summary(self) -> dict[str, Any]:
        """Totals over the log: calls, tokens, known cost, unknown-cost calls, cached share."""
        records = [r for r in self.records() if r.get("status") == "ok"]
        usage = Usage()
        for r in records:
            fields = {k: r["usage"].get(k, 0) for k in Usage.__dataclass_fields__}
            usage = usage + Usage(**fields)
        known = [r["cost"] for r in records if r["cost"] != UNKNOWN]
        return {"calls": len(records), "errors": len(self.records()) - len(records),
                "usage": usage.to_dict(), "cost_usd": round(sum(known), 6),
                "unknown_cost_calls": len(records) - len(known)}
