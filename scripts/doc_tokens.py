"""Real token sizes of the prompts and docs, as the provider counts them.

    uv run python scripts/doc_tokens.py

Sends each file once as a user message to the cheap model (output capped at 1 token) and
subtracts a baseline call with a one-word message. The files are what every agent run already
sends to the provider. Costs well under $0.01.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm import Budget, User, get_model  # noqa: E402

FILES = ["prompts/system.md", "prompts/v0/system.md", "docs/sdk/quickstart.md",
         "prompts/v0/docs/quickstart.md", "docs/sdk/shapes.md", "docs/sdk/joints_and_loops.md",
         "docs/sdk/testing.md", "docs/sdk/README.md"]


def main() -> None:
    run_dir = Path(tempfile.mkdtemp(prefix="doc_tokens_"))
    model = get_model("cheap", run_dir=run_dir, budget=Budget(max_cost_usd=0.05, max_calls=20))

    def count(text: str) -> int:
        return model.query([User(text)], (), max_output_tokens=1).usage.input_tokens

    base = count("x")
    print(f"model {model.model}; baseline {base} tokens\n")
    print("| File | Characters | Tokens (provider) | Characters per token |")
    print("| --- | --- | --- | --- |")
    for name in FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        n = count(text) - base + 1
        print(f"| `{name}` | {len(text):,} | {n:,} | {len(text) / n:.2f} |")


if __name__ == "__main__":
    main()
