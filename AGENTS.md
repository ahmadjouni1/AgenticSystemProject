# Replica: guidance for coding agents

A step-by-step rebuild of Articraft: an LLM agent that writes articulated 3D objects as
Python (`main.py` with a small SDK), which a compiler builds, checks and exports. It runs on
Windows, with uv and Python 3.13. Step 13 added our teammate's dimensional synthesis solver
as a `synthesize` tool.

## Read in this order

1. `README.md`: what is here, the status of every step, how to run.
2. `docs/HANDOFF.md`: the prioritized to-do list, with evidence.
3. `docs/SOLVERS.md`: the solver plug-ins, the skeleton, the known gaps.
4. `BUILD_LOG.md`: the full history, step by step.

## Rules

- **No live API calls without the user's approval.** The DeepSeek budget is nearly gone,
  and runs go off-peak only (not 01-04 or 06-10 UTC on weekdays). The tests need no
  network.
- **Never edit `solvers/nested/`.** Those are our teammate's originals, checked byte for
  byte. Wrap them instead (`solvers/search.py` and the plug-ins).
- **Keep Windows support.** Paths, process kills and line endings must work there.
- **Run `uv run pytest -q` before committing.**
- **Never commit `.env`.** `git ls-files .env` must print nothing.
- **Keep the docs pages within their token limits.** The system prompt and every
  `docs/sdk` page are limited to 2k tokens, and `synthesis.md` to 1.5k; `tests/test_docs.py`
  checks them.
