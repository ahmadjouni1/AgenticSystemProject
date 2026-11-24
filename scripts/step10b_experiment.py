"""Step 10b experiment: the fixes from the step 10 failure analysis, with and without the
two-phase build. 2 variants x the 4 step 10 prompts x N runs (default 3).

    uv run python scripts/step10b_experiment.py --runs 3 --jobs 4
    uv run python scripts/step10_report.py --folder runs/step10b --variants B2 D

| Variant | What the model gets |
| --- | --- |
| B2 | step 10's B (prompt set v1) plus fixes 1-4: 25-minute limit with a final turn and a fix reserve, the joint contact rule and recipes, soft-failing helpers, the compiler fix |
| D | B2 plus ``--two-phase`` (skeleton first, then detail) |

Fixes 1-4 are the defaults now, so B2 needs no flag. Everything else is as in step 10: the
builder role (deepseek-v4-pro), $1 per run, context policy v1, adaptive effort, 4 runs at a
time, jobs interleaved, Windows kept awake, no new run after an HTTP 402. Run folders go to
``runs/step10b/<variant>/``, console logs to ``runs/step10b/logs/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step10_experiment as base  # noqa: E402

base.OUT = base.ROOT / "runs" / "step10b"
base.VARIANTS = {
    "B2": ["--prompt-set", "v1"],
    "D": ["--prompt-set", "v1", "--two-phase"],
}
base.__doc__ = __doc__

if __name__ == "__main__":
    base.main()
