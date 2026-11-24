"""Step 10 experiment: 3 prompt variants x the step 10 evaluation prompts x N runs.

    uv run python scripts/step10_experiment.py --runs 3 --jobs 4
    uv run python scripts/step10_experiment.py --only A --prompt "a folding table" --runs 1

| Variant | What the model gets |
| --- | --- |
| A | step 9's system prompt and docs (``--prompt-set v0``) |
| B | step 10's system prompt and docs (``--prompt-set v1``) |
| C | B, plus ``reasoning_effort=low`` before the first compile (``--planning-effort low``) |

All variants: the builder role (deepseek-v4-pro), $1 per run, the context policy v1 and
adaptive effort (the defaults). Jobs are interleaved (repetition, then prompt, then variant) so
that time-of-day and API load fall on every variant alike. Each run is ``python -m agent`` in
its own process; its console output goes to ``runs/step10/logs/``. Run folders go to
``runs/step10/<variant>/``. If any run gets HTTP 402 (out of credit), no further run starts. Then ``scripts/step10_report.py`` makes the tables and the render
folder.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "step10"
VARIANTS = {
    "A": ["--prompt-set", "v0"],
    "B": ["--prompt-set", "v1"],
    "C": ["--prompt-set", "v1", "--planning-effort", "low"],
}
PROMPTS = json.loads((ROOT / "prompts" / "eval_prompts.json").read_text(encoding="utf-8"))["step10"]


def slug(text: str) -> str:
    sys.path.insert(0, str(ROOT))
    from agent.loop import slugify

    return slugify(text)


STOP = threading.Event()  # set when the provider account is out of credit


def job(variant: str, prompt: str, rep: int, max_cost: float) -> tuple[str, int, float]:
    log = OUT / "logs" / f"{variant}-{slug(prompt)}-{rep}.log"
    if STOP.is_set():
        return log.name, -1, 0.0  # not started
    command = [sys.executable, "-m", "agent", prompt, "--runs", str(OUT / variant),
               "--max-cost", str(max_cost), *VARIANTS[variant]]
    started = time.time()
    with log.open("w", encoding="utf-8") as out:
        code = subprocess.call(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if "HTTP 402" in log.read_text(encoding="utf-8", errors="replace"):
        STOP.set()  # out of credit: every later run would fail too
    return log.name, code, time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="runs per prompt and variant")
    parser.add_argument("--jobs", type=int, default=4,
                        help="runs at the same time (DeepSeek limits concurrency by balance; "
                             "9 got HTTP 429 on 2026-11-23)")
    parser.add_argument("--only", nargs="*", default=sorted(VARIANTS), help="variants")
    parser.add_argument("--prompt", nargs="*", default=PROMPTS, help="prompts")
    parser.add_argument("--max-cost", type=float, default=1.0, help="USD per run")
    args = parser.parse_args()
    unknown = set(args.only) - set(VARIANTS)
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(v, p, rep) for rep in range(1, args.runs + 1) for p in args.prompt
            for v in args.only]
    print(f"{len(jobs)} runs, {args.jobs} at a time -> {OUT}", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(job, v, p, rep, args.max_cost) for v, p, rep in jobs]
        for future in futures:
            name, code, seconds = future.result()
            print(f"{name}: " + ("not started (out of credit)" if code == -1 else
                                 f"exit {code} after {seconds:.0f} s"), flush=True)
    if STOP.is_set():
        print("STOPPED: the provider returned HTTP 402 (insufficient balance)", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
