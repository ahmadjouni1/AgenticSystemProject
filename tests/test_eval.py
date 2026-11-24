"""Step 12: the evaluation harness. The benchmark file, scripts/eval.py (runs, the total-spend
cap, resume on a rerun, HTTP 402), scripts/eval_report.py, the blind rating sheets and the
Articraft baseline adapter. ScriptedModel only: no network, no WSL."""

from __future__ import annotations

import contextlib
import csv
import importlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from agent import storage
from agent.loop import PROMPT_SETS, REPLICA_ROOT
from fileio import read_jsonl
from llm import ScriptedModel

sys.path.insert(0, str(REPLICA_ROOT / "scripts"))
eval_script = importlib.import_module("eval")
eval_report = importlib.import_module("eval_report")
eval_rating = importlib.import_module("eval_rating")
baseline = importlib.import_module("articraft_baseline")

BENCHMARK = REPLICA_ROOT / "prompts" / "benchmark_v1.json"
GOOD = re.findall(r"```python\n(.*?)```",
                  PROMPT_SETS["v0"].quickstart.read_text(encoding="utf-8"), re.S)[-1]
BAD = GOOD.replace("limits=(0.0, math.radians(135))", "limits=(0.5, 1.0)")
COMPILE = {"tool_calls": [{"name": "compile", "arguments": {}}]}


def write(content: str) -> dict:
    return {"tool_calls": [{"name": "write", "arguments": {"path": "main.py",
                                                           "content": content}}]}


SUCCESS = [write(GOOD), COMPILE, "Done: a base and a lid on a hinge."]
FAILED_CHECKS = [write(BAD), COMPILE, "done", "done", "done"]


# ---- the benchmark ------------------------------------------------------------------------------

# The objects of every docs example and recipe, examples/ program and earlier evaluation prompt.
EARLIER_OBJECTS = {"laptop", "drawer", "cabinet", "linkage", "lamp", "toolbox",
                   "table", "pump", "pliers", "mailbox", "hatch", "cupboard", "bench", "carcass",
                   "door", "hinge", "mouse", "chair", "gantry", "toaster", "kitchen", "legs"}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def test_the_benchmark_has_12_prompts_2_to_3_per_category() -> None:
    data = eval_script.load_benchmark(BENCHMARK)
    prompts = data["prompts"]
    assert len(prompts) == 12
    counts = {c: sum(p["category"] == c for p in prompts) for c in data["categories"]}
    assert set(counts) == {"single", "chain", "independent", "loop", "tool"}
    assert all(2 <= n <= 3 for n in counts.values()), counts
    for p in prompts:
        assert p["motion"] and len(p["motion"].splitlines()) == 1
        assert re.fullmatch(r"[a-z_]+", p["id"])


def test_no_benchmark_object_is_a_docs_example_or_an_earlier_prompt() -> None:
    data = eval_script.load_benchmark(BENCHMARK)
    earlier = json.loads((REPLICA_ROOT / "prompts" / "eval_prompts.json")
                         .read_text(encoding="utf-8"))
    examples = {w for path in (REPLICA_ROOT / "examples").glob("*.py")
                for w in path.stem.split("_")}
    assert {"desk", "lamp", "toaster", "gantry"} <= examples  # the list below is complete
    examples -= {"four", "bar"}  # four_bar: checked as a phrase ("four spinning wheels" is fine)
    for p in data["prompts"]:
        clash = (words(p["object"]) | words(p["prompt"])) & (EARLIER_OBJECTS | examples)
        assert not clash, (p["id"], clash)
        assert not re.search(r"four[- ]bar", p["prompt"] + p["object"])
        for step, prompts in earlier.items():
            if step not in ("_comment", "step12"):
                assert p["prompt"] not in prompts
    # The docs test protects the benchmark from future docs examples.
    assert earlier["step12"] == [p["prompt"] for p in data["prompts"]]


# ---- eval.py ------------------------------------------------------------------------------------


def mini_benchmark(tmp_path: Path, n: int = 3) -> Path:
    data = eval_script.load_benchmark(BENCHMARK)
    picked = [next(p for p in data["prompts"] if p["category"] == c)
              for c in ("single", "chain", "loop")][:n]
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({**data, "prompts": picked}), encoding="utf-8")
    return path


def settings(tmp_path: Path, **kwargs) -> eval_script.Settings:
    return eval_script.Settings(name="t", runs_root=tmp_path / "runs",
                                benchmark=mini_benchmark(tmp_path), jobs=1, **kwargs)


def cli_launcher(monkeypatch, scripts: list[list]):
    """Runs ``python -m agent`` in this process, the builder replies taken from ``scripts``
    in launch order; returns the launcher and the argv of each launch."""
    import agent.__main__ as cli

    def fake_get_model(role, *, run_dir, budget):
        return ScriptedModel(scripts.pop(0) if role == "builder" else [], budget=budget,
                             role=role)

    monkeypatch.setattr(cli, "get_model", fake_get_model)
    launches: list[list[str]] = []

    def launch(argv: list[str], log: Path) -> int:
        launches.append(argv)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as out, contextlib.redirect_stdout(out):
            return cli.main(argv)

    return launch, launches


def test_an_evaluation_runs_every_prompt_and_a_rerun_keeps_finished_runs(tmp_path,
                                                                        monkeypatch) -> None:
    s = settings(tmp_path)
    launch, launches = cli_launcher(monkeypatch, [SUCCESS, FAILED_CHECKS, SUCCESS])
    summary = eval_script.run_eval(s, launch, log=lambda text: None)
    assert [j["status"] for j in summary["jobs"]] == ["success", "failed_checks", "success"]
    assert summary["stopped"] is None and len(launches) == 3
    first = launches[0]
    assert first[0] == "a jewelry box with a lid hinged at the back"
    assert first[first.index("--max-cost") + 1] == "0.3"
    assert first[first.index("--index") + 1] == str(s.index)
    # runs/index.jsonl: one line per run, read by the next invocation.
    index = read_jsonl(s.index)
    assert [e["run"].split("/")[:3] for e in index] == [
        ["eval_t", "jobs", "jewelry_box-r1"], ["eval_t", "jobs", "robot_arm-r1"],
        ["eval_t", "jobs", "scissor_lift-r1"]]
    assert json.loads((s.eval_dir / "eval.json").read_text(encoding="utf-8"))["jobs"]
    assert (s.eval_dir / "benchmark.json").is_file()
    again = eval_script.run_eval(s, launch, log=lambda text: None)
    assert len(launches) == 3  # nothing started: every run had finished
    assert all(j["this_invocation"]["action"] == "kept" for j in again["jobs"])
    assert "keep" in eval_script.dry_run(s) or "done" in eval_script.dry_run(s)


class Crash(BaseException):
    pass


def test_a_rerun_resumes_an_unfinished_run_with_what_is_left(tmp_path, monkeypatch) -> None:
    from agent.loop import run
    from fileio import atomic_write_json

    s = settings(tmp_path, only=["jewelry_box"])

    def crashing(argv: list[str], log: Path) -> int:
        def dying_compiler(run_dir):
            raise Crash()

        model = ScriptedModel([write(GOOD), COMPILE], role="builder")
        folder = Path(argv[argv.index("--runs") + 1])
        try:
            run(argv[0], model, model.budget or __import__("llm").Budget(None, None),
                run_dir=folder / "run", compiler=dying_compiler, index=s.index)
        except Crash:
            pass
        atomic_write_json(folder / "run" / storage.LOCK, {"pid": 999_999_999})  # dead
        return -9

    eval_script.run_eval(s, crashing, log=lambda text: None)
    plan_text = eval_script.dry_run(s)
    assert "resume" in plan_text and "unfinished (crashed)" in plan_text
    launch, launches = cli_launcher(monkeypatch, [[COMPILE, "Done: a box with a lid."]])
    summary = eval_script.run_eval(s, launch, log=lambda text: None)
    (argv,) = launches
    assert argv[0] == "--resume" and argv[argv.index("--max-cost") + 1] == "0.3"
    assert float(argv[argv.index("--max-time") + 1]) <= s.max_time
    assert summary["jobs"][0]["status"] == "success"
    record = storage.read_record(eval_script.run_dir_of(eval_script.plan(
        eval_script.load_benchmark(s.benchmark), s.eval_dir, 1, ["jewelry_box"])[0]))
    assert [x["status"] for x in record["sessions"]] == ["crashed", "success"]


def fake_launcher(cost: float, seen: list, text: str = ""):
    """Writes a finished run record costing ``cost``; checks the cap's books on the way."""
    def launch(argv: list[str], log: Path) -> int:
        folder = Path(argv[argv.index("--runs") + 1])
        seen.append(float(argv[argv.index("--max-cost") + 1]))
        (folder / "run").mkdir(parents=True)
        (folder / "run" / "record.json").write_text(json.dumps(
            {"status": "success", "cost_usd": cost, "duration_s": 10, "turns": 3,
             "sessions": [{}]}), encoding="utf-8")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")
        return 0
    return launch


@pytest.mark.parametrize("jobs", [1, 2])
def test_no_run_starts_that_could_pass_the_total_cap(tmp_path, jobs) -> None:
    s = settings(tmp_path, max_cost=0.30, max_total=0.70)
    s.jobs = jobs
    seen: list[float] = []
    summary = eval_script.run_eval(s, fake_launcher(0.25, seen), log=lambda text: None)
    assert len(seen) == 2  # 0.25 + 0.25 spent; a third run could reach 0.80 > 0.70
    assert summary["spent_usd"] == pytest.approx(0.50)
    assert "total-spend cap" in summary["stopped"]
    assert [j["status"] for j in summary["jobs"]] == ["success", "success", "not_started"]


def test_the_report_lists_runs_the_cap_never_started(tmp_path) -> None:
    s = settings(tmp_path, max_cost=0.30, max_total=0.70)
    eval_script.run_eval(s, fake_launcher(0.25, []), log=lambda text: None)
    _benchmark, rows = eval_report.load(s.eval_dir)
    assert len(rows) == 3
    assert [r["status"] for r in rows].count("not_started") == 1


def test_http_402_stops_new_runs(tmp_path) -> None:
    s = settings(tmp_path)
    seen: list[float] = []
    summary = eval_script.run_eval(s, fake_launcher(0.01, seen, "ProviderError: HTTP 402"),
                                   log=lambda text: None)
    assert len(seen) == 1 and "402" in summary["stopped"]


def test_an_old_402_in_a_resumed_runs_log_does_not_stop_new_runs(tmp_path) -> None:
    """Logs are appended to across sessions; only this launch's output counts."""
    s = settings(tmp_path)
    logs = s.eval_dir / "logs"
    logs.mkdir(parents=True)
    jobs = eval_script.plan(eval_script.load_benchmark(s.benchmark), s.eval_dir, s.runs)
    for job in jobs:
        (logs / f"{job.key}.log").write_text("ProviderError: HTTP 402 (session 1)\n",
                                             encoding="utf-8")
    seen: list[float] = []
    write = fake_launcher(0.01, seen)

    def append(argv: list[str], log: Path) -> int:
        old = log.read_text(encoding="utf-8")
        code = write(argv, log)
        log.write_text(old + "session 2: fine\n", encoding="utf-8")
        return code

    summary = eval_script.run_eval(s, append, log=lambda text: None)
    assert len(seen) == len(jobs) and summary["stopped"] is None


def test_off_peak_only_starts_no_run_at_peak_price(tmp_path, monkeypatch) -> None:
    s = settings(tmp_path, off_peak_only=True)
    real_peak_now = eval_script.peak_now
    peak = iter([False, True, True])  # the second run would start at peak price
    monkeypatch.setattr(eval_script, "peak_now", lambda at=None: next(peak))
    seen: list[float] = []
    summary = eval_script.run_eval(s, fake_launcher(0.01, seen), log=lambda text: None)
    assert len(seen) == 1 and summary["stopped"].startswith("peak price")
    assert [j["status"] for j in summary["jobs"]] == ["success", "not_started", "not_started"]
    from datetime import UTC, datetime

    assert real_peak_now(datetime(2026, 11, 25, 1, 30, tzinfo=UTC))  # a Wednesday
    assert not real_peak_now(datetime(2026, 11, 24, 16, 0, tzinfo=UTC))


def test_the_main_dry_run_prints_the_plan_and_the_worst_case(tmp_path, capsys) -> None:
    assert eval_script.main(["--name", "x", "--dry-run", "--runs-root",
                             str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.count(" start ") == 12
    assert "worst case if every run used its whole cap: $3.60" in out


# ---- eval_report.py -----------------------------------------------------------------------------


def test_runs_cut_off_from_outside_are_not_counted() -> None:
    out_of_credit = {"reason": "model call failed: deepseek refused the request (HTTP 402): "
                               "Insufficient Balance; best build: result/last_failed"}
    assert eval_report.interrupted("error", out_of_credit) == "out of credit (HTTP 402)"
    assert eval_report.interrupted("crashed", {}) == "crashed"
    assert eval_report.interrupted("error", {"reason": "ValueError: a loop bug"}) is None
    assert eval_report.interrupted("time_exceeded", out_of_credit) is None


def test_wilson_intervals() -> None:
    low, high = eval_report.wilson(0, 3)
    assert low == 0 and high == pytest.approx(0.5615, abs=1e-4)
    low, high = eval_report.wilson(3, 3)
    assert low == pytest.approx(0.4385, abs=1e-4) and high == 1
    low, high = eval_report.wilson(6, 12)
    assert (low, high) == (pytest.approx(0.2538, abs=1e-4), pytest.approx(0.7462, abs=1e-4))


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """One small evaluation (2 successes, 1 failed_checks) for the report and rating tests."""
    tmp_path = tmp_path_factory.mktemp("eval")
    patch = pytest.MonkeyPatch()
    try:
        s = settings(tmp_path)
        launch, _launches = cli_launcher(patch, [SUCCESS, FAILED_CHECKS, SUCCESS])
        eval_script.run_eval(s, launch, log=lambda text: None)
    finally:
        patch.undo()
    return s


def test_the_report_tables_sheets_and_build_log(evaluated, tmp_path) -> None:
    s = evaluated
    log = tmp_path / "BUILD_LOG.md"
    log.write_text("# Build log\n", encoding="utf-8")
    argv = ["--name", "t", "--runs-root", str(s.runs_root), "--log", str(log)]
    assert eval_report.main(argv) == 0
    text = (s.eval_dir / "report.md").read_text(encoding="utf-8")
    assert "| **overall** | 2/3 | 67% | 21%-94% |" in text
    assert "| single (a single hinge or slide) | 1/1 | 100% | 21%-100% |" in text
    assert "indicative only" in text
    assert "| failed_checks | 1 |" in text  # failure end states
    assert "| validation | 1 | 1 |" in text  # failure kinds
    assert re.search(r"\| robot_arm-r1 \| chain \| failed_checks \| 5 \|", text)
    for category in ("single", "chain", "loop"):
        assert Image.open(s.eval_dir / "renders" / f"{category}.png").size == (400, 300)
    assert eval_report.main(argv) == 0  # a second report replaces the section
    logged = log.read_text(encoding="utf-8")
    assert logged.count("<!-- eval_t -->") == 1 and logged.startswith("# Build log")
    assert "#### Success (Wilson 95% intervals)" in logged


def test_blind_rating_sheets_and_scores(evaluated, tmp_path) -> None:
    s = evaluated
    out = tmp_path / "rating"
    rows = eval_rating.make(s.runs_root, ["t"], out)
    assert len(rows) == 2  # the successful runs only
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    assert set(key) == {"item_01", "item_02"}
    from agent.tools.render import HEIGHT, WIDTH

    assert Image.open(out / "sheets" / "item_01.png").size == (2 * WIDTH, 2 * HEIGHT)
    with (out / "ratings.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert [r["item"] for r in table] == ["item_01", "item_02"]
    assert all(r["prompt"] and r["expected_motion"] and not r["realistic"] for r in table)
    assert all(k["eval"] == "t" for k in key.values())
    assert not any(name in "".join(r["prompt"] for r in table) for name in ("eval_t", "B2"))
    table[0].update(realistic="4", motion_correct="yes")
    table[1].update(realistic="7", motion_correct="maybe")
    with (out / "ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, eval_rating.FIELDS)
        writer.writeheader()
        writer.writerows(table)
    text = eval_rating.score(out)
    assert "| eval_t | 2 | 1 | 4.00 | 1/1 |" in text
    assert "realistic must be 1-5, got '7'" in text
    assert "motion_correct must be yes or no" in text


def test_the_blind_sheet_has_no_text(evaluated) -> None:
    from agent.tools.render import BACKGROUND

    rows = eval_rating.successful_runs(evaluated.runs_root, ["t"])
    sheet = eval_rating.blind_sheet(rows[0]["version_dir"]).convert("RGB")
    corner = sheet.crop((0, 0, 200, 24))  # where the label and legend would be drawn
    assert corner.getcolors() == [(200 * 24, tuple(BACKGROUND))]


# ---- articraft_baseline.py ----------------------------------------------------------------------


def job(tmp_path: Path) -> eval_script.Job:
    return eval_script.Job("stapler", "tool", "a stapler", 1,
                           tmp_path / "eval_articraft" / "jobs" / "stapler-r1")


def test_the_wsl_command_carries_no_key(tmp_path) -> None:
    s = baseline.Settings("articraft", runs_root=tmp_path)
    j = job(tmp_path)
    argv = baseline.command(j, s)
    assert argv[:5] == ["wsl.exe", "-d", "Ubuntu", "--", "bash"]
    assert "uv run articraft generate --no-tui --provider deepseek --model deepseek-v4-pro " \
           "--effort high" in argv[-1]
    assert "'a stapler'" in argv[-1]
    env = baseline.environment(s, {"DEEPSEEK_API_KEY": "secret-value", "WSLENV": "X/p"})
    assert "secret-value" not in " ".join(argv)
    assert env["WSLENV"].split(":") == ["X/p", "DEEPSEEK_API_KEY/u", "ARTICRAFT_MAX_TURNS/u",
                                        "ARTICRAFT_DEEPSEEK_MAX_OUTPUT_TOKENS/u"]
    assert env["ARTICRAFT_MAX_TURNS"] == "60"
    assert baseline.wsl_path(Path(r"C:\Users\me\runs\x")) == "/mnt/c/Users/me/runs/x"


def fake_articraft_run(folder: Path, status: str, error: str = "", tokens: int = 100_000) -> Path:
    run = folder / "articraft" / "20261124-120000-a-stapler"
    (run / "result" / "usdz").mkdir(parents=True)
    (run / "result" / "usdz" / "0002.usdz").write_bytes(b"usdz")
    (run / "result" / "model.json").write_text(json.dumps({
        "rigid_bodies": [{"name": "base", "shapes": [{"geometry_type": "Box"}]},
                         {"name": "arm", "shapes": [{"geometry_type": "Box"},
                                                    {"geometry_type": "Cylinder"}]}],
        "joints": [{"name": "pivot", "dofs": [{"axis": "Y", "limits": [0, 0.5]}]}]}),
        encoding="utf-8")
    (run / "record.json").write_text(json.dumps({
        "status": status, "error": error, "result": "result/usdz/0002.usdz" if status ==
        "success" else "", "cost": 0.0,
        "token_usage": {"input_tokens": tokens, "cached_input_tokens": tokens // 2,
                        "output_tokens": tokens // 10, "total_tokens": tokens}}),
        encoding="utf-8")
    (run / "conversation.jsonl").write_text(
        "\n".join(json.dumps({"role": r}) for r in ("user", "assistant", "tool", "assistant")),
        encoding="utf-8")
    return run


def test_an_articraft_run_becomes_a_replica_record(tmp_path) -> None:
    s = baseline.Settings("articraft", runs_root=tmp_path)
    j = job(tmp_path)
    run = fake_articraft_run(j.folder, "success")
    started = datetime(2026, 11, 24, 12, tzinfo=UTC)
    record = baseline.convert(j, run, None, 321.0, started, s)
    assert record["status"] == "success" and record["turns"] == 2
    assert record["cost_usd"] == pytest.approx(baseline.cost_of(
        {"input_tokens": 100_000, "cached_input_tokens": 50_000, "output_tokens": 10_000},
        started), abs=1e-6)
    assert record["cost_usd"] > 0  # Articraft itself records $0 for DeepSeek
    assert record["best"]["path"].endswith("result/usdz/0002.usdz")
    assert record["quality"] == {"bodies": 2, "shapes": 3, "joints": 1, "movable_joints": 1,
                                 "non_box_share": 0.3333, "helper_count": None,
                                 "triangles": None, "source": "articraft model.json"}
    assert baseline.convert(j, run, "cost", 50, started, s)["status"] == "budget_exceeded"
    assert baseline.convert(j, run, "time", 1501, started, s)["status"] == "time_exceeded"
    other = job(tmp_path / "b")
    maxed = fake_articraft_run(other.folder, "error", "agent hit max turns limit")
    assert baseline.convert(other, maxed, None, 9, started, s)["status"] == "max_turns"


class FakeProcess:
    def __init__(self) -> None:
        self.code = None

    def poll(self):
        return self.code

    def wait(self):
        self.code = -9
        return self.code


def test_the_watcher_stops_articraft_past_the_cost_cap_or_the_time_limit(tmp_path) -> None:
    s = baseline.Settings("articraft", runs_root=tmp_path, max_cost=0.30, max_time=1500)
    j = job(tmp_path)
    process = FakeProcess()
    signals: list[str] = []
    now = [0.0]

    def stop(signal: str) -> None:
        signals.append(signal)
        if signal == "KILL":
            process.code = -9

    def sleep(seconds: float) -> None:
        now[0] += seconds
        if now[0] >= 20 and baseline.articraft_run(j) is None:
            fake_articraft_run(j.folder, "running", tokens=3_000_000)  # well past $0.30

    why, _seconds = baseline.watch(j, process, s, stop, clock=lambda: now[0], sleep=sleep)
    assert why == "cost" and signals == ["TERM", "KILL"]

    other = job(tmp_path / "t")
    process, signals[:] = FakeProcess(), []
    why, seconds = baseline.watch(other, process, s, stop, clock=lambda: now[0],
                                  sleep=lambda x: now.__setitem__(0, now[0] + 100))
    assert why == "time" and seconds > 1500


def test_the_baseline_starts_no_run_that_could_pass_the_total_or_reach_peak(tmp_path) -> None:
    s = baseline.Settings("articraft", runs_root=tmp_path, max_cost=0.30, max_time=1500,
                          off_peak_only=True)
    evening = datetime(2026, 11, 24, 21, 0, tzinfo=UTC)  # a Tuesday, off-peak
    worst = baseline.worst_run(s, evening)
    assert worst == pytest.approx(0.30 + 2 * baseline.overshoot(evening))
    assert 0.05 < baseline.overshoot(evening) < 0.10
    assert baseline.may_start(s, 0.0, 3.0, evening) is None
    assert "total cap" in baseline.may_start(s, 3.0 - worst + 0.01, 3.0, evening)
    late = datetime(2026, 11, 25, 0, 40, tzinfo=UTC)  # peak at 01:00, within 25 minutes
    assert "peak price" in baseline.may_start(s, 0.0, 3.0, late)
    s.off_peak_only = False
    assert baseline.may_start(s, 0.0, 3.0, late) is None


def test_a_rerun_keeps_the_failed_attempt_and_counts_its_cost(tmp_path) -> None:
    j = job(tmp_path)
    j.folder.mkdir(parents=True)
    (j.folder / "record.json").write_text(json.dumps({"status": "error", "cost_usd": 0.12}),
                                          encoding="utf-8")
    assert baseline.spent_so_far([j]) == pytest.approx(0.12)
    baseline.keep_earlier_attempt(j)
    assert not (j.folder / "record.json").exists()
    (j.folder / "record.json").write_text(json.dumps({"status": "success", "cost_usd": 0.2}),
                                          encoding="utf-8")
    assert [p.name for p in baseline.attempts(j)] == ["record.01.json", "record.json"]
    assert baseline.spent_so_far([j]) == pytest.approx(0.32)


def test_the_baseline_dry_run_shows_plan_estimate_and_balance(tmp_path) -> None:
    s = baseline.Settings("articraft", runs_root=tmp_path, benchmark=BENCHMARK,
                          off_peak_only=True)
    jobs = eval_script.plan(eval_script.load_benchmark(BENCHMARK), s.eval_dir, 1)
    text = baseline.dry_run(s, jobs, 3.0, balance=lambda: "9.99 USD (available: True)",
                            now=datetime(2026, 11, 24, 21, 0, tzinfo=UTC))
    assert len(re.findall(r" start$", text, re.M)) == 12 and "articraft generate" in text
    assert "start now: yes" in text and "DeepSeek balance: 9.99 USD" in text
    assert "6 of 12 runs start even if every run ends at" in text


def test_a_call_articraft_never_recorded_is_priced_and_only_interruptions_rerun(
        tmp_path) -> None:
    s = baseline.Settings("articraft", runs_root=tmp_path)
    j = job(tmp_path)
    run = fake_articraft_run(j.folder, "error",
                             "model query failed: ModelError: DeepSeek output was truncated.")
    (run / "conversation.jsonl").write_text(json.dumps(
        {"role": "assistant", "token_usage": {"input_tokens": 26_000, "output_tokens": 500}}),
        encoding="utf-8")
    started = datetime(2026, 11, 24, 21, 0, tzinfo=UTC)
    record = baseline.convert(j, run, None, 255, started, s)
    assert record["cost_estimated_usd"] == pytest.approx(baseline.unrecorded_call(run, started))
    assert 0.05 < record["cost_estimated_usd"] < 0.10
    assert record["cost_usd"] == pytest.approx(record["cost_recorded_usd"]
                                               + record["cost_estimated_usd"], abs=1e-6)
    assert not baseline.rerun(record)  # Articraft's own error is a result
    stopped = baseline.convert(j, run, "outside", 300, started, s)
    assert stopped["cost_estimated_usd"] > 0 and baseline.rerun(stopped)
    assert baseline.rerun({}) and baseline.rerun({"status": "error", "reason":
                                                  "articraft: DeepSeek request failed "
                                                  "(HTTP 402): Insufficient Balance"})
    assert not baseline.rerun({"status": "success"})
    other = job(tmp_path / "c")
    clean = baseline.convert(other, fake_articraft_run(other.folder, "success"), None, 100,
                             started, s)
    assert clean["cost_estimated_usd"] == 0
