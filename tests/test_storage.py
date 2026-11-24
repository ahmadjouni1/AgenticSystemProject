"""Step 11: run storage. Messages to and from dicts, crash-safe writes, resume (after
time_exceeded, after a crash mid-turn, after budget_exceeded), the run index and --list.
All with ScriptedModel: no network."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

import agent.loop as loop
import fileio
from agent import storage
from agent.loop import PROMPT_SETS, resume, run
from llm import Assistant, Budget, ScriptedModel, System, ToolCall, ToolResult, User
from llm.messages import MessageError, from_dict, to_dict, validate

LAPTOP_DOC = PROMPT_SETS["v0"].quickstart
GOOD = re.findall(r"```python\n(.*?)```", LAPTOP_DOC.read_text(encoding="utf-8"), re.S)[-1]
BAD = GOOD.replace("limits=(0.0, math.radians(135))", "limits=(0.5, 1.0)")
COMPILE = {"tool_calls": [{"name": "compile", "arguments": {}}]}
DONE = "Done: a base and a lid on a hinge."


def write(content: str) -> dict:
    return {"tool_calls": [{"name": "write", "arguments": {"path": "main.py",
                                                           "content": content}}]}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class TimedModel(ScriptedModel):
    def __init__(self, replies, seconds: float, clock: Clock, **kwargs):
        super().__init__(replies, **kwargs)
        self.seconds, self.clock = seconds, clock

    def _complete(self, messages, tools, options):
        self.clock.now += self.seconds
        return super()._complete(messages, tools, options)


def scripted(replies, budget: Budget | None = None) -> tuple[ScriptedModel, Budget]:
    budget = budget or Budget(max_cost_usd=None, max_calls=None)
    return ScriptedModel(replies, budget=budget), budget


# ---- messages: to_dict / from_dict -------------------------------------------------------------


def test_every_message_kind_round_trips_through_a_dict() -> None:
    broken = ToolCall.parse("c2", "write", '{"path": "main.py", "content": "x')  # cut off
    messages = [
        System("rules"), User("build it", ("renders/a.png",)),
        Assistant("plan", (ToolCall("c1", "read", {"path": "main.py"},
                                    '{"path":  "main.py"}'), broken), "thinking..."),
        ToolResult("c1", "read", '{"text": "..."}'),
        ToolResult("c2", "write", '{"error_kind": "bad_arguments"}', True),
        Assistant("done"),
    ]
    data = [to_dict(m) for m in messages]
    assert json.loads(json.dumps(data)) == data  # plain JSON
    back = [from_dict(d) for d in data]
    assert back == messages
    assert back[2].tool_calls[0].arguments_json() == '{"path":  "main.py"}'  # exact bytes
    assert back[2].tool_calls[1].error is not None  # the parse error comes back too
    validate(back)


def test_from_dict_ignores_log_keys_and_refuses_unknown_roles() -> None:
    entry = {"turn": 3, "time": "2026-11-24T10:00:00", "reminder": "compile_required",
             **to_dict(User("<compile_required>"))}
    assert from_dict(entry) == User("<compile_required>")
    with pytest.raises(MessageError, match="unknown message role"):
        from_dict({"role": "developer", "content": "x"})


# ---- crash-safe writes -------------------------------------------------------------------------


def test_an_atomic_write_that_fails_keeps_the_old_file_and_leaves_no_temp(tmp_path,
                                                                          monkeypatch) -> None:
    target = tmp_path / "main.py"
    fileio.atomic_write_text(target, "old\n", newline="\n")

    def broken_replace(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(fileio.os, "replace", broken_replace)
    with pytest.raises(OSError):
        fileio.atomic_write_text(target, "new\n", newline="\n")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["main.py"]


def test_a_locked_target_is_retried(tmp_path, monkeypatch) -> None:
    target = tmp_path / "record.json"
    real, calls = os.replace, []

    def flaky(source, destination):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("in use by another process")
        real(source, destination)

    monkeypatch.setattr(fileio.os, "replace", flaky)
    monkeypatch.setattr(fileio.time, "sleep", lambda s: None)
    fileio.atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True} and len(calls) == 3


def test_read_jsonl_skips_only_a_cut_off_last_line(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    fileio.append_line(path, '{"a": 1}')
    fileio.append_line(path, '{"a": 2}')
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"a": 3, "b"')  # a crash in the middle of a write
    assert fileio.read_jsonl(path) == [{"a": 1}, {"a": 2}]
    path.write_text('{"a": 1}\nnot json\n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        fileio.read_jsonl(path)  # damage in the middle is not hidden


def test_the_write_tool_and_the_record_leave_no_temporary_files(tmp_path) -> None:
    model, budget = scripted([write(GOOD), COMPILE, DONE])
    result = run("a laptop", model, budget, run_dir=tmp_path / "run")
    assert result.status == "success"
    leftovers = [p for p in result.run_dir.rglob("*.tmp")]
    assert leftovers == []
    assert not (result.run_dir / storage.LOCK).exists()  # released at the end


# ---- resume ------------------------------------------------------------------------------------


def stopped_on_time(tmp_path, monkeypatch):
    """Session 1: a failing compile, then the 1500 s run out (each call takes 800 s)."""
    clock = Clock()
    monkeypatch.setattr(loop.time, "time", clock)
    budget = Budget(max_cost_usd=None, max_calls=None)
    model = TimedModel([write(BAD), COMPILE], 800, clock, budget=budget)
    first = run("a laptop", model, budget, run_dir=tmp_path / "run", max_time_s=1500,
                fix_reserve_s=0)
    assert first.status == "time_exceeded", first.reason
    return first, model, clock


def test_resume_after_time_exceeded_continues_and_succeeds(tmp_path, monkeypatch) -> None:
    first, old_model, _clock = stopped_on_time(tmp_path, monkeypatch)
    sent_before = old_model.requests[-1][0]
    model, budget = scripted([write(GOOD), COMPILE, DONE])
    result = resume(first.run_dir, model, budget, max_time_s=900)
    assert result.status == "success", result.reason
    # The conversation goes on: same prefix (prompt cache), then the resume note.
    sent = model.requests[0][0]
    assert [to_dict(m) for m in sent[:len(sent_before)]] == [to_dict(m) for m in sent_before]
    note = sent[-1].content
    assert note.startswith("<resumed>") and "time_exceeded" in note and "15 min" in note
    # Counters continue; one entry per session.
    record = result.record
    assert record["turns"] == 5 and record["compiles"]["calls"] == 2
    assert record["duration_s"] >= 1600
    one, two = record["sessions"]
    assert one["status"] == "time_exceeded" and one["turns"] == [1, 2]
    assert two["status"] == "success" and two["turns"] == [3, 5]
    assert two["resumed_from"] == "time_exceeded" and two["allowance"]["max_time_s"] == 900
    assert record["best"]["passed_checks"]
    # The log has the resume note; the index has one line per session.
    log = fileio.read_jsonl(first.run_dir / "conversation.jsonl")
    assert any(e.get("reminder") == "resumed" for e in log)
    index = fileio.read_jsonl(tmp_path / "index.jsonl")
    assert [(e["run"], e["session"], e["status"]) for e in index] == [
        ("run", 1, "time_exceeded"), ("run", 2, "success")]


class Crash(BaseException):
    """Stands in for the process dying (not an Exception: the loop cannot catch it)."""


def crash_mid_turn(tmp_path):
    """Session 1 dies while its compile runs: the call has no result, the record says
    'running', and the lock names a process that no longer exists."""
    def dying_compiler(run_dir):
        raise Crash()

    model, budget = scripted([write(GOOD), COMPILE])
    with pytest.raises(Crash):
        run("a laptop", model, budget, run_dir=tmp_path / "run", compiler=dying_compiler)
    run_dir = (tmp_path / "run").resolve()
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True)
    fileio.atomic_write_json(run_dir / storage.LOCK, {"pid": int(dead.stdout)})
    return run_dir


def test_resume_after_a_crash_mid_turn(tmp_path) -> None:
    run_dir = crash_mid_turn(tmp_path)
    assert storage.read_record(run_dir)["status"] == "running"
    assert storage.status_of(run_dir, storage.read_record(run_dir)) == "crashed"
    with (run_dir / "conversation.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"turn": 2, "role": "tool", "call_id": "call_2", "na')  # cut off
    stale = run_dir / "workspace" / ".main.py.0123abcd.tmp"
    stale.write_text("half a file", encoding="utf-8")

    model, budget = scripted([COMPILE, DONE])
    result = resume(run_dir, model, budget)
    assert result.status == "success", result.reason
    assert not stale.exists()
    sent = model.requests[0][0]
    interrupted, note = sent[-2], sent[-1]
    assert isinstance(interrupted, ToolResult) and interrupted.name == "compile"
    assert json.loads(interrupted.content)["error_kind"] == "interrupted"
    assert note.content.startswith("<resumed>") and "crashed" in note.content
    one, two = result.record["sessions"]
    assert one["status"] == "crashed" and two["status"] == "success"
    # The repaired result was logged, so the log is valid for the next resume too.
    messages, _turns, repaired = storage.load_conversation(run_dir)
    assert repaired == []
    validate(messages)


def test_resume_after_budget_exceeded_gets_a_fresh_budget(tmp_path) -> None:
    model, budget = scripted([write(BAD), COMPILE, write(GOOD)],
                             Budget(max_cost_usd=None, max_calls=2))
    first = run("a laptop", model, budget, run_dir=tmp_path / "run")
    assert first.status == "budget_exceeded"
    model, budget = scripted([write(GOOD), COMPILE, DONE], Budget(max_cost_usd=1.0,
                                                                  max_calls=10))
    result = resume(first.run_dir, model, budget)
    assert result.status == "success"
    assert result.record["budget"]["calls"] == 3  # this session's budget
    assert result.record["sessions"][1]["allowance"]["max_calls"] == 10


def test_calls_a_stopped_run_never_ran_are_answered_as_not_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    entries = [{"turn": 0, **to_dict(System("s"))}, {"turn": 0, **to_dict(User("task"))},
               {"turn": 1, **to_dict(Assistant("", (ToolCall("c9", "compile", {}, "{}"),))),
                "stopped": "budget_exceeded"}]
    for entry in entries:
        fileio.append_line(run_dir / "conversation.jsonl", json.dumps(entry))
    messages, turns, repaired = storage.load_conversation(run_dir)
    assert json.loads(messages[-1].content)["error_kind"] == "not_run"
    assert turns[-1] == 1 and repaired[0]["repaired"] == "not_run"
    validate(messages)


def test_a_run_that_succeeded_or_is_still_running_is_not_resumed(tmp_path) -> None:
    model, budget = scripted([write(GOOD), COMPILE, DONE])
    done = run("a laptop", model, budget, run_dir=tmp_path / "done")
    with pytest.raises(storage.ResumeError, match="already succeeded"):
        resume(done.run_dir, *scripted([]))

    run_dir = crash_mid_turn(tmp_path)
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        fileio.atomic_write_json(run_dir / storage.LOCK, {"pid": alive.pid})
        assert storage.pid_alive(alive.pid)
        with pytest.raises(storage.ResumeError, match="is running in process"):
            resume(run_dir, *scripted([]))
    finally:
        alive.kill()
        alive.wait()
    assert not storage.pid_alive(alive.pid)


# ---- the index and the list --------------------------------------------------------------------


def test_the_index_and_the_list(tmp_path, capsys) -> None:
    import agent.__main__ as cli

    runs = tmp_path / "runs"
    ok = run("a laptop", *scripted([write(GOOD), COMPILE, DONE]), run_dir=runs / "exp" / "ok",
             runs_root=runs)
    model, budget = scripted([write(BAD), COMPILE], Budget(max_cost_usd=None, max_calls=2))
    stopped = run("a lamp", model, budget, run_dir=runs / "exp" / "stopped", runs_root=runs)
    crash_mid_turn(runs / "exp")  # runs/exp/run, a crash: never indexed

    index = fileio.read_jsonl(runs / "index.jsonl")  # one index for the runs folder
    assert [(e["run"], e["status"], e["best"], e["best_passed"]) for e in index] == [
        ("exp/ok", "success", "v0000", True),
        ("exp/stopped", "budget_exceeded", None, None)]  # BAD fails validation: no build
    assert index[0]["turns"] == ok.turns and index[0]["prompt"] == "a laptop"
    assert index[1]["reason"].startswith("call cap reached")

    rows = {r["run"]: r for r in storage.list_runs(runs)}
    assert rows["exp/ok"]["status"] == "success" and rows["exp/ok"]["best"] == "v0000"
    assert rows["exp/stopped"]["best"] is None
    assert rows["exp/stopped"]["turns"] == stopped.turns
    assert rows["exp/run"]["status"] == "crashed"

    assert cli.main(["--list", str(runs)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["run", "status", "turns", "cost", "best",
                                           "sessions", "prompt"]
    assert re.search(r"exp/ok\s+success\s+3\s+\$0\.0000\s+v0000\s+1\s+a laptop", out)
    assert "3 runs: 1 budget_exceeded, 1 crashed, 1 success" in out


def test_the_cli_resumes_a_run(tmp_path, monkeypatch, capsys) -> None:
    import agent.__main__ as cli

    scripts = [[write(BAD), COMPILE], [write(GOOD), COMPILE, DONE]]

    def fake_get_model(role, *, run_dir, budget):
        return ScriptedModel(scripts.pop(0) if role == "builder" else [], budget=budget,
                             role=role)

    monkeypatch.setattr(cli, "get_model", fake_get_model)
    assert cli.main(["a laptop", "--runs", str(tmp_path), "--max-calls", "2"]) == 1
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert cli.main(["--resume", str(run_dir), "--max-calls", "10"]) == 0
    out = capsys.readouterr().out
    assert "resuming:" in out and "was: budget_exceeded after 3 turns" in out
    assert "success: finished after a passing, fresh compile" in out
    record = storage.read_record(run_dir)
    assert [s["status"] for s in record["sessions"]] == ["budget_exceeded", "success"]
    assert [s["turns"] for s in record["sessions"]] == [[1, 3], [4, 6]]  # no turn reused
    assert record["options"]["context"] == "v1" and record["options"]["prompt_set"] == "v1"
    assert cli.main(["--resume", str(run_dir)]) == 2  # it succeeded: nothing to resume
    with pytest.raises(SystemExit):
        cli.main(["a laptop", "--list", str(tmp_path)])  # one mode at a time
