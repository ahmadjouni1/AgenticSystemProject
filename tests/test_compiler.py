"""The compiler, end to end, with real subprocesses (no LLM)."""

from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path

import pytest

from compiler import compile
from compiler.feedback import SignalKind

TIMEOUT = 60.0


def make_run(tmp_path: Path, source: str, name: str = "run") -> Path:
    run_dir = tmp_path / name
    (run_dir / "workspace").mkdir(parents=True)
    (run_dir / "workspace" / "main.py").write_text(textwrap.dedent(source).lstrip("\n"),
                                                   encoding="utf-8")
    return run_dir


def versions(run_dir: Path) -> list[str]:
    result = run_dir / "result"
    return sorted(p.name for p in result.iterdir() if p.name.startswith("v")) \
        if result.is_dir() else []


LAPTOP = """
from examples.laptop import build

object_model = build()
"""


def test_valid_main_compiles_and_exports_v0000(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, LAPTOP)
    result = compile(run_dir, timeout=TIMEOUT)
    assert result.ok, result.signals_text
    assert result.version == 0 and versions(run_dir) == ["v0000"]
    assert (run_dir / "result" / "v0000" / "main.py").read_text() == \
        (run_dir / "workspace" / "main.py").read_text()  # source snapshot
    assert "Compile passed: exported v0000 (bodies=2, joints=1" in result.signals_text
    assert list(result.phases) == ["loading main.py", "getting object_model",
                                   "validating the assembly", "running baseline checks",
                                   "exporting"]
    log = [json.loads(line) for line in result.log_path.read_text().splitlines()]
    assert log[-1]["status"] == "success" and log[-1]["export"]["version"] == 0


# ---- every failure kind: the right signal at the right main.py line ---------------------

FAILURES = {
    "syntax": ("""
        from examples.laptop import build

        object_model = = build()
        """, SignalKind.SYNTAX_ERROR, 3, "SyntaxError"),
    "import": ("""
        import math
        import not_a_real_module_xyz
        """, SignalKind.IMPORT_ERROR, 2, "No module named 'not_a_real_module_xyz'"),
    "runtime": ("""
        from examples.laptop import build

        def width():
            return 1 / 0

        object_model = build()
        size = width()
        """, SignalKind.RUNTIME_ERROR, 4, "ZeroDivisionError: division by zero"),
    "missing_object_model": ("""
        from examples.laptop import build

        model = build()
        """, SignalKind.OBJECT_MODEL, None, "does not define object_model"),
    "wrong_object_model": ("""
        import math

        object_model = 42
        """, SignalKind.OBJECT_MODEL, 3, "object_model is a int"),
    "validation_while_authoring": ("""
        from sdk import Assembly, Box
        a = Assembly("x")
        a.body("base").add("box", Box((0.1, 0.1, 0.1)))
        a.body("base")
        object_model = a
        """, SignalKind.VALIDATION, 4, "duplicate body name 'base'"),
    "validation_at_validate": ("""
        from sdk import Assembly, Box
        a = Assembly("x")
        base = a.body("base")
        base.add("box", Box((0.1, 0.1, 0.1)))
        hollow = a.body("hollow")
        a.joint("j", "fixed", base.at(), hollow.at())
        object_model = a
        """, SignalKind.VALIDATION, 5, "bodies with no shapes: ['hollow']"),
    "pose": ("""
        from examples.laptop import build

        object_model = build()


        def run_tests():
            object_model.solve({"hinge": 5.0})
        """, SignalKind.POSE, 7, "PoseError: joint 'hinge' value 5 is outside"),
    "loop_closure": ("""
        from examples.four_bar import build

        object_model = build(rocker_limits=(-0.05, 0.05))


        def run_tests():
            object_model.solve({"crank_joint": 3.14159})
        """, SignalKind.POSE, 7, "LoopClosureError"),
    "test_assert": ("""
        from examples.laptop import build

        object_model = build()


        def run_tests():
            lid = object_model.forward_kinematics({"hinge": 1.0})["lid"]
            assert lid[2, 3] > 1.0, "lid should be above 1 m"
        """, SignalKind.TEST_FAILURE, 8, "check failed: lid should be above 1 m"),
    "test_returned_failures": ("""
        from examples.laptop import build

        object_model = build()


        def run_tests():
            return ["lid gap is 3 mm, expected 1 mm"]
        """, SignalKind.TEST_FAILURE, 6, "check failed: lid gap is 3 mm"),
    "export": ("""
        from sdk import Assembly, MeshGeometry
        a = Assembly("flat")
        sheet = a.body("sheet")
        sheet.add("tri", MeshGeometry([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)]))
        object_model = a
        """, SignalKind.EXPORT_ERROR, 3, "body 'sheet' shape 'tri' encloses no volume"),
    "sys_exit": ("""
        import sys

        sys.exit(2)
        """, SignalKind.RUNTIME_ERROR, 3, "main.py called sys.exit(2)"),
    "crash": ("""
        import os

        os._exit(3)
        """, SignalKind.CRASH, None, "exited with code 3"),
}


@pytest.mark.parametrize("case", list(FAILURES))
def test_failure_kinds(tmp_path: Path, case: str) -> None:
    source, kind, line, message = FAILURES[case]
    run_dir = make_run(tmp_path, source)
    result = compile(run_dir, timeout=TIMEOUT)
    assert result.status == "error"
    (signal,) = result.signals
    assert signal.kind == kind, result.signals_text
    assert message in signal.message
    if line is not None:
        assert (signal.file, signal.line) == ("main.py", line), result.signals_text
        assert f"at main.py:{line}" in result.signals_text
    assert signal.hint
    assert "sdk" not in (signal.file or "")  # never points into SDK internals
    assert versions(run_dir) == []  # a failed compile leaves no version folder
    assert result.signals_text.startswith("<compile_signals>")
    assert "Traceback" not in result.signals_text  # tracebacks go to the log only
    log = json.loads(result.log_path.read_text().splitlines()[-1])
    assert log["status"] == "error"


def test_missing_main(tmp_path: Path) -> None:
    (tmp_path / "run" / "workspace").mkdir(parents=True)
    result = compile(tmp_path / "run", timeout=TIMEOUT)
    assert result.signals[0].kind == SignalKind.MISSING_MAIN


# ---- timeout and process-tree kill ---------------------------------------------------------


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_infinite_loop_times_out_and_the_whole_tree_dies(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, """
        import pathlib
        import subprocess
        import sys

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        pathlib.Path("grandchild.pid").write_text(str(child.pid))
        total = 0
        while True:
            total += 1
        """)
    started = time.perf_counter()
    result = compile(run_dir, timeout=4.0)
    assert time.perf_counter() - started < 20
    (signal,) = result.signals
    assert signal.kind == SignalKind.TIMEOUT
    assert "timed out after 4s while loading main.py" in signal.message
    assert (signal.file, signal.line) in {("main.py", 8), ("main.py", 9)}  # inside the loop
    pid = int((run_dir / "workspace" / "grandchild.pid").read_text())
    deadline = time.time() + 5
    while process_alive(pid) and time.time() < deadline:
        time.sleep(0.1)
    assert not process_alive(pid), "the grandchild process survived the timeout"
    assert versions(run_dir) == []


# ---- environment -----------------------------------------------------------------------------


def test_api_keys_are_not_visible_inside_main(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPLICA_FAKE_API_KEY", "secret-key")
    monkeypatch.setenv("REPLICA_FAKE_TOKEN", "secret-token")
    monkeypatch.setenv("REPLICA_VISIBLE", "yes")
    run_dir = make_run(tmp_path, """
        import os

        leaked = sorted(k for k in os.environ if k.upper().endswith(("_API_KEY", "_TOKEN")))
        assert not leaked, f"leaked: {leaked}"
        assert os.environ.get("REPLICA_VISIBLE") == "yes"
        assert os.getcwd() == os.path.dirname(os.path.abspath(__file__))  # cwd = workspace

        from examples.laptop import build

        object_model = build()
        """)
    result = compile(run_dir, timeout=TIMEOUT)
    assert result.ok, result.signals_text


# ---- cache, repeats, leftovers -----------------------------------------------------------------


def test_unchanged_workspace_is_cached_and_an_edit_recompiles(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, LAPTOP)
    first = compile(run_dir, timeout=TIMEOUT)
    second = compile(run_dir, timeout=TIMEOUT)
    assert not first.cached and second.cached
    assert second.version == 0 and versions(run_dir) == ["v0000"]
    assert "cached=true" in second.signals_text
    main = run_dir / "workspace" / "main.py"
    main.write_text(main.read_text() + "# a comment\n")
    third = compile(run_dir, timeout=TIMEOUT)
    assert not third.cached and third.version == 1
    assert versions(run_dir) == ["v0000", "v0001"]
    assert compile(run_dir, timeout=TIMEOUT, force=True).version == 2


def test_cache_is_dropped_when_the_toolchain_changes(tmp_path: Path, monkeypatch) -> None:
    from compiler import runner

    run_dir = make_run(tmp_path, LAPTOP)
    compile(run_dir, timeout=TIMEOUT)
    assert compile(run_dir, timeout=TIMEOUT).cached
    monkeypatch.setattr(runner, "_TOOLCHAIN_DIGEST", "a different sdk")  # as if sdk/ changed
    again = compile(run_dir, timeout=TIMEOUT)
    assert not again.cached and again.version == 1


def test_three_identical_failures_add_the_inspect_hint(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, FAILURES["runtime"][0])
    main = run_dir / "workspace" / "main.py"
    results = []
    for attempt in range(3):
        results.append(compile(run_dir, timeout=TIMEOUT))
        main.write_text(main.read_text() + f"# attempt {attempt}\n")  # same error, new digest
    assert [r.repeat_count for r in results] == [1, 2, 3]
    assert "Inspect before editing" not in results[0].signals_text
    assert "matches the previous compile" in results[1].signals_text
    assert "same failure 3 times in a row" in results[2].signals_text
    assert "Inspect before editing again: read the code around main.py:4" in \
        results[2].signals_text
    # A different failure resets the streak.
    main.write_text("object_model = 42\n")
    assert compile(run_dir, timeout=TIMEOUT).repeat_count == 1


def test_a_version_written_during_a_failed_compile_is_removed(tmp_path: Path) -> None:
    """Even if main.py exports by itself and then fails, no version folder survives."""
    run_dir = make_run(tmp_path, """
        from pathlib import Path
        from examples.laptop import build
        from sdk.export import export

        object_model = build()
        export(object_model, Path.cwd().parent / "result")
        raise RuntimeError("fails after exporting")
        """)
    result = compile(run_dir, timeout=TIMEOUT)
    assert result.status == "error" and versions(run_dir) == []


def test_baseline_check_failure_blocks_export(tmp_path: Path) -> None:
    from check_fixtures import DRAWER_SOURCE

    run_dir = make_run(tmp_path, DRAWER_SOURCE)
    result = compile(run_dir, timeout=TIMEOUT)
    assert result.status == "error" and versions(run_dir) == []
    (signal,) = [s for s in result.signals if s.severity == "failure"]
    assert signal.kind == SignalKind.SWEPT_COLLISION
    assert "first contact at 260.0 mm" in signal.message
    assert signal.source.startswith('cabinet.joint("slide"')  # pointed at the joint
    assert "Check failures are design evidence" in result.signals_text


def test_allowances_pass_and_show_their_reason(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, """
        import check_fixtures

        object_model = check_fixtures.laptop(lid_sink=0.005)


        def run_tests(ctx):
            ctx.allow_overlap("base", "lid", reason="the lid is recessed into the deck")
            ctx.expect_contact("base", "lid")
        """)
    (run_dir / "workspace" / "check_fixtures.py").write_text(
        (Path(__file__).parent / "check_fixtures.py").read_text(encoding="utf-8"),
        encoding="utf-8")
    result = compile(run_dir, timeout=TIMEOUT)
    assert result.ok, result.signals_text
    assert "allowed=1" in result.signals_text
    assert "(allowed: the lid is recessed into the deck)" in result.signals_text


def test_failed_tests_block_export_until_fixed(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, FAILURES["test_assert"][0])
    assert compile(run_dir, timeout=TIMEOUT).status == "error"
    assert versions(run_dir) == []
    main = run_dir / "workspace" / "main.py"
    main.write_text(main.read_text().replace("> 1.0", "> 0.0"))
    fixed = compile(run_dir, timeout=TIMEOUT)
    assert fixed.ok and fixed.version == 0


# ---- internal failures (step 10b) ----------------------------------------------------------


def test_a_denied_progress_write_does_not_fail_the_compile(tmp_path: Path, monkeypatch) -> None:
    """Step 10, C pliers run 2: on Windows, os.replace of the progress file raised
    PermissionError in run_tests() and failed the compile. Progress writes are best effort."""
    from compiler import worker

    real_replace = os.replace

    def denied(source, target, *args, **kwargs):
        if str(target).endswith(worker.PROGRESS_FILE):
            raise PermissionError(13, "Access is denied", str(target))
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(worker.os, "replace", denied)
    monkeypatch.setattr(worker.time, "sleep", lambda _s: None)
    run_dir = make_run(tmp_path, LAPTOP + "\ndef run_tests(ctx):\n    return None\n")
    payload = worker.compile_workspace(run_dir.resolve())
    assert payload["status"] == "success", payload["signals"]
    assert payload["internal"] is False and payload["progress_writes_skipped"] >= 5


def test_internal_failures_are_told_apart_from_main_py_failures(tmp_path: Path) -> None:
    from compiler.worker import is_internal
    from sdk.errors import ValidationError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("def boom():\n    raise RuntimeError('main')\n")
    namespace: dict = {}
    exec(compile_source(workspace / "main.py"), namespace)  # noqa: S102 - our own test code
    try:
        namespace["boom"]()
    except RuntimeError as exc:
        assert not is_internal(exc, workspace)  # raised in main.py
    try:
        raise OSError("disk")
    except OSError as exc:
        assert is_internal(exc, workspace)  # no frame in the workspace
    assert not is_internal(ValidationError("bad limits"), workspace)
    assert not is_internal(SyntaxError("bad"), workspace)


def compile_source(path: Path):
    import builtins  # this module's ``compile`` is the replica's compiler

    return builtins.compile(path.read_text(), str(path), "exec")


@pytest.mark.parametrize("payload_line", [
    json.dumps({"status": "error", "phase": "running baseline checks", "phases": {},
                "internal": True, "signals": [{"kind": "runtime_error", "message":
                                               "internal compiler error: OSError: disk",
                                               "hint": "compile again"}]}),
    "",  # the worker died without a result: a crash
], ids=["internal", "crash"])
def test_internal_failures_are_not_cached(tmp_path: Path, monkeypatch, payload_line) -> None:
    from compiler import runner

    calls = []

    def fake_worker(run_dir, workspace, timeout):
        calls.append(1)
        return {"returncode": 1, "timed_out": False, "payload_line": payload_line,
                "stray_stdout": "", "stderr": "boom"}

    monkeypatch.setattr(runner, "_run_worker", fake_worker)
    run_dir = make_run(tmp_path, LAPTOP)
    first = compile(run_dir, timeout=TIMEOUT)
    second = compile(run_dir, timeout=TIMEOUT)
    assert not first.ok and not second.cached and len(calls) == 2
    assert not (run_dir / "result" / runner.CACHE_FILE).exists()


def test_a_normal_failure_is_still_cached(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, FAILURES["runtime"][0])
    first = compile(run_dir, timeout=TIMEOUT)
    assert not first.ok and compile(run_dir, timeout=TIMEOUT).cached
