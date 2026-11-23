# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Compile one workspace in a fresh interpreter:

    python -m compiler.worker <run_dir> [--timeout SECONDS]

The phases are:

1. load ``workspace/main.py`` (runpy);
2. get ``object_model``;
3. validate it;
4. run ``run_tests()`` if it is defined;
5. baseline checks (a stub until step 4);
6. export to ``result/vNNNN`` with ``main.py`` as the source snapshot.

Prints ONE JSON payload as the last line of stdout. Anything main.py prints is captured
into the payload instead.

``result/.compile-progress.json`` always names the current phase. Shortly before the
timeout, ``faulthandler`` dumps every thread's stack to ``result/.compile-hang.txt``, so
the runner can say which line of main.py was running when it had to kill the process.

``run_tests()`` passes if it returns None or True. It fails if it raises
``AssertionError``, returns False, or returns a non-empty list of failure messages.
"""

from __future__ import annotations

import faulthandler
import sys
import time

_STARTED = time.perf_counter()
PROGRESS_FILE = ".compile-progress.json"
HANG_FILE = ".compile-hang.txt"


def _arm_hang_dump(argv: list[str]) -> object | None:
    """Arm faulthandler before the heavy imports, so even a slow import is caught."""
    if "--timeout" not in argv or len(argv) < 2:
        return None
    from pathlib import Path

    timeout = float(argv[argv.index("--timeout") + 1])
    result_dir = Path(argv[0]) / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    handle = open(result_dir / HANG_FILE, "w", encoding="utf-8")  # noqa: SIM115 (kept open)
    faulthandler.dump_traceback_later(max(0.5, timeout - min(1.0, 0.2 * timeout)), file=handle)
    return handle


_HANG_HANDLE = _arm_hang_dump(sys.argv[1:]) if __name__ == "__main__" else None

import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import runpy  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from compiler import feedback  # noqa: E402
from compiler.feedback import CompileSignal  # noqa: E402


class Progress:
    """Writes the current phase to a small JSON file at every phase change."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.phase: str | None = None
        self.phases: dict[str, float] = {}
        self._phase_started = time.perf_counter()

    @contextlib.contextmanager
    def phase_named(self, name: str):
        self.phase, self._phase_started = name, time.perf_counter()
        self._write()
        try:
            yield
        finally:
            self.phases[name] = round(time.perf_counter() - self._phase_started, 4)

    def _write(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "phase": self.phase,
            "phase_started_unix": time.time(),
            "phases": self.phases,
            "pid": os.getpid(),
        }), encoding="utf-8")
        os.replace(temporary, self.path)


def baseline_checks(model: Any) -> list[CompileSignal]:
    """Compiler-owned checks (collisions, floating parts, ...). Step 4 fills this in."""
    del model
    return []


def compile_workspace(run_dir: Path) -> dict[str, Any]:
    workspace, result_dir = run_dir / "workspace", run_dir / "result"
    main_path = workspace / "main.py"
    progress = Progress(result_dir / PROGRESS_FILE)
    out, err = io.StringIO(), io.StringIO()
    signals: list[CompileSignal] = []
    export_info = model_info = None
    traceback_text = ""

    sys.path.insert(0, str(workspace))
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            signals, export_info, model_info = _run_phases(workspace, result_dir, main_path,
                                                           progress)
    except BaseException as exc:  # noqa: BLE001 - every failure becomes a signal
        import traceback

        traceback_text = "".join(traceback.format_exception(exc))
        signals = [feedback.signal_from_exception(exc, progress.phase or "", workspace)]
    finally:
        if sys.path and sys.path[0] == str(workspace):
            sys.path.pop(0)

    failed = any(s.severity == "failure" for s in signals)
    return {
        "status": "error" if failed else "success",
        "phase": progress.phase,
        "phases": progress.phases,
        "worker_seconds": round(time.perf_counter() - _STARTED, 4),
        "signals": [s.to_dict() for s in signals],
        "export": export_info,
        "model": model_info,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "traceback": traceback_text,
    }


def _run_phases(workspace: Path, result_dir: Path, main_path: Path, progress: Progress):
    from sdk import Assembly
    from sdk.export import export

    with progress.phase_named("loading main.py"):
        namespace = runpy.run_path(str(main_path), run_name="__main__")
    with progress.phase_named("getting object_model"):
        model = namespace.get("object_model")
        if not isinstance(model, Assembly):
            return [feedback.signal_object_model(model, workspace)], None, None
    with progress.phase_named("validating the assembly"):
        art = model.validate()
    model_info = {"name": model.name, "bodies": len(model.bodies), "joints": len(model.joints),
                  "closures": len(art.closures), "mimics": len(art.mimic_order)}
    run_tests = namespace.get("run_tests")
    if callable(run_tests):
        with progress.phase_named(feedback.TESTS_PHASE):
            outcome = run_tests()
        failures = _test_failures(outcome)
        if failures:
            return feedback.signal_test_failures(failures, workspace), None, model_info
    with progress.phase_named("running baseline checks"):
        signals = baseline_checks(model)
    if any(s.severity == "failure" for s in signals):
        return signals, None, model_info
    with progress.phase_named(feedback.EXPORT_PHASE):
        result = export(model, result_dir, source=main_path)
    export_info = {"version": result.version, "directory": str(result.directory),
                   "urdf": str(result.urdf), "glb": str(result.glb),
                   "manifest": str(result.manifest)}
    return signals, export_info, model_info


def _test_failures(outcome: object) -> list[str]:
    if outcome is None or outcome is True:
        return []
    if outcome is False:
        return ["run_tests() returned False"]
    if isinstance(outcome, (list, tuple)):
        return [str(item) for item in outcome]
    return [f"run_tests() returned {type(outcome).__name__}; return None or a list of failures"]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(json.dumps({"status": "error", "signals": [], "phase": None,
                          "traceback": "usage: python -m compiler.worker <run_dir>"}))
        return 2
    payload = compile_workspace(Path(args[0]).resolve())
    faulthandler.cancel_dump_traceback_later()
    sys.stdout.write("\n" + json.dumps(payload) + "\n")
    sys.stdout.flush()
    return 0 if payload["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
