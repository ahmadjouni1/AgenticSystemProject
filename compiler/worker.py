# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Compile one workspace in a fresh interpreter:

    python -m compiler.worker <run_dir> [--timeout SECONDS]

The phases are:

1. load ``workspace/main.py`` (runpy);
2. get ``object_model``;
3. validate it;
4. run ``run_tests(ctx)`` if it is defined (``ctx`` is an ``sdk.testing.TestContext``:
   expectations and ``allow_*`` with reasons);
5. baseline checks (``sdk.testing.run_checks``), using the allowances from ``ctx``;
   then, if main.py declares ``target_path``, the trajectory check (``sdk.target_path``);
6. export to ``result/vNNNN`` with ``main.py`` as the source snapshot and the signals
   in ``signals.json``, but only if nothing blocks. A valid model whose checks fail is
   exported to ``result/last_failed`` instead (not a version), for the viewer.

Prints ONE JSON payload as the last line of stdout. Anything main.py prints is captured
into the payload instead.

``result/.compile-progress.json`` always names the current phase. Shortly before the
timeout, ``faulthandler`` dumps every thread's stack to ``result/.compile-hang.txt``, so
the runner can say which line of main.py was running when it had to kill the process.

``run_tests`` may also take no argument (step-3 style). Its failed expectations block the
compile. Raising ``AssertionError``, returning False, or returning a non-empty list of
messages also count as failures.
"""

from __future__ import annotations

import faulthandler
import sys
import time

_STARTED = time.perf_counter()
PROGRESS_FILE = ".compile-progress.json"
WRITE_ATTEMPTS = 5
INTERNAL_HINT = ("This failure comes from the compiler, not from main.py, and is not cached: "
                 "compile again without changes. If it repeats, simplify the shape it was "
                 "checking.")
HANG_FILE = ".compile-hang.txt"
SIGNALS_FILE = "signals.json"
LAST_FAILED = "last_failed"


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
import dataclasses  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import runpy  # noqa: E402
import shutil  # noqa: E402
import traceback as traceback_module  # noqa: E402
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
        self.skipped_writes = 0

    @contextlib.contextmanager
    def phase_named(self, name: str):
        self.phase, self._phase_started = name, time.perf_counter()
        self._write()
        try:
            yield
        finally:
            self.phases[name] = round(time.perf_counter() - self._phase_started, 4)

    def _write(self) -> None:
        """Best effort: the file only helps the runner explain a timeout. On Windows,
        ``os.replace`` fails while another process (an indexer, an antivirus scan, the
        runner) has the file open; one such PermissionError used to fail a whole compile,
        and the failure was then cached (step 10, C pliers run 2). Retry, then skip."""
        text = json.dumps({
            "phase": self.phase,
            "phase_started_unix": time.time(),
            "phases": self.phases,
            "pid": os.getpid(),
        })
        temporary = self.path.with_suffix(".tmp")
        for attempt in range(WRITE_ATTEMPTS):
            try:
                temporary.write_text(text, encoding="utf-8")
                os.replace(temporary, self.path)
                return
            except OSError:
                if attempt + 1 < WRITE_ATTEMPTS:
                    time.sleep(0.05 * (attempt + 1))
        self.skipped_writes += 1


def compile_workspace(run_dir: Path) -> dict[str, Any]:
    workspace, result_dir = run_dir / "workspace", run_dir / "result"
    main_path = workspace / "main.py"
    shutil.rmtree(result_dir / LAST_FAILED, ignore_errors=True)  # describes the latest compile
    progress = Progress(result_dir / PROGRESS_FILE)
    out, err = io.StringIO(), io.StringIO()
    signals: list[CompileSignal] = []
    export_info = model_info = None
    traceback_text = ""
    internal = False

    sys.path.insert(0, str(workspace))
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            signals, export_info, model_info = _run_phases(workspace, result_dir, main_path,
                                                           progress)
    except BaseException as exc:  # noqa: BLE001 - every failure becomes a signal
        import traceback

        traceback_text = "".join(traceback.format_exception(exc))
        signals = [feedback.signal_from_exception(exc, progress.phase or "", workspace)]
        internal = is_internal(exc, workspace)
        if internal:
            signals = [dataclasses.replace(
                signals[0], message=f"internal compiler error: {signals[0].message}",
                hint=INTERNAL_HINT)]
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
        "internal": internal,
        "progress_writes_skipped": progress.skipped_writes,
    }


def notice_signals(workspace: Path) -> list[CompileSignal]:
    """Warnings for the adjustments shape helpers made (sdk.notices), each at the deepest
    workspace line of the call."""
    from sdk.notices import drain

    out = []
    for item in drain():
        where = None
        for filename, line in item.stack:
            relative = feedback.relative_path(filename, workspace)
            if relative is not None:
                where = (relative, line)
        signal = feedback.make_signal(feedback.SignalKind.HELPER_ADJUSTED, item.message,
                                      item.hint, workspace, where)
        out.append(dataclasses.replace(signal, severity="warning"))
    return out


def is_internal(exc: BaseException, workspace: Path) -> bool:
    """True when the failure is the compiler's own, not main.py's: no frame of the
    traceback is in the workspace, and it is not a syntax error or an SDK error (those
    report a problem with the model, raised from SDK code on main.py's behalf)."""
    from sdk.errors import SDKError

    if isinstance(exc, (SyntaxError, SDKError)):
        return False
    root = str(workspace.resolve()).lower()
    frames = traceback_module.extract_tb(exc.__traceback__)
    return not any(str(Path(f.filename).resolve()).lower().startswith(root) for f in frames)


def _run_phases(workspace: Path, result_dir: Path, main_path: Path, progress: Progress):
    from sdk import Assembly
    from sdk.export import export
    from sdk.testing import TestContext, call_run_tests, run_checks

    with progress.phase_named("loading main.py"):
        namespace = runpy.run_path(str(main_path), run_name="__main__")
    adjusted = notice_signals(workspace)  # helpers that repaired an argument
    with progress.phase_named("getting object_model"):
        model = namespace.get("object_model")
        if not isinstance(model, Assembly):
            return [*adjusted, feedback.signal_object_model(model, workspace)], None, None
    with progress.phase_named("validating the assembly"):
        art = model.validate()
    model_info = {"name": model.name, "bodies": len(model.bodies), "joints": len(model.joints),
                  "closures": len(art.closures), "mimics": len(art.mimic_order)}
    ctx = TestContext(model)
    signals: list[CompileSignal] = list(adjusted)
    run_tests = namespace.get("run_tests")
    if callable(run_tests):
        with progress.phase_named(feedback.TESTS_PHASE):
            outcome = call_run_tests(run_tests, ctx)
        signals += feedback.signal_test_failures(_test_failures(outcome), workspace)
    with progress.phase_named(feedback.CHECKS_PHASE):
        report = run_checks(model, ctx)
    signals += feedback.signals_from_report(report, workspace)
    target = namespace.get("target_path")
    if target is not None:  # step 13: the output point must follow the declared path
        from sdk.target_path import check_target_path

        with progress.phase_named(feedback.TARGET_PHASE):
            findings, model_info["target_path"] = check_target_path(model, target)
        signals += feedback.signals_from_report(_Findings(findings), workspace)
    signals += notice_signals(workspace)  # helpers called in run_tests() or later
    model_info["check_seconds"] = round(sum(report.timings.values()), 4)
    signals_json = json.dumps([s.to_dict() for s in signals], indent=2)
    if any(s.severity == "failure" for s in signals):
        _export_last_failed(model, result_dir, main_path, signals_json)
        return signals, None, model_info
    with progress.phase_named(feedback.EXPORT_PHASE):
        result = export(model, result_dir, source=main_path,
                        extra_files={SIGNALS_FILE: signals_json})
    export_info = {"version": result.version, "directory": str(result.directory),
                   "urdf": str(result.urdf), "glb": str(result.glb),
                   "manifest": str(result.manifest)}
    return signals, export_info, model_info


@dataclasses.dataclass
class _Findings:
    findings: list


def _export_last_failed(model: Any, result_dir: Path, main_path: Path, signals_json: str) -> None:
    """Keep the model of a compile whose checks failed in ``result/last_failed`` (not a
    version), so the viewer can show the problem at the pose where it happens."""
    from sdk.export import export

    build = result_dir / ".last_failed_build"
    shutil.rmtree(build, ignore_errors=True)
    try:
        exported = export(model, build, version=0, source=main_path,
                          extra_files={SIGNALS_FILE: signals_json})
        os.replace(exported.directory, result_dir / LAST_FAILED)
    except Exception:  # noqa: BLE001 - a preview is a convenience, never a new failure
        pass
    finally:
        shutil.rmtree(build, ignore_errors=True)


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
