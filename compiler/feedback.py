# Portions adapted from Articraft (https://github.com/mattzh72/articraft),
# licensed under Apache-2.0, and modified for this replica. See NOTICE.
"""Compile signals: what the agent is told about a compile, and nothing more.

Every failure becomes a typed ``CompileSignal`` with:

- a kind;
- a message;
- a location in the workspace (``main.py`` line, never SDK internals);
- a one-line fix hint.

``render_signals`` turns them into the compact ``<compile_signals>`` block the model
reads. Full stdout, stderr and tracebacks go to ``result/compile_log.jsonl`` instead.
"""

from __future__ import annotations

import hashlib
import json
import re
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

MAIN = "main.py"
TESTS_PHASE = "running run_tests()"
CHECKS_PHASE = "running baseline checks"
EXPORT_PHASE = "exporting"

Where = tuple[str, int]


class SignalKind(StrEnum):
    MISSING_MAIN = "missing_main"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    OBJECT_MODEL = "object_model"
    VALIDATION = "validation"
    POSE = "pose"
    TEST_FAILURE = "test_failure"
    EXPORT_ERROR = "export_error"
    TIMEOUT = "timeout"
    CRASH = "crash"
    # Baseline checks (step 4).
    OVERLAP = "overlap"
    FLOATING = "floating"
    SWEPT_COLLISION = "swept_collision"
    SEPARATION = "separation"
    LOOP = "loop"
    MESH_HEALTH = "mesh_health"
    ISLANDS = "islands"
    DIMENSIONS = "dimensions"
    ALLOWANCE = "allowance"
    # A shape helper repaired an argument instead of failing (step 10b; a warning).
    HELPER_ADJUSTED = "helper_adjusted"


@dataclass(frozen=True)
class CompileSignal:
    kind: str
    message: str
    hint: str
    file: str | None = None
    """Workspace-relative path of the location, usually ``main.py``."""
    line: int | None = None
    source: str | None = None
    """The text of that line, stripped."""
    severity: str = "failure"
    bodies: tuple[str, ...] = ()
    """For check findings: the bodies involved (the viewer highlights them)."""
    pose: dict[str, float] | None = None
    """For check findings: joint values where it happens (the viewer jumps there)."""
    amount: float | None = None
    """For check findings: meters of penetration, gap or gap growth."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "bodies", tuple(self.bodies))

    @property
    def location(self) -> str | None:
        if self.file is None:
            return None
        return f"{self.file}:{self.line}" if self.line is not None else self.file

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CompileSignal:
        return cls(**data)


# ---- locating an error in the workspace --------------------------------------------------


def _relative(path: str | Path | None, workspace: Path) -> str | None:
    """Workspace-relative path, or None. Only absolute paths count: compiled modules
    (e.g. scipy's ``_qhull.pyx``) report relative names that are not workspace files."""
    if not path or not Path(path).is_absolute():
        return None
    try:
        return Path(path).resolve().relative_to(workspace.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def _lines(workspace: Path, file: str) -> list[str]:
    try:
        return (workspace / file).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _first_match(workspace: Path, pattern: re.Pattern[str], file: str = MAIN) -> Where | None:
    for number, text in enumerate(_lines(workspace, file), start=1):
        if pattern.search(text):
            return file, number
    return None


def locate_traceback(exc: BaseException, workspace: Path) -> Where | None:
    """The deepest traceback frame inside the workspace (main.py or a helper module)."""
    found = None
    for frame in traceback.extract_tb(exc.__traceback__):
        relative = _relative(frame.filename, workspace)
        if relative is not None:
            found = (relative, frame.lineno)
    return found


def locate_by_name(message: str, workspace: Path) -> Where | None:
    """For errors raised after main.py ran (e.g. in validate()): the first main.py line
    that mentions an entity quoted in the message, such as 'pin' or 'hollow'."""
    for name in re.findall(r"'([A-Za-z_]\w*)'", message):
        found = _first_match(workspace, re.compile(rf"""["']{re.escape(name)}["']"""))
        if found:
            return found
    return None


def locate_assignment(name: str, workspace: Path) -> Where | None:
    return _first_match(workspace, re.compile(rf"^\s*{re.escape(name)}\s*[:=]"))


def locate_definition(name: str, workspace: Path) -> Where | None:
    return _first_match(workspace, re.compile(rf"^\s*def\s+{re.escape(name)}\b"))


def locate_hang(dump: str, workspace: Path) -> Where | None:
    """The innermost workspace frame in a faulthandler dump (most recent call first)."""
    for match in re.finditer(r'File "(.+?)", line (\d+)', dump):
        relative = _relative(match.group(1), workspace)
        if relative is not None:
            return relative, int(match.group(2))
    return None


def _signal(kind: SignalKind, message: str, hint: str, workspace: Path,
            where: Where | None) -> CompileSignal:
    if where is None:
        return CompileSignal(kind.value, message, hint)
    file, line = where
    lines = _lines(workspace, file)
    source = lines[line - 1].strip() if 0 < line <= len(lines) else None
    return CompileSignal(kind.value, message, hint, file, line, source)


# ---- classification ------------------------------------------------------------------------

HINTS = {
    SignalKind.MISSING_MAIN: "Write workspace/main.py that defines object_model (an sdk.Assembly).",
    SignalKind.SYNTAX_ERROR: "Fix the Python syntax at this line; nothing runs until main.py parses.",
    SignalKind.IMPORT_ERROR: "Import only sdk, build123d, numpy, scipy, trimesh, the standard "
                             "library, or your own workspace modules; check the exact name.",
    SignalKind.RUNTIME_ERROR: "Fix the exception at this line; no model checks run until "
                              "main.py runs cleanly.",
    SignalKind.OBJECT_MODEL: "Define object_model = build_object_model() at module level, "
                             "returning an sdk.Assembly.",
    SignalKind.VALIDATION: "Fix the model structure named in the message before tuning geometry.",
    SignalKind.POSE: "Drive only independent tree joints, with values inside their limits.",
    SignalKind.TEST_FAILURE: "Fix the geometry or motion this check names; do not delete or "
                             "weaken a check that reports a real defect.",
    SignalKind.EXPORT_ERROR: "The model validated but could not be exported; fix the named body "
                             "or shape (it needs closed geometry with volume).",
    SignalKind.TIMEOUT: "Make the slow step faster (fewer segments, coarser tolerance, fewer "
                        "booleans) or remove the endless loop at this line.",
    SignalKind.CRASH: "The compile process died without a result; look for os._exit, deep "
                      "recursion, or a huge allocation in main.py.",
    SignalKind.OVERLAP: "Move or resize one shape so the two only touch; if they interpenetrate "
                        "by design (a pin in a bore), add ctx.allow_overlap(a, b, reason=...) in "
                        "run_tests(ctx).",
    SignalKind.FLOATING: "Give the body real support so it touches its neighbour (no gap), or "
                         "ctx.allow_isolated(body, reason=...) if it is meant to be loose.",
    SignalKind.SWEPT_COLLISION: "Change the joint limits, axis or geometry so the parts clear "
                                "each other over the whole range.",
    SignalKind.SEPARATION: "Put the joint origin on the hinge line and the axis along it, so the "
                           "child stays in contact while it turns.",
    SignalKind.LOOP: "Fix the link lengths or the driver's limits so the loop closes over the "
                     "whole range.",
    SignalKind.MESH_HEALTH: "Rebuild the named shape as a closed, consistently wound mesh.",
    SignalKind.ISLANDS: "Make the shapes of one body touch or overlap slightly.",
    SignalKind.DIMENSIONS: "Check the units: the SDK works in meters.",
    SignalKind.ALLOWANCE: "Remove the allowance; nothing needs it.",
    SignalKind.HELPER_ADJUSTED: "A shape helper changed an argument to build the shape; pass a "
                                "valid value to choose it yourself.",
}

_CHECK_KINDS = {
    "authored": SignalKind.TEST_FAILURE, "overlap": SignalKind.OVERLAP,
    "floating": SignalKind.FLOATING, "swept_collision": SignalKind.SWEPT_COLLISION,
    "combined_collision": SignalKind.SWEPT_COLLISION,
    "separation": SignalKind.SEPARATION, "loop": SignalKind.LOOP,
    "mesh_health": SignalKind.MESH_HEALTH, "islands": SignalKind.ISLANDS,
    "dimensions": SignalKind.DIMENSIONS, "allowance": SignalKind.ALLOWANCE,
}


def signals_from_report(report: object, workspace: Path) -> list[CompileSignal]:
    """One signal per check finding. A blocking finding is a failure; an allowed one is a
    note that carries its reason; anything else is a warning."""
    signals = []
    for finding in report.findings:  # type: ignore[attr-defined]
        kind = _CHECK_KINDS.get(finding.check, SignalKind.TEST_FAILURE)
        message = finding.message
        if finding.allowed:
            message += f" (allowed: {finding.allowed})"
        where = None
        if finding.joint:
            where = locate_by_name(f"'{finding.joint}'", workspace)
        where = where or locate_by_name(message, workspace)
        signal = _signal(kind, message, HINTS[kind], workspace, where)
        if finding.blocking:
            severity = "failure"
        elif finding.allowed:
            severity = "note"
        else:
            severity = "warning"
        signals.append(CompileSignal(signal.kind, signal.message, signal.hint, signal.file,
                                     signal.line, signal.source, severity, finding.bodies,
                                     finding.pose, finding.amount))
    return signals

_VALIDATION_HINTS = (
    ("open in the rest pose", "Author the loop closed: build one end of the closure with "
                              "assembly.frame_in(...)."),
    ("must contain 0", "Author the parts at a pose inside the joint range, so 0 is within the "
                       "limits."),
    ("no shapes", "Give every body geometry with body.add(name, shape)."),
    ("exactly one root", "Connect every body to the root with a joint (parent -> child)."),
    ("duplicate", "Use a unique name for each body, joint and shape."),
)


MAX_MESSAGE = 400


def compact(text: str) -> str:
    """First non-empty line, at most MAX_MESSAGE characters (the log keeps everything)."""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first if len(first) <= MAX_MESSAGE else first[: MAX_MESSAGE - 3] + "..."


def _classify(exc: BaseException, phase: str) -> tuple[SignalKind, str, str]:
    """(kind, message, hint) for an exception raised in ``phase``."""
    from sdk.errors import PoseError, ValidationError

    text = compact(str(exc)) or type(exc).__name__
    typed = f"{type(exc).__name__}: {text}"
    if phase == EXPORT_PHASE:
        return SignalKind.EXPORT_ERROR, text, HINTS[SignalKind.EXPORT_ERROR]
    if isinstance(exc, PoseError):  # LoopClosureError is a PoseError
        return SignalKind.POSE, typed, HINTS[SignalKind.POSE]
    if isinstance(exc, ValidationError):
        hint = next((h for needle, h in _VALIDATION_HINTS if needle in text),
                    HINTS[SignalKind.VALIDATION])
        return SignalKind.VALIDATION, text, hint
    if isinstance(exc, ImportError):
        return SignalKind.IMPORT_ERROR, typed, HINTS[SignalKind.IMPORT_ERROR]
    if isinstance(exc, AssertionError) and phase == TESTS_PHASE:
        message = f"check failed: {exc}" if str(exc) else "check failed (assert without message)"
        return SignalKind.TEST_FAILURE, message, HINTS[SignalKind.TEST_FAILURE]
    if isinstance(exc, SystemExit):
        return (SignalKind.RUNTIME_ERROR, f"main.py called sys.exit({exc.code!r})",
                "Do not call sys.exit() or exit() in main.py.")
    return SignalKind.RUNTIME_ERROR, typed, HINTS[SignalKind.RUNTIME_ERROR]


def signal_from_exception(exc: BaseException, phase: str, workspace: Path) -> CompileSignal:
    """Turn an exception raised while compiling into one signal."""
    if isinstance(exc, SyntaxError):  # includes IndentationError
        file = _relative(exc.filename, workspace) or MAIN
        signal = _signal(SignalKind.SYNTAX_ERROR, f"{type(exc).__name__}: {exc.msg}",
                         HINTS[SignalKind.SYNTAX_ERROR], workspace, (file, exc.lineno or 1))
        return signal
    kind, message, hint = _classify(exc, phase)
    if phase == TESTS_PHASE and kind is not SignalKind.TEST_FAILURE:
        message = f"in run_tests(): {message}"
    where = locate_traceback(exc, workspace) or locate_by_name(message, workspace)
    return _signal(kind, message, hint, workspace, where)


def signal_missing_main() -> CompileSignal:
    return CompileSignal(SignalKind.MISSING_MAIN.value, "workspace/main.py does not exist",
                         HINTS[SignalKind.MISSING_MAIN])


def signal_object_model(value: object, workspace: Path) -> CompileSignal:
    if value is None:
        return CompileSignal(SignalKind.OBJECT_MODEL.value, "main.py does not define object_model",
                             HINTS[SignalKind.OBJECT_MODEL], MAIN)
    message = f"object_model is a {type(value).__name__}, expected an sdk.Assembly"
    return _signal(SignalKind.OBJECT_MODEL, message, HINTS[SignalKind.OBJECT_MODEL], workspace,
                   locate_assignment("object_model", workspace))


def signal_test_failures(failures: Iterable[str], workspace: Path) -> list[CompileSignal]:
    """``run_tests()`` may return failure messages instead of raising."""
    return [_signal(SignalKind.TEST_FAILURE, f"check failed: {text}",
                    HINTS[SignalKind.TEST_FAILURE], workspace,
                    locate_by_name(str(text), workspace) or locate_definition("run_tests",
                                                                              workspace))
            for text in failures]


def signal_timeout(timeout: float, phase: str | None, phase_seconds: float | None,
                   where: Where | None, workspace: Path) -> CompileSignal:
    phase = phase or "starting the compile worker"
    spent = f" ({phase_seconds:.1f}s in that phase)" if phase_seconds is not None else ""
    message = f"compile timed out after {timeout:g}s while {phase}{spent}"
    return _signal(SignalKind.TIMEOUT, message, HINTS[SignalKind.TIMEOUT], workspace, where)


def signal_crash(returncode: int | None, stderr: str) -> CompileSignal:
    tail = stderr.strip().splitlines()[-1] if stderr.strip() else "no error output"
    return CompileSignal(SignalKind.CRASH.value,
                         f"compile process exited with code {returncode} and no result ({tail})",
                         HINTS[SignalKind.CRASH])


# ---- repeat detection and rendering ----------------------------------------------------------


def failure_signature(signals: Sequence[CompileSignal]) -> str | None:
    """A hash of the failure set (kind, message, location); None when nothing failed."""
    failures = sorted((s.kind, s.message, s.file or "", s.line or 0)
                      for s in signals if s.severity == "failure")
    if not failures:
        return None
    return hashlib.sha256(json.dumps(failures).encode()).hexdigest()[:16]


_RULES = {
    **dict.fromkeys(
        (SignalKind.MISSING_MAIN, SignalKind.SYNTAX_ERROR, SignalKind.IMPORT_ERROR,
         SignalKind.RUNTIME_ERROR, SignalKind.OBJECT_MODEL, SignalKind.CRASH),
        "- Fix this error first: nothing else is checked until main.py runs and defines "
        "object_model.",
    ),
    SignalKind.VALIDATION: "- Fix the model structure first; local geometry tuning comes after.",
    SignalKind.POSE: "- Check joint names, limits and loops; drive the loop's input joint, "
                     "never a closure or a mimic.",
    SignalKind.TEST_FAILURE: "- Treat failed checks as design evidence: repair the model, keep "
                             "the check.",
    SignalKind.EXPORT_ERROR: "- The structure is valid; repair the geometry named in the "
                             "message.",
    SignalKind.TIMEOUT: "- Find the slow or endless step at the reported line before editing "
                        "anything else.",
    **dict.fromkeys(
        (SignalKind.OVERLAP, SignalKind.FLOATING, SignalKind.SWEPT_COLLISION,
         SignalKind.SEPARATION, SignalKind.LOOP, SignalKind.MESH_HEALTH),
        "- Check failures are design evidence: fix the geometry, joint origin, axis or limits. "
        "Use allow_* only for real, intended contact, with a concrete reason.",
    ),
}


def _section(tag: str, title: str, signals: Sequence[CompileSignal],
             hints: bool = True) -> list[str]:
    if not signals:
        return []
    lines = ["", f"<{tag}>", title]
    for s in signals:
        lines.append(f"- {s.severity.upper()} [{s.kind}] {s.message}")
        if s.location:
            lines.append(f"  at {s.location}" + (f": {s.source}" if s.source else ""))
        if hints:
            lines.append(f"  fix: {s.hint}")
    return [*lines, f"</{tag}>"]


def _rules(first: CompileSignal, repeat_count: int) -> list[str]:
    rules = [_RULES.get(SignalKind(first.kind), "- Fix the first failure first.")]
    if repeat_count >= 3:
        where = first.location or "the failing step"
        rules.append(f"- Inspect before editing again: read the code around {where}, or run "
                     "one short check of the values involved, then make one targeted change.")
    return ["", "<response_rules>", "Suggested next steps:", *rules, "</response_rules>"]


def render_signals(
    status: str,
    signals: Sequence[CompileSignal],
    *,
    success_summary: str = "",
    repeat_count: int = 0,
    cached: bool = False,
) -> str:
    """The ``<compile_signals>`` block: summary, failures, warnings, response rules."""
    failures = [s for s in signals if s.severity == "failure"]
    warnings = [s for s in signals if s.severity == "warning"]
    notes = [s for s in signals if s.severity == "note"]
    header = f"status={status} failures={len(failures)} warnings={len(warnings)}"
    header += f" allowed={len(notes)}" if notes else ""
    header += " cached=true" if cached else ""
    if failures:
        where = f" at {failures[0].location}" if failures[0].location else ""
        primary = f"Primary issue: [{failures[0].kind}]{where}: {failures[0].message}"
    else:
        primary = success_summary or "Compile passed."
    lines = ["<compile_signals>", "<summary>", header, primary]
    if repeat_count == 2:
        lines.append("This failure matches the previous compile.")
    elif repeat_count >= 3:
        lines.append(f"This is the same failure {repeat_count} times in a row.")
    lines.append("</summary>")
    lines += _section("failures", "Failures (blocking):", failures)
    lines += _section("warnings", "Warnings (non-blocking):", warnings)
    lines += _section("allowed", "Allowed by run_tests(ctx), with reasons:", notes, hints=False)
    if failures:
        lines += _rules(failures[0], repeat_count)
    lines.append("</compile_signals>")
    return "\n".join(lines)


# Public names for the worker's notice signals (step 10b).
relative_path = _relative
make_signal = _signal
