"""One solve in a fresh interpreter (``solvers.base.solve_isolated`` starts it):

    python -m solvers.worker   < {"mechanism": ..., "target": ..., "constraints": ...}

Reads one JSON request from stdin and prints one JSON line as the last line of stdout:
``{"result": {...}}`` or ``{"error": "message for the model"}``. Nothing from the request is
executed; it is parsed as JSON and checked as data.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"the request is not JSON: {exc}"}))
        return 2
    from solvers.base import SolverError, solve

    noise = io.StringIO()  # anything a solver prints must not corrupt the reply line
    try:
        with contextlib.redirect_stdout(noise):
            result = solve(request.get("mechanism"), request.get("target"),
                           request.get("constraints") or {})
        reply = {"result": result.to_dict()}
    except SolverError as exc:
        reply = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not raised
        reply = {"error": f"solver error: {type(exc).__name__}: {exc}"[:500]}
    sys.stdout.write("\n" + json.dumps(reply, allow_nan=False) + "\n")
    return 0 if "result" in reply else 1


if __name__ == "__main__":
    raise SystemExit(main())
