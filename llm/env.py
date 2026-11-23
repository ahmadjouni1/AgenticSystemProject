"""Load ``replica/.env`` into the environment: a tiny loader, no dependency.

- Lines are ``NAME=value``. Also accepted: ``export NAME=value``, values in single or double
  quotes, blank lines, and ``#`` comments (whole-line, or after an unquoted value).
- A variable that is already set is never overridden: the real environment wins.
- Values are never printed or logged; ``load_env`` returns only the names it set.

``.env`` is git-ignored. The compiler strips ``*_API_KEY`` / ``*_TOKEN`` variables from the
environment of model-written code (compiler/runner.py), so keys loaded here do not reach it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue  # not NAME=value: ignored (never echoed, it may hold a secret)
        name, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[name] = value
    return values


def load_env(path: str | Path | None = None) -> list[str]:
    """Set the variables of ``path`` (default ``replica/.env``) that are not set yet.
    Returns the names it set. A missing file is not an error."""
    path = Path(path) if path is not None else ENV_FILE
    if not path.is_file():
        return []
    loaded = []
    for name, value in parse_env(path.read_text(encoding="utf-8-sig")).items():
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded
