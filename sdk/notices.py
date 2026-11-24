"""Notices: things a shape helper adjusted instead of failing (step 10b).

A helper that can repair a bad argument (a rounded box whose radius is too large) does so
and calls ``notice(...)``. The compiler drains the notices after main.py has run and turns
each into a warning signal at the main.py line that called the helper. Warnings never block
a compile; they tell the model what was changed.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass

_SDK_DIR = __file__.rsplit("notices.py", 1)[0]


@dataclass(frozen=True)
class Notice:
    message: str
    hint: str
    stack: tuple[tuple[str, int], ...]  # (file, line), outermost first, SDK frames dropped


_NOTICES: list[Notice] = []


def notice(message: str, hint: str) -> None:
    """Record an adjustment made by a helper, with the caller's stack."""
    stack = tuple((frame.filename, frame.lineno or 0)
                  for frame in traceback.extract_stack()[:-1]
                  if not frame.filename.startswith(_SDK_DIR))
    _NOTICES.append(Notice(message, hint, stack))


def drain() -> list[Notice]:
    """Every notice since the last drain (and forget them)."""
    out = list(_NOTICES)
    _NOTICES.clear()
    return out
