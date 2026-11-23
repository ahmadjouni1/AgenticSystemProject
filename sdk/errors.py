"""Errors raised by the SDK.

Two families matter to callers:

- ``ValidationError``: the model itself is wrong (bad name, bad limits, broken tree).
- ``PoseError``: the model is fine but the requested joint values are not.
  ``LoopClosureError`` is the pose error raised when a closed loop cannot be closed.
"""

from __future__ import annotations


class SDKError(Exception):
    """Base class for every SDK error."""


class ValidationError(SDKError, ValueError):
    """The assembly, a body, a joint, or a shape is malformed."""


class PoseError(SDKError, ValueError):
    """Joint values passed to forward kinematics are invalid."""


class LoopClosureError(PoseError):
    """A loop-closure joint stays open after solving."""
