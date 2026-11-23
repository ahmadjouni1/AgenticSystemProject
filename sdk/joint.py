"""Joints between two attachments.

At joint value ``q`` the child is placed so that

    T_child = T_parent @ F_parent @ motion(q) @ inverse(F_child)

where ``F_parent`` and ``F_child`` are the attachment frames. At ``q = 0`` the two
frames coincide. ``axis`` is expressed in the parent attachment frame.

``q = 0`` is the authored rest pose, so every limit range must contain 0 (Articraft's
rule). The rest pose is then always a valid pose: rest-pose checks, collision checks
and the viewer all start from a real state of the mechanism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from sdk.errors import ValidationError
from sdk.frames import Attachment
from sdk.transforms import Vec3, axis_angle_matrix, unit


class JointType(StrEnum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


@dataclass(frozen=True)
class Mimic:
    """``value = multiplier * value(source) + offset``."""

    source: str
    multiplier: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValidationError("mimic source must be a joint name")
        multiplier, offset = float(self.multiplier), float(self.offset)
        if not (math.isfinite(multiplier) and math.isfinite(offset)):
            raise ValidationError("mimic multiplier and offset must be finite")
        if abs(multiplier) < 1e-12:
            raise ValidationError("mimic multiplier must be non-zero (use a fixed joint)")
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "offset", offset)


@dataclass(frozen=True, eq=False)
class Joint:
    name: str
    type: JointType
    parent: Attachment
    child: Attachment
    axis: Vec3 | None = None
    limits: tuple[float, float] | None = None
    mimic: Mimic | tuple[str, float, float] | None = None
    """A ``Mimic`` or a ``(source_joint, multiplier, offset)`` tuple; stored as ``Mimic``."""

    def __post_init__(self) -> None:
        try:
            kind = JointType(self.type)
        except ValueError:
            options = ", ".join(t.value for t in JointType)
            raise ValidationError(
                f"joint {self.name!r} type must be one of {options}, got {self.type!r}"
            ) from None
        object.__setattr__(self, "type", kind)
        if not isinstance(self.parent, Attachment) or not isinstance(self.child, Attachment):
            raise ValidationError(f"joint {self.name!r} ends must be body.at(...) attachments")
        if self.parent.body is self.child.body:
            body = self.parent.body.name
            raise ValidationError(f"joint {self.name!r} connects body {body!r} to itself")
        if kind is JointType.FIXED:
            if self.axis is not None or self.limits is not None or self.mimic is not None:
                raise ValidationError(f"fixed joint {self.name!r} takes no axis, limits or mimic")
            return
        if self.axis is None:
            raise ValidationError(f"{kind} joint {self.name!r} needs an axis")
        object.__setattr__(self, "axis", unit(self.axis, field=f"joint {self.name!r} axis"))
        if self.limits is not None:
            object.__setattr__(self, "limits", _limits(self.name, self.limits))
        if self.mimic is not None:
            mimic = self.mimic if isinstance(self.mimic, Mimic) else Mimic(*self.mimic)
            if mimic.source == self.name:
                raise ValidationError(f"joint {self.name!r} cannot mimic itself")
            object.__setattr__(self, "mimic", mimic)

    def __repr__(self) -> str:
        ends = f"{self.parent.body.name!r} -> {self.child.body.name!r}"
        return f"Joint({self.name!r}, {self.type.value}, {ends})"

    @property
    def movable(self) -> bool:
        return self.type is not JointType.FIXED

    @property
    def periodic(self) -> bool:
        """A revolute joint whose range covers a full turn has no hard stops."""
        if self.type is not JointType.REVOLUTE:
            return False
        return self.limits is None or self.limits[1] - self.limits[0] >= 2.0 * math.pi - 1e-9

    def within_limits(self, value: float, tolerance: float = 1e-9) -> bool:
        if self.limits is None:
            return True
        return self.limits[0] - tolerance <= value <= self.limits[1] + tolerance

    def motion(self, value: float) -> np.ndarray:
        """The 4x4 transform from the parent frame to the child frame at ``value``."""
        t = np.eye(4)
        if self.type is JointType.REVOLUTE:
            t[:3, :3] = axis_angle_matrix(self.axis, value)
        elif self.type is JointType.PRISMATIC:
            t[:3, 3] = np.asarray(self.axis) * value
        return t


def _limits(name: str, limits: object) -> tuple[float, float]:
    try:
        lower, upper = (float(v) for v in limits)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        raise ValidationError(f"joint {name!r} limits must be (lower, upper)") from None
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValidationError(f"joint {name!r} limits must be finite (use None for no limits)")
    if lower > upper:
        raise ValidationError(f"joint {name!r} lower limit {lower:g} exceeds upper {upper:g}")
    if not lower <= 0.0 <= upper:
        raise ValidationError(
            f"joint {name!r} limits ({lower:g}, {upper:g}) must contain 0, the authored rest "
            "pose; author the parts at a pose inside the range"
        )
    return (lower, upper)
