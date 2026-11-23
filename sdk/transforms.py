"""Rigid-transform math. Meters and radians, 4x4 homogeneous matrices.

Roll-pitch-yaw follows the URDF convention: fixed axes X, then Y, then Z,
so ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from sdk.errors import ValidationError

Vec3 = tuple[float, float, float]


def vec3(value: Sequence[float], *, field: str) -> Vec3:
    """Coerce a 3-sequence to a tuple of finite floats."""
    try:
        x, y, z = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} must be three numbers, got {value!r}") from exc
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValidationError(f"{field} must be finite, got {value!r}")
    return (x, y, z)


def unit(value: Sequence[float], *, field: str) -> Vec3:
    x, y, z = vec3(value, field=field)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValidationError(f"{field} must be a non-zero vector")
    return (x / norm, y / norm, z / norm)


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rpy_from_matrix(rotation: np.ndarray) -> Vec3:
    r = np.asarray(rotation, dtype=float)
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(r[2, 0]) < 1.0 - 1e-12:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: only roll +- yaw is defined, so put it all in yaw
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return (roll, pitch, yaw)


def axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rotation by ``angle`` radians about a unit ``axis`` (Rodrigues)."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def make_transform(
    xyz: Sequence[float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    t = np.eye(4)
    if rotation is not None:
        t[:3, :3] = rotation
    t[:3, 3] = xyz
    return t


def invert(transform: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, without a general matrix inverse."""
    r = transform[:3, :3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ transform[:3, 3]
    return out


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Axis times angle (radians) of a 3x3 rotation."""
    return Rotation.from_matrix(rotation).as_rotvec()


def apply(transform: np.ndarray, point: Sequence[float]) -> np.ndarray:
    """Transform one 3D point."""
    return transform[:3, :3] @ np.asarray(point, dtype=float) + transform[:3, 3]
