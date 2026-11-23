"""A minimal URDF reader and forward kinematics, independent of the SDK (standard library
and numpy only). Used to check that exported URDFs move like the assemblies they came from.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return default if text is None else tuple(float(v) for v in text.split())


def origin_matrix(element: ET.Element | None) -> np.ndarray:
    """URDF <origin xyz rpy>: rotation R = Rz(yaw) Ry(pitch) Rx(roll), then translation."""
    xyz = _floats(None if element is None else element.get("xyz"), (0.0, 0.0, 0.0))
    roll, pitch, yaw = _floats(None if element is None else element.get("rpy"), (0.0, 0.0, 0.0))
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    t = np.eye(4)
    t[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    t[:3, 3] = xyz
    return t


@dataclass
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    limits: tuple[float, float] | None
    mimic: tuple[str, float, float] | None


@dataclass
class Urdf:
    name: str
    links: dict[str, ET.Element] = field(default_factory=dict)
    joints: dict[str, UrdfJoint] = field(default_factory=dict)

    @property
    def root(self) -> str:
        children = {j.child for j in self.joints.values()}
        roots = [name for name in self.links if name not in children]
        assert len(roots) == 1, roots
        return roots[0]

    def visual_origin(self, link: str) -> np.ndarray:
        return origin_matrix(self.links[link].find("visual/origin"))

    def joint_value(self, name: str, values: dict[str, float]) -> float:
        joint = self.joints[name]
        if joint.mimic is not None:
            source, multiplier, offset = joint.mimic
            return multiplier * self.joint_value(source, values) + offset
        return float(values.get(name, 0.0))

    def link_poses(self, values: dict[str, float]) -> dict[str, np.ndarray]:
        poses = {self.root: np.eye(4)}
        pending = list(self.joints.values())
        while pending:
            joint = next(j for j in pending if j.parent in poses)
            pending.remove(joint)
            motion = np.eye(4)
            q = self.joint_value(joint.name, values)
            if joint.type in ("revolute", "continuous"):
                k = joint.axis
                skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                motion[:3, :3] = np.eye(3) + math.sin(q) * skew + (1 - math.cos(q)) * skew @ skew
            elif joint.type == "prismatic":
                motion[:3, 3] = joint.axis * q
            poses[joint.child] = poses[joint.parent] @ joint.origin @ motion
        return poses

    def body_poses(self, values: dict[str, float]) -> dict[str, np.ndarray]:
        """Where each link's geometry (the SDK body frame) is: link pose @ visual origin."""
        return {name: pose @ self.visual_origin(name)
                for name, pose in self.link_poses(values).items()}


def read_urdf(path: str | Path) -> Urdf:
    robot = ET.parse(path).getroot()
    assert robot.tag == "robot"
    urdf = Urdf(robot.get("name", ""))
    for link in robot.findall("link"):
        urdf.links[link.get("name")] = link
    for element in robot.findall("joint"):
        limit = element.find("limit")
        mimic = element.find("mimic")
        urdf.joints[element.get("name")] = UrdfJoint(
            name=element.get("name"),
            type=element.get("type"),
            parent=element.find("parent").get("link"),
            child=element.find("child").get("link"),
            origin=origin_matrix(element.find("origin")),
            axis=np.array(_floats(None if element.find("axis") is None
                                  else element.find("axis").get("xyz"), (1.0, 0.0, 0.0))),
            limits=None if limit is None or limit.get("lower") is None
            else (float(limit.get("lower")), float(limit.get("upper"))),
            mimic=None if mimic is None else (
                mimic.get("joint"), float(mimic.get("multiplier", 1)), float(mimic.get("offset", 0))),
        )
    return urdf
