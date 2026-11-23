"""body.at(build123d feature) and Assembly.frame_in."""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Axis, Box, Cylinder, GeomType, Plane, Pos

from sdk import Assembly, ValidationError
from sdk.transforms import apply


def z_axis(attachment) -> np.ndarray:
    return attachment.frame.matrix[:3, 2]


def test_flat_face_gives_center_and_outward_normal() -> None:
    body = Assembly("x").body("b")
    box = Box(0.2, 0.1, 0.05)
    top = body.at(box.faces().sort_by(Axis.Z)[-1])
    assert top.frame.xyz == pytest.approx((0, 0, 0.025))
    assert z_axis(top) == pytest.approx((0, 0, 1))
    side = body.at(box.faces().sort_by(Axis.X)[-1])
    assert side.frame.xyz == pytest.approx((0.1, 0, 0))
    assert z_axis(side) == pytest.approx((1, 0, 0))
    bottom = body.at(box.faces().sort_by(Axis.Z)[0])
    assert z_axis(bottom) == pytest.approx((0, 0, -1))


def test_round_face_gives_its_axis_of_symmetry() -> None:
    body = Assembly("x").body("b")
    shaft = Pos(0.3, 0.2, 0.0) * Cylinder(0.02, 0.1)
    side = shaft.faces().filter_by(GeomType.CYLINDER)[0]
    frame = body.at(side)
    assert frame.frame.xyz == pytest.approx((0.3, 0.2, 0.0), abs=1e-9)  # on the axis, mid-height
    assert abs(z_axis(frame) @ (0, 0, 1)) == pytest.approx(1.0)


def test_edges() -> None:
    body = Assembly("x").body("b")
    box = Box(0.2, 0.1, 0.05)
    # The top back edge runs along x.
    edge = [e for e in box.edges().filter_by(Axis.X) if e.center().Y > 0 and e.center().Z > 0][0]
    frame = body.at(edge)
    assert frame.frame.xyz == pytest.approx((0, 0.05, 0.025))
    assert abs(z_axis(frame) @ (1, 0, 0)) == pytest.approx(1.0)
    rim = Cylinder(0.02, 0.1).edges().filter_by(GeomType.CIRCLE).sort_by(Axis.Z)[-1]
    frame = body.at(rim)
    assert frame.frame.xyz == pytest.approx((0, 0, 0.05))
    assert abs(z_axis(frame) @ (0, 0, 1)) == pytest.approx(1.0)


def test_axis_plane_vertex_and_shape() -> None:
    body = Assembly("x").body("b")
    along_y = body.at(Axis((0.1, 0, 0), (0, 1, 0)))
    assert along_y.frame.xyz == pytest.approx((0.1, 0, 0))
    assert z_axis(along_y) == pytest.approx((0, 1, 0))
    assert z_axis(body.at(Plane.XZ)) == pytest.approx((0, -1, 0))
    corner = Box(0.2, 0.1, 0.05).vertices().sort_by(Axis.X)[-1]
    assert body.at(corner).frame.xyz[0] == pytest.approx(0.1)
    assert body.at(Pos(1, 2, 3) * Box(1, 1, 1)).frame.xyz == pytest.approx((1, 2, 3))
    with pytest.raises(ValidationError, match="already has an orientation"):
        body.at(Axis.Z, rpy=(0, 0, 1))


def test_features_of_an_offset_rotated_shape_use_its_placement() -> None:
    body = Assembly("x").body("b")
    box = Box(0.2, 0.1, 0.05)
    body.add("plate", box, xyz=(1.0, 0.0, 0.0), rpy=(0.0, 0.0, math.pi / 2))
    # Top face: lifted by the shape offset, still facing up.
    top = body.at(box.faces().sort_by(Axis.Z)[-1])
    assert top.frame.xyz == pytest.approx((1.0, 0.0, 0.025), abs=1e-12)
    assert z_axis(top) == pytest.approx((0, 0, 1), abs=1e-12)
    # The shape's +x face is turned 90 degrees: in body coordinates it faces +y.
    side = body.at(box.faces().sort_by(Axis.X)[-1])
    assert side.frame.xyz == pytest.approx((1.0, 0.1, 0.0), abs=1e-12)
    assert z_axis(side) == pytest.approx((0, 1, 0), abs=1e-12)
    # A straight edge along the shape's x runs along body y.
    edge = [e for e in box.edges().filter_by(Axis.X) if e.center().Y > 0 and e.center().Z > 0][0]
    hinge = body.at(edge)
    assert hinge.frame.xyz == pytest.approx((0.95, 0.0, 0.025), abs=1e-12)
    assert abs(z_axis(hinge) @ (0, 1, 0)) == pytest.approx(1.0)
    # An Axis cannot be traced to a shape, so shape= says whose coordinates it is in.
    along = body.at(Axis((0.1, 0, 0), (1, 0, 0)), shape="plate")
    assert along.frame.xyz == pytest.approx((1.0, 0.1, 0.0), abs=1e-12)
    assert z_axis(along) == pytest.approx((0, 1, 0), abs=1e-12)
    # A feature from an unrelated shape is read in body coordinates, as before.
    loose = body.at(Box(0.2, 0.1, 0.05).faces().sort_by(Axis.Z)[-1])
    assert loose.frame.xyz == pytest.approx((0, 0, 0.025))


def test_feature_of_a_shape_added_twice_needs_a_name() -> None:
    body = Assembly("x").body("b")
    box = Box(0.1, 0.1, 0.1)
    body.add("left", box, xyz=(-1, 0, 0))
    body.add("right", box, xyz=(1, 0, 0))
    top = box.faces().sort_by(Axis.Z)[-1]
    with pytest.raises(ValidationError, match="several shapes"):
        body.at(top)
    assert body.at(top, shape="right").frame.xyz == pytest.approx((1, 0, 0.05))


def test_mesh_shape_bounds_center_uses_its_placement() -> None:
    from sdk import BoxGeometry

    body = Assembly("x").body("b")
    mesh = BoxGeometry((0.1, 0.1, 0.1))
    body.add("block", mesh, xyz=(0, 0, 0.5))
    assert body.at(mesh).frame.xyz == pytest.approx((0, 0, 0.5))


def test_hinge_on_a_box_edge() -> None:
    """A lid hinged on the base's top back edge, with the axis read from the edge."""
    asm = Assembly("box_lid")
    base = asm.body("base")
    shape = Box(0.2, 0.1, 0.05)
    base.add("shell", shape)
    lid = asm.body("lid")
    lid.add("panel", Pos(0, 0, 0.03) * Box(0.2, 0.1, 0.01))
    edge = [e for e in shape.edges().filter_by(Axis.X) if e.center().Y > 0 and e.center().Z > 0][0]
    hinge = base.at(edge)
    asm.joint("hinge", "revolute", hinge, lid.at(hinge.frame), axis=(0, 0, 1),
              limits=(-math.pi, math.pi))
    for angle in (0.5, -1.0):
        lid_pose = asm.forward_kinematics({"hinge": angle})["lid"]
        # Points on the hinge line do not move; the front edge swings on a circle.
        assert apply(lid_pose, (0.07, 0.05, 0.025)) == pytest.approx((0.07, 0.05, 0.025))
        front = apply(lid_pose, (0.0, -0.05, 0.035))
        radius = math.hypot(0.1, 0.01)
        assert np.linalg.norm(front - (0, 0.05, 0.025)) == pytest.approx(radius)
        assert front[0] == pytest.approx(0.0)


def test_frame_in_expresses_a_frame_on_another_body() -> None:
    asm = Assembly("arm")
    base, upper = asm.body("base"), asm.body("upper")
    for body in (base, upper):
        body.add("box", Box(0.05, 0.05, 0.05))
    asm.joint("shoulder", "revolute", base.at((0, 0, 0.1), rpy=(0, 0, math.pi / 2)), upper.at(),
              axis=(0, 0, 1), limits=(-1, 1))
    tip = upper.at((0.3, 0, 0))  # 0.3 m along upper's x, which points along base's +y at rest
    on_base = asm.frame_in(tip, base)
    assert on_base.body is base
    assert on_base.frame.xyz == pytest.approx((0, 0.3, 0.1))
    assert on_base.frame.rpy == pytest.approx((0, 0, math.pi / 2))
    back = asm.frame_in(on_base, upper)
    assert back.frame.xyz == pytest.approx((0.3, 0, 0), abs=1e-12)
