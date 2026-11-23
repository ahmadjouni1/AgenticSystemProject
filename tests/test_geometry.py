from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box as CadBox
from build123d import Cylinder as CadCylinder
from build123d import Location, Vector

from sdk import Assembly, Box, Build123dShape, Cylinder, Frame, Mesh, Sphere, ValidationError
from sdk.transforms import rpy_matrix


def test_primitive_volumes() -> None:
    assert Box((0.1, 0.2, 0.3)).to_trimesh().volume == pytest.approx(0.006)
    assert Cylinder(0.05, 0.2).to_trimesh().volume == pytest.approx(
        math.pi * 0.05**2 * 0.2, rel=0.01
    )
    assert Sphere(0.1).to_trimesh().volume == pytest.approx(4 / 3 * math.pi * 0.1**3, rel=0.02)


def test_primitives_reject_bad_sizes() -> None:
    with pytest.raises(ValidationError, match="positive"):
        Box((0.1, 0.0, 0.1))
    with pytest.raises(ValidationError, match="positive"):
        Cylinder(-1, 1)


def test_mesh_from_vertices_and_faces() -> None:
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    mesh = Mesh(vertices, faces).to_trimesh()
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(1 / 6)
    with pytest.raises(ValidationError, match="out of range"):
        Mesh(vertices, [(0, 1, 9)])
    with pytest.raises(ValidationError, match="shape"):
        Mesh([(0, 0)], faces)


def test_build123d_shape_is_tessellated() -> None:
    asm = Assembly("cad")
    body = asm.body("part")
    shape = body.add("plate", CadBox(0.1, 0.05, 0.01) - CadCylinder(0.01, 0.01))
    assert isinstance(shape.geometry, Build123dShape)
    mesh = shape.to_trimesh()
    expected = 0.1 * 0.05 * 0.01 - math.pi * 0.01**2 * 0.01
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(expected, rel=1e-3)


def test_unsupported_geometry() -> None:
    with pytest.raises(ValidationError, match="unsupported geometry"):
        Assembly("x").body("b").add("s", "not a shape")


def test_shape_placement_and_body_mesh() -> None:
    body = Assembly("x").body("b")
    body.add("one", Box((0.1, 0.1, 0.1)), xyz=(1, 0, 0))
    body.add("two", Box((0.1, 0.1, 0.1)), xyz=(0, 0, 2), rpy=(0, 0, math.pi / 4))
    mesh = body.trimesh()
    assert mesh.bounds[1][0] == pytest.approx(1.05)
    assert mesh.bounds[1][2] == pytest.approx(2.05)
    assert len(mesh.vertices) == 16


def test_frame_round_trip() -> None:
    rpy = (0.3, -0.7, 2.1)
    frame = Frame.from_matrix(Frame((1, 2, 3), rpy).matrix)
    assert frame.xyz == pytest.approx((1, 2, 3))
    assert np.allclose(rpy_matrix(frame.rpy), rpy_matrix(rpy))


def test_rpy_order_is_roll_then_pitch_then_yaw_about_fixed_axes() -> None:
    # yaw 90 deg alone turns +x into +y
    assert rpy_matrix((0, 0, math.pi / 2)) @ (1, 0, 0) == pytest.approx((0, 1, 0))
    # roll 90 then yaw 90: +z -> -y (roll) -> +x (yaw)
    assert rpy_matrix((math.pi / 2, 0, math.pi / 2)) @ (0, 0, 1) == pytest.approx((1, 0, 0))


def test_build123d_degrees_are_converted_at_the_boundary() -> None:
    body = Assembly("x").body("b")
    attached = body.at(Location((0.1, 0.2, 0.3), (0, 0, 90)))  # build123d: degrees
    assert attached.frame.xyz == pytest.approx((0.1, 0.2, 0.3))
    assert attached.frame.rpy == pytest.approx((0, 0, math.pi / 2))  # SDK: radians
    assert body.at(Vector(1, 2, 3), rpy=(0.5, 0, 0)).frame == Frame((1, 2, 3), (0.5, 0, 0))
    with pytest.raises(ValidationError, match="already has an orientation"):
        body.at(Location((0, 0, 0)), rpy=(0, 0, 1))
